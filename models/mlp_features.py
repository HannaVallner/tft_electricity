import torch
import torch.nn as nn

from src.layers import (
    TimeDistributed,
    GateAddNorm,
    GatedResidualNetwork,
    InterpretableMultiHeadAttention,
    get_decoder_mask,
)
from src.data_formatter import DataTypes, InputTypes


class MLPFeatureBlock(nn.Module):
    """
    Replaces variable selection with an MLP over concatenated feature embeddings.

    Static case:
        input:  (B, N, H)
        output: (B, H)

    Temporal case:
        input:  (B, T, H, N)
        output: (B, T, H)
    """

    def __init__(self, hidden_dim, num_inputs, dropout_rate, time_distributed=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_inputs = num_inputs
        self.time_distributed = time_distributed

        input_dim = hidden_dim * num_inputs

        if time_distributed:
            self.net = nn.Sequential(
                TimeDistributed(nn.Linear(input_dim, hidden_dim)),
                nn.ReLU(),
                TimeDistributed(nn.Dropout(dropout_rate)),
                TimeDistributed(nn.Linear(hidden_dim, hidden_dim)),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim),
            )

    def forward(self, x):
        if self.time_distributed:
            # x: (B, T, H, N)
            b, t, h, n = x.shape
            x = x.reshape(b, t, h * n)
            return self.net(x)  # (B, T, H)
        else:
            # x: (B, N, H)
            b, n, h = x.shape
            x = x.reshape(b, n * h)
            return self.net(x)  # (B, H)


class TemporalFusionTransformer(nn.Module):
    """
    Full TFT architecture where the feature-selection / feature-fusion stage
    is replaced by MLP feature blocks instead of Variable Selection Networks.
    """

    def __init__(self, formatter):
        super().__init__()

        self.formatter = formatter

        fixed_params = formatter.get_fixed_params()
        model_params = formatter.get_default_model_params()

        self.total_time_steps = fixed_params["total_time_steps"]
        self.num_encoder_steps = fixed_params["num_encoder_steps"]
        self.decoder_steps = self.total_time_steps - self.num_encoder_steps

        self.hidden_dim = model_params["hidden_layer_size"]
        self.dropout_rate = model_params["dropout_rate"]
        self.num_heads = model_params["num_heads"]
        self.quantiles = [0.1, 0.5, 0.9]

        self.column_definition = formatter.get_column_definition()

        self.input_definition = [
            tup for tup in self.column_definition
            if tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]

        self.input_columns = [name for name, _, _ in self.input_definition]

        self.real_inputs = [
            tup for tup in self.input_definition
            if tup[1] == DataTypes.REAL_VALUED
        ]

        self.categorical_inputs = [
            tup for tup in self.input_definition
            if tup[1] == DataTypes.CATEGORICAL
        ]

        self.num_regular_variables = len(self.real_inputs)
        self.num_categorical_variables = len(self.categorical_inputs)
        self.input_size = len(self.input_definition)

        self.output_size = len([
            tup for tup in self.input_definition
            if tup[2] == InputTypes.TARGET
        ])

        self.obs_real_indices = [
            i for i, (_, _, role) in enumerate(self.real_inputs)
            if role == InputTypes.TARGET
        ]

        self.known_regular_indices = [
            i for i, (_, _, role) in enumerate(self.real_inputs)
            if role in {InputTypes.KNOWN_INPUT, InputTypes.STATIC_INPUT}
        ]

        self.known_categorical_indices = [
            i for i, (_, _, role) in enumerate(self.categorical_inputs)
            if role in {InputTypes.KNOWN_INPUT, InputTypes.STATIC_INPUT}
        ]

        self.static_regular_indices = [
            i for i, (_, _, role) in enumerate(self.real_inputs)
            if role == InputTypes.STATIC_INPUT
        ]

        self.static_categorical_indices = [
            i for i, (_, _, role) in enumerate(self.categorical_inputs)
            if role == InputTypes.STATIC_INPUT
        ]

        self.unknown_regular_indices = [
            i for i in range(self.num_regular_variables)
            if i not in self.known_regular_indices and i not in self.obs_real_indices
        ]

        self.unknown_categorical_indices = [
            i for i in range(self.num_categorical_variables)
            if i not in self.known_categorical_indices
        ]

        category_counts = formatter.num_classes_per_cat_input
        if category_counts is None:
            raise ValueError(
                "formatter.num_classes_per_cat_input is None. "
                "Call formatter.split_data(...) or formatter.set_scalers(...) first."
            )

        self.category_counts = category_counts

        # Real-valued variable projections
        self.real_variable_projections = nn.ModuleList(
            [TimeDistributed(nn.Linear(1, self.hidden_dim)) for _ in range(self.num_regular_variables)]
        )

        # Categorical embeddings
        self.categorical_variable_embeddings = nn.ModuleList(
            [nn.Embedding(num_classes, self.hidden_dim) for num_classes in self.category_counts]
        )

        # Counts
        num_static_inputs = len(self.static_regular_indices) + len(self.static_categorical_indices)
        if num_static_inputs == 0:
            raise ValueError("Expected at least one static input.")

        num_historical_inputs = (
            len(self.unknown_regular_indices)
            + len(self.unknown_categorical_indices)
            + len([i for i in self.known_regular_indices if i not in self.static_regular_indices])
            + len([i for i in self.known_categorical_indices if i not in self.static_categorical_indices])
            + len(self.obs_real_indices)
        )

        num_future_inputs = (
            len([i for i in self.known_regular_indices if i not in self.static_regular_indices])
            + len([i for i in self.known_categorical_indices if i not in self.static_categorical_indices])
        )

        # MLP feature blocks replacing VSNs
        self.static_feature_block = MLPFeatureBlock(
            hidden_dim=self.hidden_dim,
            num_inputs=num_static_inputs,
            dropout_rate=self.dropout_rate,
            time_distributed=False,
        )

        self.historical_feature_block = MLPFeatureBlock(
            hidden_dim=self.hidden_dim,
            num_inputs=num_historical_inputs,
            dropout_rate=self.dropout_rate,
            time_distributed=True,
        )

        self.future_feature_block = MLPFeatureBlock(
            hidden_dim=self.hidden_dim,
            num_inputs=num_future_inputs,
            dropout_rate=self.dropout_rate,
            time_distributed=True,
        )

        # Static context generators
        self.static_context_enrichment = GatedResidualNetwork(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            context_dim=None,
            dropout_rate=self.dropout_rate,
            time_distributed=False,
        )

        self.static_context_state_h = GatedResidualNetwork(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            context_dim=None,
            dropout_rate=self.dropout_rate,
            time_distributed=False,
        )

        self.static_context_state_c = GatedResidualNetwork(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            context_dim=None,
            dropout_rate=self.dropout_rate,
            time_distributed=False,
        )

        # LSTM encoder/decoder
        self.history_lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )

        self.future_lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )

        self.post_lstm_gate_add_norm = GateAddNorm(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            time_distributed=True,
        )

        # Static enrichment
        self.static_enrichment = GatedResidualNetwork(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            context_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            time_distributed=True,
        )

        # Self-attention
        self.self_attention = InterpretableMultiHeadAttention(
            n_head=self.num_heads,
            d_model=self.hidden_dim,
            dropout_rate=self.dropout_rate,
        )

        self.post_attention_gate_add_norm = GateAddNorm(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            time_distributed=True,
        )

        # Position-wise processing
        self.positionwise_grn = GatedResidualNetwork(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            context_dim=None,
            dropout_rate=self.dropout_rate,
            time_distributed=True,
        )

        self.final_gate_add_norm = GateAddNorm(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            dropout_rate=None,
            time_distributed=True,
        )

        self.output_layer = TimeDistributed(
            nn.Linear(self.hidden_dim, self.output_size * len(self.quantiles))
        )

    def _split_inputs(self, all_inputs):
        regular_inputs = all_inputs[:, :, :self.num_regular_variables]
        categorical_inputs = all_inputs[:, :, self.num_regular_variables:]
        return regular_inputs, categorical_inputs

    def _embed_inputs(self, all_inputs):
        regular_inputs, categorical_inputs = self._split_inputs(all_inputs)

        real_embeddings = []
        for i in range(self.num_regular_variables):
            emb = self.real_variable_projections[i](regular_inputs[:, :, i:i + 1])
            real_embeddings.append(emb)

        categorical_embeddings = []
        for i in range(self.num_categorical_variables):
            cat_values = categorical_inputs[:, :, i].long()
            emb = self.categorical_variable_embeddings[i](cat_values)
            categorical_embeddings.append(emb)

        static_inputs_list = []
        for i in self.static_regular_indices:
            static_inputs_list.append(real_embeddings[i][:, 0, :])
        for i in self.static_categorical_indices:
            static_inputs_list.append(categorical_embeddings[i][:, 0, :])
        static_inputs = torch.stack(static_inputs_list, dim=1)

        obs_inputs_list = [real_embeddings[i] for i in self.obs_real_indices]
        obs_inputs = torch.stack(obs_inputs_list, dim=-1)

        unknown_inputs_list = []
        for i in self.unknown_regular_indices:
            unknown_inputs_list.append(real_embeddings[i])
        for i in self.unknown_categorical_indices:
            unknown_inputs_list.append(categorical_embeddings[i])

        if len(unknown_inputs_list) > 0:
            unknown_inputs = torch.stack(unknown_inputs_list, dim=-1)
        else:
            unknown_inputs = None

        known_inputs_list = []
        for i in self.known_regular_indices:
            if i not in self.static_regular_indices:
                known_inputs_list.append(real_embeddings[i])
        for i in self.known_categorical_indices:
            if i not in self.static_categorical_indices:
                known_inputs_list.append(categorical_embeddings[i])

        known_combined_layer = torch.stack(known_inputs_list, dim=-1)

        return unknown_inputs, known_combined_layer, obs_inputs, static_inputs

    def forward(self, all_inputs, return_attention=False):
        unknown_inputs, known_combined_layer, obs_inputs, static_inputs = self._embed_inputs(all_inputs)

        encoder_steps = self.num_encoder_steps

        historical_parts = []
        if unknown_inputs is not None:
            historical_parts.append(unknown_inputs[:, :encoder_steps, :, :])

        historical_parts.append(known_combined_layer[:, :encoder_steps, :, :])
        historical_parts.append(obs_inputs[:, :encoder_steps, :, :])

        historical_inputs = torch.cat(historical_parts, dim=-1)
        future_inputs = known_combined_layer[:, encoder_steps:, :, :]

        # MLP-based feature fusion
        static_encoder = self.static_feature_block(static_inputs)
        historical_features = self.historical_feature_block(historical_inputs)
        future_features = self.future_feature_block(future_inputs)

        # Static contexts
        static_context_enrichment = self.static_context_enrichment(static_encoder)
        static_context_state_h = self.static_context_state_h(static_encoder)
        static_context_state_c = self.static_context_state_c(static_encoder)

        # LSTM
        init_h = static_context_state_h.unsqueeze(0)
        init_c = static_context_state_c.unsqueeze(0)

        history_lstm, (state_h, state_c) = self.history_lstm(
            historical_features,
            (init_h, init_c),
        )

        future_lstm, _ = self.future_lstm(
            future_features,
            (state_h, state_c),
        )

        lstm_layer = torch.cat([history_lstm, future_lstm], dim=1)
        input_embeddings = torch.cat([historical_features, future_features], dim=1)

        temporal_feature_layer, _ = self.post_lstm_gate_add_norm(
            lstm_layer,
            input_embeddings,
        )

        # Static enrichment
        expanded_static_context_enrichment = static_context_enrichment.unsqueeze(1).expand(
            -1, self.total_time_steps, -1
        )

        enriched = self.static_enrichment(
            temporal_feature_layer,
            context=expanded_static_context_enrichment,
        )

        # Self-attention
        mask = get_decoder_mask(self.total_time_steps, device=enriched.device)

        attention_out, self_attention = self.self_attention(
            enriched,
            enriched,
            enriched,
            mask=mask,
        )

        x, _ = self.post_attention_gate_add_norm(attention_out, enriched)

        # Position-wise processing
        decoder = self.positionwise_grn(x)
        transformer_layer, _ = self.final_gate_add_norm(decoder, temporal_feature_layer)

        decoder_output = transformer_layer[:, self.num_encoder_steps:, :]
        predictions = self.output_layer(decoder_output)

        if return_attention:
            diagnostics = {
                "decoder_self_attn": self_attention,
                "static_flags": None,
                "historical_flags": None,
                "future_flags": None,
            }
            return predictions, diagnostics
        
        return predictions



def quantile_loss(y_true, y_pred, quantiles=(0.1, 0.5, 0.9)):
    output_size = y_true.size(-1)
    losses = []

    for i, q in enumerate(quantiles):
        start = i * output_size
        end = (i + 1) * output_size

        pred_q = y_pred[:, :, start:end]
        errors = y_true - pred_q

        loss_q = torch.max((q - 1) * errors, q * errors)
        losses.append(loss_q)

    return torch.mean(torch.cat(losses, dim=-1))