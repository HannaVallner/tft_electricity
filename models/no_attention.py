import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers import (
    TimeDistributed,
    GateAddNorm,
    GatedResidualNetwork,
)
from src.data_formatter import DataTypes, InputTypes


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN) used by TFT.
    """

    def __init__(
        self,
        hidden_dim,
        num_inputs,
        dropout_rate,
        context_dim=None,
        time_distributed=True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_inputs = num_inputs
        self.time_distributed = time_distributed

        self.flattened_grn = GatedResidualNetwork(
            input_dim=hidden_dim * num_inputs,
            hidden_dim=hidden_dim,
            output_dim=num_inputs,
            context_dim=context_dim,
            dropout_rate=dropout_rate,
            time_distributed=time_distributed,
        )

        self.single_variable_grns = nn.ModuleList(
            [
                GatedResidualNetwork(
                    input_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim,
                    context_dim=None,
                    dropout_rate=dropout_rate,
                    time_distributed=time_distributed,
                )
                for _ in range(num_inputs)
            ]
        )

    def forward(self, embedding, context=None):
        if self.time_distributed:
            batch_size, time_steps, hidden_dim, num_inputs = embedding.shape

            flattened = embedding.reshape(
                batch_size, time_steps, hidden_dim * num_inputs
            )

            sparse_weights = self.flattened_grn(flattened, context=context)
            sparse_weights = F.softmax(sparse_weights, dim=-1)
            sparse_weights = sparse_weights.unsqueeze(2)  # (B, T, 1, N)

            transformed_list = []
            for i in range(num_inputs):
                transformed = self.single_variable_grns[i](embedding[..., i])
                transformed_list.append(transformed)

            transformed_embedding = torch.stack(transformed_list, dim=-1)
            combined = torch.sum(sparse_weights * transformed_embedding, dim=-1)

            return combined, sparse_weights.squeeze(2)

        else:
            batch_size, num_inputs, hidden_dim = embedding.shape

            flattened = embedding.reshape(batch_size, num_inputs * hidden_dim)

            sparse_weights = self.flattened_grn(flattened, context=context)
            sparse_weights = F.softmax(sparse_weights, dim=-1)
            sparse_weights = sparse_weights.unsqueeze(-1)  # (B, N, 1)

            transformed_list = []
            for i in range(num_inputs):
                transformed = self.single_variable_grns[i](embedding[:, i, :])
                transformed_list.append(transformed)

            transformed_embedding = torch.stack(transformed_list, dim=1)
            combined = torch.sum(sparse_weights * transformed_embedding, dim=1)

            return combined, sparse_weights.squeeze(-1)


class TemporalFusionTransformer(nn.Module):
    """
    TFT ablation removing the temporal self-attention module while preserving
    all other architectural components (VSN, LSTM, static enrichment, gating).

    Keeps:
    - embeddings
    - static context
    - variable selection networks
    - LSTM encoder/decoder
    - quantile output head

    Removes:
    - interpretable multi-head self-attention
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

        self.real_input_positions = list(range(self.num_regular_variables))
        self.categorical_input_positions = list(
            range(self.num_regular_variables, self.input_size)
        )

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

        self.real_variable_projections = nn.ModuleList(
            [
                TimeDistributed(nn.Linear(1, self.hidden_dim))
                for _ in range(self.num_regular_variables)
            ]
        )

        self.categorical_variable_embeddings = nn.ModuleList(
            [nn.Embedding(num_classes, self.hidden_dim) for num_classes in self.category_counts]
        )

        num_static_inputs = len(self.static_regular_indices) + len(self.static_categorical_indices)
        if num_static_inputs == 0:
            raise ValueError(
                "This model expects at least one static input, e.g. categorical_id."
            )

        self.static_variable_selection = VariableSelectionNetwork(
            hidden_dim=self.hidden_dim,
            num_inputs=num_static_inputs,
            dropout_rate=self.dropout_rate,
            context_dim=None,
            time_distributed=False,
        )

        self.static_context_variable_selection = GatedResidualNetwork(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            context_dim=None,
            dropout_rate=self.dropout_rate,
            time_distributed=False,
        )

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

        self.historical_variable_selection = VariableSelectionNetwork(
            hidden_dim=self.hidden_dim,
            num_inputs=num_historical_inputs,
            dropout_rate=self.dropout_rate,
            context_dim=self.hidden_dim,
            time_distributed=True,
        )

        self.future_variable_selection = VariableSelectionNetwork(
            hidden_dim=self.hidden_dim,
            num_inputs=num_future_inputs,
            dropout_rate=self.dropout_rate,
            context_dim=self.hidden_dim,
            time_distributed=True,
        )

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

        self.static_enrichment = GatedResidualNetwork(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            context_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            time_distributed=True,
        )

        # No attention block here

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
        encoder_steps = self.num_encoder_steps

        unknown_inputs, known_combined_layer, obs_inputs, static_inputs = self._embed_inputs(
            all_inputs
        )

        historical_parts = []

        if unknown_inputs is not None:
            historical_parts.append(unknown_inputs[:, :encoder_steps, :, :])

        historical_parts.append(known_combined_layer[:, :encoder_steps, :, :])
        historical_parts.append(obs_inputs[:, :encoder_steps, :, :])

        historical_inputs = torch.cat(historical_parts, dim=-1)
        future_inputs = known_combined_layer[:, encoder_steps:, :, :]

        static_encoder, static_weights = self.static_variable_selection(static_inputs)

        static_context_variable_selection = self.static_context_variable_selection(
            static_encoder
        )
        static_context_enrichment = self.static_context_enrichment(static_encoder)
        static_context_state_h = self.static_context_state_h(static_encoder)
        static_context_state_c = self.static_context_state_c(static_encoder)

        expanded_static_context_hist = static_context_variable_selection.unsqueeze(1).expand(
            -1, encoder_steps, -1
        )
        expanded_static_context_future = static_context_variable_selection.unsqueeze(1).expand(
            -1, self.decoder_steps, -1
        )

        historical_features, historical_flags = self.historical_variable_selection(
            historical_inputs,
            context=expanded_static_context_hist,
        )

        future_features, future_flags = self.future_variable_selection(
            future_inputs,
            context=expanded_static_context_future,
        )

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

        expanded_static_context_enrichment = static_context_enrichment.unsqueeze(1).expand(
            -1, self.total_time_steps, -1
        )

        enriched = self.static_enrichment(
            temporal_feature_layer,
            context=expanded_static_context_enrichment,
        )

        # No self-attention:
        # enriched goes straight into position-wise processing
        decoder = self.positionwise_grn(enriched)

        transformer_layer, _ = self.final_gate_add_norm(
            decoder,
            temporal_feature_layer,
        )

        decoder_output = transformer_layer[:, self.num_encoder_steps:, :]
        predictions = self.output_layer(decoder_output)

        if return_attention:
            diagnostics = {
                "decoder_self_attn": None,
                "static_flags": static_weights,
                "historical_flags": historical_flags,
                "future_flags": future_flags,
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