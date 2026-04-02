import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import (
    TimeDistributed,
    GateAddNorm,
    GatedResidualNetwork,
    InterpretableMultiHeadAttention,
    get_decoder_mask,
)
from data_formatter import DataTypes, InputTypes


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN) used by TFT.

    Static case:
        input:  (batch, num_inputs, hidden_dim)
        output: (batch, hidden_dim)

    Temporal case:
        input:  (batch, time, hidden_dim, num_inputs)
        output: (batch, time, hidden_dim)
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
            # embedding: (B, T, H, N)
            batch_size, time_steps, hidden_dim, num_inputs = embedding.shape

            flattened = embedding.reshape(
                batch_size, time_steps, hidden_dim * num_inputs
            )

            sparse_weights = self.flattened_grn(flattened, context=context)
            sparse_weights = F.softmax(sparse_weights, dim=-1)  # (B, T, N)
            sparse_weights = sparse_weights.unsqueeze(2)        # (B, T, 1, N)

            transformed_list = []
            for i in range(num_inputs):
                transformed = self.single_variable_grns[i](embedding[..., i])  # (B, T, H)
                transformed_list.append(transformed)

            transformed_embedding = torch.stack(transformed_list, dim=-1)  # (B, T, H, N)
            combined = torch.sum(sparse_weights * transformed_embedding, dim=-1)  # (B, T, H)

            return combined, sparse_weights.squeeze(2)

        else:
            # embedding: (B, N, H)
            batch_size, num_inputs, hidden_dim = embedding.shape

            flattened = embedding.reshape(batch_size, num_inputs * hidden_dim)

            sparse_weights = self.flattened_grn(flattened, context=context)
            sparse_weights = F.softmax(sparse_weights, dim=-1)  # (B, N)
            sparse_weights = sparse_weights.unsqueeze(-1)       # (B, N, 1)

            transformed_list = []
            for i in range(num_inputs):
                transformed = self.single_variable_grns[i](embedding[:, i, :])  # (B, H)
                transformed_list.append(transformed)

            transformed_embedding = torch.stack(transformed_list, dim=1)  # (B, N, H)
            combined = torch.sum(sparse_weights * transformed_embedding, dim=1)  # (B, H)

            return combined, sparse_weights.squeeze(-1)


class TemporalFusionTransformerNoLSTM(nn.Module):
    """
    TFT-style model with the LSTM block removed.

    Kept:
    - rolling windows
    - static variable selection
    - temporal variable selection
    - static enrichment
    - interpretable self-attention
    - quantile outputs (0.1, 0.5, 0.9)

    Removed:
    - encoder LSTM
    - decoder LSTM

    Input:
        (batch, total_time_steps, input_size)

    Output:
        (batch, decoder_steps, output_size * num_quantiles)
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
            [
                nn.Embedding(num_classes, self.hidden_dim)
                for num_classes in self.category_counts
            ]
        )

        num_static_inputs = len(self.static_regular_indices) + len(self.static_categorical_indices)
        if num_static_inputs == 0:
            raise ValueError(
                "TFT expects at least one static input. "
                "For electricity this should usually be categorical_id."
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

        # Kept for closeness to baseline architecture, though not used for recurrence now.
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

        # This replaces the old "post-LSTM residual path" usage.
        self.post_sequence_gate_add_norm = GateAddNorm(
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
            real_embeddings.append(emb)  # (B, T, H)

        categorical_embeddings = []
        for i in range(self.num_categorical_variables):
            cat_values = categorical_inputs[:, :, i].long()
            emb = self.categorical_variable_embeddings[i](cat_values)  # (B, T, H)
            categorical_embeddings.append(emb)

        static_inputs_list = []

        for i in self.static_regular_indices:
            static_inputs_list.append(real_embeddings[i][:, 0, :])

        for i in self.static_categorical_indices:
            static_inputs_list.append(categorical_embeddings[i][:, 0, :])

        static_inputs = torch.stack(static_inputs_list, dim=1)  # (B, N_static, H)

        obs_inputs_list = [real_embeddings[i] for i in self.obs_real_indices]
        obs_inputs = torch.stack(obs_inputs_list, dim=-1)  # (B, T, H, N_obs)

        unknown_inputs_list = []

        for i in self.unknown_regular_indices:
            unknown_inputs_list.append(real_embeddings[i])

        for i in self.unknown_categorical_indices:
            unknown_inputs_list.append(categorical_embeddings[i])

        if len(unknown_inputs_list) > 0:
            unknown_inputs = torch.stack(unknown_inputs_list, dim=-1)  # (B, T, H, N_unknown)
        else:
            unknown_inputs = None

        known_inputs_list = []

        for i in self.known_regular_indices:
            if i not in self.static_regular_indices:
                known_inputs_list.append(real_embeddings[i])

        for i in self.known_categorical_indices:
            if i not in self.static_categorical_indices:
                known_inputs_list.append(categorical_embeddings[i])

        known_combined_layer = torch.stack(known_inputs_list, dim=-1)  # (B, T, H, N_known)

        return unknown_inputs, known_combined_layer, obs_inputs, static_inputs

    def forward(self, all_inputs, return_attention=False):
        batch_size = all_inputs.size(0)

        unknown_inputs, known_combined_layer, obs_inputs, static_inputs = self._embed_inputs(all_inputs)

        encoder_steps = self.num_encoder_steps

        historical_parts = []
        if unknown_inputs is not None:
            historical_parts.append(unknown_inputs[:, :encoder_steps, :, :])

        historical_parts.append(known_combined_layer[:, :encoder_steps, :, :])
        historical_parts.append(obs_inputs[:, :encoder_steps, :, :])

        historical_inputs = torch.cat(historical_parts, dim=-1)   # (B, enc, H, N_hist)
        future_inputs = known_combined_layer[:, encoder_steps:, :, :]  # (B, dec, H, N_future)

        static_encoder, static_weights = self.static_variable_selection(static_inputs)

        static_context_variable_selection = self.static_context_variable_selection(static_encoder)
        static_context_enrichment = self.static_context_enrichment(static_encoder)

        # Kept for closeness with baseline, though unused downstream for recurrence.
        _ = self.static_context_state_h(static_encoder)
        _ = self.static_context_state_c(static_encoder)

        expanded_static_context_hist = static_context_variable_selection.unsqueeze(1).expand(
            -1, encoder_steps, -1
        )
        expanded_static_context_future = static_context_variable_selection.unsqueeze(1).expand(
            -1, self.decoder_steps, -1
        )

        historical_features, historical_flags = self.historical_variable_selection(
            historical_inputs,
            context=expanded_static_context_hist,
        )  # (B, enc, H)

        future_features, future_flags = self.future_variable_selection(
            future_inputs,
            context=expanded_static_context_future,
        )  # (B, dec, H)

        # No LSTM here: just combine selected temporal features directly.
        sequence_layer = torch.cat([historical_features, future_features], dim=1)  # (B, total, H)
        input_embeddings = torch.cat([historical_features, future_features], dim=1)

        temporal_feature_layer, _ = self.post_sequence_gate_add_norm(
            sequence_layer,
            input_embeddings,
        )

        expanded_static_context_enrichment = static_context_enrichment.unsqueeze(1).expand(
            -1, self.total_time_steps, -1
        )

        enriched = self.static_enrichment(
            temporal_feature_layer,
            context=expanded_static_context_enrichment,
        )

        mask = get_decoder_mask(self.total_time_steps, device=enriched.device)

        attention_out, self_attention = self.self_attention(
            enriched,
            enriched,
            enriched,
            mask=mask,
        )

        x, _ = self.post_attention_gate_add_norm(attention_out, enriched)

        decoder = self.positionwise_grn(x)
        transformer_layer, _ = self.final_gate_add_norm(decoder, temporal_feature_layer)

        decoder_output = transformer_layer[:, self.num_encoder_steps:, :]  # (B, dec, H)
        predictions = self.output_layer(decoder_output)  # (B, dec, output_size * n_quantiles)

        if return_attention:
            diagnostics = {
                "decoder_self_attn": self_attention,
                "static_flags": static_weights,
                "historical_flags": historical_flags,
                "future_flags": future_flags,
            }
            return predictions, diagnostics

        return predictions


def quantile_loss(y_true, y_pred, quantiles=(0.1, 0.5, 0.9)):
    """
    Multi-quantile loss.

    y_true:
        (batch, decoder_steps, output_size)

    y_pred:
        (batch, decoder_steps, output_size * num_quantiles)
    """
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


# Optional alias so the rest of the code can keep importing TemporalFusionTransformer
TemporalFusionTransformer = TemporalFusionTransformerNoLSTM