import torch
import torch.nn as nn

from src.layers import (
    TimeDistributed,
    InterpretableMultiHeadAttention,
    GateAddNorm,
    GatedResidualNetwork,
    get_decoder_mask,
)
from src.data_formatter import DataTypes, InputTypes


class TemporalFusionTransformer(nn.Module):
    """
    TFT-style transformer-only ablation.

    Preserved from TFT:
    - rolling window setup
    - per-variable embeddings
    - causal interpretable self-attention
    - gated residual/feedforward processing
    - quantile output head

    Removed:
    - variable selection networks (VSN)
    - static context pathway
    - LSTM encoder/decoder

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

        # --------------------------------------------------
        # Column structure
        # --------------------------------------------------
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

        self.static_regular_indices = [
            i for i, (_, _, role) in enumerate(self.real_inputs)
            if role == InputTypes.STATIC_INPUT
        ]

        self.static_categorical_indices = [
            i for i, (_, _, role) in enumerate(self.categorical_inputs)
            if role == InputTypes.STATIC_INPUT
        ]

        self.known_regular_indices = [
            i for i, (_, _, role) in enumerate(self.real_inputs)
            if role == InputTypes.KNOWN_INPUT
        ]

        self.known_categorical_indices = [
            i for i, (_, _, role) in enumerate(self.categorical_inputs)
            if role == InputTypes.KNOWN_INPUT
        ]

        self.unknown_regular_indices = [
            i for i in range(self.num_regular_variables)
            if i not in self.known_regular_indices
            and i not in self.obs_real_indices
            and i not in self.static_regular_indices
        ]

        self.unknown_categorical_indices = [
            i for i in range(self.num_categorical_variables)
            if i not in self.known_categorical_indices
            and i not in self.static_categorical_indices
        ]

        # --------------------------------------------------
        # Category counts for embeddings
        # --------------------------------------------------
        category_counts = formatter.num_classes_per_cat_input
        if category_counts is None:
            raise ValueError(
                "formatter.num_classes_per_cat_input is None. "
                "Call formatter.split_data(...) or formatter.set_scalers(...) first."
            )

        self.category_counts = category_counts

        # --------------------------------------------------
        # Per-variable embeddings
        #
        # Keep TFT-style input handling:
        # - one projection per real variable
        # - one embedding per categorical variable
        # --------------------------------------------------
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

        # --------------------------------------------------
        # Variable mixing without VSN
        #
        # We preserve embeddings, but instead of VSN we concatenate all
        # variable embeddings and project them back to hidden_dim.
        # --------------------------------------------------
        self.num_static_inputs = (
            len(self.static_regular_indices)
            + len(self.static_categorical_indices)
        )

        self.num_historical_inputs = (
            len(self.unknown_regular_indices)
            + len(self.unknown_categorical_indices)
            + len(self.known_regular_indices)
            + len(self.known_categorical_indices)
            + len(self.obs_real_indices)
            + self.num_static_inputs
        )

        self.num_future_inputs = (
            len(self.known_regular_indices)
            + len(self.known_categorical_indices)
            + self.num_static_inputs
        )

        self.historical_input_mixer = TimeDistributed(
            nn.Linear(self.hidden_dim * self.num_historical_inputs, self.hidden_dim)
        )

        self.future_input_mixer = TimeDistributed(
            nn.Linear(self.hidden_dim * self.num_future_inputs, self.hidden_dim)
        )

        # --------------------------------------------------
        # Transformer block
        # --------------------------------------------------
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

        # --------------------------------------------------
        # Output layer
        # --------------------------------------------------
        self.output_layer = TimeDistributed(
            nn.Linear(self.hidden_dim, self.output_size * len(self.quantiles))
        )

    def _split_inputs(self, all_inputs):
        """
        Split raw inputs into real and categorical components.

        Input:
            (B, T, input_size)

        Returns:
            regular_inputs:     (B, T, num_regular_variables)
            categorical_inputs: (B, T, num_categorical_variables)
        """
        regular_inputs = all_inputs[:, :, :self.num_regular_variables]
        categorical_inputs = all_inputs[:, :, self.num_regular_variables:]
        return regular_inputs, categorical_inputs

    def _embed_inputs(self, all_inputs):
        """
        Apply per-variable embeddings and separate them by role.

        Returns:
            unknown_inputs: (B, T, H, N_unknown) or None
            known_inputs:   (B, T, H, N_known)
            obs_inputs:     (B, T, H, N_obs)
            static_inputs:  (B, H, N_static) or None
        """
        regular_inputs, categorical_inputs = self._split_inputs(all_inputs)

        real_embeddings = []
        for i in range(self.num_regular_variables):
            emb = self.real_variable_projections[i](regular_inputs[:, :, i:i + 1])  # (B, T, H)
            real_embeddings.append(emb)

        categorical_embeddings = []
        for i in range(self.num_categorical_variables):
            cat_values = categorical_inputs[:, :, i].long()
            emb = self.categorical_variable_embeddings[i](cat_values)  # (B, T, H)
            categorical_embeddings.append(emb)

        static_inputs_list = []

        for i in self.static_regular_indices:
            static_inputs_list.append(real_embeddings[i][:, 0, :])  # (B, H)

        for i in self.static_categorical_indices:
            static_inputs_list.append(categorical_embeddings[i][:, 0, :])  # (B, H)

        if len(static_inputs_list) > 0:
            static_inputs = torch.stack(static_inputs_list, dim=-1)  # (B, H, N_static)
        else:
            static_inputs = None

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
            known_inputs_list.append(real_embeddings[i])

        for i in self.known_categorical_indices:
            known_inputs_list.append(categorical_embeddings[i])

        known_inputs = torch.stack(known_inputs_list, dim=-1)  # (B, T, H, N_known)

        return unknown_inputs, known_inputs, obs_inputs, static_inputs

    def forward(self, all_inputs, return_attention=False):
        """
        Forward pass.

        Parameters
        ----------
        all_inputs:
            (batch, total_time_steps, input_size)

        return_attention:
            If True, return diagnostics too.

        Returns
        -------
        predictions:
            (batch, decoder_steps, output_size * num_quantiles)
        """
        # --------------------------------------------------
        # Step 1: embed inputs and split by role
        # --------------------------------------------------
        unknown_inputs, known_inputs, obs_inputs, static_inputs = self._embed_inputs(all_inputs)

        encoder_steps = self.num_encoder_steps

        # --------------------------------------------------
        # Step 2: build historical and future inputs separately
        # historical: unknown + known + observed target
        # future: known only
        # --------------------------------------------------
        historical_parts = []
        future_parts = []

        if unknown_inputs is not None:
            historical_parts.append(unknown_inputs[:, :encoder_steps, :, :])

        historical_parts.append(known_inputs[:, :encoder_steps, :, :])
        historical_parts.append(obs_inputs[:, :encoder_steps, :, :])
        future_parts.append(known_inputs[:, encoder_steps:, :, :])

        if static_inputs is not None:
            static_hist = static_inputs.unsqueeze(1).expand(-1, encoder_steps, -1, -1)       # (B, enc, H, N_static)
            static_future = static_inputs.unsqueeze(1).expand(-1, self.decoder_steps, -1, -1)  # (B, dec, H, N_static)

            historical_parts.append(static_hist)
            future_parts.append(static_future)

        historical_inputs = torch.cat(historical_parts, dim=-1)   # (B, enc, H, N_hist)
        future_inputs = torch.cat(future_parts, dim=-1)           # (B, dec, H, N_future)

        # --------------------------------------------------
        # Step 3: mix historical and future variables separately
        # --------------------------------------------------
        batch_size, hist_steps, hidden_dim, num_hist_vars = historical_inputs.shape
        historical_flat = historical_inputs.reshape(
            batch_size, hist_steps, hidden_dim * num_hist_vars
        )
        historical_features = self.historical_input_mixer(historical_flat)  # (B, enc, H)

        batch_size, fut_steps, hidden_dim, num_future_vars = future_inputs.shape
        future_flat = future_inputs.reshape(
            batch_size, fut_steps, hidden_dim * num_future_vars
        )
        future_features = self.future_input_mixer(future_flat)  # (B, dec, H)

        temporal_features = torch.cat([historical_features, future_features], dim=1)  # (B, T, H)

        # --------------------------------------------------
        # Step 4: causal self-attention
        # --------------------------------------------------
        mask = get_decoder_mask(self.total_time_steps, device=temporal_features.device)

        attention_out, self_attention = self.self_attention(
            temporal_features,
            temporal_features,
            temporal_features,
            mask=mask,
        )

        x, _ = self.post_attention_gate_add_norm(attention_out, temporal_features)

        # --------------------------------------------------
        # Step 5: position-wise TFT feedforward block
        # --------------------------------------------------
        decoder = self.positionwise_grn(x)
        transformer_layer, _ = self.final_gate_add_norm(decoder, x)

        # --------------------------------------------------
        # Step 6: keep decoder horizon only
        # --------------------------------------------------
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