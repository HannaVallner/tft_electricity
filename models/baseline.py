import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers import (
    TimeDistributed,
    GateAddNorm,
    GatedResidualNetwork,
    InterpretableMultiHeadAttention,
    get_decoder_mask,
)
from src.data_formatter import DataTypes, InputTypes


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN) used by TFT.

    This block takes a set of variable embeddings and learns:
    1. a softmax weight for each variable
    2. a transformed version of each variable
    3. a weighted combination of those transformed variables

    It is used in two places:
    - static variable selection
    - temporal variable selection

    Shapes
    ------
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

        # This GRN produces the variable-selection weights
        self.flattened_grn = GatedResidualNetwork(
            input_dim=hidden_dim * num_inputs,
            hidden_dim=hidden_dim,
            output_dim=num_inputs,
            context_dim=context_dim,
            dropout_rate=dropout_rate,
            time_distributed=time_distributed,
        )

        # One GRN per input variable
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
        """
        Parameters
        ----------
        embedding:
            Static case:
                (batch, num_inputs, hidden_dim)

            Temporal case:
                (batch, time, hidden_dim, num_inputs)

        context:
            Optional context vector.

            Static case:
                (batch, hidden_dim)

            Temporal case:
                (batch, time, hidden_dim)

        Returns
        -------
        combined:
            Static case:   (batch, hidden_dim)
            Temporal case: (batch, time, hidden_dim)

        sparse_weights:
            Static case:   (batch, num_inputs)
            Temporal case: (batch, time, num_inputs)
        """
        if self.time_distributed:
            # ----------------------------------------------------------
            # Temporal case
            # embedding shape: (batch, time, hidden_dim, num_inputs)
            # ----------------------------------------------------------
            batch_size, time_steps, hidden_dim, num_inputs = embedding.shape

            # Flatten variable dimension into the feature dimension
            flattened = embedding.reshape(batch_size, time_steps, hidden_dim * num_inputs)

            # Variable-selection weights
            sparse_weights = self.flattened_grn(flattened, context=context)
            sparse_weights = F.softmax(sparse_weights, dim=-1)  # (B, T, N)
            sparse_weights = sparse_weights.unsqueeze(2)        # (B, T, 1, N)

            # Transform each variable separately
            transformed_list = []
            for i in range(num_inputs):
                transformed = self.single_variable_grns[i](embedding[..., i])  # (B, T, H)
                transformed_list.append(transformed)

            transformed_embedding = torch.stack(transformed_list, dim=-1)  # (B, T, H, N)

            # Weighted sum across variables
            combined = torch.sum(sparse_weights * transformed_embedding, dim=-1)  # (B, T, H)

            return combined, sparse_weights.squeeze(2)

        else:
            # ----------------------------------------------------------
            # Static case
            # embedding shape: (batch, num_inputs, hidden_dim)
            # ----------------------------------------------------------
            batch_size, num_inputs, hidden_dim = embedding.shape

            flattened = embedding.reshape(batch_size, num_inputs * hidden_dim)

            sparse_weights = self.flattened_grn(flattened, context=context)
            sparse_weights = F.softmax(sparse_weights, dim=-1)   # (B, N)
            sparse_weights = sparse_weights.unsqueeze(-1)        # (B, N, 1)

            transformed_list = []
            for i in range(num_inputs):
                transformed = self.single_variable_grns[i](embedding[:, i, :])  # (B, H)
                transformed_list.append(transformed)

            transformed_embedding = torch.stack(transformed_list, dim=1)  # (B, N, H)

            combined = torch.sum(sparse_weights * transformed_embedding, dim=1)  # (B, H)

            return combined, sparse_weights.squeeze(-1)


class TemporalFusionTransformer(nn.Module):
    """
    PyTorch implementation of Temporal Fusion Transformer (TFT).

    Important assumptions
    ---------------------
    - The input tensor has shape:
          (batch, total_time_steps, input_size)

    - Input columns are ordered exactly like the formatter's
      get_column_definition(), excluding ID and TIME columns.

    - Categorical columns are already integer-encoded by the formatter.

    - This model outputs quantile forecasts:
          shape = (batch, decoder_steps, output_size * num_quantiles)

      For electricity:
          output_size = 1
          num_quantiles = 3
          decoder_steps = 24
    """

    def __init__(self, formatter):
        super().__init__()

        self.formatter = formatter

        # --------------------------------------------------------------
        # Fixed params from the formatter
        # --------------------------------------------------------------
        fixed_params = formatter.get_fixed_params()
        model_params = formatter.get_default_model_params()

        self.total_time_steps = fixed_params["total_time_steps"]
        self.num_encoder_steps = fixed_params["num_encoder_steps"]
        self.decoder_steps = self.total_time_steps - self.num_encoder_steps

        self.hidden_dim = model_params["hidden_layer_size"]
        self.dropout_rate = model_params["dropout_rate"]
        self.num_heads = model_params["num_heads"]
        self.quantiles = [0.1, 0.5, 0.9]

        # --------------------------------------------------------------
        # Column information from the formatter
        # --------------------------------------------------------------
        self.column_definition = formatter.get_column_definition()

        # Remove special ID and TIME columns because they are not direct model inputs
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

        # Number of target columns
        self.output_size = len([
            tup for tup in self.input_definition
            if tup[2] == InputTypes.TARGET
        ])

        # --------------------------------------------------------------
        # Index bookkeeping
        # --------------------------------------------------------------
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

        # --------------------------------------------------------------
        # Category counts are needed for embeddings
        # This is set by the formatter after fitting encoders.
        # --------------------------------------------------------------
        category_counts = formatter.num_classes_per_cat_input
        if category_counts is None:
            raise ValueError(
                "formatter.num_classes_per_cat_input is None. "
                "Call formatter.split_data(...) or formatter.set_scalers(...) first."
            )

        self.category_counts = category_counts

        # --------------------------------------------------------------
        # Input embedding layers
        # Each real-valued variable gets its own small linear projection.
        # Each categorical variable gets its own embedding layer.
        # --------------------------------------------------------------
        self.real_variable_projections = nn.ModuleList(
            [TimeDistributed(nn.Linear(1, self.hidden_dim)) for _ in range(self.num_regular_variables)]
        )

        self.categorical_variable_embeddings = nn.ModuleList(
            [nn.Embedding(num_classes, self.hidden_dim) for num_classes in self.category_counts]
        )

        # --------------------------------------------------------------
        # Static variable selection
        # --------------------------------------------------------------
        num_static_inputs = len(self.static_regular_indices) + len(self.static_categorical_indices)
        if num_static_inputs == 0:
            raise ValueError("TFT expects at least one static input. Here that should be categorical_id.")

        self.static_variable_selection = VariableSelectionNetwork(
            hidden_dim=self.hidden_dim,
            num_inputs=num_static_inputs,
            dropout_rate=self.dropout_rate,
            context_dim=None,
            time_distributed=False,
        )

        # Context vectors produced from the static representation
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

        # --------------------------------------------------------------
        # Temporal variable selection
        # --------------------------------------------------------------
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

        # --------------------------------------------------------------
        # LSTM encoder / decoder
        # --------------------------------------------------------------
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

        # After LSTM: gated residual add + norm
        self.post_lstm_gate_add_norm = GateAddNorm(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            time_distributed=True,
        )

        # --------------------------------------------------------------
        # Static enrichment
        # --------------------------------------------------------------
        self.static_enrichment = GatedResidualNetwork(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            context_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            time_distributed=True,
        )

        # --------------------------------------------------------------
        # Interpretable self-attention
        # --------------------------------------------------------------
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

        # --------------------------------------------------------------
        # Position-wise processing
        # --------------------------------------------------------------
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

        # --------------------------------------------------------------
        # Final prediction layer
        # Output is quantiles for each target
        # For electricity:
        #   output_size = 1
        #   quantiles = [0.1, 0.5, 0.9]
        # so final dim = 3
        # --------------------------------------------------------------
        self.output_layer = TimeDistributed(
            nn.Linear(self.hidden_dim, self.output_size * len(self.quantiles))
        )

    def _split_inputs(self, all_inputs):
        """
        Split raw model inputs into:
        - real-valued inputs
        - categorical inputs

        Input shape:
            (batch, time, input_size)

        Returns
        -------
        regular_inputs:
            (batch, time, num_regular_variables)

        categorical_inputs:
            (batch, time, num_categorical_variables)
        """
        regular_inputs = all_inputs[:, :, :self.num_regular_variables]
        categorical_inputs = all_inputs[:, :, self.num_regular_variables:]

        return regular_inputs, categorical_inputs

    def _embed_inputs(self, all_inputs):
        """
        Transforms raw inputs into TFT-style embeddings.

        Returns
        -------
        unknown_inputs:
            (batch, time, hidden_dim, num_unknown_inputs) or None

        known_combined_layer:
            (batch, time, hidden_dim, num_known_inputs)

        obs_inputs:
            (batch, time, hidden_dim, num_observed_target_inputs)

        static_inputs:
            (batch, num_static_inputs, hidden_dim)
        """
        regular_inputs, categorical_inputs = self._split_inputs(all_inputs)

        # --------------------------------------------------------------
        # Real-valued variable embeddings
        # Each real variable is projected independently to hidden_dim
        # --------------------------------------------------------------
        real_embeddings = []
        for i in range(self.num_regular_variables):
            emb = self.real_variable_projections[i](regular_inputs[:, :, i:i + 1])
            real_embeddings.append(emb)  # each: (B, T, H)

        # --------------------------------------------------------------
        # Categorical variable embeddings
        # Categorical values come in as float tensors from the dataset,
        # so we cast to long before embedding lookup.
        # --------------------------------------------------------------
        categorical_embeddings = []
        for i in range(self.num_categorical_variables):
            cat_values = categorical_inputs[:, :, i].long()
            emb = self.categorical_variable_embeddings[i](cat_values)  # (B, T, H)
            categorical_embeddings.append(emb)

        # --------------------------------------------------------------
        # Static inputs
        # These do not change across time for an entity.
        # We use the first time step for static variables.
        # --------------------------------------------------------------
        static_inputs_list = []

        for i in self.static_regular_indices:
            static_inputs_list.append(real_embeddings[i][:, 0, :])

        for i in self.static_categorical_indices:
            static_inputs_list.append(categorical_embeddings[i][:, 0, :])

        static_inputs = torch.stack(static_inputs_list, dim=1)  # (B, N_static, H)

        # --------------------------------------------------------------
        # Observed target inputs
        # These are the target values from the past, embedded as variables.
        # --------------------------------------------------------------
        obs_inputs_list = [real_embeddings[i] for i in self.obs_real_indices]
        obs_inputs = torch.stack(obs_inputs_list, dim=-1)  # (B, T, H, N_obs)

        # --------------------------------------------------------------
        # Unknown inputs
        # These are observed in the past but not known in advance.
        # --------------------------------------------------------------
        unknown_inputs_list = []

        for i in self.unknown_regular_indices:
            unknown_inputs_list.append(real_embeddings[i])

        for i in self.unknown_categorical_indices:
            unknown_inputs_list.append(categorical_embeddings[i])

        if len(unknown_inputs_list) > 0:
            unknown_inputs = torch.stack(unknown_inputs_list, dim=-1)  # (B, T, H, N_unknown)
        else:
            unknown_inputs = None

        # --------------------------------------------------------------
        # Known inputs
        # These are available for future time steps too.
        # Static variables are excluded here because they already go into
        # the static input path.
        # --------------------------------------------------------------
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
        """
        Forward pass of TFT.

        Parameters
        ----------
        all_inputs : torch.Tensor
            Shape:
                (batch, total_time_steps, input_size)

        return_attention : bool
            If True, also return attention / variable-selection information.

        Returns
        -------
        predictions:
            Shape:
                (batch, decoder_steps, output_size * num_quantiles)

        Optional:
            diagnostics dictionary
        """

        # --------------------------------------------------------------
        # Step 1: input embeddings
        # --------------------------------------------------------------
        unknown_inputs, known_combined_layer, obs_inputs, static_inputs = self._embed_inputs(all_inputs)

        # --------------------------------------------------------------
        # Step 2: build historical and future input blocks
        # historical:
        #   past unknown + past known + past observed target
        #
        # future:
        #   only future known inputs
        # --------------------------------------------------------------
        encoder_steps = self.num_encoder_steps

        historical_parts = []

        if unknown_inputs is not None:
            historical_parts.append(unknown_inputs[:, :encoder_steps, :, :])

        historical_parts.append(known_combined_layer[:, :encoder_steps, :, :])
        historical_parts.append(obs_inputs[:, :encoder_steps, :, :])

        historical_inputs = torch.cat(historical_parts, dim=-1)
        future_inputs = known_combined_layer[:, encoder_steps:, :, :]

        # --------------------------------------------------------------
        # Step 3: static variable selection
        # --------------------------------------------------------------
        static_encoder, static_weights = self.static_variable_selection(static_inputs)

        static_context_variable_selection = self.static_context_variable_selection(static_encoder)
        static_context_enrichment = self.static_context_enrichment(static_encoder)
        static_context_state_h = self.static_context_state_h(static_encoder)
        static_context_state_c = self.static_context_state_c(static_encoder)

        # Expand static context across time for temporal VSN
        expanded_static_context_hist = static_context_variable_selection.unsqueeze(1).expand(
            -1, encoder_steps, -1
        )
        expanded_static_context_future = static_context_variable_selection.unsqueeze(1).expand(
            -1, self.decoder_steps, -1
        )

        # --------------------------------------------------------------
        # Step 4: temporal variable selection
        # --------------------------------------------------------------
        historical_features, historical_flags = self.historical_variable_selection(
            historical_inputs,
            context=expanded_static_context_hist,
        )

        future_features, future_flags = self.future_variable_selection(
            future_inputs,
            context=expanded_static_context_future,
        )

        # --------------------------------------------------------------
        # Step 5: LSTM encoder / decoder
        # Initial state is conditioned on static context
        # --------------------------------------------------------------
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

        # Residual reference
        input_embeddings = torch.cat([historical_features, future_features], dim=1)

        temporal_feature_layer, _ = self.post_lstm_gate_add_norm(
            lstm_layer,
            input_embeddings,
        )

        # --------------------------------------------------------------
        # Step 6: static enrichment
        # --------------------------------------------------------------
        expanded_static_context_enrichment = static_context_enrichment.unsqueeze(1).expand(
            -1, self.total_time_steps, -1
        )

        enriched = self.static_enrichment(
            temporal_feature_layer,
            context=expanded_static_context_enrichment,
        )

        # --------------------------------------------------------------
        # Step 7: decoder self-attention
        # --------------------------------------------------------------
        mask = get_decoder_mask(self.total_time_steps, device=enriched.device)

        attention_out, self_attention = self.self_attention(
            enriched,
            enriched,
            enriched,
            mask=mask,
        )

        x, _ = self.post_attention_gate_add_norm(attention_out, enriched)

        # --------------------------------------------------------------
        # Step 8: position-wise processing
        # --------------------------------------------------------------
        decoder = self.positionwise_grn(x)
        transformer_layer, _ = self.final_gate_add_norm(decoder, temporal_feature_layer)

        # --------------------------------------------------------------
        # Step 9: final output layer
        # Only forecast on the decoder horizon
        # --------------------------------------------------------------
        decoder_output = transformer_layer[:, self.num_encoder_steps:, :]  # (B, 24, H)
        predictions = self.output_layer(decoder_output)  # (B, 24, output_size * num_quantiles)

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
    Computes multi-quantile loss.

    Parameters
    ----------
    y_true : torch.Tensor
        Shape:
            (batch, decoder_steps, output_size)

    y_pred : torch.Tensor
        Shape:
            (batch, decoder_steps, output_size * num_quantiles)

    quantiles : tuple
        Quantiles to optimize.

    Returns
    -------
    loss : torch.Tensor
        Scalar loss
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