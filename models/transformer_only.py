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
        self.input_mixer = TimeDistributed(
            nn.Linear(self.hidden_dim * self.input_size, self.hidden_dim)
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
        Apply TFT-style per-variable embeddings.

        Returns:
            combined_embeddings: (B, T, H, N_variables)
        """
        regular_inputs, categorical_inputs = self._split_inputs(all_inputs)

        embedded_variables = []

        # Real-valued variables
        for i in range(self.num_regular_variables):
            emb = self.real_variable_projections[i](regular_inputs[:, :, i:i + 1])  # (B, T, H)
            embedded_variables.append(emb)

        # Categorical variables
        for i in range(self.num_categorical_variables):
            cat_values = categorical_inputs[:, :, i].long()
            emb = self.categorical_variable_embeddings[i](cat_values)  # (B, T, H)
            embedded_variables.append(emb)

        combined_embeddings = torch.stack(embedded_variables, dim=-1)  # (B, T, H, N)
        return combined_embeddings

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
        # Step 1: embed all variables independently
        # --------------------------------------------------
        embedded_inputs = self._embed_inputs(all_inputs)  # (B, T, H, N)

        # --------------------------------------------------
        # Step 2: flatten variables and mix into one hidden representation
        # --------------------------------------------------
        batch_size, time_steps, hidden_dim, num_vars = embedded_inputs.shape
        flattened = embedded_inputs.reshape(batch_size, time_steps, hidden_dim * num_vars)
        temporal_features = self.input_mixer(flattened)  # (B, T, H)

        # --------------------------------------------------
        # Step 3: causal self-attention
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
        # Step 4: position-wise TFT feedforward block
        # --------------------------------------------------
        decoder = self.positionwise_grn(x)
        transformer_layer, _ = self.final_gate_add_norm(decoder, x)

        # --------------------------------------------------
        # Step 5: keep decoder horizon only
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