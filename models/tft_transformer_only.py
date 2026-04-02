import torch
import torch.nn as nn

from layers import (
    TimeDistributed,
    InterpretableMultiHeadAttention,
    GateAddNorm,
    GatedResidualNetwork,
    get_decoder_mask,
)
from data_formatter import DataTypes, InputTypes


class TemporalFusionTransformerTransformerOnly(nn.Module):
    """
    No VSN + No LSTM
    Transformer handles everything
    """

    def __init__(self, formatter):
        super().__init__()

        fixed_params = formatter.get_fixed_params()
        model_params = formatter.get_default_model_params()

        self.total_time_steps = fixed_params["total_time_steps"]
        self.num_encoder_steps = fixed_params["num_encoder_steps"]

        self.hidden_dim = model_params["hidden_layer_size"]
        self.num_heads = model_params["num_heads"]

        self.input_definition = [
            tup for tup in formatter.get_column_definition()
            if tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]

        self.input_size = len(self.input_definition)

        self.output_size = len([
            tup for tup in self.input_definition
            if tup[2] == InputTypes.TARGET
        ])

        # --------------------------------------------------
        # Simple projection instead of embeddings+VSN
        # --------------------------------------------------
        self.input_projection = TimeDistributed(
            nn.Linear(self.input_size, self.hidden_dim)
        )

        # --------------------------------------------------
        # Transformer
        # --------------------------------------------------
        self.self_attention = InterpretableMultiHeadAttention(
            n_head=self.num_heads,
            d_model=self.hidden_dim,
            dropout_rate=0.1,
        )

        self.post_attention = GateAddNorm(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            dropout_rate=0.1,
            time_distributed=True,
        )

        self.feedforward = GatedResidualNetwork(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            context_dim=None,
            dropout_rate=0.1,
            time_distributed=True,
        )

        self.output_layer = TimeDistributed(
            nn.Linear(self.hidden_dim, self.output_size * 3)
        )

    def forward(self, x):
        # x: (B, T, input_size)

        x = self.input_projection(x)

        mask = get_decoder_mask(x.size(1), device=x.device)

        attn_out, _ = self.self_attention(x, x, x, mask)

        x, _ = self.post_attention(attn_out, x)

        x = self.feedforward(x)

        decoder = x[:, self.num_encoder_steps:, :]

        return self.output_layer(decoder)