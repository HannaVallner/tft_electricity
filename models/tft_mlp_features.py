import torch
import torch.nn as nn

from layers import TimeDistributed, GateAddNorm, GatedResidualNetwork
from data_formatter import DataTypes, InputTypes


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


class TemporalFusionTransformerMLP(nn.Module):
    """
    TFT variant where VSN is replaced with simple MLP.
    """

    def __init__(self, formatter):
        super().__init__()

        fixed_params = formatter.get_fixed_params()
        model_params = formatter.get_default_model_params()

        self.total_time_steps = fixed_params["total_time_steps"]
        self.num_encoder_steps = fixed_params["num_encoder_steps"]
        self.decoder_steps = self.total_time_steps - self.num_encoder_steps

        self.hidden_dim = model_params["hidden_layer_size"]
        self.quantiles = [0.1, 0.5, 0.9]

        self.column_definition = formatter.get_column_definition()

        self.input_definition = [
            tup for tup in self.column_definition
            if tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]

        self.input_size = len(self.input_definition)

        self.output_size = len([
            tup for tup in self.input_definition
            if tup[2] == InputTypes.TARGET
        ])

        # --------------------------------------------------
        # Replace VSN with MLP
        # --------------------------------------------------
        self.feature_mlp = TimeDistributed(
            SimpleMLP(self.input_size, self.hidden_dim)
        )

        # LSTM stays
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

        self.post_lstm = GateAddNorm(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            dropout_rate=0.1,
            time_distributed=True,
        )

        self.final_grn = GatedResidualNetwork(
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
        # x shape: (B, T, input_size)

        features = self.feature_mlp(x)  # (B, T, H)

        hist = features[:, :self.num_encoder_steps, :]
        fut = features[:, self.num_encoder_steps:, :]

        hist_out, (h, c) = self.history_lstm(hist)
        fut_out, _ = self.future_lstm(fut, (h, c))

        combined = torch.cat([hist_out, fut_out], dim=1)

        enriched, _ = self.post_lstm(combined, features)

        processed = self.final_grn(enriched)

        decoder = processed[:, self.num_encoder_steps:, :]

        return self.output_layer(decoder)