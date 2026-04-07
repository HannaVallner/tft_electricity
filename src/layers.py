import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeDistributed(nn.Module):
    """
    Applies a module independently to each time step.

    Example:
        input shape:  (batch, time, input_dim)
        output shape: (batch, time, output_dim)
    """

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        # If x is already 2D, just apply the module directly.
        if x.dim() <= 2:
            return self.module(x)

        batch_size, time_steps = x.shape[0], x.shape[1]

        # Flatten (batch, time, ...) -> (batch * time, ...)
        reshaped = x.contiguous().view(batch_size * time_steps, *x.shape[2:])

        output = self.module(reshaped)

        # Restore (batch, time, ...)
        output = output.contiguous().view(batch_size, time_steps, *output.shape[1:])
        return output


class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit used in TFT.

    The idea:
    - one linear path produces transformed features
    - another linear path produces a sigmoid gate
    - output = transformed_features * gate

    If dropout is provided, it is applied before the linear layers.
    """

    def __init__(self, input_dim, hidden_dim, dropout_rate=None, time_distributed=True):
        super().__init__()

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None else None

        linear = nn.Linear(input_dim, hidden_dim)
        gate = nn.Linear(input_dim, hidden_dim)

        self.activation_layer = TimeDistributed(linear) if time_distributed else linear
        self.gate_layer = TimeDistributed(gate) if time_distributed else gate

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)

        activation = self.activation_layer(x)
        gate = torch.sigmoid(self.gate_layer(x))

        return activation * gate, gate


class AddNorm(nn.Module):
    """
    Skips connection followed by LayerNorm.

    This corresponds to:
        add_and_norm(x_list)

    In TFT this is used repeatedly after gated blocks.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, skip):
        return self.layer_norm(x + skip)


class GateAddNorm(nn.Module):
    """
    Convenience block:
        GLU -> residual add -> layer norm

    It combines:
    - gating layer
    - residual connection
    - normalization
    """

    def __init__(self, input_dim, hidden_dim, dropout_rate=None, time_distributed=True):
        super().__init__()
        self.glu = GatedLinearUnit(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            time_distributed=time_distributed,
        )
        self.add_norm = AddNorm(hidden_dim)

    def forward(self, x, skip):
        gated_output, gate = self.glu(x)
        output = self.add_norm(gated_output, skip)
        return output, gate


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN), one of the core TFT blocks.

    Structure:
    1. optional linear projection for skip connection
    2. first linear layer
    3. optional additional context projection
    4. ELU activation
    5. second linear layer
    6. GLU gating
    7. residual add + layer norm

    Parameters
    ----------
    input_dim : int
        Size of input features.
    hidden_dim : int
        Internal hidden size of the GRN.
    output_dim : int or None
        Final output dimension. If None, uses hidden_dim.
    context_dim : int or None
        Additional context size if this GRN receives context.
    dropout_rate : float or None
        Dropout before gating.
    time_distributed : bool
        Whether input shape is (batch, time, dim) or (batch, dim).
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        context_dim=None,
        dropout_rate=None,
        time_distributed=True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.time_distributed = time_distributed

        # ------------------------------------------------------------------
        # Skip connection
        # If input_dim != output_dim, we project input so the residual add works.
        # ------------------------------------------------------------------
        if input_dim != self.output_dim:
            skip_linear = nn.Linear(input_dim, self.output_dim)
            self.skip_layer = TimeDistributed(skip_linear) if time_distributed else skip_linear
        else:
            self.skip_layer = None

        # Main feedforward layers
        fc1 = nn.Linear(input_dim, hidden_dim)
        fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.fc1 = TimeDistributed(fc1) if time_distributed else fc1
        self.fc2 = TimeDistributed(fc2) if time_distributed else fc2

        # Optional context projection
        if context_dim is not None:
            context_linear = nn.Linear(context_dim, hidden_dim, bias=False)
            self.context_layer = (
                TimeDistributed(context_linear) if time_distributed else context_linear
            )
        else:
            self.context_layer = None

        self.elu = nn.ELU()

        self.gate = GatedLinearUnit(
            input_dim=hidden_dim,
            hidden_dim=self.output_dim,
            dropout_rate=dropout_rate,
            time_distributed=time_distributed,
        )

        self.add_norm = AddNorm(self.output_dim)

    def forward(self, x, context=None, return_gate=False):
        # Skip path
        if self.skip_layer is not None:
            skip = self.skip_layer(x)
        else:
            skip = x

        # Main GRN path
        hidden = self.fc1(x)

        if context is not None:
            hidden = hidden + self.context_layer(context)

        hidden = self.elu(hidden)
        hidden = self.fc2(hidden)

        gated_output, gate = self.gate(hidden)
        output = self.add_norm(gated_output, skip)

        if return_gate:
            return output, gate

        return output


def get_decoder_mask(sequence_length, device=None):
    """
    Creates a causal attention mask for decoder self-attention.

    Shape:
        (1, sequence_length, sequence_length)

    Meaning:
    - position t can attend to positions <= t
    - position t cannot attend to the future

    Example for length 4:
        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]
    """
    mask = torch.tril(torch.ones(sequence_length, sequence_length, device=device))
    return mask.unsqueeze(0)


class ScaledDotProductAttention(nn.Module):
    """
    Standard scaled dot-product attention.

    Inputs:
        q: (batch, time, d_k)
        k: (batch, time, d_k)
        v: (batch, time, d_v)
        mask: (batch or 1, time, time)

    Outputs:
        output: (batch, time, d_v)
        attn_weights: (batch, time, time)
    """

    def __init__(self, attn_dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)

        # Attention scores = QK^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # Fill masked positions with a very negative number
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)
        return output, attn_weights


class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable multi-head attention used in TFT.
    - each head has its own Q and K projection
    - all heads SHARE the same V projection
    - head outputs are averaged, not concatenated
    - then projected back to d_model
    """

    def __init__(self, n_head, d_model, dropout_rate):
        super().__init__()

        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head.")

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head

        # Separate Q and K projection for each head
        self.q_layers = nn.ModuleList(
            [nn.Linear(d_model, self.d_k, bias=False) for _ in range(n_head)]
        )
        self.k_layers = nn.ModuleList(
            [nn.Linear(d_model, self.d_k, bias=False) for _ in range(n_head)]
        )

        # Shared V projection across heads
        self.v_layer = nn.Linear(d_model, self.d_v, bias=False)

        self.attention = ScaledDotProductAttention(attn_dropout=dropout_rate)

        self.output_projection = nn.Linear(self.d_v, d_model, bias=False)
        self.output_dropout = nn.Dropout(dropout_rate)

    def forward(self, q, k, v, mask=None):
        head_outputs = []
        attention_weights = []

        # Shared V projection
        v_projected = self.v_layer(v)

        for i in range(self.n_head):
            q_proj = self.q_layers[i](q)
            k_proj = self.k_layers[i](k)

            head_output, attn = self.attention(q_proj, k_proj, v_projected, mask)

            head_output = self.output_dropout(head_output)

            head_outputs.append(head_output)
            attention_weights.append(attn)

        # Stack heads:
        # head_outputs -> (n_head, batch, time, d_v)
        head_outputs = torch.stack(head_outputs, dim=0)
        attention_weights = torch.stack(attention_weights, dim=0)

        combined = torch.mean(head_outputs, dim=0)

        # Project back to d_model
        output = self.output_projection(combined)
        output = self.output_dropout(output)

        return output, attention_weights