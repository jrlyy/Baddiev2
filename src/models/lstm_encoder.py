"""
LSTM encoder for skeleton sequences.
Flattens joints per frame → LSTM over time → embedding.
Matches the (B, C, T, V) → (B, embedding_dim) interface.
"""
import torch
import torch.nn as nn


class SkeletonLSTM(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_nodes: int = 34,
        hidden_dim: int = 256,
        num_layers: int = 2,
        embedding_dim: int = 256,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_nodes = num_nodes
        self.input_dim = in_channels * num_nodes  # flatten C*V per frame

        self.input_bn = nn.BatchNorm1d(self.input_dim)

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_out_dim, embedding_dim)

    def forward(self, x):
        """
        Args:
            x: (B, C, T, V) — batch, channels, frames, nodes
        Returns:
            (B, embedding_dim)
        """
        B, C, T, V = x.shape
        # (B, C, T, V) → (B, T, C*V)
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * V)

        # Batch norm over feature dim: (B*T, C*V)
        x = x.reshape(B * T, -1)
        x = self.input_bn(x)
        x = x.reshape(B, T, -1)

        # LSTM
        out, _ = self.lstm(x)  # (B, T, hidden*dirs)

        # Mean pool over time
        out = out.mean(dim=1)  # (B, hidden*dirs)

        return self.fc(out)  # (B, embedding_dim)