"""
1D-CNN encoder for skeleton sequences.
Flattens joints per frame → temporal 1D convolutions → embedding.
Matches the (B, C, T, V) → (B, embedding_dim) interface.
"""
import torch
import torch.nn as nn


class SkeletonCNN1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_nodes: int = 34,
        channels: tuple = (128, 256, 256),
        kernel_size: int = 3,
        embedding_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_nodes = num_nodes
        self.input_dim = in_channels * num_nodes

        self.input_bn = nn.BatchNorm1d(self.input_dim)

        layers = []
        prev_ch = self.input_dim
        for ch in channels:
            layers.extend([
                nn.Conv1d(prev_ch, ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(ch),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_ch = ch

        self.conv_layers = nn.Sequential(*layers)
        self.fc = nn.Linear(channels[-1], embedding_dim)

    def forward(self, x):
        """
        Args:
            x: (B, C, T, V) — batch, channels, frames, nodes
        Returns:
            (B, embedding_dim)
        """
        B, C, T, V = x.shape
        # (B, C, T, V) → (B, C*V, T) for Conv1d
        x = x.permute(0, 1, 3, 2).reshape(B, C * V, T)

        # Batch norm over channel dim
        x = self.input_bn(x)

        # Temporal convolutions
        x = self.conv_layers(x)  # (B, channels[-1], T)

        # Global average pool over time
        x = x.mean(dim=2)  # (B, channels[-1])

        return self.fc(x)  # (B, embedding_dim)