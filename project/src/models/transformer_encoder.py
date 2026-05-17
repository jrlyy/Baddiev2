"""
Transformer-based skeleton encoder for architectural ablation (BST-style).

Flattens joint features per frame and applies temporal self-attention,
as an alternative to ST-GCN's explicit graph structure.

Reference:
    - Chang, 2025: "BST: Badminton Stroke-type Transformer"
"""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal position."""

    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        """x: (B, T, d_model)"""
        return x + self.pe[:, :x.size(1)]


class SkeletonTransformer(nn.Module):
    """
    Transformer encoder for skeleton sequences.

    Flattens all joint coordinates per frame into a single vector,
    projects to d_model, applies positional encoding and transformer
    encoder layers, then pools to a fixed embedding.

    Input:  (B, C, T, V) where C=2 (x,y), T=16 frames, V=34 nodes
    Output: (B, embedding_dim)
    """

    def __init__(self, in_channels, d_model=256, nhead=8, num_layers=4,
                 dim_feedforward=512, dropout=0.1, max_seq_len=16,
                 embedding_dim=256):
        super().__init__()

        self.embedding_dim = embedding_dim

        # Input projection: flatten (C * V) per frame → d_model
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.fc = nn.Linear(d_model, embedding_dim)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: (B, C, T, V) skeleton sequence
        Returns:
            (B, embedding_dim) feature embedding
        """
        B, C, T, V = x.shape

        # Flatten joints per frame: (B, T, C*V)
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T, C * V)

        # Project and add positional encoding
        x = self.input_proj(x)        # (B, T, d_model)
        x = self.pos_encoding(x)

        # Transformer encoding
        x = self.transformer(x)       # (B, T, d_model)
        x = self.norm(x)

        # Mean pooling over time
        x = x.mean(dim=1)             # (B, d_model)

        # Project to embedding
        x = self.fc(x)                # (B, embedding_dim)
        return x
