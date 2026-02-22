"""
Spatial-Temporal Graph Convolutional Network (ST-GCN) for skeleton-based
tactical strategy recognition.

References:
    - Yan et al., 2018: "Spatial Temporal Graph Convolutional Networks
      for Skeleton-Based Action Recognition"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpatialGraphConv(nn.Module):
    """Graph convolution over the spatial dimension (joints at a single frame)."""

    def __init__(self, in_channels, out_channels, adjacency, bias=True):
        super().__init__()
        self.num_subsets = adjacency.shape[0]  # number of adjacency partitions
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * self.num_subsets,
            kernel_size=1,
            bias=bias,
        )
        self.register_buffer("A", adjacency.clone())
        # Learnable edge importance weights (one per partition)
        self.edge_importance = nn.Parameter(torch.ones(self.num_subsets))

    def forward(self, x):
        """
        Args:
            x: (B, C, T, V) — batch, channels, frames, nodes
        Returns:
            (B, C_out, T, V)
        """
        B, C, T, V = x.shape
        # Apply convolution: (B, C_out * num_subsets, T, V)
        h = self.conv(x)
        # Reshape to (B, num_subsets, C_out, T, V)
        C_out = h.shape[1] // self.num_subsets
        h = h.view(B, self.num_subsets, C_out, T, V)

        # Aggregate over adjacency partitions
        out = torch.zeros(B, C_out, T, V, device=x.device)
        for k in range(self.num_subsets):
            # h_k: (B, C_out, T, V), A_k: (V, V)
            A_k = self.A[k] * self.edge_importance[k]
            out += torch.einsum("bctv,vw->bctw", h[:, k], A_k)

        return out


class STGCNBlock(nn.Module):
    """One ST-GCN block: spatial graph conv → temporal conv → residual."""

    def __init__(self, in_channels, out_channels, adjacency,
                 temporal_kernel=9, stride=1, dropout=0.0):
        super().__init__()

        self.spatial = SpatialGraphConv(in_channels, out_channels, adjacency)
        self.bn_s = nn.BatchNorm2d(out_channels)

        # Temporal convolution
        padding = (temporal_kernel - 1) // 2
        self.temporal = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=(temporal_kernel, 1),
                      stride=(stride, 1),
                      padding=(padding, 0)),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        # Residual connection
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        x = self.relu(self.bn_s(self.spatial(x)))
        x = self.temporal(x)
        x = self.dropout(x)
        x = self.relu(x + res)
        return x


class STGCN(nn.Module):
    """
    Full ST-GCN encoder.

    Takes skeleton graph sequences and produces fixed-size embeddings.

    Input:  (B, C_in, T, V) where C_in=2 (x,y), T=16 frames, V=34 nodes
    Output: (B, embedding_dim)
    """

    def __init__(self, in_channels, num_nodes, adjacency,
                 num_layers=9, base_channels=64, embedding_dim=256,
                 temporal_kernel=9, dropout=0.3):
        super().__init__()

        self.embedding_dim = embedding_dim

        # Batch norm on input
        self.bn_input = nn.BatchNorm1d(in_channels * num_nodes)

        # Build ST-GCN blocks with channel progression:
        # 64 → 64 → 64 → 128 → 128 → 128 → 256 → 256 → 256
        channels = []
        for i in range(num_layers):
            if i < 3:
                channels.append(base_channels)
            elif i < 6:
                channels.append(base_channels * 2)
            else:
                channels.append(base_channels * 4)

        self.layers = nn.ModuleList()
        c_in = in_channels
        for i, c_out in enumerate(channels):
            stride = 2 if i in [3, 6] else 1  # downsample at layer 3 and 6
            self.layers.append(
                STGCNBlock(c_in, c_out, adjacency,
                           temporal_kernel=temporal_kernel,
                           stride=stride,
                           dropout=dropout)
            )
            c_in = c_out

        # Global average pooling → embedding
        self.fc = nn.Linear(channels[-1], embedding_dim)

    def forward(self, x):
        """
        Args:
            x: (B, C, T, V) skeleton sequence
        Returns:
            (B, embedding_dim) feature embedding
        """
        B, C, T, V = x.shape

        # Input normalization
        x = x.permute(0, 2, 3, 1).contiguous().view(B, T, -1)  # (B, T, V*C)
        x = x.view(B * T, -1)
        x = self.bn_input(x)
        x = x.view(B, T, V, C).permute(0, 3, 1, 2).contiguous()  # (B, C, T, V)

        # ST-GCN blocks
        for layer in self.layers:
            x = layer(x)

        # Global average pooling over time and nodes
        x = x.mean(dim=[2, 3])  # (B, C_last)

        # Project to embedding
        x = self.fc(x)
        return x

    def get_attention_maps(self, x):
        """Return per-layer spatial attention for visualization."""
        maps = []
        B, C, T, V = x.shape

        x = x.permute(0, 2, 3, 1).contiguous().view(B, T, -1)
        x = x.view(B * T, -1)
        x = self.bn_input(x)
        x = x.view(B, T, V, C).permute(0, 3, 1, 2).contiguous()

        for layer in self.layers:
            x = layer(x)
            # Store activation magnitude per node as proxy attention
            attention = x.abs().mean(dim=[0, 1])  # (T', V) → mean over batch & channels
            maps.append(attention.detach().cpu())

        return maps
