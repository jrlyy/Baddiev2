"""
Cross-attention shuttle fusion module (BST-style).

Instead of encoding the shuttle as a virtual graph node (fixed wrist topology),
the shuttle trajectory is processed by a lightweight temporal encoder and fused
with the player skeleton embedding via cross-attention.

    Q = player skeleton embedding (from ST-GCN or Transformer)
    K, V = shuttle trajectory embedding (from 1D conv encoder)

This lets the model learn *which* shuttle positions are informative for
classifying each player's shot, rather than forcing a fixed graph topology.

Reference: Chang 2025, "BST: Badminton Stroke-type Transformer" (arXiv 2502.21085)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ShuttleTCN(nn.Module):
    """
    Lightweight 1D temporal encoder for shuttle trajectory.

    Input:  (B, 2, T) — x, y positions over T frames (zeros where not visible)
    Output: (B, T, d_model) — per-frame shuttle embeddings
    """

    def __init__(self, in_channels=2, d_model=128, num_layers=3, kernel_size=5):
        super().__init__()
        layers = []
        c_in = in_channels
        for i in range(num_layers):
            c_out = d_model if i == num_layers - 1 else d_model // 2
            padding = (kernel_size - 1) // 2
            layers.extend([
                nn.Conv1d(c_in, c_out, kernel_size, padding=padding),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
            ])
            c_in = c_out
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (B, 2, T) shuttle trajectory [x, y] over time
        Returns:
            (B, T, d_model)
        """
        h = self.net(x)  # (B, d_model, T)
        return h.permute(0, 2, 1)  # (B, T, d_model)


class ShuttleCrossAttention(nn.Module):
    """
    Cross-attention fusion: player skeleton queries attend to shuttle trajectory.

    Takes a skeleton embedding (B, d_skel) from the ST-GCN/Transformer encoder
    and a shuttle trajectory (B, 2, T), and produces a fused embedding.
    """

    def __init__(self, d_skel=256, d_shuttle=128, nhead=4, dropout=0.1,
                 img_w=1920.0, img_h=1080.0):
        """
        Args:
            d_skel: skeleton embedding dimension (from ST-GCN output)
            d_shuttle: shuttle TCN output dimension
            nhead: number of attention heads
            dropout: attention dropout
            img_w: frame width for normalizing shuttle x coords to [0, 1]
            img_h: frame height for normalizing shuttle y coords to [0, 1]
        """
        super().__init__()

        self.register_buffer('_norm_scale',
                             torch.tensor([img_w, img_h]).view(1, 2, 1))

        self.shuttle_tcn = ShuttleTCN(in_channels=2, d_model=d_shuttle)

        # Project skeleton embedding to query (expand to sequence for cross-attn)
        self.skel_proj = nn.Linear(d_skel, d_shuttle)

        # Cross-attention: Q=skeleton, K/V=shuttle
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_shuttle,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_shuttle)

        # Final projection back to skeleton dim
        self.out_proj = nn.Sequential(
            nn.Linear(d_shuttle, d_skel),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, skel_emb, shuttle_traj):
        """
        Args:
            skel_emb: (B, d_skel) — skeleton encoder output
            shuttle_traj: (B, 2, T) — shuttle x,y over time

        Returns:
            (B, d_skel) — fused embedding (additive residual)
        """
        # Normalize pixel coords to [0, 1]
        shuttle_traj = shuttle_traj / self._norm_scale

        # Mask: True where position is zero (not visible)
        shuttle_mask = (shuttle_traj.abs().sum(dim=1) == 0)  # (B, T)

        # If ALL positions are masked for a sample, skip cross-attention
        # (softmax over all -inf → NaN)
        all_masked = shuttle_mask.all(dim=1)  # (B,)

        # Encode shuttle trajectory
        shuttle_emb = self.shuttle_tcn(shuttle_traj)  # (B, T, d_shuttle)

        # Project skeleton to query space: (B, 1, d_shuttle)
        q = self.skel_proj(skel_emb).unsqueeze(1)

        if all_masked.all():
            # No shuttle data in entire batch — pass through skeleton unchanged
            return skel_emb

        # For all-masked rows, unmask one position to avoid NaN softmax
        # (the shuttle_emb will be near-zero anyway since input was zeros)
        safe_mask = shuttle_mask.clone()
        safe_mask[all_masked, 0] = False

        # Cross-attention (capture weights for analysis; averaged over heads)
        attn_out, attn_weights = self.cross_attn(
            query=q,
            key=shuttle_emb,
            value=shuttle_emb,
            key_padding_mask=safe_mask,
            need_weights=True,
            average_attn_weights=True,
        )  # attn_out: (B, 1, d_shuttle); attn_weights: (B, 1, T)
        self._last_attn = attn_weights.squeeze(1).detach().cpu()  # (B, T)
        self._last_all_masked = all_masked.detach().cpu()  # (B,)

        attn_out = self.norm(attn_out.squeeze(1))  # (B, d_shuttle)

        # Zero out the cross-attn contribution for all-masked samples
        fused = self.out_proj(attn_out)
        if all_masked.any():
            fused[all_masked] = 0.0

        return skel_emb + fused