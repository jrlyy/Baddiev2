"""
Contrastive learning components for skeleton sequence pre-training.

Includes:
    - NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss — self-supervised
    - SupConLoss (Supervised Contrastive Loss) — uses shot-type labels
    - Projection head (MLP)
    - Skeleton augmentation pipeline

References:
    - Chen et al., 2020: "A Simple Framework for Contrastive Learning
      of Visual Representations (SimCLR)"
    - Khosla et al., 2020: "Supervised Contrastive Learning"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class ProjectionHead(nn.Module):
    """MLP projection head: embedding_dim → hidden → projection_dim."""

    def __init__(self, embedding_dim, hidden_dim=256, projection_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x):
        return self.net(x)


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross-Entropy Loss.

    For a batch of N skeleton sequences, creates 2N augmented views.
    Positive pairs are the two views of the same sequence.
    All other 2(N-1) samples in the batch serve as negatives.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        Args:
            z_i: (B, D) projections from augmentation view 1
            z_j: (B, D) projections from augmentation view 2

        Returns:
            loss: scalar NT-Xent loss
        """
        B = z_i.shape[0]

        # Normalize projections
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate: [z_i; z_j] → (2B, D)
        z = torch.cat([z_i, z_j], dim=0)

        # Similarity matrix: (2B, 2B)
        sim = torch.mm(z, z.t()) / self.temperature

        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, -1e9)

        # Positive pairs: (i, i+B) and (i+B, i)
        pos_i = torch.arange(B, device=z.device)
        pos_j = pos_i + B
        labels = torch.cat([pos_j, pos_i], dim=0)  # (2B,)

        loss = F.cross_entropy(sim, labels)
        return loss


class SkeletonAugmentor:
    """
    Augmentation pipeline for skeleton contrastive learning.

    Augmentations:
        - Joint jittering (Gaussian noise on coordinates)
        - Temporal crop and resample
        - Spatial rotation (court-relative)
        - Joint masking (random zeroing)
    """

    def __init__(self, jitter_std=0.01, mask_ratio=0.15,
                 speed_range=0.2, rotation_range=15.0):
        self.jitter_std = jitter_std
        self.mask_ratio = mask_ratio
        self.speed_range = speed_range
        self.rotation_range = rotation_range

    def __call__(self, x):
        """
        Apply random augmentations to a skeleton sequence.

        Args:
            x: (C, T, V) single skeleton sequence

        Returns:
            augmented: (C, T, V)
        """
        x = x.clone()
        x = self._jitter(x)
        x = self._speed_perturb(x)
        x = self._rotate(x)
        x = self._mask_joints(x)
        return x

    def _jitter(self, x):
        """Add Gaussian noise to joint coordinates."""
        noise = torch.randn_like(x) * self.jitter_std
        return x + noise

    def _speed_perturb(self, x):
        """Resample the full sequence at a random speed, preserving temporal order.

        Unlike temporal crop, all phases of the stroke remain present — the
        sequence is simply played faster or slower. A speed > 1 compresses
        the motion (fast-forward effect); speed < 1 stretches it (slow-motion).

        Args:
            x: (C, T, V)

        Returns:
            (C, T, V) with the same number of frames but different temporal density
        """
        C, T, V = x.shape
        speed = 1.0 + (torch.rand(1).item() * 2 - 1) * self.speed_range
        new_len = max(2, int(round(T * speed)))
        if new_len == T:
            return x
        # Interpolate only along the temporal axis: reshape to (1, C*V, T)
        x_2d = x.reshape(1, C * V, T)
        x_rs = F.interpolate(x_2d, size=new_len, mode='linear', align_corners=False)
        x_back = F.interpolate(x_rs, size=T, mode='linear', align_corners=False)
        return x_back.reshape(C, T, V)

    def _rotate(self, x):
        """Apply random 2D rotation to (x, y) coordinates."""
        angle = (torch.rand(1).item() * 2 - 1) * self.rotation_range
        theta = np.radians(angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # x[0] = x-coords, x[1] = y-coords
        x_rot = x[0] * cos_t - x[1] * sin_t
        y_rot = x[0] * sin_t + x[1] * cos_t
        x[0] = x_rot
        x[1] = y_rot
        return x

    def _mask_joints(self, x):
        """Randomly zero out a fraction of joints across all frames."""
        C, T, V = x.shape
        num_mask = max(1, int(V * self.mask_ratio))
        mask_indices = torch.randperm(V)[:num_mask]
        x[:, :, mask_indices] = 0.0
        return x


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., 2020).

    Uses shot-type labels to define positives: any two samples with the
    same shot-type label form a positive pair. Both augmented views of the
    same sample also form a positive pair, so this subsumes NT-Xent when
    every sample has a unique label.

    Usage in SSL ablation:
        - SimCLR:  NTXentLoss()(z_i, z_j)
        - SupCon:  SupConLoss()(z_i, z_j, labels)
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j, labels):
        """
        Args:
            z_i: (B, D) projections from augmentation view 1
            z_j: (B, D) projections from augmentation view 2
            labels: (B,) integer shot-type labels

        Returns:
            loss: scalar SupCon loss
        """
        B = z_i.shape[0]
        device = z_i.device

        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate both views: (2B, D) and (2B,) labels
        z = torch.cat([z_i, z_j], dim=0)
        labels_2x = torch.cat([labels, labels], dim=0)

        # Similarity matrix scaled by temperature: (2B, 2B)
        sim = torch.mm(z, z.t()) / self.temperature

        # Self-mask: exclude diagonal (anchor vs itself)
        self_mask = torch.eye(2 * B, dtype=torch.bool, device=device)

        # Positive mask: same label, excluding self
        pos_mask = (
            labels_2x.unsqueeze(1) == labels_2x.unsqueeze(0)
        ) & ~self_mask  # (2B, 2B)

        # Log-denominator: log sum over all non-self pairs
        sim_no_self = sim.masked_fill(self_mask, float('-inf'))
        log_denom = torch.logsumexp(sim_no_self, dim=1)  # (2B,)

        # Log-prob for every pair: sim[i,j] - log_denom[i]
        log_probs = sim - log_denom.unsqueeze(1)  # (2B, 2B)

        # Mean loss over anchors that have at least one positive
        n_pos = pos_mask.sum(dim=1).float()  # (2B,)
        valid = n_pos > 0

        if not valid.any():
            # Fallback: batch has no within-class pairs — shouldn't happen
            return torch.tensor(0.0, device=device, requires_grad=True)

        per_anchor = -(log_probs * pos_mask.float()).sum(dim=1)  # (2B,)
        loss = (per_anchor[valid] / n_pos[valid]).mean()
        return loss


class AuxiliaryShotTypeHead(nn.Module):
    """Auxiliary head for shot-type prediction during SSL (kept for reference)."""

    def __init__(self, embedding_dim, num_shot_types=18):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, num_shot_types)

    def forward(self, x):
        return self.fc(x)
