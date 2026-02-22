"""
SimCLR-style contrastive learning components for self-supervised
pre-training on unlabeled skeleton sequences.

Includes:
    - NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss
    - Projection head (MLP)
    - Skeleton augmentation pipeline

Reference:
    - Chen et al., 2020: "A Simple Framework for Contrastive Learning
      of Visual Representations (SimCLR)"
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
                 temporal_crop_ratio=0.8, rotation_range=15.0):
        self.jitter_std = jitter_std
        self.mask_ratio = mask_ratio
        self.temporal_crop_ratio = temporal_crop_ratio
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
        x = self._temporal_crop(x)
        x = self._rotate(x)
        x = self._mask_joints(x)
        return x

    def _jitter(self, x):
        """Add Gaussian noise to joint coordinates."""
        noise = torch.randn_like(x) * self.jitter_std
        return x + noise

    def _temporal_crop(self, x):
        """Randomly crop temporal dimension and resample to original length."""
        C, T, V = x.shape
        crop_len = max(1, int(T * self.temporal_crop_ratio))
        start = torch.randint(0, T - crop_len + 1, (1,)).item()
        cropped = x[:, start:start + crop_len, :]

        # Resample back to original length T
        if crop_len != T:
            cropped = cropped.unsqueeze(0)  # (1, C, crop_len, V)
            cropped = F.interpolate(cropped, size=(T, V), mode="nearest")
            cropped = cropped.squeeze(0)
        return cropped

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


class AuxiliaryShotTypeHead(nn.Module):
    """Optional auxiliary head for shot-type prediction during SSL."""

    def __init__(self, embedding_dim, num_shot_types=18):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, num_shot_types)

    def forward(self, x):
        return self.fc(x)
