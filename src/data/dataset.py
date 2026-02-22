"""
Data loading, episode sampling, and fold splitting for both
FineBadminton (labeled) and ShuttleSet (unlabeled) datasets.
"""
import json
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from pathlib import Path
from typing import List, Optional
from sklearn.model_selection import StratifiedKFold

from ..config import (
    FB_ANNOTATIONS, FB_SKELETONS, FB_FRAMES,
    SS_SKELETONS, SS_OUTPUTS, SS_FRAMES,
    STRATEGY_TO_IDX, FB_STRATEGY_MAP, FB_EXCLUDED_STRATEGIES,
    SS_SHOT_TYPE_TO_IDX, NUM_CLASSES, NUM_JOINTS,
)
from .feature_eng import FeatureEngineer


class FineBadmintonDataset(Dataset):
    """
    Labeled dataset for few-shot strategy classification.

    Loads FineBadminton annotations, maps strategy labels to indices,
    and pairs with skeleton .npy files (or raw frame paths for extraction).
    """

    def __init__(self, skeleton_dir=None, annotations_path=None,
                 shot_window=16, feature_layer="L2", transform=None):
        self.skeleton_dir = Path(skeleton_dir or FB_SKELETONS)
        self.annotations_path = Path(annotations_path or FB_ANNOTATIONS)
        self.shot_window = shot_window
        self.transform = transform
        self.feature_eng = FeatureEngineer(feature_layer=feature_layer)

        self.samples = []       # (skeleton_path, label_idx)
        self.rally_info = []    # metadata for each sample
        self.raw_annotations = None
        self._load_annotations()

    def _load_annotations(self):
        """Parse FineBadminton annotations and build sample list."""
        if not self.annotations_path.exists():
            print(f"[WARN] Annotations not found: {self.annotations_path}")
            return

        with open(self.annotations_path) as f:
            self.raw_annotations = json.load(f)

        for rally in self.raw_annotations:
            video_name = rally.get("video", "")
            rally_id = video_name.replace(".mp4", "")

            for hit_idx, hit in enumerate(rally.get("hitting", [])):
                strategies = hit.get("strategies", [])
                if not strategies:
                    continue

                raw_label = strategies[0].lower().strip()
                if raw_label in FB_EXCLUDED_STRATEGIES:
                    continue
                canonical = FB_STRATEGY_MAP.get(raw_label)
                if canonical is None:
                    continue
                label_idx = STRATEGY_TO_IDX[canonical]

                skeleton_path = self.skeleton_dir / f"{rally_id}.npy"
                hit_frame = hit.get("hit_frame")
                start_frame = hit.get("start_frame")
                end_frame = hit.get("end_frame")

                sample_info = {
                    "rally_id": rally_id,
                    "hit_idx": hit_idx,
                    "hit_frame": hit_frame,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "skeleton_path": str(skeleton_path),
                    "has_skeleton": skeleton_path.exists(),
                    "hit_type": hit.get("hit_type", ""),
                    "player": hit.get("player", ""),
                    "strategy": canonical,
                    "raw_strategy": raw_label,
                    "quality": hit.get("quality"),
                }

                self.samples.append((skeleton_path, label_idx))
                self.rally_info.append(sample_info)

        print(f"[INFO] FineBadminton: {len(self.samples)} labeled shots "
              f"across {len(self.raw_annotations)} rallies")

        from collections import Counter
        dist = Counter(info["strategy"] for info in self.rally_info)
        for s, c in dist.most_common():
            print(f"  {s}: {c}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        skeleton_path, label = self.samples[idx]
        info = self.rally_info[idx]

        skeleton_path = Path(skeleton_path)
        if skeleton_path.exists():
            raw_skel = np.load(skeleton_path)  # (2, T_full, 34)
            x = self._extract_shot_window(raw_skel, info)
        else:
            x = np.zeros((2, self.shot_window, 34), dtype=np.float32)

        # Compute enriched features
        if x.shape[0] == 2:
            x = self.feature_eng.compute(x)

        x = torch.tensor(x, dtype=torch.float32)

        if self.transform:
            x = self.transform(x)

        return x, label

    def _extract_shot_window(self, full_skeleton, info):
        """Extract a T=shot_window segment centered on the hit frame."""
        C, T_full, V = full_skeleton.shape
        hit_frame = info.get("hit_frame")
        rally_start = None

        if hit_frame is not None and self.raw_annotations:
            for rally in self.raw_annotations:
                if rally.get("video", "").replace(".mp4", "") == info["rally_id"]:
                    rally_start = rally.get("start_frame", 0)
                    break

        if hit_frame is not None and rally_start is not None:
            rel_hit = hit_frame - rally_start
            half = self.shot_window // 2
            start = max(0, rel_hit - half)
            end = start + self.shot_window
            if end > T_full:
                end = T_full
                start = max(0, end - self.shot_window)
        else:
            start = 0
            end = min(self.shot_window, T_full)

        segment = full_skeleton[:, start:end, :]

        if segment.shape[1] < self.shot_window:
            pad_len = self.shot_window - segment.shape[1]
            pad = np.zeros((C, pad_len, V), dtype=segment.dtype)
            segment = np.concatenate([segment, pad], axis=1)

        return segment

    def get_labels(self):
        return [s[1] for s in self.samples]

    def get_fold_splits(self, n_folds=5, seed=42):
        """Generate stratified k-fold splits."""
        if len(self.samples) == 0:
            return []

        labels = self.get_labels()
        indices = np.arange(len(self.samples))
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        splits = []
        for train_val_idx, test_idx in skf.split(indices, labels):
            train_val_labels = [labels[i] for i in train_val_idx]
            inner_skf = StratifiedKFold(
                n_splits=max(2, n_folds - 1), shuffle=True, random_state=seed
            )
            train_idx, val_idx = next(inner_skf.split(train_val_idx, train_val_labels))
            train_idx = train_val_idx[train_idx]
            val_idx = train_val_idx[val_idx]
            splits.append((train_idx.tolist(), val_idx.tolist(), test_idx.tolist()))

        return splits


class ShuttleSetDataset(Dataset):
    """
    Unlabeled dataset for self-supervised pre-training.

    Loads skeleton sequences from processed ShuttleSet match data.
    Each sample is one shot segment with optional shot-type label.
    """

    def __init__(self, skeleton_dir=None, outputs_dir=None,
                 shot_window=16, feature_layer="L2",
                 transform=None, load_shot_types=True):
        self.skeleton_dir = Path(skeleton_dir or SS_SKELETONS)
        self.outputs_dir = Path(outputs_dir or SS_OUTPUTS)
        self.shot_window = shot_window
        self.transform = transform
        self.load_shot_types = load_shot_types
        self.feature_eng = FeatureEngineer(feature_layer=feature_layer)

        self.samples = []
        self._load_data()

    def _load_data(self):
        """Load skeleton data from processed ShuttleSet outputs."""
        if self.skeleton_dir.exists():
            for npy_file in sorted(self.skeleton_dir.rglob("*.npy")):
                self.samples.append(("npy", str(npy_file), None))
            if self.samples:
                print(f"[INFO] ShuttleSet: {len(self.samples)} skeleton files")
                return

        # Fallback: index from JSON outputs
        if self.outputs_dir.exists():
            total = 0
            for json_file in sorted(self.outputs_dir.glob("*.json")):
                with open(json_file) as f:
                    records = json.load(f)
                for record in records:
                    shot_type_idx = None
                    if self.load_shot_types:
                        shot_type_idx = SS_SHOT_TYPE_TO_IDX.get(
                            record.get("type", ""), None
                        )
                    self.samples.append(("record", record, shot_type_idx))
                    total += 1
            print(f"[INFO] ShuttleSet: {total} shot records from "
                  f"{len(list(self.outputs_dir.glob('*.json')))} matches "
                  f"(skeletons pending extraction)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_type, data, shot_type_idx = self.samples[idx]

        if sample_type == "npy":
            raw_skel = np.load(data)
        else:
            raw_skel = np.zeros((2, self.shot_window, 34), dtype=np.float32)
            shot_type_idx = data if isinstance(data, int) else shot_type_idx

        C, T, V = raw_skel.shape
        if T < self.shot_window:
            pad = np.zeros((C, self.shot_window - T, V), dtype=raw_skel.dtype)
            raw_skel = np.concatenate([raw_skel, pad], axis=1)
        elif T > self.shot_window:
            start = np.random.randint(0, T - self.shot_window)
            raw_skel = raw_skel[:, start:start + self.shot_window, :]

        if raw_skel.shape[0] == 2:
            features = self.feature_eng.compute(raw_skel)
        else:
            features = raw_skel

        x = torch.tensor(features, dtype=torch.float32)

        if self.transform:
            x = self.transform(x)

        if shot_type_idx is not None:
            return x, shot_type_idx
        return x


class EpisodicSampler(Sampler):
    """Samples balanced N-way K-shot episodes for meta-learning."""

    def __init__(self, labels, n_way, k_shot, n_query, episodes_per_epoch):
        self.labels = np.array(labels)
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.episodes_per_epoch = episodes_per_epoch

        self.class_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)
        self.classes = list(self.class_indices.keys())

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            episode_classes = np.random.choice(
                self.classes, size=min(self.n_way, len(self.classes)), replace=False
            )
            episode_indices = []
            for c in episode_classes:
                avail = self.class_indices[c]
                needed = self.k_shot + self.n_query
                selected = np.random.choice(
                    avail, size=needed, replace=len(avail) < needed,
                )
                episode_indices.extend(selected.tolist())
            yield episode_indices

    def __len__(self):
        return self.episodes_per_epoch


def collate_episode(batch, n_way, k_shot, n_query):
    """Collate function for episodic batches."""
    xs, ys = zip(*batch)
    xs = torch.stack(xs)
    ys = torch.tensor(ys)

    unique_labels = ys.unique()
    label_map = {int(l): i for i, l in enumerate(unique_labels)}
    ys_mapped = torch.tensor([label_map[int(y)] for y in ys])

    per_class = k_shot + n_query
    support_x, support_y, query_x, query_y = [], [], [], []

    for c in range(min(n_way, len(unique_labels))):
        start = c * per_class
        support_x.append(xs[start:start + k_shot])
        support_y.append(ys_mapped[start:start + k_shot])
        query_x.append(xs[start + k_shot:start + per_class])
        query_y.append(ys_mapped[start + k_shot:start + per_class])

    return (torch.cat(support_x), torch.cat(support_y),
            torch.cat(query_x), torch.cat(query_y))
