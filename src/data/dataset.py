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
    FB_HIT_TYPE_TO_UNIFIED, SS_SHOT_TYPE_TO_UNIFIED, UNIFIED_SHOT_TO_IDX,
    PROJECT_ROOT,
)
from .feature_eng import FeatureEngineer


def _reorder_hitter_first(segment: np.ndarray, hitter_position: str) -> np.ndarray:
    """
    Ensure the hitting player occupies nodes 0–16 in the skeleton.

    After Y-sorting in pose_extractor:
      player 0 (nodes  0-16) = top court  (smaller Y in image)
      player 1 (nodes 17-33) = bottom court (larger Y in image)

    If the annotation says the hitter is on the bottom court, swap both halves
    so downstream models always see the hitter at player-0 nodes.

    Args:
        segment: (2, T, 34) raw skeleton — C=2 (x,y), T=frames, V=34 joints
        hitter_position: "top", "bottom", or "" (unknown → no-op)

    Returns:
        (2, T, 34) skeleton with hitter at nodes 0-16
    """
    if hitter_position == "bottom":
        segment = segment.copy()
        tmp = segment[:, :, :17].copy()
        segment[:, :, :17] = segment[:, :, 17:]
        segment[:, :, 17:] = tmp
    return segment


def _reorder_hitter_first_by_location(
    segment: np.ndarray, player_location_y: float
) -> np.ndarray:
    """
    Hitter-first reordering for ShuttleSet using the annotated player Y position.

    Compares player_location_y (pixel Y of the hitter from the JSON annotation)
    against the Y centroid of each skeleton player to identify which
    skeleton player (0 or 1) is the hitter, then swaps if needed.

    Args:
        segment: (2, T, 34) raw skeleton
        player_location_y: pixel Y coordinate of the hitter from JSON record

    Returns:
        (2, T, 34) skeleton with hitter at nodes 0-16
    """
    if player_location_y is None or np.isnan(player_location_y):
        return segment
    # C=1 is the Y channel; mean over time and joints for each player
    p0_y = segment[1, :, :17].mean()   # player 0 (top court)
    p1_y = segment[1, :, 17:].mean()   # player 1 (bottom court)
    # Which player centroid is closer to the annotated hitter Y?
    if abs(player_location_y - p1_y) < abs(player_location_y - p0_y):
        # Hitter is player 1 → swap
        segment = segment.copy()
        tmp = segment[:, :, :17].copy()
        segment[:, :, :17] = segment[:, :, 17:]
        segment[:, :, 17:] = tmp
    return segment


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

        # Load court homography (pixel → court metres) if available
        h_path = PROJECT_ROOT / "datasets_preprocessing" / "court_homographies" / "H_img_to_court_m.npy"
        homography = np.load(h_path) if h_path.exists() else None
        if homography is not None:
            print(f"[INFO] FineBadminton: loaded court homography from {h_path.name}")
        self.feature_eng = FeatureEngineer(feature_layer=feature_layer, homography=homography)

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

                hit_type_raw = hit.get("hit_type", "")
                unified = FB_HIT_TYPE_TO_UNIFIED.get(hit_type_raw)
                sample_info = {
                    "rally_id": rally_id,
                    "hit_idx": hit_idx,
                    "hit_frame": hit_frame,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "skeleton_path": str(skeleton_path),
                    "has_skeleton": skeleton_path.exists(),
                    "hit_type": hit_type_raw,
                    "unified_shot_type": unified,
                    "unified_shot_type_idx": UNIFIED_SHOT_TO_IDX.get(unified) if unified else None,
                    "player": hit.get("player", ""),
                    # "top" = upper court (player 0 after Y-sort), "bottom" = lower court (player 1)
                    "hitter": hit.get("hitter", ""),
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

        # Reorder so the hitting player is always at nodes 0–16.
        # Pose extractor Y-sorts: player 0 = top court, player 1 = bottom court.
        # If hitter is "bottom", swap the two halves.
        hitter = info.get("hitter", "")
        segment = _reorder_hitter_first(segment, hitter)

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

    Supports two skeleton formats auto-detected from skeleton_dir:
    1. Per-shot files: {skeleton_dir}/{match_id}/r????_b????.npy  shape (2, T, 34)
       Already centred on hit_frame and hitter-first ordered.
    2. Per-rally files: {skeleton_dir}/{match_id}/r????.npy  shape (2, T_rally, 34)
       Full rally skeletons; this loader slices T=shot_window windows
       centred on hit_frame using rally_frame_start from JSON records
       and applies hitter-first reordering via player_location_y.

    Falls back to placeholder zeros (indexed from JSON records) if no
    skeleton files are found.
    """

    def __init__(self, skeleton_dir=None, outputs_dir=None,
                 shot_window=16, feature_layer="L2",
                 transform=None, load_shot_types=True):
        self.skeleton_dir = Path(skeleton_dir or SS_SKELETONS)
        self.outputs_dir = Path(outputs_dir or SS_OUTPUTS)
        self.shot_window = shot_window
        self.transform = transform
        self.load_shot_types = load_shot_types

        # Load per-match homographies from ShuttleSet CSV
        # {match_name: (3,3) ndarray}
        self._homography_dict = self._load_ss_homographies()
        # For single-match usage (most common), pick the first available H
        h = next(iter(self._homography_dict.values()), None) if self._homography_dict else None
        if h is not None:
            print(f"[INFO] ShuttleSet: loaded homograph(ies) for "
                  f"{len(self._homography_dict)} match(es)")
        self.feature_eng = FeatureEngineer(feature_layer=feature_layer, homography=h)

        # samples entries are either:
        #   (str_path, shot_type_idx)  — per-shot mode / zeros mode
        #   dict with keys: npy_path, frame_num, rally_frame_start,
        #                   player_location_y, shot_type_idx  — per-rally mode
        self.samples = []
        self._mode = 'zeros'
        self._rally_cache: dict = {}  # path → ndarray, avoids repeated disk reads
        self._load_data()

    # ── data loading ──────────────────────────────────────────────────────────

    @staticmethod
    def _load_ss_homographies():
        """
        Parse ShuttleSet homography.csv → {match_name: (3,3) ndarray}.
        Each row's homography_matrix column is a JSON-encoded 3×3 list.
        """
        import json as _json
        csv_path = PROJECT_ROOT / "datasets" / "ShuttleSet" / "set" / "homography.csv"
        if not csv_path.exists():
            return {}
        result = {}
        with open(csv_path) as f:
            header = f.readline().strip().split(",")
            video_col = header.index("video")
            h_col = header.index("homography_matrix")
            for line in f:
                # The homography_matrix cell may contain commas inside the JSON list,
                # so we can't simply split on commas — parse carefully.
                # Format: id,video,"[[...]]",upleft_x,...
                # Find the quoted JSON block.
                parts = line.strip()
                quote_start = parts.index('"')
                quote_end = parts.index('"', quote_start + 1)
                h_str = parts[quote_start + 1: quote_end]
                # Video name is the field before the quoted block
                before_quote = parts[:quote_start].rstrip(",").split(",")
                video_name = before_quote[video_col].strip()
                try:
                    h_matrix = np.array(_json.loads(h_str), dtype=np.float64)
                    if h_matrix.shape == (3, 3):
                        result[video_name] = h_matrix
                except Exception:
                    pass
        return result

    def _load_data(self):
        """Auto-detect skeleton format and populate self.samples."""
        if not self.skeleton_dir.exists():
            self._fallback_zeros()
            return

        # 1. Per-shot files take priority (r????_b????.npy)
        per_shot = sorted(self.skeleton_dir.rglob("r????_b????.npy"))
        if per_shot:
            self._mode = 'per_shot'
            for f in per_shot:
                self.samples.append((str(f), None))
            print(f"[INFO] ShuttleSet: {len(self.samples)} per-shot skeleton files")
            return

        # 2. Per-rally files (r????.npy) — slice windows using JSON metadata
        per_rally = sorted(self.skeleton_dir.rglob("r????.npy"))
        if per_rally and self.outputs_dir.exists():
            self._mode = 'per_rally'
            self._load_per_rally_index()
            return

        # 3. Zeros fallback
        self._fallback_zeros()

    def _load_per_rally_index(self):
        """Build sample list from JSON records cross-referenced with per-rally npys."""
        for json_file in sorted(self.outputs_dir.glob("*.json")):
            if json_file.name == 'pipeline_summary.json':
                continue
            match_id = json_file.stem
            rally_dir = self.skeleton_dir / match_id
            if not rally_dir.exists():
                continue

            with open(json_file) as f:
                records = json.load(f)

            for rec in records:
                rally = rec.get('rally')
                frame_num = rec.get('frame_num')
                rally_frame_start = rec.get('rally_frame_start')
                if any(v is None for v in [rally, frame_num, rally_frame_start]):
                    continue

                npy_path = rally_dir / f"r{rally:04d}.npy"
                if not npy_path.exists():
                    continue

                if self.load_shot_types:
                    unified = SS_SHOT_TYPE_TO_UNIFIED.get(rec.get('type', ''))
                    shot_type_idx = UNIFIED_SHOT_TO_IDX.get(unified) if unified else None
                else:
                    shot_type_idx = None
                self.samples.append({
                    'npy_path': str(npy_path),
                    'frame_num': int(frame_num),
                    'rally_frame_start': int(rally_frame_start),
                    'player_location_y': rec.get('player_location_y'),
                    'shot_type_idx': shot_type_idx,
                })

        n_matches = len({Path(s['npy_path']).parent.parent.name
                         for s in self.samples})
        print(f"[INFO] ShuttleSet: {len(self.samples)} shots from per-rally GDINO "
              f"skeletons across {n_matches} match(es)")

    def _fallback_zeros(self):
        """Index shot records from JSON; return zeros until skeletons are extracted."""
        self._mode = 'zeros'
        if not self.outputs_dir.exists():
            return
        total = 0
        for json_file in sorted(self.outputs_dir.glob("*.json")):
            if json_file.name == 'pipeline_summary.json':
                continue
            with open(json_file) as f:
                records = json.load(f)
            for rec in records:
                if self.load_shot_types:
                    unified = SS_SHOT_TYPE_TO_UNIFIED.get(rec.get('type', ''))
                    shot_type_idx = UNIFIED_SHOT_TO_IDX.get(unified) if unified else None
                else:
                    shot_type_idx = None
                self.samples.append(("zero", shot_type_idx))
                total += 1
        print(f"[INFO] ShuttleSet: {total} shot records from "
              f"{len(list(self.outputs_dir.glob('*.json')))} matches "
              f"(skeletons pending extraction)")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _load_rally(self, npy_path: str) -> np.ndarray:
        """Load rally skeleton with in-memory cache (rally npys are small ~300KB)."""
        if npy_path not in self._rally_cache:
            self._rally_cache[npy_path] = np.load(npy_path)
        return self._rally_cache[npy_path]

    def _slice_window(self, rally_skel: np.ndarray, hit_idx: int) -> np.ndarray:
        """Extract shot_window frames centred on hit_idx, with zero-padding."""
        C, T_rally, V = rally_skel.shape
        half = self.shot_window // 2
        start = max(0, hit_idx - half)
        end = start + self.shot_window
        if end > T_rally:
            end = T_rally
            start = max(0, end - self.shot_window)
        segment = rally_skel[:, start:end, :].copy()
        if segment.shape[1] < self.shot_window:
            pad = np.zeros((C, self.shot_window - segment.shape[1], V),
                           dtype=segment.dtype)
            segment = np.concatenate([segment, pad], axis=1)
        return segment

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        if self._mode == 'per_rally':
            info = sample  # dict
            rally_skel = self._load_rally(info['npy_path'])
            hit_idx = info['frame_num'] - info['rally_frame_start']
            raw_skel = self._slice_window(rally_skel, hit_idx)

            player_loc_y = info.get('player_location_y')
            if player_loc_y is not None:
                try:
                    y = float(player_loc_y)
                    if not np.isnan(y):
                        raw_skel = _reorder_hitter_first_by_location(raw_skel, y)
                except (ValueError, TypeError):
                    pass

            shot_type_idx = info.get('shot_type_idx')

        elif self._mode == 'per_shot':
            npy_path, shot_type_idx = sample
            raw_skel = np.load(npy_path)
            C, T, V = raw_skel.shape
            if T < self.shot_window:
                pad = np.zeros((C, self.shot_window - T, V), dtype=raw_skel.dtype)
                raw_skel = np.concatenate([raw_skel, pad], axis=1)
            elif T > self.shot_window:
                raw_skel = raw_skel[:, :self.shot_window, :]

        else:  # zeros
            npy_path, shot_type_idx = sample
            raw_skel = np.zeros((2, self.shot_window, 34), dtype=np.float32)

        features = self.feature_eng.compute(raw_skel)
        x = torch.tensor(features, dtype=torch.float32)

        if self.transform:
            x = self.transform(x)

        # Always return a tuple; -1 signals "no auxiliary label" to training loop
        return x, (shot_type_idx if shot_type_idx is not None else -1)


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
