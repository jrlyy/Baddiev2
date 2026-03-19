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
    FB_ANNOTATIONS, FB_SKELETONS, FB_FRAMES, FB_SHUTTLES,
    SS_SKELETONS, SS_OUTPUTS, SS_FRAMES, SS_SHUTTLES, SS_CSV_ROOT, SS_SPLIT_JSON,
    STRATEGY_TO_IDX, FB_STRATEGY_MAP, FB_EXCLUDED_STRATEGIES,
    SS_SHOT_TYPE_TO_IDX, NUM_CLASSES, NUM_JOINTS,
    FB_SUBTYPE_TO_SHOT_TYPE, FB_HIT_TYPE_TO_SHOT_TYPE,
    SS_TYPE_TO_SHOT_TYPE, SHOT_TYPE_TO_IDX,
    PROJECT_ROOT,
)
from .feature_eng import FeatureEngineer





class FineBadmintonDataset(Dataset):
    """
    Labeled dataset for few-shot strategy classification.

    Loads FineBadminton annotations, maps strategy labels to indices,
    and pairs with skeleton .npy files (or raw frame paths for extraction).
    """

    def __init__(self, skeleton_dir=None, annotations_path=None,
                 shot_window=16, feature_layer="L2", transform=None,
                 use_shuttle=False, shuttle_dir=None, use_hitter=False):
        self.skeleton_dir = Path(skeleton_dir or FB_SKELETONS)
        self.annotations_path = Path(annotations_path or FB_ANNOTATIONS)
        self.shot_window = shot_window
        self.transform = transform
        self.use_shuttle = use_shuttle
        self.shuttle_dir = Path(shuttle_dir or FB_SHUTTLES)
        self.use_hitter = use_hitter

        # Load court homography (pixel → court metres) if available
        h_path = PROJECT_ROOT / "datasets_preprocessing" / "court_homographies" / "H_img_to_court_m.npy"
        homography = np.load(h_path) if h_path.exists() else None
        if homography is not None:
            print(f"[INFO] FineBadminton: loaded court homography from {h_path.name}")
        self.feature_eng = FeatureEngineer(feature_layer=feature_layer,
                                           homography=homography,
                                           use_hitter=use_hitter)

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
                subtypes = hit.get("subtype", [])
                # Prefer subtype (fine-grained) over hit_type for shot type mapping
                shot_type = None
                subtype_used = None
                for st in subtypes:
                    shot_type = FB_SUBTYPE_TO_SHOT_TYPE.get(st)
                    if shot_type is not None:
                        subtype_used = st
                        break
                if shot_type is None:
                    shot_type = FB_HIT_TYPE_TO_SHOT_TYPE.get(hit_type_raw)
                sample_info = {
                    "rally_id": rally_id,
                    "hit_idx": hit_idx,
                    "hit_frame": hit_frame,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "skeleton_path": str(skeleton_path),
                    "has_skeleton": skeleton_path.exists(),
                    "hit_type": hit_type_raw,
                    "subtype": subtype_used or (subtypes[0] if subtypes else None),
                    "shot_type": shot_type,
                    "shot_type_idx": SHOT_TYPE_TO_IDX.get(shot_type) if shot_type else None,
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

        # Append shuttle as virtual node 34 (before feature engineering so homography applies)
        if self.use_shuttle:
            x = self._append_shuttle(x, info)  # (2, T, 35)

        # Compute enriched features
        if x.shape[0] == 2:
            hitter = info.get('hitter') if self.use_hitter else None
            x = self.feature_eng.compute(x, hitter=hitter)

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

    def _append_shuttle(self, skeleton, info):
        """
        Append shuttle position as virtual node 34.

        Loads the rally's shuttle trajectory .npy (shape: (T_rally, 3) [x, y, vis]),
        extracts the same shot window as the skeleton, and appends as a (2, T, 1) node.
        Positions where visibility < 0.5 are zeroed out.

        Args:
            skeleton: (2, T, V) raw skeleton (pixel coords, pre-homography)
            info: sample metadata dict with 'rally_id', 'hit_frame', etc.

        Returns:
            (2, T, V+1) skeleton with shuttle appended as the last node
        """
        C, T, V = skeleton.shape
        shuttle_node = np.zeros((C, T, 1), dtype=skeleton.dtype)

        shuttle_path = self.shuttle_dir / f"{info['rally_id']}.npy"
        if shuttle_path.exists():
            shuttle = np.load(shuttle_path)  # (T_full, 3): [x, y, vis]

            # Find same window as skeleton extraction
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
                T_full = len(shuttle)
                if end > T_full:
                    end = T_full
                    start = max(0, end - self.shot_window)
            else:
                start = 0
                end = min(self.shot_window, len(shuttle))

            segment = shuttle[start:end]  # (T_seg, 3)
            T_seg = segment.shape[0]
            xy = segment[:, :2].T  # (2, T_seg)
            vis = segment[:, 2]    # (T_seg,)

            # Zero out low-visibility positions
            xy[:, vis < 0.5] = 0.0

            # Pad if short
            if T_seg < T:
                pad = np.zeros((C, T - T_seg), dtype=skeleton.dtype)
                xy = np.concatenate([xy, pad], axis=1)
            elif T_seg > T:
                xy = xy[:, :T]

            shuttle_node[:, :, 0] = xy

        return np.concatenate([skeleton, shuttle_node], axis=2)  # (C, T, V+1)

    def get_labels(self):
        return [s[1] for s in self.samples]

    def get_rally_splits(self, n_train_rallies=30, seed=42):
        """
        Split dataset by rally to prevent data leakage between train and held-out sets.

        All shots from a given rally appear in only one split (train or held-out),
        unlike get_fold_splits() which can put shots from the same rally in both.

        Args:
            n_train_rallies: number of rallies for training (rest go to held-out)
            seed: random seed for reproducibility

        Returns:
            train_idx: list of sample indices for training
            holdout_idx: list of sample indices for held-out evaluation
        """
        rng = np.random.RandomState(seed)
        rally_ids = sorted({info['rally_id'] for info in self.rally_info})
        rally_ids = list(rally_ids)
        rng.shuffle(rally_ids)

        train_rallies = set(rally_ids[:n_train_rallies])
        holdout_rallies = set(rally_ids[n_train_rallies:])

        train_idx = [i for i, info in enumerate(self.rally_info)
                     if info['rally_id'] in train_rallies]
        holdout_idx = [i for i, info in enumerate(self.rally_info)
                       if info['rally_id'] in holdout_rallies]

        print(f"Rally split: {len(train_rallies)} train rallies ({len(train_idx)} shots), "
              f"{len(holdout_rallies)} held-out rallies ({len(holdout_idx)} shots)")
        return train_idx, holdout_idx

    def get_fold_splits(self, n_folds=5, seed=42):
        """
        Rally-level stratified k-fold splits.

        Rallies (not individual shots) are assigned to folds so that no rally
        appears in both train and test — preventing leakage from shared player
        skeletons across shots of the same rally.  The dominant strategy label
        per rally is used as the stratification key.

        Each fold also carves out a val set (1 inner fold from the train rallies)
        for checkpoint selection.

        Returns:
            list of (train_idx, val_idx, test_idx) — sample-level indices.
        """
        if len(self.samples) == 0:
            return []

        from collections import Counter as _Counter
        rally_ids = sorted({info['rally_id'] for info in self.rally_info})
        rally_label = {
            rid: _Counter(
                info['strategy'] for info in self.rally_info
                if info['rally_id'] == rid
            ).most_common(1)[0][0]
            for rid in rally_ids
        }

        rally_arr = np.array(rally_ids)
        strat_arr = np.array([rally_label[r] for r in rally_ids])

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        splits = []
        for tr_r_idx, te_r_idx in skf.split(rally_arr, strat_arr):
            test_rallies = set(rally_arr[te_r_idx])
            tv_rallies   = rally_arr[tr_r_idx]
            tv_strat     = strat_arr[tr_r_idx]

            inner_skf = StratifiedKFold(
                n_splits=max(2, n_folds - 1), shuffle=True, random_state=seed
            )
            tr_inner, val_inner = next(inner_skf.split(tv_rallies, tv_strat))
            train_rallies = set(tv_rallies[tr_inner])
            val_rallies   = set(tv_rallies[val_inner])

            train_idx = [i for i, info in enumerate(self.rally_info)
                         if info['rally_id'] in train_rallies]
            val_idx   = [i for i, info in enumerate(self.rally_info)
                         if info['rally_id'] in val_rallies]
            test_idx  = [i for i, info in enumerate(self.rally_info)
                         if info['rally_id'] in test_rallies]

            splits.append((train_idx, val_idx, test_idx))

        return splits


class ShuttleSetDataset(Dataset):
    """
    Unlabeled dataset for self-supervised pre-training.

    Supports two skeleton formats auto-detected from skeleton_dir:
    1. Per-shot files: {skeleton_dir}/{match_id}/r????_b????.npy  shape (2, T, 34)
       Already centred on hit_frame. Player ordering: P0 (nodes 0-16) = top court,
       P1 (nodes 17-33) = bottom court.
    2. Per-rally files: {skeleton_dir}/{match_id}/r????.npy  shape (2, T_rally, 34)
       Full rally skeletons; this loader slices T=shot_window windows
       centred on hit_frame using rally_frame_start from JSON records.
       Hitter identity is stored as metadata ('top'/'bottom') but does NOT
       affect node ordering — P0 is always top court, P1 is always bottom court.

    Falls back to placeholder zeros (indexed from JSON records) if no
    skeleton files are found.
    """

    def __init__(self, skeleton_dir=None, outputs_dir=None,
                 shot_window=16, feature_layer="L2",
                 transform=None, load_shot_types=True,
                 split=None, split_json=None,
                 use_shuttle=False, shuttle_dir=None,
                 use_hitter=False, variable_window=False,
                 shuttle_fusion="graph"):
        """
        Args:
            split: None (all matches) | 'train' | 'val'
                   Filters to only matches in the specified split using
                   shuttleset_split.json.
            split_json: path to split manifest; defaults to SS_SPLIT_JSON.
                        On Colab, pass the path on your Drive.
            use_shuttle: append shuttle position as virtual node 34 (35 nodes total)
            shuttle_dir: path to SS shuttle .npy files; defaults to SS_SHUTTLES
            use_hitter: append is_hitter channel (1=hitter's joints, 0=opponent's)
            variable_window: use prev/next shot hit frames instead of fixed window
            shuttle_fusion: "graph" (virtual node in skeleton) or
                           "cross_attn" (separate trajectory tensor returned)
        """
        self.skeleton_dir = Path(skeleton_dir or SS_SKELETONS)
        self.outputs_dir = Path(outputs_dir or SS_OUTPUTS)
        self.shot_window = shot_window
        self.transform = transform
        self.load_shot_types = load_shot_types
        self.split = split
        self.use_shuttle = use_shuttle
        self.shuttle_dir = Path(shuttle_dir or SS_SHUTTLES)
        self.use_hitter = use_hitter
        self.variable_window = variable_window
        self.shuttle_fusion = shuttle_fusion
        self.allowed_matches = self._load_split(split, split_json)

        # Load per-match homographies (Roboflow-based, from notebook 07)
        self._homography_dict = self._load_ss_homographies()
        self._feature_layer = feature_layer
        self._use_hitter = use_hitter
        # Default feature engineer (no H) — used when match H is unavailable
        self.feature_eng = FeatureEngineer(feature_layer=feature_layer, homography=None,
                                           use_hitter=use_hitter)
        # Per-match feature engineers (cached lazily)
        self._fe_cache: dict = {}
        if self._homography_dict:
            print(f"[INFO] ShuttleSet: {len(self._homography_dict)} per-match homographies")

        # samples entries are either:
        #   (str_path, shot_type_idx)  — per-shot mode / zeros mode
        #   dict with keys: npy_path, frame_num, rally_frame_start,
        #                   player_location_y, shot_type_idx  — per-rally mode
        self.samples = []
        self._mode = 'zeros'
        self._rally_cache: dict = {}  # path → ndarray, avoids repeated disk reads
        self._load_data()

        if self.variable_window and self._mode in ('whole_match', 'per_rally'):
            self._compute_variable_windows()

    # ── data loading ──────────────────────────────────────────────────────────

    @staticmethod
    def _load_split(split: Optional[str], split_json=None) -> Optional[set]:
        """Load allowed match names from split manifest. Returns None = all matches."""
        if split is None:
            return None
        path = Path(split_json) if split_json else SS_SPLIT_JSON
        if not path.exists():
            print(f"[WARN] Split manifest not found: {path} — using all matches")
            return None
        data = json.loads(path.read_text())
        matches = set(data.get(split, []))
        if not matches:
            print(f"[WARN] No matches found for split='{split}' in {path}")
        print(f"[INFO] ShuttleSet split='{split}': {len(matches)} allowed matches")
        return matches

    @staticmethod
    def _load_ss_homographies():
        """
        Load ShuttleSet per-match homographies (pixel → court metres).

        Load pre-computed ss_per_match_H.npy (saved by notebook 07).
        These are Roboflow-detected homographies mapping pixel → court metres.
        """
        npy_path = (PROJECT_ROOT / "datasets_preprocessing"
                    / "court_homographies" / "ss_per_match_H.npy")
        if npy_path.exists():
            h_dict = np.load(str(npy_path), allow_pickle=True).item()
            print(f"[INFO] Loaded SS homographies from {npy_path.name} "
                  f"({len(h_dict)} matches)")
            return h_dict
        return {}

    def _match_allowed(self, match_name: str) -> bool:
        """True if this match is in the allowed set (or no filter is applied)."""
        return self.allowed_matches is None or match_name in self.allowed_matches

    def _load_data(self):
        """Auto-detect skeleton format and populate self.samples."""
        if not self.skeleton_dir.exists():
            self._fallback_zeros()
            return

        # 1. Whole-match format: {match}/skeletons.npy + {match}/frame_nums.npy
        #    (produced by the Colab GDINO extraction pipeline)
        whole_match_dirs = [
            d for d in sorted(self.skeleton_dir.iterdir())
            if d.is_dir()
            and (d / 'skeletons.npy').exists()
            and (d / 'frame_nums.npy').exists()
            and self._match_allowed(d.name)
        ]
        if whole_match_dirs:
            self._mode = 'whole_match'
            self._load_whole_match_index(whole_match_dirs)
            return

        # 2. Per-shot files (r????_b????.npy)
        per_shot = [
            f for f in sorted(self.skeleton_dir.rglob("r????_b????.npy"))
            if self._match_allowed(f.parent.name)
        ]
        if per_shot:
            self._mode = 'per_shot'
            for f in per_shot:
                self.samples.append((str(f), None))
            print(f"[INFO] ShuttleSet: {len(self.samples)} per-shot skeleton files")
            return

        # 3. Per-rally files (r????.npy) — slice windows using JSON metadata
        per_rally = [
            f for f in sorted(self.skeleton_dir.rglob("r????.npy"))
            if self._match_allowed(f.parent.name)
        ]
        if per_rally and self.outputs_dir.exists():
            self._mode = 'per_rally'
            self._load_per_rally_index()
            return

        # 4. Zeros fallback
        self._fallback_zeros()

    @staticmethod
    def _build_rally_hitter_map(csv_dir):
        """Per-rally mapping of player A/B → 'top'/'bottom' using median Y.

        For each rally, the player with the smaller median player_location_y
        is on the top (far) court. Returns {(set_num, rally): {'A': ..., 'B': ...}}.
        """
        import csv as _csv
        from collections import defaultdict
        rally_ys = defaultdict(lambda: defaultdict(list))
        for csv_path in sorted(Path(csv_dir).glob('set*.csv')):
            try:
                set_num = int(''.join(filter(str.isdigit, csv_path.name)) or '0')
            except ValueError:
                set_num = 0
            with open(csv_path) as f:
                for row in _csv.DictReader(f):
                    try:
                        rally = int(row['rally'])
                        player = row.get('player', '')
                        ply = float(row.get('player_location_y', 'nan'))
                    except (ValueError, TypeError):
                        continue
                    if player in ('A', 'B') and ply == ply:  # not NaN
                        rally_ys[(set_num, rally)][player].append(ply)
        result = {}
        for key, players in rally_ys.items():
            if 'A' in players and 'B' in players:
                a_is_top = float(np.median(players['A'])) < float(np.median(players['B']))
                result[key] = {
                    'A': 'top' if a_is_top else 'bottom',
                    'B': 'bottom' if a_is_top else 'top',
                }
            else:
                result[key] = {'A': '', 'B': ''}
        return result

    def _load_whole_match_index(self, match_dirs):
        """
        Build sample list from whole-match skeletons (skeletons.npy + frame_nums.npy)
        cross-referenced with ShuttleSet CSV annotations.

        Each sample is a dict:
            skel_dir, frame_num, frame_nums_arr (for lookup), shot_type_idx,
            player_location_y
        """
        import csv as _csv
        total = 0
        for match_dir in match_dirs:
            match_name = match_dir.name
            csv_dir = Path(SS_CSV_ROOT) / match_name
            if not csv_dir.exists():
                continue

            frame_nums_arr = np.load(str(match_dir / 'frame_nums.npy'))
            fn_set = set(frame_nums_arr.tolist())

            # Build per-rally hitter map using median Y of each player
            hitter_map = self._build_rally_hitter_map(csv_dir)

            for csv_path in sorted(csv_dir.glob('set*.csv')):
                try:
                    set_num = int(''.join(filter(str.isdigit, csv_path.name))) or 0
                except ValueError:
                    set_num = 0
                with open(csv_path) as f:
                    for row in _csv.DictReader(f):
                        try:
                            frame_num = int(float(row['frame_num']))
                            rally = int(row['rally'])
                        except (KeyError, ValueError, TypeError):
                            continue
                        # Find nearest extracted frame (stride-4 tolerance)
                        if frame_num not in fn_set:
                            candidates = frame_nums_arr[
                                np.abs(frame_nums_arr - frame_num) <= 4
                            ]
                            if len(candidates) == 0:
                                continue
                            frame_num = int(candidates[np.argmin(np.abs(candidates - frame_num))])

                        if self.load_shot_types:
                            shot_type = SS_TYPE_TO_SHOT_TYPE.get(row.get('type', ''))
                            shot_type_idx = SHOT_TYPE_TO_IDX.get(shot_type) if shot_type else None
                        else:
                            shot_type_idx = None

                        # Hitter from per-rally median-Y mapping
                        player = row.get('player', '')
                        rally_sides = hitter_map.get((set_num, rally), {})
                        hitter = rally_sides.get(player, '')

                        self.samples.append({
                            'skel_dir':         str(match_dir),
                            'frame_num':        frame_num,
                            'frame_nums_arr':   frame_nums_arr,
                            'hitter':           hitter,
                            'shot_type_idx':    shot_type_idx,
                            'match_name':       match_name,
                            'rally_key':        f's{set_num}r{rally}',
                        })
                        total += 1

        n_matches = len(match_dirs)
        print(f"[INFO] ShuttleSet split='{self.split}': {total} shots from "
              f"whole-match skeletons across {n_matches} match(es)")

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

            # Build per-rally hitter map from JSON records (same median-Y approach)
            from collections import defaultdict
            rally_ys = defaultdict(lambda: defaultdict(list))
            for rec in records:
                player = rec.get('player', '')
                rally = rec.get('rally')
                try:
                    ply = float(rec.get('player_location_y', 'nan'))
                except (ValueError, TypeError):
                    continue
                if player in ('A', 'B') and rally is not None and ply == ply:
                    set_file = rec.get('set_file', '')
                    try:
                        set_num = int(''.join(filter(str.isdigit, set_file)) or '0')
                    except ValueError:
                        set_num = 0
                    rally_ys[(set_num, rally)][player].append(ply)
            hitter_map = {}
            for key, players in rally_ys.items():
                if 'A' in players and 'B' in players:
                    a_top = float(np.median(players['A'])) < float(np.median(players['B']))
                    hitter_map[key] = {
                        'A': 'top' if a_top else 'bottom',
                        'B': 'bottom' if a_top else 'top',
                    }
                else:
                    hitter_map[key] = {'A': '', 'B': ''}

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
                    shot_type = SS_TYPE_TO_SHOT_TYPE.get(rec.get('type', ''))
                    shot_type_idx = SHOT_TYPE_TO_IDX.get(shot_type) if shot_type else None
                else:
                    shot_type_idx = None

                # Hitter from per-rally median-Y mapping
                player = rec.get('player', '')
                set_file = rec.get('set_file', '')
                try:
                    set_num = int(''.join(filter(str.isdigit, set_file)) or '0')
                except ValueError:
                    set_num = 0
                rally_sides = hitter_map.get((set_num, rally), {})
                hitter = rally_sides.get(player, '')

                rally_frame_end = rec.get('rally_frame_end')
                self.samples.append({
                    'npy_path': str(npy_path),
                    'frame_num': int(frame_num),
                    'rally_frame_start': int(rally_frame_start),
                    'rally_frame_end': int(rally_frame_end) if rally_frame_end else None,
                    'hitter': hitter,
                    'shot_type_idx': shot_type_idx,
                    'match_name': match_id,
                    'rally_key': f's{set_num}r{rally}',
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
                    shot_type = SS_TYPE_TO_SHOT_TYPE.get(rec.get('type', ''))
                    shot_type_idx = SHOT_TYPE_TO_IDX.get(shot_type) if shot_type else None
                else:
                    shot_type_idx = None
                self.samples.append(("zero", shot_type_idx))
                total += 1
        print(f"[INFO] ShuttleSet: {total} shot records from "
              f"{len(list(self.outputs_dir.glob('*.json')))} matches "
              f"(skeletons pending extraction)")

    def _compute_variable_windows(self):
        """
        For each sample, compute frame_prev_hit and frame_next_hit from
        adjacent shots in the same rally. Used for variable-width windowing.

        Groups samples by (match_name, rally_key), sorts by frame_num,
        and sets:
            frame_prev_hit: previous shot's frame_num (or rally_frame_start)
            frame_next_hit: next shot's frame_num (or rally_frame_end)
        """
        from collections import defaultdict
        rally_groups = defaultdict(list)
        for i, s in enumerate(self.samples):
            if not isinstance(s, dict):
                continue
            key = (s.get('match_name', ''), s.get('rally_key', ''))
            rally_groups[key].append(i)

        for key, indices in rally_groups.items():
            indices.sort(key=lambda i: self.samples[i]['frame_num'])
            for pos, idx in enumerate(indices):
                s = self.samples[idx]
                if pos > 0:
                    s['frame_prev_hit'] = self.samples[indices[pos - 1]]['frame_num']
                else:
                    s['frame_prev_hit'] = s.get('rally_frame_start', s['frame_num'])
                if pos < len(indices) - 1:
                    s['frame_next_hit'] = self.samples[indices[pos + 1]]['frame_num']
                else:
                    s['frame_next_hit'] = s.get('rally_frame_end', s['frame_num'])

    # ── helpers ───────────────────────────────────────────────────────────────

    def _load_rally(self, npy_path: str) -> np.ndarray:
        """Load rally skeleton with in-memory cache (rally npys are small ~300KB)."""
        if npy_path not in self._rally_cache:
            self._rally_cache[npy_path] = np.load(npy_path)
        return self._rally_cache[npy_path]

    def _slice_window(self, rally_skel: np.ndarray, hit_idx: int,
                       var_start_idx: int = None, var_end_idx: int = None) -> np.ndarray:
        """
        Extract shot_window frames centred on hit_idx.

        If var_start_idx and var_end_idx are provided (variable-width mode),
        the full hit-to-hit span is extracted and then resampled to
        shot_window frames via linear interpolation, so every sample has
        the same temporal length regardless of the original span.

        Fixed-width mode centres on hit_idx with zero-padding if needed.
        """
        C, T_rally, V = rally_skel.shape

        if var_start_idx is not None and var_end_idx is not None:
            # Variable-width: extract the full hit-to-hit span, then resample.
            vstart = max(0, min(var_start_idx, T_rally - 1))
            vend = min(var_end_idx, T_rally)
            if vend <= vstart:
                vend = vstart + 1
            segment = rally_skel[:, vstart:vend, :].copy()
            return self._resample_temporal(segment, self.shot_window)
        else:
            # Fixed-width: centre on hit_idx
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

    @staticmethod
    def _resample_temporal(segment: np.ndarray, target_len: int) -> np.ndarray:
        """
        Resample (C, T_src, V) to (C, target_len, V) via linear interpolation.

        Handles variable-width windows that may be shorter or longer than
        the fixed shot_window.
        """
        C, T_src, V = segment.shape
        if T_src == target_len:
            return segment
        if T_src == 0:
            return np.zeros((C, target_len, V), dtype=segment.dtype)

        # Linear interpolation along temporal axis
        src_indices = np.linspace(0, T_src - 1, target_len)
        lower = np.floor(src_indices).astype(int)
        upper = np.minimum(lower + 1, T_src - 1)
        frac = (src_indices - lower).reshape(1, -1, 1)  # (1, target_len, 1)

        return segment[:, lower, :] * (1 - frac) + segment[:, upper, :] * frac

    def _load_shuttle_rally(self, match_name: str, rally_key: str):
        """
        Load and cache per-rally shuttle trajectory + frame numbers.

        SS shuttle files are per-rally:
            {match}_s{set}r{rally}.npy        — shape (T, 3) [x, y, vis]
            {match}_s{set}r{rally}_frames.npy — shape (T,)   absolute frame nums

        Returns:
            (traj, frames) or (None, None) if not found
        """
        cache_key = f'_shuttle_{match_name}_{rally_key}'
        if cache_key not in self._rally_cache:
            traj_path = self.shuttle_dir / f"{match_name}_{rally_key}.npy"
            frames_path = self.shuttle_dir / f"{match_name}_{rally_key}_frames.npy"
            if traj_path.exists() and frames_path.exists():
                self._rally_cache[cache_key] = (
                    np.load(traj_path),   # (T, 3)
                    np.load(frames_path), # (T,)
                )
            else:
                self._rally_cache[cache_key] = (None, None)
        return self._rally_cache[cache_key]

    def _append_shuttle_ss(self, skeleton: np.ndarray, match_name: str,
                           frame_num: int, rally_key: str = None) -> np.ndarray:
        """
        Append dense shuttle trajectory as virtual node 34 for ShuttleSet.

        Loads the per-rally trajectory (T, 3) + frame numbers (T,) and
        slices the window matching the skeleton's temporal span.

        Args:
            skeleton: (2, T, V) raw skeleton (pixel coords)
            match_name: match directory name
            frame_num: absolute frame number of the hit
            rally_key: e.g. 's1r3' for set 1 rally 3

        Returns:
            (2, T, V+1) skeleton with shuttle appended
        """
        C, T, V = skeleton.shape
        shuttle_node = np.zeros((C, T, 1), dtype=skeleton.dtype)

        if rally_key is None:
            return np.concatenate([skeleton, shuttle_node], axis=2)

        traj, frames = self._load_shuttle_rally(match_name, rally_key)
        if traj is not None and len(frames) > 0:
            half = self.shot_window // 2
            window_start = frame_num - half

            for t in range(T):
                target_frame = window_start + t
                # Find closest detected frame
                dists = np.abs(frames - target_frame)
                nearest_idx = int(np.argmin(dists))
                # Use if within 2 frames and visible
                if dists[nearest_idx] <= 2 and traj[nearest_idx, 2] >= 0.5:
                    shuttle_node[0, t, 0] = traj[nearest_idx, 0]  # x
                    shuttle_node[1, t, 0] = traj[nearest_idx, 1]  # y

        return np.concatenate([skeleton, shuttle_node], axis=2)

    def _extract_shuttle_trajectory(self, match_name: str, frame_num: int,
                                     rally_key: str = None) -> torch.Tensor:
        """
        Extract shuttle trajectory as a (2, shot_window) tensor for cross-attention.

        Returns x,y positions over the shot window. Zeros where not visible.
        """
        T = self.shot_window
        traj_out = np.zeros((2, T), dtype=np.float32)

        if rally_key is None:
            return torch.tensor(traj_out, dtype=torch.float32)

        traj, frames = self._load_shuttle_rally(match_name, rally_key)
        if traj is not None and len(frames) > 0:
            half = T // 2
            window_start = frame_num - half
            for t in range(T):
                target_frame = window_start + t
                dists = np.abs(frames - target_frame)
                nearest_idx = int(np.argmin(dists))
                if dists[nearest_idx] <= 2 and traj[nearest_idx, 2] >= 0.5:
                    traj_out[0, t] = traj[nearest_idx, 0]
                    traj_out[1, t] = traj[nearest_idx, 1]

        return torch.tensor(traj_out, dtype=torch.float32)

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def _get_feature_eng(self, match_name: str) -> 'FeatureEngineer':
        """Return a FeatureEngineer with the correct per-match homography."""
        if match_name not in self._fe_cache:
            h = self._homography_dict.get(match_name)
            if h is not None:
                self._fe_cache[match_name] = FeatureEngineer(
                    feature_layer=self._feature_layer, homography=h,
                    use_hitter=self._use_hitter)
            else:
                self._fe_cache[match_name] = self.feature_eng  # no-H fallback
        return self._fe_cache[match_name]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        match_name = None

        if self._mode == 'whole_match':
            info       = sample  # dict
            skel_dir   = Path(info['skel_dir'])
            match_name = skel_dir.name
            whole_skel = self._load_rally(str(skel_dir / 'skeletons.npy'))  # (2, T, 34)
            fn_arr     = info['frame_nums_arr']
            # Find array index for this shot's frame_num
            diffs    = np.abs(fn_arr - info['frame_num'])
            hit_idx  = int(np.argmin(diffs))

            var_start, var_end = None, None
            if self.variable_window and 'frame_prev_hit' in info:
                # Convert absolute frames to array indices
                prev_diffs = np.abs(fn_arr - info['frame_prev_hit'])
                next_diffs = np.abs(fn_arr - info['frame_next_hit'])
                var_start = int(np.argmin(prev_diffs))
                var_end = int(np.argmin(next_diffs)) + 1  # exclusive
            raw_skel = self._slice_window(whole_skel, hit_idx, var_start, var_end)

            shot_type_idx = info.get('shot_type_idx')

        elif self._mode == 'per_rally':
            info = sample  # dict
            match_name = info.get('match_name')
            rally_skel = self._load_rally(info['npy_path'])
            hit_idx = info['frame_num'] - info['rally_frame_start']

            var_start, var_end = None, None
            if self.variable_window and 'frame_prev_hit' in info:
                rfs = info['rally_frame_start']
                var_start = info['frame_prev_hit'] - rfs
                var_end = info['frame_next_hit'] - rfs + 1  # exclusive
            raw_skel = self._slice_window(rally_skel, hit_idx, var_start, var_end)

            shot_type_idx = info.get('shot_type_idx')

        elif self._mode == 'per_shot':
            npy_path, shot_type_idx = sample
            match_name = Path(npy_path).parent.name
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

        # Shuttle handling depends on fusion mode
        shuttle_tensor = None
        if self.use_shuttle and match_name and self._mode in ('whole_match', 'per_rally'):
            frame_num = info['frame_num']
            rally_key = info.get('rally_key')

            if self.shuttle_fusion == 'graph':
                # Virtual node 34 appended to skeleton graph
                raw_skel = self._append_shuttle_ss(raw_skel, match_name, frame_num,
                                                   rally_key=rally_key)
            elif self.shuttle_fusion == 'cross_attn':
                # Extract shuttle as separate (2, T) tensor for cross-attention
                shuttle_tensor = self._extract_shuttle_trajectory(
                    match_name, frame_num, rally_key)

        fe = self._get_feature_eng(match_name) if match_name else self.feature_eng
        hitter = None
        if self._use_hitter:
            if isinstance(sample, dict):
                hitter = sample.get('hitter')
        features = fe.compute(raw_skel, hitter=hitter)
        x = torch.tensor(features, dtype=torch.float32)

        if self.transform:
            x = self.transform(x)

        label = shot_type_idx if shot_type_idx is not None else -1

        if shuttle_tensor is not None:
            return x, label, shuttle_tensor
        return x, label


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
