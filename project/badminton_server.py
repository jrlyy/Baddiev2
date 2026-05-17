#!/usr/bin/env python3
"""
Badminton Pipeline Demo — Local Frame Server
=============================================
Serves actual FineBadminton frames + skeleton data to the HTML demo.

Usage:
    python badminton_server.py          # runs on http://localhost:7860
    python badminton_server.py 8080     # custom port

Then open: http://localhost:7860
"""

import csv
import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote, parse_qs, urlparse

import numpy as np

# ─── Paths (relative to this file) ──────────────────────────────────────────
ROOT         = Path(__file__).parent
IMG_DIR      = ROOT / "Datasets/FineBadminton-dataset/dataset/image"
SKEL_DIR      = ROOT / "datasets_preprocessing/finebadminton_skeletons"
GDINO_SKEL_DIR = ROOT / "datasets_preprocessing/finebadminton_skeletons_gdino_v2"
SHUTTLE_DIR  = ROOT / "datasets_preprocessing/finebadminton_shuttles"
ANN_FILE     = ROOT / "Datasets/FineBadminton-dataset/dataset/transformed_combined_rounds_output_en_evals_translated.json"
HTML_FILE    = ROOT / "badminton_pipeline_demo.html"

IMG_W, IMG_H = 1280, 720  # FineBadminton native resolution

# ─── Court homography (pixel → metres) ───────────────────────────────────────
COURT_DIR    = ROOT / "datasets_preprocessing" / "court_homographies"
COURT_LENGTH = 13.4
COURT_WIDTH  = 6.1
NET_Y        = COURT_LENGTH / 2

def _load_homography() -> np.ndarray | None:
    p = COURT_DIR / "H_img_to_court_m.npy"
    if not p.exists():
        return None
    try:
        import cv2
        H = np.load(p)
        test = np.array([[[IMG_W/2, IMG_H/2]]], dtype=np.float32)
        pt   = cv2.perspectiveTransform(test, H)[0, 0]
        if not (0 <= pt[0] <= COURT_WIDTH * 2 and 0 <= pt[1] <= COURT_LENGTH * 2):
            return None
        return H
    except Exception:
        return None

_H_IMG_TO_COURT: np.ndarray | None = _load_homography()
if _H_IMG_TO_COURT is not None:
    print(f"  Court homography (FB) loaded from {COURT_DIR}")
else:
    print(f"  Court homography (FB): not found or degenerate — run notebook 07 first")

# ShuttleSet per-match homographies
_SS_H_DICT: dict = {}
_ss_h_path = COURT_DIR / "ss_per_match_H.npy"
if _ss_h_path.exists():
    try:
        _SS_H_DICT = np.load(str(_ss_h_path), allow_pickle=True).item()
        print(f"  Court homography (SS) loaded: {len(_SS_H_DICT)} matches")
    except Exception:
        pass
if not _SS_H_DICT:
    print(f"  Court homography (SS): not found — run notebook 07 first")

# ─── ShuttleSet paths ─────────────────────────────────────────────────────────
SS_PREPROCESS      = ROOT / "datasets_preprocessing"
SS_FRAMES_DIR      = SS_PREPROCESS / "shuttleset_frames"
SS_SKEL_DIR        = SS_PREPROCESS / "shuttleset_skeletons_yolo"
SS_SKEL_GDINO_DIR  = SS_PREPROCESS / "shuttleset_skeletons_gdino"
SS_OUTPUTS_DIR     = SS_PREPROCESS / "shuttleset_outputs"   # legacy JSON (unused now)
SS_SHUTTLE_DIR     = SS_PREPROCESS / "shuttleset_shuttles"  # TrackNetV4 detections
SS_CSV_ROOT        = ROOT / "datasets" / "ShuttleSet" / "set"  # per-match CSVs
SS_MATCHES_CSV     = SS_CSV_ROOT / "match.csv"
SS_SPLIT_JSON      = SS_PREPROCESS / "shuttleset_split.json"
SS_IMG_W, SS_IMG_H = 1920, 1080

# ─── Strategy mapping ────────────────────────────────────────────────────────
# (canonical_name, excluded_from_model)
# Excluded = cannot be reliably inferred from skeleton data alone
STRAT_MAP = {
    "intercept":             ("intercept",       False),
    "defensive":             ("defensive",       False),
    "move to the net":       ("move_to_net",     False),
    "to create depth":       ("create_depth",    False),
    "passive":               ("passive",         False),
    "deception":             ("deception",       True),
    "hesitation":            ("hesitation",      True),
    "seamlessly":            ("seamlessly",      True),
    "a high net early shot": ("high_net_early",  True),
}

# ─���─ Build shot index from annotations ──────────────────────��────────────────
def build_rally_windows() -> dict[str, list]:
    """
    All annotation hit windows per rally, including unlabeled hits.
    Used by the frontend to show the correct hitter when scrubbing.

    Returns:
        { video_id: [ {"start": abs_frame, "end": abs_frame, "hitter": "top"|"bottom"}, ... ] }
    """
    with open(ANN_FILE) as f:
        ann = json.load(f)
    windows: dict[str, list] = {}
    for entry in ann:
        video_id = entry["video"].replace(".mp4", "")
        w = []
        for hit in entry.get("hitting", []):
            sf       = hit.get("start_frame")
            ef       = hit.get("end_frame")
            hf       = hit.get("hit_frame")
            hitter   = hit.get("hitter", "")
            if sf is not None and ef is not None and hitter:
                w.append({"start": sf, "end": ef,
                          "hit_frame": hf if hf is not None else sf,
                          "hitter": hitter})
        if w:
            windows[video_id] = w
    return windows


def _safe_float(v, default=0.0) -> float:
    """Convert a value (possibly NaN/None/string) to a JSON-safe float."""
    try:
        f = float(v)
        return default if (f != f) else f  # nan check: nan != nan
    except (TypeError, ValueError):
        return default


def _build_rally_hitter_map(csv_dir: Path) -> dict:
    """Build per-rally mapping of player A/B → 'top'/'bottom' using median Y.

    For each rally, the player (A or B) with the smaller median player_location_y
    is on the top (far) court. Returns {(set_num, rally): {'A': 'top'|'bottom', 'B': ...}}.
    """
    from collections import defaultdict
    # Collect Y values per (set, rally, player)
    rally_ys: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for csv_path in sorted(csv_dir.glob("set*.csv")):
        try:
            set_num = int(''.join(filter(str.isdigit, csv_path.name)) or '0')
        except ValueError:
            set_num = 0
        with open(csv_path) as f:
            for row in csv.DictReader(f):
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
            med_a = float(np.median(players['A']))
            med_b = float(np.median(players['B']))
            a_is_top = med_a < med_b
            result[key] = {
                'A': 'top' if a_is_top else 'bottom',
                'B': 'bottom' if a_is_top else 'top',
            }
        elif 'A' in players:
            result[key] = {'A': '', 'B': ''}  # can't determine with one player
        elif 'B' in players:
            result[key] = {'A': '', 'B': ''}
    return result


def _load_split_manifest() -> dict:
    """Load split manifest → {match_name: 'train'|'test'|'held_out'|'unused'}.
    Edit datasets_preprocessing/shuttleset_split.json to reassign matches."""
    if not SS_SPLIT_JSON.exists():
        return {}
    data = json.loads(SS_SPLIT_JSON.read_text())
    out = {}
    for key in ("train", "test", "held_out", "val", "unused", "removed"):
        for m in data.get(key, []):
            out[m] = key
    return out

_SS_SPLIT: dict[str, str] = _load_split_manifest()


def _has_ss_gdino(match_name: str, frame_num: int) -> bool:
    """Check if GDINO skeleton exists for this match+frame.
    New format: skeletons.npy + frame_nums.npy per match folder."""
    gdino_dir = SS_SKEL_GDINO_DIR / match_name
    return (gdino_dir / "skeletons.npy").exists() and (gdino_dir / "frame_nums.npy").exists()


def build_ss_shots() -> tuple[list[dict], dict]:
    """
    Build shot index from ShuttleSet CSV annotations (not output JSONs).
    Only includes matches that have extracted frames.

    Returns:
        shots      - list of per-shot dicts
        rally_index - {match_name: {rally_uid: {frame_min, frame_max, shots:[]}}}
    """
    if not SS_CSV_ROOT.exists():
        return [], {}

    shots: list[dict] = []
    rally_index: dict = {}

    # Find matches with frames (exclude symlink dirs like train/ val/)
    frame_matches = set()
    for d in SS_FRAMES_DIR.iterdir():
        if d.is_dir() and not d.is_symlink():
            if any(d.iterdir()):
                frame_matches.add(d.name)

    for match_name in sorted(frame_matches):
        csv_dir = SS_CSV_ROOT / match_name
        if not csv_dir.exists():
            continue
        frame_dir = SS_FRAMES_DIR / match_name

        # Build shuttle lookup (whole-match or per-rally format)
        shuttle_frames: set[int] = set()
        sh_path = SS_SHUTTLE_DIR / f"{match_name}.npy"
        if sh_path.exists():
            sh_arr = np.load(sh_path)
            shuttle_frames = {int(r[0]) for r in sh_arr if r[3] > 0}
        elif SS_SHUTTLE_DIR.exists():
            for fp in SS_SHUTTLE_DIR.glob(f"{match_name}_s*r*.npy"):
                if fp.stem.endswith("_frames"):
                    continue
                frames_fp = fp.with_name(fp.stem + "_frames.npy")
                if not frames_fp.exists():
                    continue
                coords = np.load(fp)
                frames = np.load(frames_fp)
                for i in range(len(frames)):
                    if coords[i, 2] > 0:
                        shuttle_frames.add(int(frames[i]))

        # Check GDINO availability for this match
        match_has_gdino = (SS_SKEL_GDINO_DIR / match_name / "skeletons.npy").exists()

        rally_index[match_name] = {}

        # Build per-rally hitter map: (set_num, rally) → {'A': 'top'|'bottom', 'B': ...}
        hitter_map = _build_rally_hitter_map(csv_dir)

        for csv_path in sorted(csv_dir.glob("set*.csv")):
            set_file = csv_path.name
            try:
                set_num = int(''.join(filter(str.isdigit, set_file))) or 0
            except ValueError:
                set_num = 0

            with open(csv_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            # Compute per-rally frame ranges
            rally_frames: dict[int, list[int]] = {}
            for row in rows:
                try:
                    rally = int(row["rally"])
                    fn    = int(float(row["frame_num"]))
                except (KeyError, TypeError, ValueError):
                    continue
                rally_frames.setdefault(rally, []).append(fn)

            for row in rows:
                try:
                    rally      = int(row["rally"])
                    ball_round = int(float(row["ball_round"]))
                    frame_num  = int(float(row["frame_num"]))
                except (KeyError, TypeError, ValueError):
                    continue

                # Check frame exists
                img_path = frame_dir / f"frame_{frame_num:06d}.jpg"
                if not img_path.exists():
                    img_path = frame_dir / f"frame_{frame_num:06d}.png"
                if not img_path.exists():
                    continue

                rally_uid = f"s{set_num}r{rally}"
                fns = rally_frames.get(rally, [frame_num])
                rally_frame_start = min(fns)
                rally_frame_end   = max(fns)
                rally_t           = frame_num - rally_frame_start

                # Hitter from per-rally median-Y mapping
                player = row.get("player", "")
                rally_sides = hitter_map.get((set_num, rally), {})
                hitter = rally_sides.get(player, "")

                shot = {
                    "id":                len(shots),
                    "match":             match_name,
                    "rally":             rally,
                    "rally_uid":         rally_uid,
                    "set_num":           set_num,
                    "ball_round":        ball_round,
                    "frame_num":         frame_num,
                    "rally_t":           rally_t,
                    "rally_frame_start": rally_frame_start,
                    "rally_frame_end":   rally_frame_end,
                    "player":            player,
                    "shot_type":         row.get("type", ""),
                    "player_location_y": _safe_float(row.get("player_location_y"), None),
                    "hitter":            hitter,
                    "has_skeleton":      match_has_gdino,
                    "has_gdino":         match_has_gdino,
                    "has_shuttle":       frame_num in shuttle_frames,
                    "hit_x":             _safe_float(row.get("hit_x"), None),
                    "hit_y":             _safe_float(row.get("hit_y"), None),
                    "landing_x":         _safe_float(row.get("landing_x"), None),
                    "landing_y":         _safe_float(row.get("landing_y"), None),
                    "split":             _SS_SPLIT.get(match_name, ""),
                    "img_w":             SS_IMG_W,
                    "img_h":             SS_IMG_H,
                }
                shots.append(shot)

                if rally_uid not in rally_index[match_name]:
                    rally_index[match_name][rally_uid] = {
                        "frame_min": rally_frame_start,
                        "frame_max": rally_frame_end,
                        "shots": [],
                    }
                ri = rally_index[match_name][rally_uid]
                ri["frame_min"] = min(ri["frame_min"], rally_frame_start)
                ri["frame_max"] = max(ri["frame_max"], rally_frame_end)
                ri["shots"].append({
                    "hit_frame":  frame_num,
                    "ball_round": ball_round,
                    "shot_type":  row.get("type", ""),
                    "player":     player,
                    "hitter":     hitter,
                })

    return shots, rally_index


def build_shots() -> list[dict]:
    with open(ANN_FILE) as f:
        ann = json.load(f)

    shots = []
    for entry in ann:
        video_id    = entry["video"].replace(".mp4", "")
        start_frame = entry["start_frame"]           # absolute frame index of rally start

        for hit in entry.get("hitting", []):
            strategies = hit.get("strategies", [])
            matched = next(
                (STRAT_MAP[s.lower()] for s in strategies if s.lower() in STRAT_MAP),
                None,
            )
            if not matched:
                continue

            strategy, excluded = matched
            hit_frame  = hit.get("hit_frame", hit["start_frame"])
            frame_idx  = hit_frame - start_frame     # index into skeleton array

            # Verify frame exists on disk
            if not (IMG_DIR / f"{video_id}_{hit_frame}.jpg").exists():
                continue

            # Build a ±8 frame window; clamp to annotation bounds
            seg_start = max(hit.get("start_frame", hit_frame - 8), start_frame)
            seg_end   = hit.get("end_frame", hit_frame + 8)

            shots.append({
                "id":          len(shots),
                "video":       video_id,
                "strategy":    strategy,
                "excluded":    excluded,
                "hit_frame":   hit_frame,
                "seg_start":   seg_start,
                "seg_end":     seg_end,
                "frame_idx":   frame_idx,          # relative index into .npy
                "hitter":      hit.get("hitter", ""),
                "hit_type":    hit.get("hit_type", ""),
                "img_w":       IMG_W,
                "img_h":       IMG_H,
            })

    return shots


# ─── Skeleton loader ─────────────���───────────────────────────────────────────
_SKEL_CACHE: dict[str, np.ndarray] = {}
_GDINO_SKEL_CACHE: dict[str, np.ndarray] = {}

def load_skeleton(video_id: str) -> np.ndarray | None:
    """Return (2, T, 34) original skeleton array, cached."""
    if video_id not in _SKEL_CACHE:
        path = SKEL_DIR / f"{video_id}.npy"
        _SKEL_CACHE[video_id] = np.load(path) if path.exists() else None
    return _SKEL_CACHE[video_id]

def load_gdino_skeleton(video_id: str) -> np.ndarray | None:
    """Return (2, T, 34) GDINO-guided skeleton array, cached. None if not yet extracted."""
    if video_id not in _GDINO_SKEL_CACHE:
        path = GDINO_SKEL_DIR / f"{video_id}.npy"
        _GDINO_SKEL_CACHE[video_id] = np.load(path) if path.exists() else None
    return _GDINO_SKEL_CACHE[video_id]


# ─── Shuttle loader ──────────────────────────────────────────────────────────
_SHUTTLE_CACHE: dict[str, np.ndarray] = {}

def load_shuttle(video_id: str) -> np.ndarray | None:
    """Return (T, 3) shuttle trajectory array [x, y, visible], cached."""
    if video_id not in _SHUTTLE_CACHE:
        path = SHUTTLE_DIR / f"{video_id}.npy"
        if not path.exists():
            _SHUTTLE_CACHE[video_id] = None
        else:
            _SHUTTLE_CACHE[video_id] = np.load(path)
    return _SHUTTLE_CACHE[video_id]


def get_shuttle_frame(video_id: str, frame_idx: int, trail: int = 8) -> dict:
    """
    Returns shuttle position + trail for a given frame index.
    frame_idx is the *absolute* frame number; we convert to relative via rally offset.
    Returns: {x, y, visible, trail: [{x, y}]}  (coordinates in original pixel space)
    """
    traj = load_shuttle(video_id)
    if traj is None:
        return {"visible": False, "trail": []}

    t = min(max(frame_idx, 0), len(traj) - 1)
    x, y, vis = float(traj[t, 0]), float(traj[t, 1]), float(traj[t, 2])

    trail_pts = []
    for dt in range(trail, 0, -1):
        ti = t - dt
        if ti >= 0 and traj[ti, 2] > 0:
            trail_pts.append({"x": float(traj[ti, 0]), "y": float(traj[ti, 1]), "alpha": dt / trail})

    return {"x": x, "y": y, "visible": vis > 0, "trail": trail_pts}


# ─── ShuttleSet skeleton loader ──────────────────────────────────────────────
_SS_SKEL_CACHE: dict[str, np.ndarray | None] = {}

def load_ss_skeleton(match_name: str, rally: int, ball_round: int) -> np.ndarray | None:
    """Load per-shot (2, 16, 34) original YOLOv8 skeleton, cached."""
    key = f"orig:{match_name}:r{rally:04d}b{ball_round:04d}"
    if key not in _SS_SKEL_CACHE:
        path = SS_SKEL_DIR / match_name / f"r{rally:04d}_b{ball_round:04d}.npy"
        _SS_SKEL_CACHE[key] = np.load(path) if path.exists() else None
    return _SS_SKEL_CACHE[key]

_SS_GDINO_RALLY_CACHE: dict[str, np.ndarray | None] = {}

def load_ss_skel_gdino_rally(match_name: str, rally: int) -> np.ndarray | None:
    """Load per-rally (2, T_rally, 34) GDINO skeleton, cached (old format)."""
    key = f"{match_name}:r{rally:04d}"
    if key not in _SS_GDINO_RALLY_CACHE:
        path = SS_SKEL_GDINO_DIR / match_name / f"r{rally:04d}.npy"
        _SS_GDINO_RALLY_CACHE[key] = np.load(path) if path.exists() else None
    return _SS_GDINO_RALLY_CACHE[key]

# ─── New GDINO format: skeletons.npy + frame_nums.npy per match ─────────────
_SS_GDINO_MATCH_CACHE: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}

def _load_ss_gdino_match(match_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Load whole-match GDINO data: (skeletons (2,T,34), frame_nums (T,)). Cached."""
    if match_name not in _SS_GDINO_MATCH_CACHE:
        sk_path = SS_SKEL_GDINO_DIR / match_name / "skeletons.npy"
        fn_path = SS_SKEL_GDINO_DIR / match_name / "frame_nums.npy"
        if sk_path.exists() and fn_path.exists():
            sk = np.load(sk_path)
            fn = np.load(fn_path)
            # Warn if skeletons appear to be in wrong resolution
            x_max = float(sk[0][sk[0] > 0].max()) if (sk[0] > 0).any() else 0
            if 0 < x_max < SS_IMG_W * 0.4:
                print(f"  ⚠ WARNING: {match_name} skeleton x_max={x_max:.0f} — "
                      f"likely extracted at lower resolution than {SS_IMG_W}×{SS_IMG_H}. "
                      f"Scale the .npy file!")
            _SS_GDINO_MATCH_CACHE[match_name] = (sk, fn)
        else:
            _SS_GDINO_MATCH_CACHE[match_name] = None
    return _SS_GDINO_MATCH_CACHE[match_name]

_SS_GDINO_FN_INDEX: dict[str, dict[int, int]] = {}   # match → {frame_num: time_idx}

def _get_ss_gdino_fn_index(match_name: str) -> dict[int, int]:
    """Build frame_num → array-index lookup for a match's GDINO data."""
    if match_name not in _SS_GDINO_FN_INDEX:
        data = _load_ss_gdino_match(match_name)
        if data is None:
            _SS_GDINO_FN_INDEX[match_name] = {}
        else:
            _, fn_arr = data
            _SS_GDINO_FN_INDEX[match_name] = {int(fn_arr[i]): i for i in range(len(fn_arr))}
    return _SS_GDINO_FN_INDEX[match_name]

def get_ss_skeleton_frame(match_name: str, rally: int, ball_round: int,
                          frame_num: int | None = None) -> list | None:
    """Return per-shot YOLO skeleton at the hit frame.
    Per-shot files are (2, 16, 34); hit sits at index T//2.
    P0=top, P1=bottom; no reordering."""
    sk = load_ss_skeleton(match_name, rally, ball_round)
    if sk is None:
        return None
    return _npy_to_players(sk, sk.shape[1] // 2)

def _load_ss_skel_gdino_pershot(match_name: str, rally: int, ball_round: int) -> np.ndarray | None:
    """Load old per-shot (2, 16, 34) GDINO skeleton for fallback."""
    key = f"gdino_shot:{match_name}:r{rally:04d}b{ball_round:04d}"
    if key not in _SS_SKEL_CACHE:
        path = SS_SKEL_GDINO_DIR / match_name / f"r{rally:04d}_b{ball_round:04d}.npy"
        _SS_SKEL_CACHE[key] = np.load(path) if path.exists() else None
    return _SS_SKEL_CACHE[key]



def get_ss_skel_gdino_at_frame(match_name: str, rally: int, frame_num: int,
                               rally_frame_start: int,
                               shot_index: dict | None = None,
                               hitter: str = "",
                               ) -> tuple[list | None, bool, str]:
    """Return (skeleton, has_match_file, hitter) at an absolute frame_num.

    No skeleton reordering — P0=top-court, P1=bottom-court always.
    Hitter string is passed through so the UI can color them correctly.

    Priority: 1) new whole-match format, 2) old per-rally, 3) old per-shot.
    """
    # 1) New whole-match format: skeletons.npy + frame_nums.npy
    fn_idx = _get_ss_gdino_fn_index(match_name)
    if fn_idx:
        data = _load_ss_gdino_match(match_name)
        if data is not None:
            sk, _ = data
            t = fn_idx.get(frame_num)
            if t is None:
                for offset in range(1, 5):
                    for candidate in (frame_num - offset, frame_num + offset):
                        if candidate in fn_idx:
                            t = fn_idx[candidate]
                            break
                    if t is not None:
                        break
            if t is not None:
                return _npy_to_players(sk, t), True, hitter

    # 2) Old per-rally file
    sk = load_ss_skel_gdino_rally(match_name, rally)
    if sk is not None:
        t = frame_num - rally_frame_start
        return _npy_to_players(sk, t), True, hitter

    # 3) Old per-shot GDINO files
    if shot_index:
        for hit_frame, ball_round in shot_index.get(rally, []):
            offset = frame_num - hit_frame + 8
            if 0 <= offset <= 15:
                sk_shot = _load_ss_skel_gdino_pershot(match_name, rally, ball_round)
                if sk_shot is not None:
                    return _npy_to_players(sk_shot, offset), False, hitter

    return None, False, ""


# ─── ShuttleSet shuttle loader ────────────────────────────────────────────────
_SS_SHUTTLE_CACHE: dict[str, dict] = {}  # match_name → {frame_num: (x,y,visible)}

def load_ss_shuttle_index(match_name: str) -> dict:
    """Load & index shuttle npy as {frame_num: (x,y,visible)}, cached.
    Supports whole-match (N,4) format and per-rally (N,3) + _frames.npy format."""
    if match_name not in _SS_SHUTTLE_CACHE:
        # Prefer per-rally files (dense trajectory) over old per-match files (sparse hit-only)
        merged = {}
        if SS_SHUTTLE_DIR.exists():
            for fp in sorted(SS_SHUTTLE_DIR.glob(f"{match_name}_s*r*.npy")):
                if fp.stem.endswith("_frames"):
                    continue
                frames_fp = fp.with_name(fp.stem + "_frames.npy")
                if not frames_fp.exists():
                    continue
                coords = np.load(fp)       # (N, 3) [x, y, visible]
                frames = np.load(frames_fp) # (N,)   [frame_num]
                for i in range(len(frames)):
                    merged[int(frames[i])] = (float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2]))
        if not merged:
            # Fall back to old whole-match (N,4) format
            path = SS_SHUTTLE_DIR / f"{match_name}.npy"
            if path.exists():
                arr = np.load(path)   # (N, 4) [frame_num, x, y, visible]
                merged = {
                    int(r[0]): (float(r[1]), float(r[2]), float(r[3]))
                    for r in arr
                }
        _SS_SHUTTLE_CACHE[match_name] = merged
    return _SS_SHUTTLE_CACHE[match_name]


def get_ss_shuttle(match_name: str, frame_num: int) -> dict:
    """Return shuttle position for the given hit-frame, or not-visible."""
    idx = load_ss_shuttle_index(match_name)
    if frame_num not in idx:
        return {"visible": False}
    x, y, vis = idx[frame_num]
    return {"x": x, "y": y, "visible": vis > 0}


def get_ss_shuttle_trajectory(match_name: str, rally_uid: str,
                              frame_min: int, frame_max: int) -> dict:
    """Return full shuttle trajectory for a rally.

    Tries dense per-rally file first ({match}_{rally_uid}.npy + _frames.npy),
    then falls back to the whole-match index filtered by [frame_min, frame_max].
    Returns: {points: [{frame, x, y, visible}], format: 'dense'|'sparse'}
    """
    # 1. Try dense per-rally file
    dense_path = SS_SHUTTLE_DIR / f"{match_name}_{rally_uid}.npy"
    frames_path = SS_SHUTTLE_DIR / f"{match_name}_{rally_uid}_frames.npy"
    if dense_path.exists() and frames_path.exists():
        coords = np.load(dense_path)    # (N, 3) [x, y, vis]
        frames = np.load(frames_path)   # (N,)
        points = []
        for i in range(len(frames)):
            f = int(frames[i])
            if frame_min <= f <= frame_max:
                points.append({
                    "frame": f,
                    "x": float(coords[i, 0]),
                    "y": float(coords[i, 1]),
                    "visible": float(coords[i, 2]) > 0,
                })
        return {"points": points, "format": "dense"}

    # 2. Fall back to whole-match index (sparse hit-only positions)
    idx = load_ss_shuttle_index(match_name)
    points = []
    for f in sorted(idx.keys()):
        if f < frame_min:
            continue
        if f > frame_max:
            break
        x, y, vis = idx[f]
        if vis > 0:
            points.append({"frame": f, "x": x, "y": y, "visible": True})
    return {"points": points, "format": "sparse"}


def _is_stuck_frame(sk: np.ndarray, t: int, min_run: int = 3) -> bool:
    """True if sk[:, t, :] is part of a forward-fill artifact run of ≥ min_run identical frames."""
    frame = sk[:, t, :]
    if frame.sum() == 0 or t == 0:
        return False
    if not np.array_equal(frame, sk[:, t - 1, :]):
        return False
    start = max(0, t - (min_run - 1))
    return all(np.array_equal(sk[:, t, :], sk[:, i, :]) for i in range(start, t))

def _npy_to_players(sk: np.ndarray, frame_idx: int) -> list | None:
    """Extract two-player joint list from (2, T, 34) array at frame_idx.
    Returns None if no detection (all zeros) or a forward-fill artifact (stuck run ≥ 3)."""
    t = min(max(frame_idx, 0), sk.shape[1] - 1)
    frame = sk[:, t, :]  # (2, 34)
    if frame.sum() == 0 or _is_stuck_frame(sk, t):
        return None
    p0 = [[float(frame[0, j]), float(frame[1, j])] for j in range(17)]
    p1 = [[float(frame[0, j + 17]), float(frame[1, j + 17])] for j in range(17)]
    return [p0, p1]

def get_skeleton_frame(video_id: str, frame_idx: int) -> list | None:
    """
    Returns skeleton for one frame as two players, each a list of 17 [x, y] pairs:
        [ [[x0,y0], [x1,y1], …, [x16,y16]],   # player 0 (top-court)
          [[x0,y0], …, [x16,y16]] ]             # player 1 (bottom-court)
    Coordinates are in original pixel space (IMG_W × IMG_H).
    """
    sk = load_skeleton(video_id)
    return None if sk is None else _npy_to_players(sk, frame_idx)

def get_gdino_skeleton_frame(video_id: str, frame_idx: int) -> list | None:
    """Same as get_skeleton_frame but from GDINO-guided extraction. None if not extracted yet."""
    sk = load_gdino_skeleton(video_id)
    return None if sk is None else _npy_to_players(sk, frame_idx)


def get_court_coords(video_id: str, frame_idx: int) -> dict:
    """
    Transform skeleton joints at frame_idx to court metres using the shared homography.
    Returns player centre-of-mass positions + key distances.
    """
    import cv2 as _cv2
    if _H_IMG_TO_COURT is None:
        return {"error": "no_homography"}
    sk = load_gdino_skeleton(video_id)
    if sk is None:
        sk = load_skeleton(video_id)
    if sk is None:
        return {"error": "no_skeleton"}

    t     = min(max(frame_idx, 0), sk.shape[1] - 1)
    if _is_stuck_frame(sk, t):
        return {"error": "stuck_frame"}
    frame = sk[:, t, :]   # (2, 34)  row0=x, row1=y

    def _player_court(x17, y17):
        # Use ankle joints (15, 16) for player position — they sit on the
        # floor plane so the homography maps them accurately.  Upper-body
        # joints are elevated and get projected incorrectly (especially
        # for the far-court player).
        ANKLE_L, ANKLE_R = 15, 16
        ankle_x = np.array([x17[ANKLE_L], x17[ANKLE_R]])
        ankle_y = np.array([y17[ANKLE_L], y17[ANKLE_R]])
        valid = (ankle_x > 0) | (ankle_y > 0)
        if not valid.any():
            # Fallback: try all joints if ankles are missing
            all_valid = (x17 > 0) | (y17 > 0)
            if not all_valid.any():
                return None
            pts = np.stack([x17[all_valid], y17[all_valid]], axis=1).astype(np.float32)
            ct = _cv2.perspectiveTransform(pts.reshape(-1, 1, 2), _H_IMG_TO_COURT).reshape(-1, 2)
            return ct.mean(axis=0).tolist()
        pts = np.stack([ankle_x[valid], ankle_y[valid]], axis=1).astype(np.float32)
        ct  = _cv2.perspectiveTransform(pts.reshape(-1, 1, 2), _H_IMG_TO_COURT).reshape(-1, 2)
        return ct.mean(axis=0).tolist()

    p0 = _player_court(frame[0, :17],  frame[1, :17])
    p1 = _player_court(frame[0, 17:],  frame[1, 17:])

    out: dict = {
        "frame_idx":    t,
        "p0":           p0,
        "p1":           p1,
        "court_width":  COURT_WIDTH,
        "court_length": COURT_LENGTH,
        "net_y":        NET_Y,
    }
    if p0:
        out["p0_dist_net"]    = round(abs(p0[1] - NET_Y), 3)
        out["p0_dist_center"] = round(abs(p0[0] - COURT_WIDTH / 2), 3)
    if p1:
        out["p1_dist_net"]    = round(abs(p1[1] - NET_Y), 3)
        out["p1_dist_center"] = round(abs(p1[0] - COURT_WIDTH / 2), 3)
    if p0 and p1:
        out["player_separation"] = round(
            ((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2) ** 0.5, 3)
    return out


def get_ss_court_coords(match_name: str, frame_num: int) -> dict:
    """
    Transform ShuttleSet skeleton joints at frame_num to court metres
    using the per-match homography.
    """
    import cv2 as _cv2
    H = _SS_H_DICT.get(match_name)
    if H is None:
        return {"error": "no_homography", "match": match_name}

    # Load GDINO whole-match skeleton
    fn_idx = _get_ss_gdino_fn_index(match_name)
    data = _load_ss_gdino_match(match_name) if fn_idx else None
    if data is None:
        return {"error": "no_skeleton", "match": match_name}
    sk, _ = data  # (2, T, 34)

    # Find frame index (exact or nearest within ±4)
    t = fn_idx.get(frame_num)
    if t is None:
        for offset in range(1, 5):
            for candidate in (frame_num - offset, frame_num + offset):
                if candidate in fn_idx:
                    t = fn_idx[candidate]
                    break
            if t is not None:
                break
    if t is None:
        return {"error": "frame_not_found", "frame_num": frame_num}

    frame = sk[:, t, :]  # (2, 34)  row0=x, row1=y

    def _player_court(x17, y17):
        ANKLE_L, ANKLE_R = 15, 16
        ankle_x = np.array([x17[ANKLE_L], x17[ANKLE_R]])
        ankle_y = np.array([y17[ANKLE_L], y17[ANKLE_R]])
        valid = (ankle_x > 0) | (ankle_y > 0)
        if not valid.any():
            all_valid = (x17 > 0) | (y17 > 0)
            if not all_valid.any():
                return None
            pts = np.stack([x17[all_valid], y17[all_valid]], axis=1).astype(np.float32)
            ct = _cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
            return ct.mean(axis=0).tolist()
        pts = np.stack([ankle_x[valid], ankle_y[valid]], axis=1).astype(np.float32)
        ct = _cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        return ct.mean(axis=0).tolist()

    p0 = _player_court(frame[0, :17], frame[1, :17])
    p1 = _player_court(frame[0, 17:], frame[1, 17:])

    out: dict = {
        "frame_num":    frame_num,
        "frame_idx":    t,
        "match":        match_name,
        "p0":           p0,
        "p1":           p1,
        "court_width":  COURT_WIDTH,
        "court_length": COURT_LENGTH,
        "net_y":        NET_Y,
    }
    if p0:
        out["p0_dist_net"]    = round(abs(p0[1] - NET_Y), 3)
        out["p0_dist_center"] = round(abs(p0[0] - COURT_WIDTH / 2), 3)
    if p1:
        out["p1_dist_net"]    = round(abs(p1[1] - NET_Y), 3)
        out["p1_dist_center"] = round(abs(p1[0] - COURT_WIDTH / 2), 3)
    if p0 and p1:
        out["player_separation"] = round(
            ((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2) ** 0.5, 3)

    return out


# ─── Inference Engine ────────────────────────────────────────────────────────
# Lazily loads fewshot_L2.pt and provides strategy + hit-type predictions.
# Predictions are pre-computed at startup for all SS shots with GDINO skeletons.

_INFER_LOADED   = False
_ENCODER        = None
_PROTOTYPES     = None   # {int class_idx → np.ndarray (256,)}
_HIT_TYPE_HEAD  = None   # nn.Linear(256 → 11) if checkpoint has one
_FEATURE_ENG    = None
_SHOT_WINDOW    = 16
_STRATEGY_NAMES = ['intercept', 'defensive', 'move_to_net', 'create_depth', 'passive']


def _load_inference_model() -> bool:
    global _INFER_LOADED, _ENCODER, _PROTOTYPES, _HIT_TYPE_HEAD, _FEATURE_ENG
    if _INFER_LOADED:
        return _ENCODER is not None
    _INFER_LOADED = True

    # Accept fewshot_L2.pt or any fewshot_L2_*.pt variant
    _candidates = sorted((ROOT / 'models').glob('fewshot_L2*.pt')) if (ROOT / 'models').exists() else []
    model_path  = ROOT / 'models' / 'fewshot_L2.pt'
    if not model_path.exists() and _candidates:
        model_path = _candidates[0]
    if not model_path.exists():
        print(f"  [inference] No checkpoint at {model_path} — predictions disabled")
        return False

    try:
        sys.path.insert(0, str(ROOT))
        import torch
        import torch.nn as nn
        from src.data.graph_builder import GraphBuilder
        from src.data.feature_eng import FeatureEngineer
        from src.models.stgcn_model import STGCN
        from src.config import get_config, NUM_FB_HIT_TYPES

        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        cfg  = get_config('fewshot')

        adj = GraphBuilder(use_inter_player=True, single_player=False).build_adjacency()
        encoder = STGCN(
            in_channels=cfg.stgcn.in_channels,
            num_nodes=cfg.stgcn.num_nodes,
            adjacency=adj,
            num_layers=cfg.stgcn.num_layers,
            base_channels=cfg.stgcn.base_channels,
            embedding_dim=cfg.stgcn.embedding_dim,
            temporal_kernel=cfg.stgcn.temporal_kernel,
            dropout=0.0,
        )
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        encoder.eval()

        prototypes = {int(k): v.numpy() for k, v in ckpt['prototypes'].items()}

        hit_type_head = None
        if 'hit_type_head_state_dict' in ckpt:
            head = nn.Linear(cfg.stgcn.embedding_dim, NUM_FB_HIT_TYPES)
            head.load_state_dict(ckpt['hit_type_head_state_dict'])
            head.eval()
            hit_type_head = head

        _ENCODER       = encoder
        _PROTOTYPES    = prototypes
        _HIT_TYPE_HEAD = hit_type_head
        _FEATURE_ENG   = FeatureEngineer(feature_layer='L2')

        print(f"  [inference] Loaded {model_path.name}  "
              f"({len(prototypes)} strategy classes"
              f"{', hit-type head' if hit_type_head else ''})")
        return True

    except Exception as exc:
        print(f"  [inference] Load failed: {exc}")
        return False


def predict_strategy(skeleton_window: np.ndarray) -> dict:
    """Run strategy prediction on a (2, T, 34) window. Returns dict with keys:
    strategy, confidence, all_probs, [hit_type_pred, hit_type_conf]"""
    if _ENCODER is None or _PROTOTYPES is None:
        return {}
    try:
        import torch
        feats = _FEATURE_ENG.compute(skeleton_window)          # (9, T, 34)
        x     = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)  # (1,9,T,34)
        with torch.no_grad():
            emb = _ENCODER(x).cpu().numpy()[0]                 # (256,)

        # Distance to each prototype → softmax probabilities
        n = len(_STRATEGY_NAMES)
        neg_d = np.array([-np.linalg.norm(emb - _PROTOTYPES[c])
                          for c in range(n)], dtype=np.float64)
        exp_d = np.exp(neg_d - neg_d.max())
        probs = exp_d / exp_d.sum()
        best  = int(np.argmax(probs))

        result = {
            'predicted_strategy':  _STRATEGY_NAMES[best],
            'strategy_confidence': round(float(probs[best]), 4),
            'strategy_probs':      {_STRATEGY_NAMES[i]: round(float(probs[i]), 4)
                                    for i in range(n)},
        }

        if _HIT_TYPE_HEAD is not None:
            import torch
            from src.config import FB_HIT_TYPES
            with torch.no_grad():
                ht_logits = _HIT_TYPE_HEAD(
                    torch.tensor(emb, dtype=torch.float32).unsqueeze(0))
                ht_probs  = torch.softmax(ht_logits, dim=1).cpu().numpy()[0]
            bht = int(np.argmax(ht_probs))
            result['hit_type_pred'] = FB_HIT_TYPES[bht]
            result['hit_type_conf'] = round(float(ht_probs[bht]), 4)

        return result
    except Exception as exc:
        return {'error': str(exc)}


def _extract_ss_window(match_name: str, rally: int, frame_num: int,
                       rally_frame_start: int) -> np.ndarray | None:
    """Load + slice a T=16 skeleton window for one SS shot."""
    sk = load_ss_skel_gdino_rally(match_name, rally)
    if sk is None:
        return None
    t    = frame_num - rally_frame_start
    half = _SHOT_WINDOW // 2
    s    = max(0, t - half)
    e    = s + _SHOT_WINDOW
    if e > sk.shape[1]:
        e = sk.shape[1]
        s = max(0, e - _SHOT_WINDOW)
    window = sk[:, s:e, :].copy()
    if window.shape[1] < _SHOT_WINDOW:
        pad    = np.zeros((2, _SHOT_WINDOW - window.shape[1], 34), dtype=window.dtype)
        window = np.concatenate([window, pad], axis=1)
    return window


# ─── Request handler ─────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    SHOTS: list[dict] = []
    RALLY_WINDOWS: dict[str, list] = {}
    SS_SHOTS: list[dict] = []
    SS_MATCHES: list[dict] = []        # lightweight summary, pre-computed in main()
    SS_GDINO_MATCHES: list[str] = []   # match names that have any GDINO skeletons
    SS_RALLY_INDEX: dict = {}          # {match: {rally_uid: {frame_min, frame_max, shots}}}
    SS_RALLY_FRAME_START: dict = {}    # {match: {rally_int: frame_start}}
    SS_GDINO_SHOT_INDEX: dict = {}     # {match: {rally_int: [(hit_frame, ball_round)]}}
    SS_SHOT_LOOKUP: dict = {}          # {match: {rally: {ball_round: {frame_num, player_location_y}}}}
    SS_RALLY_SHOTS_SORTED: dict = {}   # {match: {rally: [(frame_num, player_location_y), ...] sorted}}

    def log_message(self, fmt, *args):
        pass  # suppress per-request logs

    # ── Helpers ─────────────────────────────────────────────────────────────
    def _send_json(self, data: object, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, body: bytes, mime: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_404(self):
        self.send_response(404)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Not found")

    # ── Route: CORS preflight ────────────────────────────────────────────────
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.end_headers()

    # ── Route: GET ───────────────────────────────────────────────────────────
    def do_GET(self):
        parsed   = urlparse(self.path)
        path     = unquote(parsed.path.rstrip("/"))
        qs       = parse_qs(parsed.query)

        # Serve HTML demo
        if path in ("", "/", "/index.html"):
            self._send_bytes(HTML_FILE.read_bytes(), "text/html; charset=utf-8")
            return

        # List all labeled shots
        if path == "/api/shots":
            self._send_json(self.SHOTS)
            return

        # Single shot with skeleton
        if path.startswith("/api/shot/"):
            try:
                idx  = int(path.split("/")[-1])
                shot = self.SHOTS[idx]
                sk   = get_skeleton_frame(shot["video"], shot["frame_idx"])
                self._send_json({**shot, "skeleton": sk})
            except (IndexError, ValueError):
                self._send_json({"error": "not found"}, 404)
            return

        # Stream a JPEG frame
        # URL: /frame/{video_id}/{frame_number}
        if path.startswith("/frame/"):
            parts    = path.lstrip("/").split("/")
            video_id = parts[1] if len(parts) > 1 else ""
            frame_n  = parts[2] if len(parts) > 2 else ""
            img_path = IMG_DIR / f"{video_id}_{frame_n}.jpg"
            if img_path.exists():
                self._send_bytes(img_path.read_bytes(), "image/jpeg")
            else:
                self._send_404()
            return

        # Rally metadata: /api/rally/{video_id}
        # Returns frame_min, frame_max, frame_count, and all shots in this rally.
        if path.startswith("/api/rally/"):
            video_id = path.split("/")[-1]
            frame_nums = sorted(
                int(f.stem.split("_")[2])
                for f in IMG_DIR.glob(f"{video_id}_*.jpg")
            )
            if not frame_nums:
                self._send_json({"error": "not found"}, 404)
                return
            rally_shots = [s for s in self.SHOTS if s["video"] == video_id]
            self._send_json({
                "video":        video_id,
                "frame_min":    frame_nums[0],
                "frame_max":    frame_nums[-1],
                "frame_count":  len(frame_nums),
                "shots":        rally_shots,
                # All annotation windows (incl. unlabeled) for per-frame hitter lookup
                "windows":      self.RALLY_WINDOWS.get(video_id, []),
            })
            return

        # Original skeleton: /api/skeleton/{video_id}/{frame_idx}
        if path.startswith("/api/skeleton/"):
            parts     = path.lstrip("/").split("/")
            video_id  = parts[2] if len(parts) > 2 else ""
            frame_idx = int(parts[3]) if len(parts) > 3 else 0
            sk = get_skeleton_frame(video_id, frame_idx)
            self._send_json({"skeleton": sk, "has_skeleton": sk is not None,
                             "source": "original"})
            return

        # GDINO-guided skeleton: /api/skeleton_gdino/{video_id}/{frame_idx}
        # Returns has_gdino=false if that rally hasn't been GDINO-extracted yet.
        if path.startswith("/api/skeleton_gdino/"):
            parts     = path.lstrip("/").split("/")
            video_id  = parts[2] if len(parts) > 2 else ""
            frame_idx = int(parts[3]) if len(parts) > 3 else 0
            sk = get_gdino_skeleton_frame(video_id, frame_idx)
            self._send_json({"skeleton": sk, "has_skeleton": sk is not None,
                             "has_gdino": sk is not None, "source": "gdino"})
            return

        # Which rallies have GDINO skeletons: /api/gdino_rallies
        if path == "/api/gdino_rallies":
            available = [p.stem for p in sorted(GDINO_SKEL_DIR.glob("*.npy"))]
            self._send_json({"rallies": available})
            return

        # Court coords (FB): /api/court_coords/{video_id}/{frame_idx}
        # Returns player positions in court metres + key distances.
        if path.startswith("/api/court_coords/"):
            parts     = path.lstrip("/").split("/")
            video_id  = parts[2] if len(parts) > 2 else ""
            frame_idx = int(parts[3]) if len(parts) > 3 else 0
            self._send_json(get_court_coords(video_id, frame_idx))
            return

        # Court coords (SS): /api/ss/court_coords/{match_name}/{frame_num}
        # Returns player positions in court metres using per-match H.
        if path.startswith("/api/ss/court_coords/"):
            rest = path[len("/api/ss/court_coords/"):]
            parts = rest.rsplit("/", 1)
            if len(parts) == 2:
                try:
                    match_name = parts[0]
                    frame_num = int(parts[1])
                    self._send_json(get_ss_court_coords(match_name, frame_num))
                except ValueError:
                    self._send_json({"error": "invalid frame_num"}, 400)
            else:
                self._send_json({"error": "expected /api/ss/court_coords/{match}/{frame}"}, 400)
            return

        # Shuttle position: /api/shuttle/{video_id}/{frame_idx}
        # frame_idx is the relative index into the trajectory array (0-based)
        if path.startswith("/api/shuttle/"):
            parts    = path.lstrip("/").split("/")
            video_id = parts[2] if len(parts) > 2 else ""
            frame_idx = int(parts[3]) if len(parts) > 3 else 0
            data = get_shuttle_frame(video_id, frame_idx)
            self._send_json(data)
            return

        # ── ShuttleSet routes ────────────────────────────────────────────────

        # SS match summaries (lightweight): /api/ss/matches
        if path == "/api/ss/matches":
            self._send_json(self.SS_MATCHES)
            return

        # All SS shots: /api/ss/shots
        if path == "/api/ss/shots":
            self._send_json(self.SS_SHOTS)
            return

        # SS match shots: /api/ss/match/{match_name}
        if path.startswith("/api/ss/match/"):
            match_name  = path[len("/api/ss/match/"):]
            match_shots = [s for s in self.SS_SHOTS if s["match"] == match_name]
            self._send_json({"match": match_name, "shots": match_shots})
            return

        # SS skeleton: /api/ss/skeleton/{match_name}/{rally}/{ball_round}
        # Returns raw skeleton (P0=top, P1=bottom). Hitter info is in shot data.
        if path.startswith("/api/ss/skeleton/"):
            rest = path[len("/api/ss/skeleton/"):]
            parts = rest.rsplit("/", 2)  # [match_name, rally, ball_round]
            if len(parts) == 3:
                try:
                    match_name = parts[0]
                    rally      = int(parts[1])
                    ball_round = int(parts[2])
                    shot_info  = self.SS_SHOT_LOOKUP.get(match_name, {}).get(rally, {}).get(ball_round)
                    frame_num  = shot_info.get("frame_num") if shot_info else None
                    hitter_str = shot_info.get("hitter", "") if shot_info else ""
                    sk = get_ss_skeleton_frame(match_name, rally, ball_round,
                                               frame_num=frame_num)
                    self._send_json({"skeleton": sk, "has_skeleton": sk is not None,
                                     "hitter": hitter_str})
                except ValueError:
                    self._send_json({"error": "invalid params"}, 400)
            else:
                self._send_json({"error": "not found"}, 404)
            return

        # SS GDINO matches list: /api/ss/gdino_matches
        if path == "/api/ss/gdino_matches":
            self._send_json(self.SS_GDINO_MATCHES)
            return

        # SS GDINO skeleton: /api/ss/skeleton_gdino/{match_name}/{rally}/{frame_num}
        # Returns raw skeleton (P0=top, P1=bottom) + hitter string for UI coloring.
        if path.startswith("/api/ss/skeleton_gdino/"):
            rest  = path[len("/api/ss/skeleton_gdino/"):]
            parts = rest.rsplit("/", 2)
            if len(parts) == 3:
                try:
                    match_name = parts[0]
                    rally      = int(parts[1])
                    frame_num  = int(parts[2])
                    frame_start = self.SS_RALLY_FRAME_START.get(match_name, {}).get(rally)
                    if frame_start is None:
                        self._send_json({"skeleton": None, "has_skeleton": False,
                                         "has_gdino": False, "hitter": None})
                        return
                    # Find hitter for this frame: most recent shot at or before frame_num
                    import bisect
                    hitter = ""
                    rally_shots = self.SS_RALLY_SHOTS_SORTED.get(match_name, {}).get(rally, [])
                    if rally_shots:
                        # rally_shots is [(frame_num, hitter_str), ...] sorted by frame
                        frames_only = [s[0] for s in rally_shots]
                        idx = bisect.bisect_right(frames_only, frame_num) - 1
                        if idx >= 0:
                            hitter = rally_shots[idx][1]
                        elif len(rally_shots) > 0:
                            hitter = rally_shots[0][1]
                    shot_idx = self.SS_GDINO_SHOT_INDEX.get(match_name)
                    sk, is_rally_file, hitter = get_ss_skel_gdino_at_frame(
                        match_name, rally, frame_num, frame_start, shot_idx,
                        hitter=hitter)
                    self._send_json({
                        "skeleton":     sk,
                        "has_gdino":    sk is not None,
                        "has_skeleton": sk is not None,
                        "hitter":       hitter,
                    })
                except ValueError:
                    self._send_json({"error": "invalid params"}, 400)
            else:
                self._send_json({"error": "not found"}, 404)
            return

        # SS rally metadata: /api/ss/rally/{match_name}/{rally_uid}
        # rally_uid = "s{set_num}r{rally}" e.g. "s1r3"
        # Returns frame_min, frame_max, shots — same shape as FB /api/rally
        if path.startswith("/api/ss/rally/"):
            rest  = path[len("/api/ss/rally/"):]
            parts = rest.rsplit("/", 1)   # [match_name, rally_uid]
            if len(parts) == 2:
                match_name = parts[0]
                rally_uid  = parts[1]
                ri = self.SS_RALLY_INDEX.get(match_name, {}).get(rally_uid)
                if ri is None:
                    self._send_json({"error": "not found"}, 404)
                else:
                    self._send_json({
                        "match":      match_name,
                        "rally_uid":  rally_uid,
                        "frame_min":  ri["frame_min"],
                        "frame_max":  ri["frame_max"],
                        "shots":      ri["shots"],
                    })
            else:
                self._send_json({"error": "not found"}, 404)
            return

        # SS accuracy summary: /api/ss/accuracy/{match_name}
        if path.startswith("/api/ss/accuracy/"):
            match_name  = path[len("/api/ss/accuracy/"):]
            match_shots = [s for s in self.SS_SHOTS if s["match"] == match_name]
            confs = [s['strategy_confidence'] for s in match_shots
                     if s.get('strategy_confidence') is not None]
            dist: dict = {}
            for s in match_shots:
                p = s.get('predicted_strategy')
                if p:
                    dist[p] = dist.get(p, 0) + 1
            self._send_json({
                "match":              match_name,
                "total_shots":        len(match_shots),
                "predicted_shots":    len(confs),
                "avg_confidence":     round(float(np.mean(confs)), 4) if confs else None,
                "high_conf_shots":    sum(1 for c in confs if c >= 0.6),
                "strategy_distribution": dist,
            })
            return

        # SS shuttle trajectory: /api/ss/shuttle_trajectory/{match_name}/{rally_uid}
        if path.startswith("/api/ss/shuttle_trajectory/"):
            rest  = path[len("/api/ss/shuttle_trajectory/"):]
            parts = rest.rsplit("/", 1)   # [match_name, rally_uid]
            if len(parts) == 2:
                match_name = parts[0]
                rally_uid  = parts[1]
                ri = self.SS_RALLY_INDEX.get(match_name, {}).get(rally_uid)
                if ri is None:
                    self._send_json({"error": "rally not found"}, 404)
                else:
                    self._send_json(get_ss_shuttle_trajectory(
                        match_name, rally_uid, ri["frame_min"], ri["frame_max"]))
            else:
                self._send_json({"error": "expected /api/ss/shuttle_trajectory/{match}/{rally_uid}"}, 400)
            return

        # SS shuttle position: /api/ss/shuttle/{match_name}/{frame_num}
        if path.startswith("/api/ss/shuttle/"):
            rest  = path[len("/api/ss/shuttle/"):]
            parts = rest.rsplit("/", 1)   # [match_name, frame_num]
            if len(parts) == 2:
                try:
                    match_name = parts[0]
                    frame_num  = int(parts[1])
                    self._send_json(get_ss_shuttle(match_name, frame_num))
                except ValueError:
                    self._send_json({"visible": False})
            else:
                self._send_json({"visible": False})
            return

        # SS frame image: /ss/frame/{match_name}/{frame_num}
        if path.startswith("/ss/frame/"):
            rest = path[len("/ss/frame/"):]
            parts = rest.rsplit("/", 1)   # [match_name, frame_num]
            if len(parts) == 2:
                try:
                    match_name = parts[0]
                    frame_num  = int(parts[1])
                    frame_dir  = SS_FRAMES_DIR / match_name
                    # Search outward for nearest saved frame (handles any stride)
                    found = None
                    for delta in range(0, 8):
                        for fn in ([frame_num - delta, frame_num + delta]
                                   if delta else [frame_num]):
                            for ext, mime in [(".jpg", "image/jpeg"), (".png", "image/png")]:
                                p = frame_dir / f"frame_{fn:06d}{ext}"
                                if p.exists():
                                    found = (p, mime)
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        self._send_bytes(found[0].read_bytes(), found[1])
                    else:
                        self._send_404()
                except ValueError:
                    self._send_404()
            else:
                self._send_404()
            return

        # Shot-type predictions: /api/ss/predictions
        if path == "/api/ss/predictions":
            self._send_json(_load_predictions())
            return

        # Shot-type predictions for a match: /api/ss/predictions/{match_name}
        if path.startswith("/api/ss/predictions/"):
            match_name = path[len("/api/ss/predictions/"):]
            preds = [p for p in _load_predictions() if p["match"] == match_name]
            self._send_json(preds)
            return

        # FB shot-type inference (all shots, cached)
        if path == "/api/fb/infer_all":
            self._send_json(_run_fb_inference())
            return

        # Force-clear the FB inference cache (re-runs next request)
        if path == "/api/fb/infer_reset":
            global _fb_infer_cache
            _fb_infer_cache = None
            _FB_INFER_CACHE_PATH.unlink(missing_ok=True)
            self._send_json({"status": "cache cleared"})
            return

        self._send_404()


# ─── Shot-type prediction cache ──────────────────────────────────────────
_predictions_cache: list[dict] | None = None

def _load_predictions() -> list[dict]:
    """Lazily load shot_type_predictions.json from results/."""
    global _predictions_cache
    if _predictions_cache is not None:
        return _predictions_cache
    pred_path = ROOT / "results" / "shot_type_predictions.json"
    if pred_path.exists():
        _predictions_cache = json.loads(pred_path.read_text())
        print(f"  [predictions] Loaded {len(_predictions_cache)} predictions")
    else:
        _predictions_cache = []
        print(f"  [predictions] No file at {pred_path}")
    return _predictions_cache


# ─── FB Shot-Type Inference ───────────────────────────────────────────────────
_fb_predictor      = None
_fb_infer_cache: list[dict] | None = None
_FB_INFER_CACHE_PATH = ROOT / "results" / "fb_inference.json"
_FB_EN_ANN = ROOT / "Datasets" / "FineBadminton-dataset" / "dataset" / "transformed_combined_rounds_output_en_evals_translated.json"
# Prefer run7 C3 (best single-player L3+bones+shuttle xattn); fall back to run6.
_FB_CKPT_CANDIDATES = [
    ROOT / "models" / "run7" / "C3_shuttle_xattn.pt",
    ROOT / "models" / "run6" / "D4_mlp_bn.pt",
    ROOT / "models" / "run6" / "C3_shuttle_xattn.pt",
    ROOT / "models" / "ablation_C3_shuttle_crossattn.pt",
    ROOT / "models" / "ablation_A1_dual_L2.pt",
]

def _get_fb_predictor():
    global _fb_predictor
    if _fb_predictor is not None:
        return _fb_predictor
    ckpt = next((p for p in _FB_CKPT_CANDIDATES if p.exists()), None)
    if ckpt is None:
        print("[FB infer] No checkpoint found — inference disabled")
        return None
    try:
        sys.path.insert(0, str(ROOT))
        from src.inference_shot_type import ShotTypePredictor
        _fb_predictor = ShotTypePredictor(ckpt, roboflow_api_key=None)
        print(f"[FB infer] Loaded predictor from {ckpt.name}")
        return _fb_predictor
    except Exception as exc:
        print(f"[FB infer] Predictor load failed: {exc}")
        return None


def _run_fb_inference() -> list[dict]:
    global _fb_infer_cache
    if _fb_infer_cache is not None:
        return _fb_infer_cache
    if _FB_INFER_CACHE_PATH.exists():
        _fb_infer_cache = json.loads(_FB_INFER_CACHE_PATH.read_text())
        print(f"[FB infer] Loaded {len(_fb_infer_cache)} cached predictions from disk")
        return _fb_infer_cache
    predictor = _get_fb_predictor()
    if predictor is None:
        _fb_infer_cache = []
        return _fb_infer_cache
    print("[FB infer] Running inference over all annotated shots (TTA=5)…")
    results = predictor.run_fb_inference(
        json_path=_FB_EN_ANN,
        skel_dir=GDINO_SKEL_DIR,
        shuttle_dir=SHUTTLE_DIR,
        img_dir=None,
        default_homography=None,  # FB H uses different origin — skip to avoid bad L3 features
        tta=5,                    # test-time augmentation: 5 passes, averaged softmax
    )
    _FB_INFER_CACHE_PATH.parent.mkdir(exist_ok=True)
    _FB_INFER_CACHE_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    _fb_infer_cache = results
    print(f"[FB infer] Done — {len(results)} predictions cached to {_FB_INFER_CACHE_PATH.name}")
    return _fb_infer_cache


def _write_data_table(ss_matches: list[dict]) -> None:
    """Write datasets_preprocessing/data_table.md — readable match split table.
    Auto-generated on server startup; reflects shuttleset_split.json exactly."""
    DATA_TABLE = SS_PREPROCESS / "data_table.md"
    split_data  = json.loads(SS_SPLIT_JSON.read_text()) if SS_SPLIT_JSON.exists() else {}
    train_set   = set(split_data.get("train",    []))
    test_set    = set(split_data.get("test",     []))
    held_set    = set(split_data.get("held_out", []))
    unused_set  = set(split_data.get("unused",   []))

    known_splits = ("train", "test", "held_out", "val", "unused")
    unknwn = [m for m in ss_matches if m["split"] not in known_splits]

    def _section(title: str, rows: list[dict]) -> list[str]:
        out = [f"## {title} ({len(rows)})", ""]
        if not rows:
            return out + ["*(none yet)*", ""]
        out += [
            "| # | Split | Match | Winner | Loser | Rallies | Shots | Skeleton |",
            "|---|-------|-------|--------|-------|---------|-------|----------|",
        ]
        for i, m in enumerate(sorted(rows, key=lambda x: x["match"]), 1):
            sp   = m["split"].upper() if m["split"] else "?"
            skel = "GDINO" if m.get("has_gdino") else ("yes" if m["skeleton_count"] > 0 else "—")
            out.append(
                f"| {i} | {sp} | `{m['match']}` | {m.get('winner','')} "
                f"| {m.get('loser','')} | {m['rally_count']} | {m['shot_count']} | {skel} |"
            )
        return out + [""]

    lines = [
        "# ShuttleSet Match Split Table",
        "",
        "> **Config:** `datasets_preprocessing/shuttleset_split.json`  ",
        "> 8 train (SSL pre-training) · 1 test (SSL monitoring) · 2 held-out (strategy prediction + expert eval)",
        "",
        f"**Current:** {len(train_set)} train · {len(test_set)} test · {len(held_set)} held-out · {len(unused_set)} unused = {len(train_set)+len(test_set)+len(held_set)+len(unused_set)} total",
        "",
    ]
    lines += _section("Train (SSL pre-training)",                   [m for m in ss_matches if m["split"] == "train"])
    lines += _section("Test (SSL monitoring / checkpoint selection)",[m for m in ss_matches if m["split"] == "test"])
    lines += _section("Held-out (strategy prediction + expert eval)",[m for m in ss_matches if m["split"] == "held_out"])
    lines += _section("Unused",                                      [m for m in ss_matches if m["split"] == "unused"])
    if unknwn:
        lines += _section("Unassigned (not in split.json)", unknwn)

    DATA_TABLE.write_text("\n".join(lines))
    print(f"  data_table.md       : written ({len(ss_matches)} matches)")


# ─── Entry point ────────────────────────────────────────────────────��────────
def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7860
    shots         = build_shots()
    rally_windows = build_rally_windows()
    ss_shots, ss_rally_index = build_ss_shots()

    # Pre-compute strategy predictions for all SS shots with GDINO skeletons
    if _load_inference_model():
        pred_count = 0
        for shot in ss_shots:
            if not shot.get('has_gdino'):
                continue
            window = _extract_ss_window(
                shot['match'], shot['rally'],
                shot['frame_num'], shot['rally_frame_start'])
            if window is None:
                continue
            pred = predict_strategy(window)
            shot.update(pred)
            pred_count += 1
        if pred_count:
            print(f"  [inference] Predicted {pred_count} SS shots")

    # Merge shot-type classifier predictions into ss_shots
    _preds = _load_predictions()
    if _preds:
        # Build lookup: (match, frame_num) → prediction
        _pred_map = {}
        for p in _preds:
            key = (p["match"], p.get("frame_num", 0))
            _pred_map[key] = p
        _merged = 0
        for shot in ss_shots:
            key = (shot["match"], shot.get("frame_num", 0))
            pred = _pred_map.get(key)
            if pred:
                shot["pred_shot_type"] = pred["pred"]
                shot["pred_correct"]   = pred["correct"]
                shot["pred_probs"]     = pred["probs"]
                _merged += 1
        print(f"  [predictions] Merged {_merged}/{len(_preds)} shot-type predictions into SS shots")

    from collections import Counter
    print("=" * 55)
    print("  Badminton Pipeline Demo  –  Local Frame Server")
    print("=" * 55)
    print(f"  FineBadminton shots : {len(shots)}")
    dist = dict(Counter(s["strategy"] for s in shots))
    for k, v in dist.items():
        print(f"    {k:<18} {v:>3} shots")
    sk_rallies = sorted(set(
        s["video"] for s in shots
        if (SKEL_DIR / f"{s['video']}.npy").exists()
    ))
    sh_rallies = sorted(p.stem for p in SHUTTLE_DIR.glob("*.npy")) if SHUTTLE_DIR.exists() else []
    print(f"  FB skeleton data    : {len(sk_rallies)} rallies")
    print(f"  FB shuttle data     : {len(sh_rallies)} rallies")
    ss_match_count = len({s["match"] for s in ss_shots})
    ss_skel        = sum(1 for s in ss_shots if s["has_skeleton"])
    ss_shuttle     = sum(1 for s in ss_shots if s.get("has_shuttle"))
    print(f"  ShuttleSet shots    : {len(ss_shots)} ({ss_match_count} matches)")
    print(f"  SS skeleton data    : {ss_skel} shots")
    print(f"  SS shuttle data     : {ss_shuttle} shots ({len(list(SS_SHUTTLE_DIR.glob('*.npy'))) if SS_SHUTTLE_DIR.exists() else 0} matches)")
    print(f"\n  Open: http://localhost:{port}")
    print("  (Ctrl-C to stop)\n")

    # Load match.csv for player/tournament info
    _csv_info_map: dict[str, dict] = {}
    if SS_MATCHES_CSV.exists():
        with open(SS_MATCHES_CSV) as f:
            for row in csv.DictReader(f):
                vid = row["video"].strip()
                _csv_info_map[vid] = {
                    "winner":     row.get("winner", "").strip(),
                    "loser":      row.get("loser", "").strip(),
                    "tournament": row.get("tournament", "").strip(),
                    "round":      row.get("round", "").strip(),
                    "year":       row.get("year", "").strip(),
                }

    # Pre-compute per-match summaries for the lightweight /api/ss/matches endpoint
    _ss_match_meta: dict[str, dict] = {}
    for s in ss_shots:
        m = s["match"]
        if m not in _ss_match_meta:
            _ss_match_meta[m] = {"match": m, "rally_count": 0, "shot_count": 0, "skeleton_count": 0, "rallies": set()}
        _ss_match_meta[m]["shot_count"] += 1
        _ss_match_meta[m]["skeleton_count"] += int(s.get("has_skeleton", False))
        _ss_match_meta[m]["rallies"].add(s["rally"])

    # SS numbering = alphabetical order of the matches that have extracted data (SS01…SS25)
    _sorted_match_names = sorted(_ss_match_meta.keys())
    _ss_num_map = {name: f"SS{i:02d}" for i, name in enumerate(_sorted_match_names, 1)}

    ss_matches = []
    for name in _sorted_match_names:
        meta     = _ss_match_meta[name]
        csv_info = _csv_info_map.get(name, {})
        ss_matches.append({
            "match":          name,
            "ss_num":         _ss_num_map[name],
            "winner":         csv_info.get("winner", ""),
            "loser":          csv_info.get("loser", ""),
            "tournament":     csv_info.get("tournament", ""),
            "round":          csv_info.get("round", ""),
            "year":           csv_info.get("year", ""),
            "rally_count":    len(meta["rallies"]),
            "shot_count":     meta["shot_count"],
            "skeleton_count": meta["skeleton_count"],
            "split":          _SS_SPLIT.get(name, ""),
            "has_gdino":      (SS_SKEL_GDINO_DIR / name / "skeletons.npy").exists(),
        })

    # Matches that have at least one GDINO skeleton extracted
    ss_gdino_matches = sorted({
        d.name for d in SS_SKEL_GDINO_DIR.iterdir() if d.is_dir()
    }) if SS_SKEL_GDINO_DIR.exists() else []
    if ss_gdino_matches:
        print(f"  SS GDINO data       : {len(ss_gdino_matches)} match(es) — {ss_gdino_matches}")

    # Build {match: {rally_int: frame_start}} for GDINO per-rally indexing
    ss_rally_frame_start: dict = {}
    # Build {match: {rally_int: [(hit_frame, ball_round)]}} for per-shot GDINO fallback
    ss_gdino_shot_index: dict = {}
    for s in ss_shots:
        m   = s["match"]
        ral = s.get("rally")
        fst = s.get("rally_frame_start")
        if ral is not None and fst is not None:
            # Use .setdefault() so first-seen wins — matches records[0] in the notebook
            ss_rally_frame_start.setdefault(m, {}).setdefault(int(ral), int(fst))
        if s.get("has_gdino") and ral is not None:
            ss_gdino_shot_index.setdefault(m, {}).setdefault(int(ral), []).append(
                (s["frame_num"], s["ball_round"])
            )

    # Build {match: {rally: {ball_round: {frame_num, player_location_y}}}}
    ss_shot_lookup: dict = {}
    # Build {match: {rally: [(frame_num, hitter_str), ...] sorted by frame}}
    # hitter_str is "top"|"bottom"|"" — same convention as FineBadminton
    ss_rally_shots_sorted: dict = {}
    for s in ss_shots:
        ss_shot_lookup \
            .setdefault(s["match"], {}) \
            .setdefault(s["rally"], {})[s["ball_round"]] = {
                "frame_num":         s["frame_num"],
                "player_location_y": s.get("player_location_y"),
                "hitter":            s.get("hitter", ""),
            }
        hitter = s.get("hitter", "")
        if hitter:
            ss_rally_shots_sorted \
                .setdefault(s["match"], {}) \
                .setdefault(s["rally"], []).append((s["frame_num"], hitter))
    # Sort each rally's shots by frame_num for bisect lookup
    for m in ss_rally_shots_sorted:
        for r in ss_rally_shots_sorted[m]:
            ss_rally_shots_sorted[m][r].sort()

    Handler.SHOTS                = shots
    Handler.RALLY_WINDOWS        = rally_windows
    # Auto-generate data_table.md as a readable view of the current split
    _write_data_table(ss_matches)

    Handler.SS_SHOTS             = ss_shots
    Handler.SS_MATCHES           = ss_matches
    Handler.SS_GDINO_MATCHES     = ss_gdino_matches
    Handler.SS_RALLY_INDEX       = ss_rally_index
    Handler.SS_RALLY_FRAME_START = ss_rally_frame_start
    Handler.SS_GDINO_SHOT_INDEX  = ss_gdino_shot_index
    Handler.SS_SHOT_LOOKUP       = ss_shot_lookup
    Handler.SS_RALLY_SHOTS_SORTED = ss_rally_shots_sorted

    server = HTTPServer(("", port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
