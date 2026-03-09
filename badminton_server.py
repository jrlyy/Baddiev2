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
    print(f"  Court homography loaded from {COURT_DIR}")
else:
    print(f"  Court homography: not found or degenerate — run notebook 07 first")

# ─── ShuttleSet paths ─────────────────────────────────────────────────────────
SS_PREPROCESS      = ROOT / "datasets_preprocessing"
SS_FRAMES_DIR      = SS_PREPROCESS / "shuttleset_frames"
SS_SKEL_DIR        = SS_PREPROCESS / "shuttleset_skeletons"
SS_SKEL_GDINO_DIR  = SS_PREPROCESS / "shuttleset_skeletons_gdino"
SS_OUTPUTS_DIR     = SS_PREPROCESS / "shuttleset_outputs"   # pre-built JSON shot metadata
SS_SHUTTLE_DIR     = SS_PREPROCESS / "shuttleset_shuttles"  # TrackNetV4 detections
SS_MATCHES_CSV     = ROOT / "datasets" / "ShuttleSet" / "set" / "match.csv"
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

# Placeholder confidences vary realistically per shot so the demo looks live
def _fake_conf(i):
    return round(0.62 + (i % 7) * 0.045, 3)

def _fake_margin(i):
    return round(0.10 + (i % 9) * 0.036, 3)

# ─── Build shot index from annotations ───────────────────────────────────────
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


def build_ss_shots() -> tuple[list[dict], dict]:
    """
    Build shot index from the pre-processed shuttleset_outputs JSON files.

    Returns:
        shots      - list of per-shot dicts
        rally_index - {match_name: {rally_num: {frame_min, frame_max, shots:[]}}}
                      used by /api/ss/rally to serve the frame range + shot list
    """
    if not SS_OUTPUTS_DIR.exists():
        return [], {}

    shots: list[dict] = []
    rally_index: dict = {}   # match → rally → {frame_min, frame_max, shots}

    for json_file in sorted(SS_OUTPUTS_DIR.glob("*.json")):
        match_name = json_file.stem
        frame_dir  = SS_FRAMES_DIR / match_name
        if not frame_dir.exists():
            continue
        try:
            records = json.loads(json_file.read_text())
        except Exception:
            continue

        # Build frame_num→shuttle lookup for this match (if .npy exists)
        shuttle_frames: set[int] = set()
        sh_path = SS_SHUTTLE_DIR / f"{match_name}.npy"
        if sh_path.exists():
            sh_arr = np.load(sh_path)   # (N, 4) [frame_num, x, y, visible]
            shuttle_frames = {int(r[0]) for r in sh_arr if r[3] > 0}

        rally_index[match_name] = {}

        for rec in records:
            try:
                rally      = int(rec["rally"])
                ball_round = int(rec["ball_round"])
                frame_num  = int(rec["frame_num"])
            except (KeyError, TypeError, ValueError):
                continue
            img_path = frame_dir / f"frame_{frame_num:06d}.jpg"
            if not img_path.exists():
                img_path = frame_dir / f"frame_{frame_num:06d}.png"
            if not img_path.exists():
                continue

            skel_name = f"r{rally:04d}_b{ball_round:04d}.npy"
            # Check per-shot GDINO file OR per-rally GDINO file (new streaming pipeline format)
            has_gdino = (
                (SS_SKEL_GDINO_DIR / match_name / skel_name).exists() or
                (SS_SKEL_GDINO_DIR / match_name / f"r{rally:04d}.npy").exists()
            )
            has_orig  = (SS_SKEL_DIR / match_name / skel_name).exists()

            # rally_frame_start/end come from the new pipeline; fall back to frame_num
            rally_frame_start = int(rec.get("rally_frame_start") or frame_num)
            rally_frame_end   = int(rec.get("rally_frame_end")   or frame_num)
            rally_t           = int(rec.get("rally_t")           or 0)

            # Unique rally key within a match: set_file rallies restart at 1 each set,
            # so we need (set_file, rally) → e.g. "s1r3" for set1 rally 3.
            set_file_str = rec.get("_set_file") or rec.get("set_file", "")
            try:
                set_num = int(''.join(filter(str.isdigit, set_file_str))) or 0
            except (ValueError, TypeError):
                set_num = 0
            rally_uid = f"s{set_num}r{rally}"  # unique per match

            shot = {
                "id":                len(shots),
                "match":             match_name,
                "rally":             rally,
                "rally_uid":         rally_uid,   # unique per match, use for grouping in UI
                "set_num":           set_num,
                "ball_round":        ball_round,
                "frame_num":         frame_num,
                # Temporal index within the rally (= frame_num - rally_frame_start)
                "rally_t":           rally_t,
                "rally_frame_start": rally_frame_start,
                "rally_frame_end":   rally_frame_end,
                "player":            rec.get("player", ""),
                "shot_type":         rec.get("type", ""),
                "player_location_y": _safe_float(rec.get("player_location_y"), None),
                "has_skeleton":      has_gdino or has_orig,
                "has_gdino":         has_gdino,
                "has_shuttle":       frame_num in shuttle_frames,
                # Annotated shuttle positions (human GT, more reliable than TrackNet for SS)
                "hit_x":             _safe_float(rec.get("hit_x"), None),
                "hit_y":             _safe_float(rec.get("hit_y"), None),
                "landing_x":         _safe_float(rec.get("landing_x"), None),
                "landing_y":         _safe_float(rec.get("landing_y"), None),
                "img_w":             SS_IMG_W,
                "img_h":             SS_IMG_H,
            }
            shots.append(shot)

            # Build rally index — keyed by rally_uid so set1-rally1 ≠ set2-rally1
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
                "shot_type":  rec.get("type", ""),
                "player":     rec.get("player", ""),
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
                "conf":        _fake_conf(len(shots)) if not excluded else None,
                "margin":      _fake_margin(len(shots)) if not excluded else None,
                "img_w":       IMG_W,
                "img_h":       IMG_H,
            })

    return shots


# ─── Skeleton loader ─────────────────────────────────────────────────────────
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
    """Load per-rally (2, T_rally, 34) GDINO skeleton, cached."""
    key = f"{match_name}:r{rally:04d}"
    if key not in _SS_GDINO_RALLY_CACHE:
        path = SS_SKEL_GDINO_DIR / match_name / f"r{rally:04d}.npy"
        _SS_GDINO_RALLY_CACHE[key] = np.load(path) if path.exists() else None
    return _SS_GDINO_RALLY_CACHE[key]

def get_ss_skeleton_frame(match_name: str, rally: int, ball_round: int,
                          frame_offset: int = 8) -> list | None:
    """Return original per-shot skeleton at frame_offset (default 8 = centre frame)."""
    sk = load_ss_skeleton(match_name, rally, ball_round)
    return None if sk is None else _npy_to_players(sk, frame_offset)

def _load_ss_skel_gdino_pershot(match_name: str, rally: int, ball_round: int) -> np.ndarray | None:
    """Load old per-shot (2, 16, 34) GDINO skeleton for fallback."""
    key = f"gdino_shot:{match_name}:r{rally:04d}b{ball_round:04d}"
    if key not in _SS_SKEL_CACHE:
        path = SS_SKEL_GDINO_DIR / match_name / f"r{rally:04d}_b{ball_round:04d}.npy"
        _SS_SKEL_CACHE[key] = np.load(path) if path.exists() else None
    return _SS_SKEL_CACHE[key]

def get_ss_skel_gdino_at_frame(match_name: str, rally: int, frame_num: int,
                               rally_frame_start: int,
                               shot_index: dict | None = None) -> tuple[list | None, bool]:
    """Return (skeleton, has_rally_file) at an absolute frame_num.
    Tries per-rally array first; falls back to per-shot GDINO files via shot_index.
    shot_index: {rally_int: [(hit_frame, ball_round)]}
    """
    # Try per-rally file (new format)
    sk = load_ss_skel_gdino_rally(match_name, rally)
    if sk is not None:
        t = frame_num - rally_frame_start
        return _npy_to_players(sk, t), True

    # Fallback: old per-shot GDINO files
    if shot_index:
        for hit_frame, ball_round in shot_index.get(rally, []):
            offset = frame_num - hit_frame + 8
            if 0 <= offset <= 15:
                sk_shot = _load_ss_skel_gdino_pershot(match_name, rally, ball_round)
                if sk_shot is not None:
                    return _npy_to_players(sk_shot, offset), False

    return None, False


# ─── ShuttleSet shuttle loader ────────────────────────────────────────────────
_SS_SHUTTLE_CACHE: dict[str, dict] = {}  # match_name → {frame_num: (x,y,visible)}

def load_ss_shuttle_index(match_name: str) -> dict:
    """Load & index (N,4) shuttle npy as {frame_num: (x,y,visible)}, cached."""
    if match_name not in _SS_SHUTTLE_CACHE:
        path = SS_SHUTTLE_DIR / f"{match_name}.npy"
        if not path.exists():
            _SS_SHUTTLE_CACHE[match_name] = {}
        else:
            arr = np.load(path)   # (N, 4) [frame_num, x, y, visible]
            _SS_SHUTTLE_CACHE[match_name] = {
                int(r[0]): (float(r[1]), float(r[2]), float(r[3]))
                for r in arr
            }
    return _SS_SHUTTLE_CACHE[match_name]


def get_ss_shuttle(match_name: str, frame_num: int) -> dict:
    """Return shuttle position for the given hit-frame, or not-visible."""
    idx = load_ss_shuttle_index(match_name)
    if frame_num not in idx:
        return {"visible": False}
    x, y, vis = idx[frame_num]
    return {"x": x, "y": y, "visible": vis > 0}


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
        valid = (x17 > 0) | (y17 > 0)
        if not valid.any():
            return None
        pts = np.stack([x17[valid], y17[valid]], axis=1).astype(np.float32)
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

    model_path = ROOT / 'models' / 'fewshot_L2.pt'
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

        # Court coords: /api/court_coords/{video_id}/{frame_idx}
        # Returns player positions in court metres + key distances.
        if path.startswith("/api/court_coords/"):
            parts     = path.lstrip("/").split("/")
            video_id  = parts[2] if len(parts) > 2 else ""
            frame_idx = int(parts[3]) if len(parts) > 3 else 0
            self._send_json(get_court_coords(video_id, frame_idx))
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

        # SS skeleton: /api/ss/skeleton/{match_name}/{rally}/{ball_round}[?offset=N]
        # match_name may contain spaces; split from the right to extract the two ints
        if path.startswith("/api/ss/skeleton/"):
            rest = path[len("/api/ss/skeleton/"):]
            parts = rest.rsplit("/", 2)  # [match_name, rally, ball_round]
            if len(parts) == 3:
                try:
                    match_name  = parts[0]
                    rally       = int(parts[1])
                    ball_round  = int(parts[2])
                    frame_offset = int(qs.get("offset", [8])[0])
                    sk = get_ss_skeleton_frame(match_name, rally, ball_round, frame_offset)
                    self._send_json({"skeleton": sk, "has_skeleton": sk is not None})
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
        # frame_num is the absolute frame number; server looks up rally_frame_start
        # and indexes into the per-rally (2, T_rally, 34) array.
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
                        self._send_json({"skeleton": None, "has_skeleton": False, "has_gdino": False})
                        return
                    shot_idx = self.SS_GDINO_SHOT_INDEX.get(match_name)
                    sk, is_rally_file = get_ss_skel_gdino_at_frame(
                        match_name, rally, frame_num, frame_start, shot_idx)
                    self._send_json({
                        "skeleton":     sk,
                        "has_gdino":    sk is not None,
                        "has_skeleton": sk is not None,
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
                    frame_dir = SS_FRAMES_DIR / match_name
                    jpg_path  = frame_dir / f"frame_{frame_num:06d}.jpg"
                    png_path  = frame_dir / f"frame_{frame_num:06d}.png"
                    if jpg_path.exists():
                        self._send_bytes(jpg_path.read_bytes(), "image/jpeg")
                    elif png_path.exists():
                        self._send_bytes(png_path.read_bytes(), "image/png")
                    else:
                        self._send_404()
                except ValueError:
                    self._send_404()
            else:
                self._send_404()
            return

        self._send_404()


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

    Handler.SHOTS                = shots
    Handler.RALLY_WINDOWS        = rally_windows
    Handler.SS_SHOTS             = ss_shots
    Handler.SS_MATCHES           = ss_matches
    Handler.SS_GDINO_MATCHES     = ss_gdino_matches
    Handler.SS_RALLY_INDEX       = ss_rally_index
    Handler.SS_RALLY_FRAME_START = ss_rally_frame_start
    Handler.SS_GDINO_SHOT_INDEX  = ss_gdino_shot_index

    server = HTTPServer(("", port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
