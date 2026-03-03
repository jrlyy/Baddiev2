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

import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import numpy as np

# ─── Paths (relative to this file) ──────────────────────────────────────────
ROOT         = Path(__file__).parent
IMG_DIR      = ROOT / "Datasets/FineBadminton-dataset/dataset/image"
SKEL_DIR      = ROOT / "datasets_preprocessing/finebadminton_skeletons"
GDINO_SKEL_DIR = ROOT / "datasets_preprocessing/finebadminton_skeletons_gdino"
SHUTTLE_DIR  = ROOT / "datasets_preprocessing/finebadminton_shuttles"
ANN_FILE     = ROOT / "Datasets/FineBadminton-dataset/dataset/transformed_combined_rounds_output_en_evals_translated.json"
HTML_FILE    = ROOT / "badminton_pipeline_demo.html"

IMG_W, IMG_H = 1280, 720  # FineBadminton native resolution

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


def _npy_to_players(sk: np.ndarray, frame_idx: int) -> list:
    """Extract two-player joint list from (2, T, 34) array at frame_idx."""
    t = min(max(frame_idx, 0), sk.shape[1] - 1)
    frame = sk[:, t, :]  # (2, 34)
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


# ─── Request handler ─────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    SHOTS: list[dict] = []
    RALLY_WINDOWS: dict[str, list] = {}

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
        path = self.path.split("?")[0].rstrip("/")

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

        # Shuttle position: /api/shuttle/{video_id}/{frame_idx}
        # frame_idx is the relative index into the trajectory array (0-based)
        if path.startswith("/api/shuttle/"):
            parts    = path.lstrip("/").split("/")
            video_id = parts[2] if len(parts) > 2 else ""
            frame_idx = int(parts[3]) if len(parts) > 3 else 0
            data = get_shuttle_frame(video_id, frame_idx)
            self._send_json(data)
            return

        self._send_404()


# ─── Entry point ─────────────────────────────────────────────────────────────
def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7860
    shots = build_shots()
    rally_windows = build_rally_windows()

    print("=" * 55)
    print("  Badminton Pipeline Demo  –  Local Frame Server")
    print("=" * 55)
    print(f"  Shots indexed  : {len(shots)}")
    from collections import Counter
    dist = dict(Counter(s["strategy"] for s in shots))
    for k, v in dist.items():
        print(f"    {k:<18} {v:>3} shots")
    sk_rallies = sorted(set(
        s["video"] for s in shots
        if (SKEL_DIR / f"{s['video']}.npy").exists()
    ))
    sh_rallies = sorted(p.stem for p in SHUTTLE_DIR.glob("*.npy")) if SHUTTLE_DIR.exists() else []
    print(f"  Skeleton data  : {len(sk_rallies)} rallies")
    print(f"  Shuttle data   : {len(sh_rallies)} rallies")
    print(f"\n  Open: http://localhost:{port}")
    print("  (Ctrl-C to stop)\n")

    # Attach shots and windows to handler class so they're accessible per-request
    Handler.SHOTS = shots
    Handler.RALLY_WINDOWS = rally_windows

    server = HTTPServer(("", port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
