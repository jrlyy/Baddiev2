"""
Shot type inference on unseen FineBadminton rallies.

Loads any ablation checkpoint and auto-detects its architecture (in_channels,
num_classes, pooling, cross-attention) from the saved weights.  Works with
run4 checkpoints (17 cls, 9-dim L2) and run6 checkpoints (18 cls, 10-dim L2)
without any manual configuration.

Usage:
    predictor = ShotTypePredictor(
        "models/ablation_C3_shuttle_crossattn.pt",
        roboflow_api_key="...",
    )
    results = predictor.run_fb_inference(
        json_path="datasets/FineBadminton-dataset/dataset/transformed_combined_rounds_zh.json",
        skel_dir="datasets_preprocessing/finebadminton_skeletons_gdino_v2",
        shuttle_dir="datasets_preprocessing/finebadminton_shuttles",
        img_dir="datasets/FineBadminton-dataset/dataset/image",
    )
    # Each dict: rally_id, hit_frame, hitter, player,
    #            predicted, confidence, top5,
    #            fb_label, fb_label_en, H_available

    import json
    with open("results/fb_inference.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
"""
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from .config import PROJECT_ROOT, NUM_JOINTS, NUM_NODES
from .data.graph_builder import GraphBuilder
from .data.feature_eng import FeatureEngineer
from .models.stgcn_model import STGCN
from .models.shuttle_cross_attn import ShuttleCrossAttention


# ── Known class vocabularies (keyed by num_classes) ───────────────────────────
# Add new run vocabs here as training produces them.

_VOCAB = {
    17: [                       # run4 (Mar 2025)
        'short_serve', 'long_serve', 'smash', 'tap_smash', 'push_rush',
        'clear', 'slice_drop', 'net_drop', 'transition', 'drive',
        'block', 'lob_lift', 'defensive_lift', 'cross_net', 'net_shot',
        'smash_defense', 'push',
    ],
    18: [                       # run6 (May 2025)
        'net shot', 'return net', 'smash', 'wrist smash', 'lob',
        'defensive return lob', 'clear', 'drive', 'driven flight',
        'back-court drive', 'drop', 'passive drop', 'push', 'rush',
        'defensive return drive', 'cross-court net shot',
        'short service', 'long service',
    ],
}

# ── FB hit_type (English) → model class, keyed by which vocab is active ─────
# FineBadminton uses English labels natively (kill, drive, drop shot, …).
# The 'zh' annotation file is just a translation — we parse the English file.
_FB_LABEL_MAP = {
    17: {   # run4 vocab
        'serve':                'short_serve',
        'kill':                 'smash',
        'clear':                'clear',
        'drive':                'drive',
        'drop shot':            'slice_drop',
        'net shot':             'net_drop',
        'block':                'block',
        'push shot':            'push',
        'net lift':             'lob_lift',
        'net kill':             'push_rush',
        'cross-court net shot': 'cross_net',
    },
    18: {   # run6 vocab
        'serve':                'short service',
        'kill':                 'smash',
        'clear':                'clear',
        'drive':                'drive',
        'drop shot':            'drop',
        'net shot':             'net shot',
        'block':                'return net',
        'push shot':            'push',
        'net lift':             'lob',
        'net kill':             'rush',
        'cross-court net shot': 'cross-court net shot',
    },
}

# ── Feature dimension → feature_layer tag (for FeatureEngineer) ───────────────
# in_channels can also include +1 (hitter), +2 (bones), combinations.
# We infer the base layer and extra flags from in_channels.
_LAYER_BY_BASEDIM = {2: 'L0', 6: 'L1', 9: 'L2_v1', 10: 'L2', 13: 'L3_v1', 14: 'L3'}

_SHOT_WINDOW = 32

# FineBadminton frames: 1280×720; SS training frames: 1920×1080.
# Shuttle coords are rescaled so the cross-attention _norm_scale matches training.
_FB_W, _FB_H = 1280.0, 720.0
_SS_W, _SS_H = 1920.0, 1080.0

# Court geometry
_COURT_LEN = 13.4
_COURT_W   = 6.1
_NET_Y     = _COURT_LEN / 2        # 6.7 m
_SERVICE_Y = 1.98
_COURT_KP_M = np.array([
    [0.0,        0.0       ],  # 0 TL
    [_COURT_W/2, 0.0       ],  # 1 TC
    [_COURT_W,   0.0       ],  # 2 TR
    [0.0,        _SERVICE_Y],  # 3 TSL
    [_COURT_W,   _SERVICE_Y],  # 4 TSR
    [0.0,        _NET_Y    ],  # 5 NL
    [_COURT_W,   _NET_Y    ],  # 6 NR
    [0.0,        _COURT_LEN - _SERVICE_Y],  # 7 BSL
    [_COURT_W,   _COURT_LEN - _SERVICE_Y],  # 8 BSR
    [0.0,        _COURT_LEN],  # 9  BL
    [_COURT_W/2, _COURT_LEN],  # 10 BC
    [_COURT_W,   _COURT_LEN],  # 11 BR
], dtype=np.float32)
_CORNER_IDX  = [0, 2, 9, 11]      # TL, TR, BL, BR
_RF_MODEL_ID = "badminton-court-detection-cfgah/3"
_RF_API_URL  = "https://serverless.roboflow.com"

_CENTROID_JOINTS = [5, 6, 11, 12]  # hips + shoulders


# ── Feature engineering ───────────────────────────────────────────────────────

def _compute_features(skeleton, in_channels, homography=None):
    """
    Compute features matching the given checkpoint's in_channels.

    For in_channels=9 (run4 L2): uses the 3-feature court-context formula
    that was active at training time (dist_net, dist_center, dist_opp).

    For all other configs: delegates to FeatureEngineer which reflects
    the current codebase definitions.

    Args:
        skeleton:    (2, T, V) raw pixel coordinates
        in_channels: expected output feature depth (from checkpoint)
        homography:  optional (3, 3) pixel → court-metres transform

    Returns:
        (in_channels, T, V) float32
    """
    if in_channels == 9:
        return _compute_9dim_l2(skeleton, homography)

    # Infer flags from in_channels
    use_hitter = False
    use_bones  = False
    base_ch    = in_channels

    if base_ch > 14:            # L3(14) + extras
        if base_ch % 2 == 0:
            use_bones = True; base_ch -= 2
        if base_ch == 15:
            use_hitter = True; base_ch -= 1
    elif base_ch > 10:          # L2(10) or L3(13/14) + extras
        if (base_ch - 10) == 2:
            use_bones = True; base_ch = 10
        elif (base_ch - 10) == 1:
            use_hitter = True; base_ch = 10
        elif (base_ch - 10) == 3:
            use_bones = True; use_hitter = True; base_ch = 10

    layer_map = {2: 'L0', 6: 'L1', 10: 'L2', 14: 'L3'}
    feature_layer = layer_map.get(base_ch, 'L2')

    eng = FeatureEngineer(
        feature_layer=feature_layer,
        homography=homography,
        use_hitter=use_hitter,
        use_bones=use_bones,
    )
    return eng.compute(skeleton).astype(np.float32)


def _compute_9dim_l2(skeleton, homography=None):
    """
    Run4 L2 feature formula: [x,y, vx,vy, ax,ay, dist_net, dist_center, dist_opp].
    The current FeatureEngineer computes 10-dim L2 (added lateral_to_opp later).
    """
    _, T, V = skeleton.shape
    feats = skeleton.astype(np.float32)            # (2, T, V)

    vel = np.zeros_like(feats)
    vel[:, 1:] = feats[:, 1:] - feats[:, :-1]
    acc = np.zeros_like(feats)
    acc[:, 1:] = vel[:, 1:] - vel[:, :-1]
    feats = np.concatenate([feats, vel, acc], axis=0)   # (6, T, V)

    def _centroid(joints):
        # skeleton[0] → (T, V); then [:, joints] → (T, n_joints); mean(1) → (T,)
        return skeleton[0][:, joints].mean(1), skeleton[1][:, joints].mean(1)

    def _transform(cx, cy):
        if homography is None:
            return cx, cy
        pts = np.stack([cx, cy, np.ones_like(cx)])
        t   = homography @ pts
        w   = np.where(np.abs(t[2]) < 1e-10, 1.0, t[2])
        return t[0] / w, t[1] / w

    p0_cx, p0_cy = _transform(*_centroid(_CENTROID_JOINTS))
    p1_joints    = [j + NUM_JOINTS for j in _CENTROID_JOINTS]
    p1_cx, p1_cy = _transform(*_centroid(p1_joints))

    dist_net = np.zeros((T, V), dtype=np.float32)
    dist_net[:, :NUM_JOINTS] = np.abs(p0_cy - _NET_Y)[:, None]
    dist_net[:, NUM_JOINTS:] = np.abs(p1_cy - _NET_Y)[:, None]

    court_cx, court_cy = _COURT_W / 2, _NET_Y
    dist_ctr = np.zeros((T, V), dtype=np.float32)
    dist_ctr[:, :NUM_JOINTS] = np.sqrt((p0_cx - court_cx)**2 + (p0_cy - court_cy)**2)[:, None]
    dist_ctr[:, NUM_JOINTS:] = np.sqrt((p1_cx - court_cx)**2 + (p1_cy - court_cy)**2)[:, None]

    c2c      = np.sqrt((p0_cx - p1_cx)**2 + (p0_cy - p1_cy)**2)
    dist_opp = np.zeros((T, V), dtype=np.float32)
    dist_opp[:, :NUM_JOINTS] = c2c[:, None]
    dist_opp[:, NUM_JOINTS:] = c2c[:, None]

    ctx = np.stack([dist_net, dist_ctr, dist_opp], axis=0)     # (3, T, V)
    return np.concatenate([feats, ctx], axis=0)                 # (9, T, V)


def _slice_window(arr_2tv, center, window=_SHOT_WINDOW):
    """Slice (2, T_rally, V) → (2, window, V) centred on center; zero-pads at edges."""
    _, T, V = arr_2tv.shape
    half    = window // 2
    out     = np.zeros((2, window, V), dtype=np.float32)
    src_lo  = max(0, center - half)
    src_hi  = min(T, center - half + window)
    dst_lo  = src_lo - (center - half)
    out[:, dst_lo: dst_lo + (src_hi - src_lo)] = arr_2tv[:, src_lo:src_hi]
    return out


def _slice_shuttle(arr_t3, center, window=_SHOT_WINDOW):
    """Slice (T_rally, 3) shuttle → (window, 2) [x,y]; zeros where vis=0 or out of range."""
    T    = arr_t3.shape[0]
    half = window // 2
    out  = np.zeros((window, 2), dtype=np.float32)
    src_lo = max(0, center - half)
    src_hi = min(T, center - half + window)
    dst_lo = src_lo - (center - half)
    n      = src_hi - src_lo
    xy     = arr_t3[src_lo:src_hi, :2].astype(np.float32)
    vis    = arr_t3[src_lo:src_hi, 2]
    xy[vis == 0] = 0.0
    out[dst_lo: dst_lo + n] = xy
    return out


def _build_shuttle_tensor(shuttle_window, img_w=_FB_W, img_h=_FB_H):
    """
    Convert (T, 2) [x_pixel, y_pixel] → (4, T) [x_norm, y_norm, dx, dy].

    Matches the patched training extraction (v2): normalise to [0,1] using
    the frame resolution, then append finite-difference velocity channels.
    Frames where x==0 and y==0 are treated as not visible; dx/dy stay zero
    unless both the current and previous frames are visible.

    Args:
        shuttle_window: (T, 2) float32 pixel coords; zeros = not visible
        img_w, img_h:  frame resolution used for normalisation

    Returns:
        (4, T) float32
    """
    T  = shuttle_window.shape[0]
    xy = shuttle_window.astype(np.float32)
    xn = np.where(xy[:, 0] != 0, xy[:, 0] / img_w, 0.0)
    yn = np.where(xy[:, 1] != 0, xy[:, 1] / img_h, 0.0)
    dx = np.zeros(T, dtype=np.float32)
    dy = np.zeros(T, dtype=np.float32)
    for t in range(1, T):
        if xn[t] > 0 and xn[t - 1] > 0:
            dx[t] = xn[t] - xn[t - 1]
            dy[t] = yn[t] - yn[t - 1]
    return np.stack([xn, yn, dx, dy], axis=0)   # (4, T)


# ── Homography estimation ─────────────────────────────────────────────────────

def _extract_rf_corners(preds, conf_thresh=0.3):
    import cv2
    all_corners = []
    for p in preds:
        if p.get('confidence', 0) < conf_thresh:
            continue
        if p.get('keypoints'):
            pts = np.array([[kp['x'], kp['y']] for kp in p['keypoints']], dtype=np.float32)
        elif p.get('points'):
            pts = np.array([[pt['x'], pt['y']] for pt in p['points']], dtype=np.float32)
        else:
            cx, cy = p.get('x', 0), p.get('y', 0)
            bw, bh = p.get('width', 0), p.get('height', 0)
            pts = np.array([[cx-bw/2, cy-bh/2], [cx+bw/2, cy-bh/2],
                            [cx-bw/2, cy+bh/2], [cx+bw/2, cy+bh/2]], dtype=np.float32)
        if len(pts) < 4:
            continue
        hull  = cv2.convexHull(pts)
        peri  = cv2.arcLength(hull, True)
        approx = hull
        for eps in [0.02, 0.03, 0.05, 0.08, 0.10, 0.15]:
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) <= 6:
                break
        all_corners.append(approx.reshape(-1, 2))
    return np.vstack(all_corners) if all_corners else np.empty((0, 2), dtype=np.float32)


def _corners_to_H(corners):
    import cv2
    if len(corners) < 4:
        return None
    hull  = cv2.convexHull(corners)
    peri  = cv2.arcLength(hull, True)
    quad  = hull
    for eps in [0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
        quad = cv2.approxPolyDP(hull, eps * peri, True)
        if len(quad) == 4:
            break
    if len(quad) != 4:
        pts = corners
        tl  = pts[np.argmin(pts[:, 0] + pts[:, 1])]
        tr  = pts[np.argmax(pts[:, 0] - pts[:, 1])]
        bl  = pts[np.argmin(pts[:, 0] - pts[:, 1])]
        br  = pts[np.argmax(pts[:, 0] + pts[:, 1])]
        quad_pts = np.array([tl, tr, bl, br], dtype=np.float32)
    else:
        q = quad.reshape(4, 2).astype(np.float32)
        q = q[np.argsort(q[:, 1])]
        top    = q[:2][np.argsort(q[:2, 0])]
        bottom = q[2:][np.argsort(q[2:, 0])]
        quad_pts = np.array([top[0], top[1], bottom[0], bottom[1]], dtype=np.float32)

    dst = _COURT_KP_M[_CORNER_IDX].astype(np.float32)
    H, mask = cv2.findHomography(quad_pts, dst, cv2.RANSAC, 5.0)
    if H is None or (mask is not None and mask.sum() < 3):
        return None
    return H.astype(np.float32)


def estimate_homography_roboflow(img_path, rf_client):
    """
    Run Roboflow court detection and return a pixel→metres (3,3) H matrix, or None.
    """
    try:
        result  = rf_client.infer(str(img_path), model_id=_RF_MODEL_ID)
        preds   = result.get('predictions', [])
        corners = _extract_rf_corners(preds)
        return _corners_to_H(corners)
    except Exception as e:
        print(f"[WARN] Roboflow failed for {img_path}: {e}")
        return None


# ── Predictor ─────────────────────────────────────────────────────────────────

class ShotTypePredictor:
    """
    Predict shot types from pre-extracted FineBadminton skeleton + shuttle data.

    Auto-detects the checkpoint's architecture (in_channels, num_classes,
    pooling, cross-attention presence) so it works with any ablation checkpoint
    regardless of which training run produced it.
    """

    def __init__(self, checkpoint_path=None, roboflow_api_key=None, device=None):
        """
        Args:
            checkpoint_path:  path to .pt file; defaults to the best run4 model
                              models/ablation_C3_shuttle_crossattn.pt
            roboflow_api_key: API key for automatic court homography estimation.
                              Pass None to skip (L2 court features stay in pixel space).
            device:           torch device; auto-detected if None.
        """
        self.device = device or torch.device(
            'mps'  if torch.backends.mps.is_available()  else
            'cuda' if torch.cuda.is_available()          else
            'cpu'
        )
        ckpt_path = Path(checkpoint_path or
                         (PROJECT_ROOT / 'models' / 'ablation_C3_shuttle_crossattn.pt'))
        self._load_model(ckpt_path)

        self._rf_client     = None
        self._H_cache: dict = {}

        if roboflow_api_key:
            try:
                from inference_sdk import InferenceHTTPClient
                self._rf_client = InferenceHTTPClient(
                    api_url=_RF_API_URL, api_key=roboflow_api_key,
                )
            except ImportError:
                print("[WARN] inference-sdk not installed — run: pip install inference-sdk")

    # ── model loading ─────────────────────────────────────────────────────────

    def _load_model(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        enc_sd  = ckpt['encoder_state_dict']
        head_sd = ckpt['head_state_dict']

        # Auto-detect architecture from saved weights
        self.in_channels  = int(enc_sd['layers.0.spatial.conv.weight'].shape[1])
        self.num_classes  = int(head_sd['weight'].shape[0])
        self.has_cross_attn = 'cross_attn_state_dict' in ckpt
        pooling = 'attn' if 'temporal_attn.weight' in enc_sd else 'mean'

        self.shot_types = _VOCAB.get(
            self.num_classes,
            [f'class_{i}' for i in range(self.num_classes)],
        )
        self._label_map = _FB_LABEL_MAP.get(self.num_classes, {})

        graph = GraphBuilder()
        adj   = graph.build_adjacency().to(self.device)

        self.encoder = STGCN(
            in_channels=self.in_channels,
            num_nodes=NUM_NODES,
            adjacency=adj,
            num_layers=9, base_channels=64,
            embedding_dim=256, temporal_kernel=9, dropout=0.3,
            pooling=pooling,
        ).to(self.device)
        self.encoder.load_state_dict(enc_sd)

        self.cross_attn = None
        if self.has_cross_attn:
            # Run4 training patched shuttle_tcn to in_channels=4 (x,y,dx,dy).
            # The _norm_scale buffer was also not yet in the checkpoint (added later).
            # We load strict=False and override _norm_scale to a no-op so the
            # pre-normalised (4-ch) shuttle tensors pass through unchanged.
            from .models.shuttle_cross_attn import ShuttleTCN
            ca = ShuttleCrossAttention(d_skel=256, d_shuttle=128, nhead=4).to(self.device)
            ca.shuttle_tcn = ShuttleTCN(in_channels=4, d_model=128).to(self.device)
            ca.load_state_dict(ckpt['cross_attn_state_dict'], strict=False)
            # _norm_scale may have been created as (1,2,1); resize to (1,4,1) ones
            # so the division in forward() is a no-op for any number of channels.
            ca._norm_scale = torch.ones(1, 4, 1, device=self.device)
            ca.eval()
            self.cross_attn = ca

        self.head = nn.Linear(256, self.num_classes).to(self.device)
        self.head.load_state_dict(head_sd)

        self.encoder.eval()
        self.head.eval()

        print(
            f"Loaded {ckpt_path.name}  "
            f"[acc={ckpt.get('accuracy', 0):.3f}  f1={ckpt.get('macro_f1', 0):.3f}]  "
            f"in_ch={self.in_channels}  classes={self.num_classes}  "
            f"cross_attn={self.has_cross_attn}  pooling={pooling}  "
            f"device={self.device}"
        )

    # ── single-shot prediction ────────────────────────────────────────────────

    def predict_shot(self, skeleton_window, shuttle_window, homography=None):
        """
        Predict shot type for a single pre-sliced window.

        Args:
            skeleton_window: (2, T, 34) numpy array — raw pixel coords
            shuttle_window:  (T, 2) numpy array — [x, y]; zeros = not visible
            homography:      optional (3, 3) pixel→metres transform

        Returns:
            dict: predicted (str), confidence (float), top5 (list of (str, float))
        """
        feat = _compute_features(skeleton_window, self.in_channels, homography)
        x    = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.encoder(x)

            if self.has_cross_attn and shuttle_window is not None:
                sh = _build_shuttle_tensor(shuttle_window, img_w=_FB_W, img_h=_FB_H)
                sh = torch.tensor(sh, dtype=torch.float32).unsqueeze(0).to(self.device)
                emb = self.cross_attn(emb, sh)

            logits = self.head(emb)[0]
            probs  = torch.softmax(logits, dim=0).cpu().numpy()

        order = probs.argsort()[::-1]
        return {
            'predicted':  self.shot_types[int(order[0])],
            'confidence': float(probs[order[0]]),
            'top5':       [(self.shot_types[int(i)], float(probs[i])) for i in order[:5]],
        }

    # ── full FB inference pipeline ────────────────────────────────────────────

    def run_fb_inference(self, json_path, skel_dir, shuttle_dir, img_dir=None):
        """
        Run inference over all annotated shots in a FineBadminton JSON.

        Args:
            json_path:   path to transformed_combined_rounds_zh.json
            skel_dir:    directory of per-rally skeleton .npy  shape (2, T, 34)
            shuttle_dir: directory of per-rally shuttle .npy   shape (T, 3)
            img_dir:     directory of reference .jpg frames for homography.
                         Pass None to skip homography estimation.

        Returns:
            list of dicts, one per annotated shot:
                rally_id, hit_frame, hitter, player,
                fb_label, fb_label_en,
                predicted, confidence, top5,
                H_available
        """
        skel_dir    = Path(skel_dir)
        shuttle_dir = Path(shuttle_dir)
        img_dir     = Path(img_dir) if img_dir else None

        shots   = _parse_fb_annotations(json_path)
        results = []
        missing = 0

        for shot in shots:
            rally_id = shot['rally_id']
            skel_p   = skel_dir    / f"{rally_id}.npy"
            shut_p   = shuttle_dir / f"{rally_id}.npy"

            if not skel_p.exists():
                missing += 1
                continue

            match_id = rally_id.rsplit('_', 1)[0]
            H        = self._get_homography(match_id, img_dir)

            skeleton = np.load(skel_p)
            shuttle  = np.load(shut_p) if shut_p.exists() else None

            lf       = shot['local_hit_frame']
            skel_win = _slice_window(skeleton, lf)
            shut_win = _slice_shuttle(shuttle, lf) if shuttle is not None \
                       else np.zeros((_SHOT_WINDOW, 2), dtype=np.float32)

            pred = self.predict_shot(skel_win, shut_win, H)
            pred.update({
                'rally_id':    rally_id,
                'hit_frame':   shot['hit_frame'],
                'hitter':      shot['hitter'],
                'player':      shot['player'],
                'fb_label':    shot['hit_type'],
                'fb_label_en': self._label_map.get(shot['hit_type'], '?'),
                'H_available': H is not None,
            })
            results.append(pred)

        if missing:
            print(f"[WARN] {missing} shots skipped — skeleton .npy not found in {skel_dir}")
        print(f"Done: {len(results)} shots predicted  ({len(shots)} total annotations)")
        return results

    # ── homography helpers ────────────────────────────────────────────────────

    def _get_homography(self, match_id, img_dir):
        if match_id in self._H_cache:
            return self._H_cache[match_id]
        H = None
        if img_dir is not None and self._rf_client is not None:
            ref = _find_reference_frame(match_id, img_dir)
            if ref is not None:
                H      = estimate_homography_roboflow(ref, self._rf_client)
                status = "OK" if H is not None else "FAILED"
                print(f"  H [{match_id}]: {status}  ({ref.name})")
            else:
                print(f"  H [{match_id}]: no image found in {img_dir}")
        self._H_cache[match_id] = H
        return H


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_fb_annotations(json_path):
    """Parse FineBadminton JSON → list of per-shot dicts."""
    with open(json_path, encoding='utf-8') as f:
        rallies = json.load(f)
    shots = []
    for rally in rallies:
        rally_id    = rally['video'].replace('.mp4', '')
        start_frame = int(rally['start_frame'])
        for shot in rally.get('hitting', []):
            hit_frame = shot.get('hit_frame')
            hit_type  = shot.get('hit_type', '').strip()
            if hit_frame is None or not hit_type:
                continue
            shots.append({
                'rally_id':        rally_id,
                'hit_frame':       int(hit_frame),
                'local_hit_frame': int(hit_frame) - start_frame,
                'hit_type':        hit_type,
                'hitter':          shot.get('hitter', ''),
                'player':          shot.get('player', ''),
            })
    return shots


def _find_reference_frame(match_id, img_dir):
    """Return the first .jpg for match_id in img_dir, or None."""
    candidates = sorted(Path(img_dir).glob(f"{match_id}_*.jpg"))
    return candidates[0] if candidates else None