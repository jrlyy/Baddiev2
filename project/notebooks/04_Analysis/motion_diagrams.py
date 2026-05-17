"""
Per-shot-type skeleton motion diagrams.

Answers: "do all smashes / net shots / etc. share a similar hitter motion?"

For every shot type we pool the *hitter's* 17-joint skeleton over the 32-frame
shot window, normalise out court position and player size, and draw two views:

  1. xattn... no — motion_strobe_per_type.png
     A strobe sheet: one row per shot type, mean stick-figure at 5 timesteps,
     with every individual shot drawn faintly behind it. Tight grey cloud =
     consistent motion; wide cloud = the type is pose-ambiguous.

  2. motion_trails_per_type.png
     One panel per shot type: mean skeleton at contact (t=16) plus the mean
     trajectory of the wrists / ankles / head across the whole window,
     coloured by time. Emphasises the swing path.

Normalisation per shot: centre on the hip midpoint at contact, scale by median
torso length. Top-court hitters are seen front-on by the camera while
bottom-court hitters are seen from behind, so top-court x is mirrored to put
every hitter in the same view.

Run from the repo root:  python notebooks/04_Analysis/motion_diagrams.py
"""
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.config import SS_SKELETONS_GDINO, SHOT_TYPES, COCO_SKELETON
from src.data.dataset import ShuttleSetDataset

SHOT_WINDOW   = 32
CONTACT       = SHOT_WINDOW // 2          # t = 16
STROBE_FRAMES = [0, 8, 16, 24, 31]
MIN_SHOTS     = 15                        # skip types with fewer test+train shots
OUT_DIR       = REPO / "figures"

# COCO joint indices
NOSE = 0
L_SH, R_SH = 5, 6
L_HIP, R_HIP = 11, 12
TRAIL_JOINTS = {9: "L wrist", 10: "R wrist", 15: "L ankle", 16: "R ankle", 0: "head"}


# ── load + normalise ────────────────────────────────────────────────────────
def collect_hitter_skeletons():
    """Return {shot_type_name: list of (2, T, 17) normalised hitter skeletons}."""
    ds = ShuttleSetDataset(skeleton_dir=SS_SKELETONS_GDINO, shot_window=SHOT_WINDOW,
                           load_shot_types=True, split=None)

    by_type = {name: [] for name in SHOT_TYPES}
    skipped = {"no_label": 0, "no_hitter": 0, "bad_norm": 0}

    for info in ds.samples:
        label  = info.get("shot_type_idx")
        hitter = info.get("hitter", "")
        if label is None:
            skipped["no_label"] += 1
            continue
        if hitter not in ("top", "bottom"):
            skipped["no_hitter"] += 1
            continue

        whole   = ds._load_rally(str(Path(info["skel_dir"]) / "skeletons.npy"))
        fn_arr  = info["frame_nums_arr"]
        hit_idx = int(np.argmin(np.abs(fn_arr - info["frame_num"])))
        raw     = ds._slice_window(whole, hit_idx)            # (2, T, 34)

        j0 = 0 if hitter == "top" else 17
        skel = raw[:, :, j0:j0 + 17].astype(np.float64).copy()  # (2, T, 17)

        norm = _normalise(skel, mirror_x=(hitter == "top"))
        if norm is None:
            skipped["bad_norm"] += 1
            continue
        by_type[SHOT_TYPES[label]].append(norm)

    print(f"Loaded {sum(len(v) for v in by_type.values())} hitter skeletons "
          f"| skipped: {skipped}")
    return by_type


def _normalise(skel, mirror_x):
    """Centre on contact-frame hip midpoint, scale by median torso length.

    skel: (2, T, 17) pixel coords, 0 = missing joint.
    Returns (2, T, 17) with NaN where the joint is missing, or None if the
    contact frame is too sparse to anchor on.
    """
    xy   = skel.copy()
    miss = (xy[0] == 0) & (xy[1] == 0)            # (T, 17)
    xy[:, miss] = np.nan

    hips_c = xy[:, CONTACT, [L_HIP, R_HIP]]       # (2, 2)
    if np.isnan(hips_c).all():
        return None
    centre = np.nanmean(hips_c, axis=1)           # (2,)

    # torso length: shoulder-mid -> hip-mid, median over frames that have both
    sh  = np.nanmean(xy[:, :, [L_SH, R_SH]],  axis=2)   # (2, T)
    hip = np.nanmean(xy[:, :, [L_HIP, R_HIP]], axis=2)  # (2, T)
    torso = np.sqrt(((sh - hip) ** 2).sum(axis=0))      # (T,)
    scale = np.nanmedian(torso)
    if not np.isfinite(scale) or scale < 1e-3:
        return None

    out = (xy - centre[:, None, None]) / scale
    if mirror_x:
        out[0] = -out[0]
    # image y grows downward -> flip so "up" is up in the plots
    out[1] = -out[1]
    return out


# ── drawing helpers ─────────────────────────────────────────────────────────
def _draw_skeleton(ax, pose, color, lw, alpha, zorder=2):
    """pose: (2, 17) with NaN for missing joints."""
    segs = []
    for a, b in COCO_SKELETON:
        pa, pb = pose[:, a], pose[:, b]
        if np.isnan(pa).any() or np.isnan(pb).any():
            continue
        segs.append([(pa[0], pa[1]), (pb[0], pb[1])])
    if segs:
        ax.add_collection(LineCollection(segs, colors=color, linewidths=lw,
                                         alpha=alpha, zorder=zorder))
    ok = ~np.isnan(pose).any(axis=0)
    ax.scatter(pose[0, ok], pose[1, ok], s=lw * 4, c=color,
               alpha=alpha, zorder=zorder + 1)


def _mean_pose(stack, t):
    """stack: (n, 2, T, 17) -> masked mean pose (2, 17) at frame t."""
    return np.nanmean(stack[:, :, t, :], axis=0)


# ── figure 1: strobe sheet ──────────────────────────────────────────────────
def make_strobe(by_type):
    types = [(n, np.stack(v)) for n, v in by_type.items() if len(v) >= MIN_SHOTS]
    n     = len(types)
    ncol  = len(STROBE_FRAMES)
    fig, axes = plt.subplots(n, ncol, figsize=(2.2 * ncol, 2.2 * n),
                             squeeze=False)

    # shared limits across the whole sheet
    all_xy = np.concatenate([s.reshape(s.shape[0], 2, -1) for _, s in types], axis=0)
    lim = np.nanpercentile(np.abs(all_xy), 99) * 1.1

    for r, (name, stack) in enumerate(types):
        n_shots = stack.shape[0]
        show    = stack if n_shots <= 60 else stack[
            np.random.default_rng(0).choice(n_shots, 60, replace=False)]
        for c, t in enumerate(STROBE_FRAMES):
            ax = axes[r][c]
            for ind in show:
                _draw_skeleton(ax, ind[:, t, :], "#888888", 0.6, 0.10, zorder=1)
            _draw_skeleton(ax, _mean_pose(stack, t), "#c0392b", 2.0, 0.95, zorder=4)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t={t}" + ("  (contact)" if t == CONTACT else ""),
                             fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{name}\n(n={n_shots})", fontsize=9)

    fig.suptitle("Hitter motion by shot type — red = mean pose, grey = "
                 "individual shots", fontsize=12)
    plt.tight_layout()
    out = OUT_DIR / "motion_strobe_per_type.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── figure 2: motion trails ─────────────────────────────────────────────────
def make_trails(by_type):
    types = [(n, np.stack(v)) for n, v in by_type.items() if len(v) >= MIN_SHOTS]
    n     = len(types)
    ncol  = 4
    nrow  = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.4 * nrow),
                             squeeze=False)

    all_xy = np.concatenate([s.reshape(s.shape[0], 2, -1) for _, s in types], axis=0)
    lim = np.nanpercentile(np.abs(all_xy), 99) * 1.15

    for k, (name, stack) in enumerate(types):
        ax = axes[k // ncol][k % ncol]
        _draw_skeleton(ax, _mean_pose(stack, CONTACT), "#2c3e50", 2.0, 0.9, zorder=4)
        for j in TRAIL_JOINTS:
            traj = np.nanmean(stack[:, :, :, j], axis=0)        # (2, T)
            pts  = traj.T.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc   = LineCollection(segs, cmap="viridis",
                                  norm=plt.Normalize(0, SHOT_WINDOW - 1),
                                  linewidths=2.0, alpha=0.9, zorder=3)
            lc.set_array(np.arange(SHOT_WINDOW - 1))
            ax.add_collection(lc)
            ax.scatter(*traj[:, CONTACT], s=30, facecolors="none",
                       edgecolors="red", linewidths=1.5, zorder=5)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{name}  (n={stack.shape[0]})", fontsize=9)

    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    sm = plt.cm.ScalarMappable(cmap="viridis",
                               norm=plt.Normalize(0, SHOT_WINDOW - 1))
    cb = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.02)
    cb.set_label("frame t  (red ring = contact, t=16)", fontsize=9)

    fig.suptitle("Hitter joint trails by shot type — wrists / ankles / head "
                 "across the shot window", fontsize=12)
    out = OUT_DIR / "motion_trails_per_type.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    by_type = collect_hitter_skeletons()
    print("\nShots per type:")
    for name in SHOT_TYPES:
        c = len(by_type[name])
        flag = "" if c >= MIN_SHOTS else "  (skipped, < %d)" % MIN_SHOTS
        print(f"  {name:<24} {c:>5}{flag}")
    print()
    make_strobe(by_type)
    make_trails(by_type)