"""
Node feature engineering for skeleton graphs (L0–L3 layers).

All features are computed from raw skeleton coordinates using simple
numpy operations. No additional data, models, or processing needed.

Feature Layers (cumulative):
    L0: [x, y] — raw court-relative coordinates (2-dim)
    L1: [x, y, vx, vy, ax, ay] — + velocity, acceleration (6-dim)
    L2: [..., dist_net, dist_center, dist_opp] — + court context (9-dim)
    L3: [..., L_elbow, R_elbow, L_knee, R_knee] — + node-specific joint angles (13-dim)

If a homography matrix (3×3 numpy array) is provided, pixel coordinates are
transformed to court-relative coordinates before any features are computed.
This makes all distance/velocity features physically meaningful (in metres).
"""
import numpy as np
from ..config import NUM_JOINTS, NUM_NODES, FEATURE_DIMS, FEATURE_DIMS_WITH_HITTER


# COCO joint indices for angle computation
# Joint angle triplets: (parent, joint, child) — angle at 'joint'
ANGLE_TRIPLETS = {
    # Player joint angles (applied to each player with offset)
    "left_elbow": (5, 7, 9),      # shoulder → elbow → wrist
    "right_elbow": (6, 8, 10),
    "left_knee": (11, 13, 15),     # hip → knee → ankle
    "right_knee": (12, 14, 16),
    # Shoulder angles are computed differently (torso → shoulder → elbow)
}

# Centroid joints for player center of mass approximation
# Use hips (11, 12) and shoulders (5, 6)
CENTROID_JOINTS = [5, 6, 11, 12]


class FeatureEngineer:
    """Compute enriched node features from raw skeleton coordinates."""

    def __init__(self, feature_layer="L2", court_length=13.4, court_width=6.1,
                 net_y=6.7, homography=None, use_hitter=False):
        """
        Args:
            feature_layer: "L0", "L1", "L2", or "L3"
            court_length: full court length in meters (for L2)
            court_width: court width in meters (for L2)
            net_y: y-coordinate of the net in court metres.
                   Court layout: y=0 (far baseline) → y=6.7 (net) → y=13.4 (near baseline)
            homography: optional (3, 3) numpy array mapping pixel coords to
                        court-relative coords (e.g. H_img_to_court_m.npy).
                        If provided, raw (x, y) pixel coords are transformed
                        before any feature computation.
            use_hitter: if True, compute() expects a hitter arg and appends
                        a binary is_hitter channel (1=hitter's joints, 0=opponent's)
        """
        self.feature_layer = feature_layer
        self.court_length = court_length
        self.court_width = court_width
        self.net_y = net_y
        self.court_center_x = court_width / 2    # 3.05 m
        self.court_center_y = net_y               # net line = court center along length
        self.use_hitter = use_hitter
        if use_hitter:
            self.feature_dim = FEATURE_DIMS_WITH_HITTER[feature_layer]
        else:
            self.feature_dim = FEATURE_DIMS[feature_layer]
        self.homography = homography  # (3, 3) or None

    def compute(self, skeleton, hitter=None):
        """
        Compute enriched features from raw skeleton.

        Args:
            skeleton: (2, T, V) — raw [x, y] pixel coordinates
                      C=2 (x, y), T=num_frames, V=34 (17 joints x 2 players)
            hitter: 'top' or 'bottom' (required when use_hitter=True).
                    'top' = P0 (nodes 0-16), 'bottom' = P1 (nodes 17-33).

        Returns:
            features: (feature_dim, T, V)

        Homography is only used for L2 court-context features (distance to
        net, court center, opponent centroid) where physical court positions
        matter.  L0 coords, L1 kinematics, and L3 joint angles stay in pixel
        space so that body proportions and motion patterns are preserved.
        """
        C, T, V = skeleton.shape
        assert C == 2, f"Expected 2 channels (x, y), got {C}"

        # L0: raw pixel coordinates (no homography — preserves body shape)
        features = skeleton.copy()  # (2, T, V)

        if self.feature_layer in ("L1", "L2", "L3"):
            vel, acc = self._compute_kinematics(skeleton)
            features = np.concatenate([features, vel, acc], axis=0)  # (6, T, V)

        if self.feature_layer in ("L2", "L3"):
            court_ctx = self._compute_court_context(skeleton)
            features = np.concatenate([features, court_ctx], axis=0)  # (9, T, V)

        if self.feature_layer == "L3":
            # Joint angles from pixel coords (perspective warp would distort them)
            angles = self._compute_joint_angles(skeleton)
            features = np.concatenate([features, angles], axis=0)  # (12, T, V)

        if self.use_hitter:
            hitter_ch = self._compute_hitter_channel(T, V, hitter)
            features = np.concatenate([features, hitter_ch], axis=0)

        assert features.shape[0] == self.feature_dim, \
            f"Expected {self.feature_dim} features, got {features.shape[0]}"

        return features

    @staticmethod
    def _compute_hitter_channel(T, V, hitter):
        """
        Binary channel: 1.0 for the hitting player's joints, 0.0 for opponent.

        Args:
            T: number of frames
            V: number of nodes (34 or 35 with shuttle)
            hitter: 'top' (P0, nodes 0-16) or 'bottom' (P1, nodes 17-33)

        Returns:
            (1, T, V) array
        """
        ch = np.zeros((1, T, V), dtype=np.float32)
        if hitter == 'top':
            ch[:, :, :NUM_JOINTS] = 1.0
        elif hitter == 'bottom':
            ch[:, :, NUM_JOINTS:NUM_JOINTS * 2] = 1.0
        # If hitter is None/unknown, all zeros (no signal — graceful degradation)
        return ch

    def _apply_homography(self, skeleton):
        """
        Apply perspective transform to pixel coordinates.

        Args:
            skeleton: (2, T, V) in pixel space
        Returns:
            (2, T, V) in court-relative space
        """
        C, T, V = skeleton.shape
        H = self.homography  # (3, 3)

        # Flatten to (T*V, 2), add homogeneous coord → (T*V, 3)
        pts = skeleton.reshape(2, -1).T  # (T*V, 2)
        pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)  # (T*V, 3)

        # Transform: (3, T*V) = H @ (T*V, 3).T
        transformed = (H @ pts_h.T)  # (3, T*V)

        # Perspective divide
        w = transformed[2:3, :]
        w = np.where(np.abs(w) < 1e-10, 1.0, w)
        xy = transformed[:2, :] / w  # (2, T*V)

        return xy.reshape(2, T, V).astype(skeleton.dtype)

    @staticmethod
    def _compute_kinematics(skeleton):
        """
        Compute velocity and acceleration via finite differences.

        Args:
            skeleton: (2, T, V)

        Returns:
            velocity: (2, T, V)
            acceleration: (2, T, V)
        """
        # Velocity: first-order finite difference, zero-padded
        vel = np.zeros_like(skeleton)
        vel[:, 1:, :] = skeleton[:, 1:, :] - skeleton[:, :-1, :]

        # Acceleration: second-order finite difference
        acc = np.zeros_like(skeleton)
        acc[:, 1:, :] = vel[:, 1:, :] - vel[:, :-1, :]

        return vel, acc

    def _transform_point(self, px, py):
        """Transform a single (x, y) through the homography. Returns (cx, cy).

        If no homography, returns the input unchanged.
        """
        if self.homography is None:
            return px, py
        pt = np.stack([px, py, np.ones_like(px)], axis=0)  # (3, N)
        t = self.homography @ pt  # (3, N)
        w = t[2:3]
        w = np.where(np.abs(w) < 1e-10, 1.0, w)
        return t[0] / w[0], t[1] / w[0]

    def _compute_court_context(self, skeleton):
        """
        Compute court-relative distance features.

        Only player centroids (hips + shoulders) are transformed through the
        homography to get court-metre positions.  Individual joint coordinates
        stay in pixel space (handled by L0).

        Features per node per frame (broadcast from centroid):
            - dist_to_net: centroid distance to net line
            - dist_to_center: centroid distance to court center
            - dist_to_opponent: centroid-to-centroid distance

        Args:
            skeleton: (2, T, V) — raw pixel coordinates

        Returns:
            court_features: (3, T, V)
        """
        C, T, V = skeleton.shape
        x = skeleton[0]  # (T, V)
        y = skeleton[1]  # (T, V)

        # Player centroids in pixel space (hips + shoulders)
        p0_cx_px = x[:, CENTROID_JOINTS].mean(axis=1)  # (T,)
        p0_cy_px = y[:, CENTROID_JOINTS].mean(axis=1)
        p1_joints = [j + NUM_JOINTS for j in CENTROID_JOINTS]
        p1_cx_px = x[:, p1_joints].mean(axis=1)
        p1_cy_px = y[:, p1_joints].mean(axis=1)

        # Transform centroids to court metres (or keep pixel if no H)
        p0_cx, p0_cy = self._transform_point(p0_cx_px, p0_cy_px)
        p1_cx, p1_cy = self._transform_point(p1_cx_px, p1_cy_px)

        # dist_to_net — per player centroid, broadcast to all joints
        p0_net = np.abs(p0_cy - self.net_y)  # (T,)
        p1_net = np.abs(p1_cy - self.net_y)
        dist_to_net = np.zeros((T, V))
        dist_to_net[:, :NUM_JOINTS] = p0_net[:, None]
        dist_to_net[:, NUM_JOINTS:] = p1_net[:, None]

        # dist_to_center — per player centroid
        p0_ctr = np.sqrt((p0_cx - self.court_center_x)**2 +
                         (p0_cy - self.court_center_y)**2)
        p1_ctr = np.sqrt((p1_cx - self.court_center_x)**2 +
                         (p1_cy - self.court_center_y)**2)
        dist_to_center = np.zeros((T, V))
        dist_to_center[:, :NUM_JOINTS] = p0_ctr[:, None]
        dist_to_center[:, NUM_JOINTS:] = p1_ctr[:, None]

        # dist_to_opponent — centroid-to-centroid
        c2c = np.sqrt((p0_cx - p1_cx)**2 + (p0_cy - p1_cy)**2)  # (T,)
        dist_to_opponent = np.zeros((T, V))
        dist_to_opponent[:, :NUM_JOINTS] = c2c[:, None]
        dist_to_opponent[:, NUM_JOINTS:] = c2c[:, None]

        return np.stack([dist_to_net, dist_to_center, dist_to_opponent], axis=0)

    @staticmethod
    def _compute_joint_angles(skeleton):
        """
        Compute joint angles at key articulation points.

        Angles computed (node-specific — each angle is placed only on
        its own joint node, zero elsewhere):
            - Left elbow angle   → joint 7 (+ offset for player 2)
            - Right elbow angle  → joint 8
            - Left knee angle    → joint 13
            - Right knee angle   → joint 14

        Args:
            skeleton: (2, T, V)

        Returns:
            angles: (4, T, V) — node-specific angle features
        """
        C, T, V = skeleton.shape
        _ANGLE_NAMES = ["left_elbow", "right_elbow", "left_knee", "right_knee"]
        num_angle_features = len(_ANGLE_NAMES)

        angles = np.zeros((num_angle_features, T, V))

        for player_offset in [0, NUM_JOINTS]:
            for feat_idx, name in enumerate(_ANGLE_NAMES):
                if name not in ANGLE_TRIPLETS:
                    continue
                p, j, c = ANGLE_TRIPLETS[name]
                p_idx = p + player_offset
                j_idx = j + player_offset
                c_idx = c + player_offset

                # Vectors from joint to parent and child
                v1 = skeleton[:, :, p_idx] - skeleton[:, :, j_idx]  # (2, T)
                v2 = skeleton[:, :, c_idx] - skeleton[:, :, j_idx]

                # Angle between vectors
                dot = v1[0] * v2[0] + v1[1] * v2[1]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                angle = np.arctan2(np.abs(cross), dot)  # (T,)

                # Place angle only on its own joint node
                angles[feat_idx, :, j_idx] = angle

        return angles  # (4, T, V)


def compute_features_batch(skeletons, feature_layer="L2", hitters=None, **kwargs):
    """
    Compute features for a batch of skeletons.

    Args:
        skeletons: list of (2, T, V) arrays or (B, 2, T, V) array
        feature_layer: "L0", "L1", "L2", "L3"
        hitters: optional list of 'top'/'bottom'/None per skeleton

    Returns:
        features: list of (feature_dim, T, V) arrays
    """
    eng = FeatureEngineer(feature_layer=feature_layer, **kwargs)

    if hitters is None:
        hitters = [None] * (len(skeletons) if not isinstance(skeletons, np.ndarray)
                            else skeletons.shape[0])

    if isinstance(skeletons, np.ndarray) and skeletons.ndim == 4:
        return np.stack([eng.compute(s, h) for s, h in zip(skeletons, hitters)])

    return [eng.compute(s, h) for s, h in zip(skeletons, hitters)]
