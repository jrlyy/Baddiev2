"""
Node feature engineering for skeleton graphs (L0–L3 layers).

All features are computed from raw skeleton coordinates using simple
numpy operations. No additional data, models, or processing needed.

Feature Layers (cumulative):
    L0: [x, y] — raw court-relative coordinates (2-dim)
    L1: [x, y, vx, vy, ax, ay] — + velocity, acceleration (6-dim)
    L2: [..., dist_net, dist_center, depth_to_opp, lateral_to_opp] — + court context (10-dim)
    L3: [..., L_elbow, R_elbow, L_knee, R_knee] — + node-specific joint angles (13-dim)

If a homography matrix (3×3 numpy array) is provided, pixel coordinates are
transformed to court-relative coordinates before any features are computed.
This makes all distance/velocity features physically meaningful (in metres).
"""
import numpy as np
from ..config import (
    NUM_JOINTS, NUM_NODES, FEATURE_DIMS, FEATURE_DIMS_WITH_HITTER,
    BONE_CHANNELS, COCO_BONE_PARENTS,
)


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
                 net_y=6.7, homography=None, use_hitter=False,
                 use_bones=False, use_bbox_norm=False):
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
            use_bones: if True, append bone vectors (child − parent) as 2
                       extra channels per joint. Encodes limb direction/length.
            use_bbox_norm: if True, normalize each player's joint coordinates
                          relative to their bounding box before computing features.
                          Makes features scale-invariant (handles camera distance).
        """
        self.feature_layer = feature_layer
        self.court_length = court_length
        self.court_width = court_width
        self.net_y = net_y
        self.court_center_x = court_width / 2    # 3.05 m
        self.court_center_y = net_y               # net line = court center along length
        self.use_hitter = use_hitter
        self.use_bones = use_bones
        self.use_bbox_norm = use_bbox_norm
        base_dim = FEATURE_DIMS[feature_layer]
        if use_hitter:
            base_dim += 1
        if use_bones:
            base_dim += BONE_CHANNELS
        self.feature_dim = base_dim
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

        # Apply bbox normalization before any feature computation
        if self.use_bbox_norm:
            skel_for_features = self._apply_bbox_norm(skeleton)
        else:
            skel_for_features = skeleton

        # L0: coordinates (bbox-normed or raw pixel)
        features = skel_for_features.copy()  # (2, T, V)

        if self.feature_layer in ("L1", "L2", "L3"):
            vel, acc = self._compute_kinematics(skel_for_features)
            features = np.concatenate([features, vel, acc], axis=0)  # (6, T, V)

        if self.feature_layer in ("L2", "L3"):
            # Court context always uses original pixel coords (need absolute position)
            court_ctx = self._compute_court_context(skeleton)
            features = np.concatenate([features, court_ctx], axis=0)  # (9, T, V)

        if self.feature_layer == "L3":
            # Joint angles from bbox-normed coords (scale-invariant angles)
            angles = self._compute_joint_angles(skel_for_features)
            features = np.concatenate([features, angles], axis=0)  # (12, T, V)

        if self.use_hitter:
            hitter_ch = self._compute_hitter_channel(T, V, hitter)
            features = np.concatenate([features, hitter_ch], axis=0)

        if self.use_bones:
            bones = self._compute_bone_vectors(skel_for_features)
            features = np.concatenate([features, bones], axis=0)

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
            - depth_to_opp: signed depth (along court, y-axis) to opponent centroid
            - lateral_to_opp: signed lateral (across court, x-axis) to opponent centroid

        Args:
            skeleton: (2, T, V) — raw pixel coordinates

        Returns:
            court_features: (4, T, V)
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

        # depth_to_opp — signed along-court (y-axis) distance to opponent
        # Positive = opponent is further from baseline (deeper)
        depth_to_opp = np.zeros((T, V))
        depth_to_opp[:, :NUM_JOINTS] = (p1_cy - p0_cy)[:, None]  # P0's view
        depth_to_opp[:, NUM_JOINTS:] = (p0_cy - p1_cy)[:, None]  # P1's view

        # lateral_to_opp — signed across-court (x-axis) distance to opponent
        # Positive = opponent is to the right
        lateral_to_opp = np.zeros((T, V))
        lateral_to_opp[:, :NUM_JOINTS] = (p1_cx - p0_cx)[:, None]
        lateral_to_opp[:, NUM_JOINTS:] = (p0_cx - p1_cx)[:, None]

        return np.stack([dist_to_net, dist_to_center, depth_to_opp, lateral_to_opp], axis=0)

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

    @staticmethod
    def _compute_bone_vectors(skeleton):
        """
        Compute bone vectors: child_joint − parent_joint for each COCO limb.

        Bone vectors explicitly encode limb direction and length, making it
        easier to distinguish e.g. elbow-high (full smash) vs elbow-low
        (tap smash) from the same joint positions.

        For dual-player graphs (V=34 or 35), bone parents are applied per
        player with the appropriate offset. Root joints (parent=-1) and
        the shuttle virtual node get zero vectors.

        Args:
            skeleton: (2, T, V) — [x, y] coordinates (bbox-normed or raw)

        Returns:
            bones: (2, T, V) — [bone_x, bone_y] per joint
        """
        C, T, V = skeleton.shape
        bones = np.zeros_like(skeleton)  # (2, T, V)

        for player_offset in range(0, V, NUM_JOINTS):
            if player_offset + NUM_JOINTS > V:
                break  # skip shuttle node if present
            for j in range(NUM_JOINTS):
                parent = COCO_BONE_PARENTS[j]
                if parent < 0:
                    continue  # root joint → zero bone
                bones[:, :, player_offset + j] = (
                    skeleton[:, :, player_offset + j] -
                    skeleton[:, :, player_offset + parent]
                )

        return bones

    @staticmethod
    def _apply_bbox_norm(skeleton):
        """
        Normalize each player's joints relative to their bounding box.

        For each player at each frame:
            x_norm = (x - bbox_cx) / max(bbox_w, 1)
            y_norm = (y - bbox_cy) / max(bbox_h, 1)

        This makes features scale-invariant (handles players at different
        distances from camera) and translation-invariant within the frame.
        Court position information is preserved in L2 court context features
        which use the original pixel coordinates.

        Zero-valued joints (undetected) remain zero after normalization.

        Args:
            skeleton: (2, T, V) — raw pixel coordinates

        Returns:
            normed: (2, T, V) — bbox-normalized coordinates
        """
        C, T, V = skeleton.shape
        normed = skeleton.copy()

        for player_offset in range(0, V, NUM_JOINTS):
            if player_offset + NUM_JOINTS > V:
                break  # skip shuttle node
            pslice = slice(player_offset, player_offset + NUM_JOINTS)
            px = skeleton[0, :, pslice]  # (T, 17)
            py = skeleton[1, :, pslice]

            # Mask: valid joints have nonzero coords
            valid = (px > 0) | (py > 0)  # (T, 17)

            for t in range(T):
                v = valid[t]
                if v.sum() < 2:
                    continue  # not enough joints to compute bbox
                vx = px[t, v]
                vy = py[t, v]
                x_min, x_max = vx.min(), vx.max()
                y_min, y_max = vy.min(), vy.max()
                bw = max(x_max - x_min, 1.0)
                bh = max(y_max - y_min, 1.0)
                cx = (x_min + x_max) / 2
                cy = (y_min + y_max) / 2
                normed[0, t, pslice] = np.where(v, (px[t] - cx) / bw, 0.0)
                normed[1, t, pslice] = np.where(v, (py[t] - cy) / bh, 0.0)

        return normed


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
