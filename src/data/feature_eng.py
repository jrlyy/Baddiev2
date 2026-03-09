"""
Node feature engineering for skeleton graphs (L0–L3 layers).

All features are computed from raw skeleton coordinates using simple
numpy operations. No additional data, models, or processing needed.

Feature Layers (cumulative):
    L0: [x, y] — raw court-relative coordinates (2-dim)
    L1: [x, y, vx, vy, ax, ay] — + velocity, acceleration (6-dim)
    L2: [..., dist_net, dist_center, dist_opp] — + court context (9-dim)
    L3: [..., elbow_angle, shoulder_angle, knee_angle] — + joint angles (12-dim)

If a homography matrix (3×3 numpy array) is provided, pixel coordinates are
transformed to court-relative coordinates before any features are computed.
This makes all distance/velocity features physically meaningful (in metres).
"""
import numpy as np
from ..config import NUM_JOINTS, NUM_NODES, FEATURE_DIMS


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
                 homography=None):
        """
        Args:
            feature_layer: "L0", "L1", "L2", or "L3"
            court_length: full court length in meters (for L2)
            court_width: court width in meters (for L2)
            homography: optional (3, 3) numpy array mapping pixel coords to
                        court-relative coords (e.g. H_img_to_court_m.npy).
                        If provided, raw (x, y) pixel coords are transformed
                        before any feature computation.
        """
        self.feature_layer = feature_layer
        self.court_length = court_length
        self.court_width = court_width
        self.feature_dim = FEATURE_DIMS[feature_layer]
        self.homography = homography  # (3, 3) or None

    def compute(self, skeleton):
        """
        Compute enriched features from raw skeleton.

        Args:
            skeleton: (2, T, V) — raw [x, y] coordinates
                      C=2 (x, y), T=num_frames, V=34 (17 joints x 2 players)

        Returns:
            features: (feature_dim, T, V)
        """
        C, T, V = skeleton.shape
        assert C == 2, f"Expected 2 channels (x, y), got {C}"

        # Apply homography first if provided (pixel → court-relative coords)
        if self.homography is not None:
            skeleton = self._apply_homography(skeleton)

        # L0: raw coordinates
        features = skeleton.copy()  # (2, T, V)

        if self.feature_layer in ("L1", "L2", "L3"):
            vel, acc = self._compute_kinematics(skeleton)
            features = np.concatenate([features, vel, acc], axis=0)  # (6, T, V)

        if self.feature_layer in ("L2", "L3"):
            court_ctx = self._compute_court_context(skeleton)
            features = np.concatenate([features, court_ctx], axis=0)  # (9, T, V)

        if self.feature_layer == "L3":
            angles = self._compute_joint_angles(skeleton)
            features = np.concatenate([features, angles], axis=0)  # (12, T, V)

        assert features.shape[0] == self.feature_dim, \
            f"Expected {self.feature_dim} features, got {features.shape[0]}"

        return features

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

    def _compute_court_context(self, skeleton):
        """
        Compute court-relative distance features.

        Features per node per frame:
            - dist_to_net: distance from joint to net line (y=0 at net)
            - dist_to_center: distance from joint to court center
            - dist_to_opponent_centroid: distance from joint to opponent's
              center of mass

        Args:
            skeleton: (2, T, V) — assumed to be in court-relative coordinates
                      (or pixel coords if no homography applied)

        Returns:
            court_features: (3, T, V)
        """
        C, T, V = skeleton.shape

        x = skeleton[0]  # (T, V)
        y = skeleton[1]  # (T, V)

        # Net position: assume y=0 is at net (court center along length)
        # If coordinates are in pixels, this is approximate
        dist_to_net = np.abs(y)  # (T, V)

        # Court center
        court_center_x = np.mean(x)  # approximate center
        court_center_y = 0.0
        dist_to_center = np.sqrt(
            (x - court_center_x) ** 2 + (y - court_center_y) ** 2
        )  # (T, V)

        # Opponent centroid distance
        # Player 1: joints 0-16, Player 2: joints 17-33
        dist_to_opponent = np.zeros((T, V))

        # Player 1 centroid (from centroid joints)
        p1_centroid_joints = CENTROID_JOINTS
        p1_cx = x[:, p1_centroid_joints].mean(axis=1, keepdims=True)  # (T, 1)
        p1_cy = y[:, p1_centroid_joints].mean(axis=1, keepdims=True)

        # Player 2 centroid
        p2_centroid_joints = [j + NUM_JOINTS for j in CENTROID_JOINTS]
        p2_cx = x[:, p2_centroid_joints].mean(axis=1, keepdims=True)
        p2_cy = y[:, p2_centroid_joints].mean(axis=1, keepdims=True)

        # Player 1's joints → distance to player 2's centroid
        dist_to_opponent[:, :NUM_JOINTS] = np.sqrt(
            (x[:, :NUM_JOINTS] - p2_cx) ** 2 +
            (y[:, :NUM_JOINTS] - p2_cy) ** 2
        )
        # Player 2's joints → distance to player 1's centroid
        dist_to_opponent[:, NUM_JOINTS:] = np.sqrt(
            (x[:, NUM_JOINTS:] - p1_cx) ** 2 +
            (y[:, NUM_JOINTS:] - p1_cy) ** 2
        )

        court_features = np.stack([dist_to_net, dist_to_center, dist_to_opponent], axis=0)
        return court_features  # (3, T, V)

    @staticmethod
    def _compute_joint_angles(skeleton):
        """
        Compute joint angles at key articulation points.

        Angles computed:
            - Left elbow angle
            - Right elbow angle
            - Left knee angle

        Applied to both players (with joint index offset for player 2).

        Args:
            skeleton: (2, T, V)

        Returns:
            angles: (3, T, V) — angle features broadcast to all nodes
                    (each player's joints get that player's angles)
        """
        C, T, V = skeleton.shape
        num_angle_features = 3  # left_elbow, right_elbow, left_knee

        angles = np.zeros((num_angle_features, T, V))

        for player_offset in [0, NUM_JOINTS]:
            # Compute angles for this player
            player_angles = {}

            for name, (p, j, c) in ANGLE_TRIPLETS.items():
                if name in ("left_elbow", "right_elbow", "left_knee"):
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

                    player_angles[name] = angle

            # Map angles to feature channels
            angle_names = ["left_elbow", "right_elbow", "left_knee"]
            start_joint = player_offset
            end_joint = player_offset + NUM_JOINTS

            for feat_idx, name in enumerate(angle_names):
                if name in player_angles:
                    # Broadcast this angle to all joints of this player
                    angles[feat_idx, :, start_joint:end_joint] = \
                        player_angles[name][:, np.newaxis]

        return angles  # (3, T, V)


def compute_features_batch(skeletons, feature_layer="L2", **kwargs):
    """
    Compute features for a batch of skeletons.

    Args:
        skeletons: list of (2, T, V) arrays or (B, 2, T, V) array
        feature_layer: "L0", "L1", "L2", "L3"

    Returns:
        features: list of (feature_dim, T, V) arrays
    """
    eng = FeatureEngineer(feature_layer=feature_layer, **kwargs)

    if isinstance(skeletons, np.ndarray) and skeletons.ndim == 4:
        return np.stack([eng.compute(s) for s in skeletons])

    return [eng.compute(s) for s in skeletons]
