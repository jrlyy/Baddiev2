"""
Central configuration for the Badminton Shot Attribute Prediction pipeline.
All hyperparameters, file paths, and experiment settings in one place.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ─── Paths ───────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# ShuttleSet CSV annotations (in datasets/)
SS_CSV_ROOT = PROJECT_ROOT / "datasets" / "ShuttleSet" / "set"
SS_MATCH_CSV = SS_CSV_ROOT / "match.csv"

# ShuttleSet processed data (in datasets_preprocessing/)
# Note: actual directories use the shuttleset_* prefix (flat, not nested under ShuttleSet/)
SS_PREPROCESS_ROOT = PROJECT_ROOT / "datasets_preprocessing"
SS_FRAMES          = SS_PREPROCESS_ROOT / "shuttleset_frames"
SS_OUTPUTS         = SS_PREPROCESS_ROOT / "shuttleset_outputs"
SS_SKELETONS       = SS_PREPROCESS_ROOT / "shuttleset_skeletons_yolo"
SS_SKELETONS_GDINO = SS_PREPROCESS_ROOT / "shuttleset_skeletons_gdino"
SS_SHUTTLES        = SS_PREPROCESS_ROOT / "shuttleset_shuttles"
SS_SPLIT_JSON      = SS_PREPROCESS_ROOT / "shuttleset_split.json"


# ShuttleSet shot type labels (raw field name: "type", Traditional Chinese)
# Includes types found in actual shuttleset_outputs/*.json files plus
# official ShuttleSet vocabulary entries not yet seen in extracted data.
SS_SHOT_TYPES = [
    '放小球', '挑球', '擋小球', '推球', '長球', '殺球',
    '發短球', '切球', '點扣', '勾球', '過度切球', '未知球種',
    '平球', '撲球', '後場抽平球', '防守回抽', '防守回挑',
    '發長球', '小平球',
    # Official vocabulary entries (may not appear in all extracted matches)
    '推撲球', '過渡球', '接殺防守', '網前球', '未知',
]
SS_SHOT_TYPE_TO_IDX = {s: i for i, s in enumerate(SS_SHOT_TYPES)}
NUM_SS_SHOT_TYPES = len(SS_SHOT_TYPES)

# ─── Shot Type Vocabulary ──────────────────────────────────────────────────
# 15-class vocabulary using official ShuttleSet English names.
# The 4 drive subtypes (平球/後場抽平球/防守回抽/小平球) are merged into a single
# 'drive' class — individually too rare/confusable to learn (driven flight: 68
# samples, 0.00 F1 in all experiments).

SHOT_TYPES = [
    'net shot',               #  0 — 放小球
    'return net',             #  1 — 擋小球
    'smash',                  #  2 — 殺球
    'wrist smash',            #  3 — 點扣
    'lob',                    #  4 — 挑球
    'defensive return lob',   #  5 — 防守回挑
    'clear',                  #  6 — 長球
    'drive',                  #  7 — 平球 + 後場抽平球 + 防守回抽 + 小平球 (merged)
    'drop',                   #  8 — 切球
    'passive drop',           #  9 — 過渡切球
    'push',                   # 10 — 推球
    'rush',                   # 11 — 撲球
    'cross-court net shot',   # 12 — 勾球
    'short service',          # 13 — 發短球
    'long service',           # 14 — 發長球
]
SHOT_TYPE_TO_IDX = {s: i for i, s in enumerate(SHOT_TYPES)}
NUM_SHOT_TYPES = len(SHOT_TYPES)  # 15

# ShuttleSet type (Chinese) → shot type label
# '未知球種' / '未知' (unknown) intentionally absent → maps to None (no label)
SS_TYPE_TO_SHOT_TYPE = {
    '放小球':       'net shot',
    '擋小球':       'return net',
    '殺球':         'smash',
    '點扣':         'wrist smash',
    '挑球':         'lob',
    '防守回挑':     'defensive return lob',
    '長球':         'clear',
    '平球':         'drive',
    '後場抽平球':   'drive',
    '防守回抽':     'drive',
    '小平球':       'drive',
    '切球':         'drop',
    '過渡切球':     'passive drop',   # official spelling
    '過度切球':     'passive drop',   # data variant (度 instead of 渡)
    '推球':         'push',
    '撲球':         'rush',
    '推撲球':       'rush',           # compound variant seen in some annotations
    '勾球':         'cross-court net shot',
    '發短球':       'short service',
    '發長球':       'long service',
}


# ─── Feature Layer Dimensions (L0-L3) ──────────────────────────────────────

FEATURE_DIMS = {
    "L0": 2,   # [x, y]
    "L1": 6,   # [x, y, vx, vy, ax, ay]
    "L2": 10,  # [..., dist_to_net, dist_to_center, depth_to_opp, lateral_to_opp]
    "L3": 14,  # [..., L_elbow, R_elbow, L_knee, R_knee (node-specific)]
}

# When use_hitter_feature=True, one extra channel is appended:
# +1 dim: 1.0 for hitter's joints, 0.0 for opponent's joints
FEATURE_DIMS_WITH_HITTER = {k: v + 1 for k, v in FEATURE_DIMS.items()}

# Bone vector channels: (bone_x, bone_y) per joint
BONE_CHANNELS = 2

# COCO kinematic tree parents (for bone vector computation)
# -1 = root joint (no parent → zero bone vector)
COCO_BONE_PARENTS = [
    -1,  # 0: nose (root)
    0,   # 1: left_eye ← nose
    0,   # 2: right_eye ← nose
    1,   # 3: left_ear ← left_eye
    2,   # 4: right_ear ← right_eye
    0,   # 5: left_shoulder ← nose
    0,   # 6: right_shoulder ← nose
    5,   # 7: left_elbow ← left_shoulder
    6,   # 8: right_elbow ← right_shoulder
    7,   # 9: left_wrist ← left_elbow
    8,   # 10: right_wrist ← right_elbow
    5,   # 11: left_hip ← left_shoulder
    6,   # 12: right_hip ← right_shoulder
    11,  # 13: left_knee ← left_hip
    12,  # 14: right_knee ← right_hip
    13,  # 15: left_ankle ← left_knee
    14,  # 16: right_ankle ← right_knee
]


# ─── Skeleton / Graph ───────────────────────────────────────────────────────

NUM_JOINTS = 17  # COCO keypoint format
NUM_PLAYERS = 2
NUM_NODES = NUM_JOINTS * NUM_PLAYERS  # 34
JOINT_DIM = 2  # 2D (x, y) coordinates

# COCO skeleton edges (0-indexed)
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # head
    (0, 5), (0, 6),                        # nose → shoulders (connect head to body)
    (5, 6),                                 # shoulders
    (5, 7), (7, 9),                         # left arm
    (6, 8), (8, 10),                        # right arm
    (5, 11), (6, 12),                       # torso
    (11, 12),                               # hips
    (11, 13), (13, 15),                     # left leg
    (12, 14), (14, 16),                     # right leg
]

# No inter-player edges — relative positioning handled by L2 court-context
# features (depth_to_opp, lateral_to_opp) which are more expressive
INTER_PLAYER_EDGES = []

# Shuttle node index (when use_shuttle=True)
SHUTTLE_NODE = NUM_NODES  # 34
NUM_NODES_WITH_SHUTTLE = NUM_NODES + 1  # 35
# Shuttle connects to both players' wrists (joint 9=left wrist, 10=right wrist)
SHUTTLE_EDGES = [
    (SHUTTLE_NODE, 9),  (SHUTTLE_NODE, 10),   # P1 wrists
    (SHUTTLE_NODE, 26), (SHUTTLE_NODE, 27),   # P2 wrists (9+17, 10+17)
]


# ─── Data Processing ────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    shot_window: int = 32          # frames per shot segment (T)
    ss_fps: int = 30               # ShuttleSet frame rate
    target_fps: int = 20           # resample to this rate
    num_folds: int = 5             # k-fold cross-validation
    random_seed: int = 42


# ─── Model: ST-GCN ──────────────────────────────────────────────────────────

@dataclass
class FeatureConfig:
    feature_layer: str = "L2"      # L0, L1, L2, L3
    # Court dimensions for L2 features (meters)
    court_length: float = 13.4     # full court length
    court_width: float = 6.1       # doubles width

    @property
    def feature_dim(self):
        return FEATURE_DIMS[self.feature_layer]


# ─── Model: ST-GCN ──────────────────────────────────────────────────────────

@dataclass
class STGCNConfig:
    in_channels: int = FEATURE_DIMS["L2"]  # default to L2 (9-dim)
    num_nodes: int = NUM_NODES     # 34
    num_layers: int = 9
    base_channels: int = 64
    embedding_dim: int = 256       # output embedding size
    temporal_kernel: int = 9
    dropout: float = 0.3
    temporal_window: int = 1       # temporal edge connection window


# ─── Model: Transformer (BST-style ablation) ────────────────────────────────

@dataclass
class TransformerConfig:
    in_channels: int = JOINT_DIM * NUM_NODES  # flattened joint features per frame
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_seq_len: int = 32          # same as shot_window
    embedding_dim: int = 256


# ─── Self-Supervised Pre-Training ────────────────────────────────────────────

@dataclass
class SSLConfig:
    temperature: float = 0.07     # NT-Xent temperature
    projection_dim: int = 128     # projection head output
    projection_hidden: int = 256
    auxiliary_weight: float = 0.3  # weight for shot-type auxiliary loss
    num_shot_types: int = NUM_SHOT_TYPES  # 18 shot types

    # Augmentation
    jitter_std: float = 0.01      # Gaussian noise on joint coords
    mask_ratio: float = 0.15      # fraction of joints to mask
    speed_range: float = 0.2      # speed perturb: speed ~ Uniform(1-r, 1+r)
    rotation_range: float = 15.0  # degrees

    # Training
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    warmup_epochs: int = 10



# ─── Pose Extraction ────────────────────────────────────────────────────────

@dataclass
class PoseConfig:
    model: str = "yolov8x-pose"    # or "vitpose-base"
    confidence_threshold: float = 0.3
    kalman_smoothing: bool = True
    batch_size: int = 32


# ─── Ablation Flags ─────────────────────────────────────────────────────────

@dataclass
class AblationConfig:
    # Feature ablation (RQ1)
    use_spatial: bool = True       # include spatial graph edges
    use_temporal: bool = True      # include temporal connections

    # Graph structure ablation
    use_inter_player: bool = True  # include cross-player edges
    single_player: bool = False    # use only one player's skeleton

    # Pre-training ablation (RQ2)
    use_pretrained: bool = True    # load SSL pre-trained weights
    use_auxiliary_task: bool = True  # include shot-type auxiliary loss

    # Architecture ablation
    encoder: str = "stgcn"         # "stgcn", "transformer"

    # Feature layer ablation (Step 1)
    feature_layer: str = "L2"      # L0, L1, L2, L3

    # Shuttlecock ablation (Step 6)
    use_shuttle: bool = False      # append shuttle trajectory as virtual node 34
    shuttle_fusion: str = "graph"  # "graph" (virtual node) or "cross_attn" (BST-style)

    # Hitter identity ablation
    use_hitter: bool = False       # append is_hitter channel (1=hitter's joints, 0=opponent's)

    # Temporal window ablation
    variable_window: bool = False  # use prev/next shot frames instead of fixed window

    # Few-shot method ablation (Step 4)
    classifier: str = "protonet"   # "protonet", "knn", "linear_probe"
    knn_k: int = 5                 # k for k-NN classifier


# ─── Convenience: full experiment config ─────────────────────────────────────

@dataclass
class ExperimentConfig:
    name: str = "default"
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    stgcn: STGCNConfig = field(default_factory=STGCNConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    ssl: SSLConfig = field(default_factory=SSLConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)

    @property
    def encoder_config(self):
        return {
            "stgcn": self.stgcn,
            "transformer": self.transformer,
        }[self.ablation.encoder]


def get_config(name: str = "default", **overrides) -> ExperimentConfig:
    """Create a config, optionally with overrides for ablations."""
    cfg = ExperimentConfig(name=name)
    for key, value in overrides.items():
        parts = key.split(".")
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
    return cfg
