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

# FineBadminton (in datasets/)
FB_ROOT        = PROJECT_ROOT / "datasets" / "FineBadminton-dataset" / "dataset"
FB_FRAMES      = FB_ROOT / "image"
FB_ANNOTATIONS = FB_ROOT / "transformed_combined_rounds_output_en_evals_translated.json"
FB_SKELETONS       = PROJECT_ROOT / "datasets_preprocessing" / "finebadminton_skeletons"
FB_SKELETONS_GDINO    = PROJECT_ROOT / "datasets_preprocessing" / "finebadminton_skeletons_gdino"
FB_SKELETONS_GDINO_V2 = PROJECT_ROOT / "datasets_preprocessing" / "finebadminton_skeletons_gdino_v2"
FB_SHUTTLES           = PROJECT_ROOT / "datasets_preprocessing" / "finebadminton_shuttles"

# ShuttleSet CSV annotations (in datasets/)
SS_CSV_ROOT = PROJECT_ROOT / "datasets" / "ShuttleSet" / "set"
SS_MATCH_CSV = SS_CSV_ROOT / "match.csv"

# ShuttleSet processed data (in datasets_preprocessing/)
# Note: actual directories use the shuttleset_* prefix (flat, not nested under ShuttleSet/)
SS_PREPROCESS_ROOT = PROJECT_ROOT / "datasets_preprocessing"
SS_FRAMES          = SS_PREPROCESS_ROOT / "shuttleset_frames"
SS_OUTPUTS         = SS_PREPROCESS_ROOT / "shuttleset_outputs"
SS_SKELETONS       = SS_PREPROCESS_ROOT / "shuttleset_skeletons"
SS_SKELETONS_GDINO = SS_PREPROCESS_ROOT / "shuttleset_skeletons_gdino"
SS_SHUTTLES        = SS_PREPROCESS_ROOT / "shuttleset_shuttles"
SS_SPLIT_JSON      = SS_PREPROCESS_ROOT / "shuttleset_split.json"


# ─── Strategy Labels ────────────────────────────────────────────────────────

STRATEGY_CLASSES = [
    "intercept",
    "defensive",
    "move_to_net",
    "create_depth",
    "passive",
]
NUM_CLASSES = len(STRATEGY_CLASSES)
STRATEGY_TO_IDX = {s: i for i, s in enumerate(STRATEGY_CLASSES)}
IDX_TO_STRATEGY = {i: s for i, s in enumerate(STRATEGY_CLASSES)}

# Mapping from raw FineBadminton annotation strings → canonical class names
FB_STRATEGY_MAP = {
    "intercept": "intercept",
    "defensive": "defensive",
    "move to the net": "move_to_net",
    "to create depth": "create_depth",
    "passive": "passive",
}
# Excluded strategies (not classifiable from skeleton data)
FB_EXCLUDED_STRATEGIES = {"deception", "hesitation", "seamlessly", "a high net early shot"}

# FineBadminton hit type labels (raw field name: "hit_type")
FB_HIT_TYPES = [
    'block', 'clear', 'cross-court net shot', 'drive', 'drop shot',
    'kill', 'net kill', 'net lift', 'net shot', 'push shot', 'serve',
]
FB_HIT_TYPE_TO_IDX = {h: i for i, h in enumerate(FB_HIT_TYPES)}
NUM_FB_HIT_TYPES = len(FB_HIT_TYPES)

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
# 17-class canonical vocabulary based on official ShuttleSet English translations.
# FineBadminton subtypes (fine-grained) are preferred; hit_type is the fallback.

SHOT_TYPES = [
    'short_serve',    #  0 — 發短球 | FB subtype: short serve
    'long_serve',     #  1 — 發長球 | FB subtype: high serve, flick serve
    'smash',          #  2 — 殺球   | FB subtype: jump/full/common/slice/stick smash
    'tap_smash',      #  3 — 點扣   | FB: net kill
    'push_rush',      #  4 — 推撲球, 撲球
    'clear',          #  5 — 長球   | FB subtype: attacking clear
    'slice_drop',     #  6 — 切球   | FB subtype: slice drop shot, reverse slice drop shot
    'net_drop',       #  7 — 放小球 | FB subtype: stop drop shot, blocked drop shot
    'transition',     #  8 — 過渡球, 過度切球
    'drive',          #  9 — 平球, 後場抽平球, 防守回抽, 小平球 | FB subtype: high/flat drive
    'block',          # 10 — 擋小球 | FB subtype: high block
    'lob_lift',       # 11 — 挑球   | FB subtype: flat lift, high lift
    'defensive_lift', # 12 — 防守回挑
    'cross_net',      # 13 — 勾球   | FB: cross-court net shot
    'net_shot',       # 14 — 網前球 | FB subtype: spinning net
    'smash_defense',  # 15 — 接殺防守
    'push',           # 16 — 推球
]
SHOT_TYPE_TO_IDX = {s: i for i, s in enumerate(SHOT_TYPES)}
NUM_SHOT_TYPES = len(SHOT_TYPES)  # 17

# FineBadminton subtype (fine-grained) → canonical shot type
# Preferred over hit_type when available (305 of ~355 shots have subtypes).
FB_SUBTYPE_TO_SHOT_TYPE = {
    # Serve subtypes
    'short serve':              'short_serve',
    'high serve':               'long_serve',
    'flick serve':              'short_serve',    # flick = deceptive short serve variant
    # Kill (smash) subtypes
    'jump smash':               'smash',
    'full smash':               'smash',
    'common smash':             'smash',
    'slice smash':              'smash',
    'stick smash':              'smash',
    # Clear subtypes
    'attacking clear':          'clear',
    # Drop shot subtypes
    'slice drop shot':          'slice_drop',
    'reverse slice drop shot':  'slice_drop',
    'stop drop shot':           'net_drop',
    'blocked drop shot':        'net_drop',
    # Drive subtypes
    'high drive':               'drive',
    'flat drive':               'drive',
    # Net shot subtypes
    'spinning net':             'net_shot',
    # Net lift subtypes
    'flat lift':                'lob_lift',       # was wrongly mapped via push shot → push
    'high lift':                'lob_lift',
    # Block subtypes
    'high block':               'block',
}

# FineBadminton hit_type (English) → canonical shot type (fallback)
# Used only when no subtype is available for a shot.
FB_HIT_TYPE_TO_SHOT_TYPE = {
    'serve':                'short_serve',    # conservative default (most serves are short)
    'kill':                 'smash',
    'net kill':             'tap_smash',
    'clear':                'clear',
    'drop shot':            'net_drop',
    'drive':                'drive',
    'net shot':             'net_shot',
    'cross-court net shot': 'cross_net',
    'net lift':             'lob_lift',
    'push shot':            'push',
    'block':                'block',
}

# ShuttleSet type (Chinese) → canonical shot type
# '未知球種' / '未知' (unknown) intentionally absent → maps to None (no auxiliary label)
SS_TYPE_TO_SHOT_TYPE = {
    # Serve
    '發短球':    'short_serve',
    '發長球':    'long_serve',
    # Smash variants
    '殺球':      'smash',
    '點扣':      'tap_smash',
    '撲球':      'push_rush',   # net rush (data variant of 推撲球)
    '推撲球':    'push_rush',   # official term
    # Clear
    '長球':      'clear',
    # Drop / slice
    '切球':      'slice_drop',
    '放小球':    'net_drop',
    '過度切球':  'transition',  # transitional slice (data variant of 過渡球)
    '過渡球':    'transition',  # official term
    # Drive variants
    '後場抽平球':'drive',
    '平球':      'drive',
    '防守回抽':  'drive',
    '小平球':    'drive',
    # Lift variants
    '挑球':      'lob_lift',
    '防守回挑':  'defensive_lift',
    # Net play
    '勾球':      'cross_net',
    '網前球':    'net_shot',
    # Defense
    '接殺防守':  'smash_defense',
    # Push / block
    '推球':      'push',
    '擋小球':    'block',
}


# ─── Feature Layer Dimensions (L0-L3) ──────────────────────────────────────

FEATURE_DIMS = {
    "L0": 2,   # [x, y]
    "L1": 6,   # [x, y, vx, vy, ax, ay]
    "L2": 9,   # [..., dist_to_net, dist_to_center, dist_to_opponent]
    "L3": 13,  # [..., L_elbow, R_elbow, L_knee, R_knee (node-specific)]
}

# When use_hitter_feature=True, one extra channel is appended:
# +1 dim: 1.0 for hitter's joints, 0.0 for opponent's joints
FEATURE_DIMS_WITH_HITTER = {k: v + 1 for k, v in FEATURE_DIMS.items()}


# ─── Skeleton / Graph ───────────────────────────────────────────────────────

NUM_JOINTS = 17  # COCO keypoint format
NUM_PLAYERS = 2
NUM_NODES = NUM_JOINTS * NUM_PLAYERS  # 34
JOINT_DIM = 2  # 2D (x, y) coordinates

# COCO skeleton edges (0-indexed)
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # head
    (5, 6),                                 # shoulders
    (5, 7), (7, 9),                         # left arm
    (6, 8), (8, 10),                        # right arm
    (5, 11), (6, 12),                       # torso
    (11, 12),                               # hips
    (11, 13), (13, 15),                     # left leg
    (12, 14), (14, 16),                     # right leg
]

# Inter-player edges: connect corresponding joints between player 1 and 2
INTER_PLAYER_EDGES = [(j, j + NUM_JOINTS) for j in range(NUM_JOINTS)]

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
    fb_fps: int = 20               # FineBadminton frame rate
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
    num_shot_types: int = NUM_SHOT_TYPES  # 17 shot types

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


# ─── Few-Shot / Prototypical Network ────────────────────────────────────────

@dataclass
class ProtoNetConfig:
    n_way: int = NUM_CLASSES       # 5
    k_shot: int = 10               # support examples per class
    n_query: int = 5               # query examples per class
    distance: str = "euclidean"    # or "cosine"
    confidence_margin: float = 0.5  # margin threshold for low-confidence flag

    # Meta-training
    episodes_per_epoch: int = 100
    epochs: int = 50
    lr: float = 1e-4
    fine_tune_encoder: bool = True
    encoder_lr_scale: float = 0.1  # encoder LR = lr * this


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
    proto: ProtoNetConfig = field(default_factory=ProtoNetConfig)
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
