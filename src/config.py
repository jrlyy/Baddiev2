"""
Central configuration for the Badminton Tactical Strategy pipeline.
All hyperparameters, file paths, and experiment settings in one place.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ─── Paths ───────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# FineBadminton (in Datasets/)
FB_ROOT        = PROJECT_ROOT / "Datasets" / "FineBadminton-dataset" / "dataset"
FB_FRAMES      = FB_ROOT / "image"
FB_ANNOTATIONS = FB_ROOT / "transformed_combined_rounds_output_en_evals_translated.json"
FB_SKELETONS       = PROJECT_ROOT / "datasets_preprocessing" / "finebadminton_skeletons"
FB_SKELETONS_GDINO = PROJECT_ROOT / "datasets_preprocessing" / "finebadminton_skeletons_gdino"

# ShuttleSet CSV annotations (in datasets/)
SS_CSV_ROOT = PROJECT_ROOT / "datasets" / "ShuttleSet" / "set"
SS_MATCH_CSV = SS_CSV_ROOT / "match.csv"

# ShuttleSet processed data (in datasets_preprocessing/)
SS_PREPROCESS_ROOT = PROJECT_ROOT / "datasets_preprocessing" / "ShuttleSet"
SS_FRAMES = SS_PREPROCESS_ROOT / "frames"
SS_OUTPUTS = SS_PREPROCESS_ROOT / "outputs"
SS_SKELETONS = SS_PREPROCESS_ROOT / "skeletons"


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

# ShuttleSet shot type labels (Chinese → index)
SS_SHOT_TYPES = [
    '發短球', '發長球', '推撲球', '殺球', '過渡球', '防守回挑',
    '切球', '接殺防守', '長球', '平球', '擋小球', '挑球',
    '放小球', '勾球', '網前球', '點扣', '推球', '未知',
]
SS_SHOT_TYPE_TO_IDX = {s: i for i, s in enumerate(SS_SHOT_TYPES)}


# ─── Feature Layer Dimensions (L0-L3) ──────────────────────────────────────

FEATURE_DIMS = {
    "L0": 2,   # [x, y]
    "L1": 6,   # [x, y, vx, vy, ax, ay]
    "L2": 9,   # [..., dist_to_net, dist_to_center, dist_to_opponent]
    "L3": 12,  # [..., elbow_angle, shoulder_angle, knee_angle]
}


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


# ─── Data Processing ────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    shot_window: int = 16          # frames per shot segment (T)
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
    max_seq_len: int = 16          # same as shot_window
    embedding_dim: int = 256


# ─── Self-Supervised Pre-Training ────────────────────────────────────────────

@dataclass
class SSLConfig:
    temperature: float = 0.07     # NT-Xent temperature
    projection_dim: int = 128     # projection head output
    projection_hidden: int = 256
    auxiliary_weight: float = 0.3  # weight for shot-type auxiliary loss
    num_shot_types: int = 18       # ShuttleSet shot types

    # Augmentation
    jitter_std: float = 0.01      # Gaussian noise on joint coords
    mask_ratio: float = 0.15      # fraction of joints to mask
    temporal_crop_ratio: float = 0.8  # min fraction of frames to keep
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
    encoder: str = "stgcn"         # "stgcn", "transformer", "lstm", "cnn1d"

    # Feature layer ablation (Step 1)
    feature_layer: str = "L2"      # L0, L1, L2, L3

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
        if self.ablation.encoder == "transformer":
            return self.transformer
        return self.stgcn


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
