**Few-Shot Tactical Strategy Recognition in Badminton**

Using Self-Supervised Skeleton-Based Spatio-Temporal Graph Learning

Project Report

\[Author Name\] \| \[Institution\] \| \[Date\]

Abstract

This project develops an end-to-end pipeline for recognizing
fine-grained tactical strategies in badminton from raw match footage.
The system extracts dual-player skeleton sequences from video, learns
generalizable spatio-temporal representations through self-supervised
contrastive pre-training on unlabeled data, and classifies tactical
patterns using few-shot prototypical networks with only 40
expert-labeled rallies. We target five classifiable strategies
(intercept, defensive, move to net, create depth, passive) and
investigate two core research questions: (1) the relative contribution
of spatial versus temporal features to strategy recognition, and (2) the
effectiveness of self-supervised pre-training in reducing label
dependency. Our target is to achieve 65--75% macro-F1 on 5-class
strategy classification, representing 85--90% of estimated supervised
upper-bound performance while using fewer than 5% of the labels that
full supervision would require.

1\. Introduction

1.1 Problem Statement

Tactical analysis in badminton remains a manually intensive process.
Coaches review match footage frame-by-frame to identify strategic
patterns such as interceptions, defensive formations, and net
approaches. This process is time-prohibitive, subjective, and
inaccessible to amateur and semi-professional players who lack dedicated
analyst support.

Automated approaches face a fundamental bottleneck: fine-grained
tactical annotations require domain expertise and are expensive to
produce at scale. The FineBadminton dataset, the only publicly available
resource with expert-level strategy labels, contains just 40 annotated
rallies. Fully supervised deep learning is infeasible at this scale.

1.2 Research Questions

This project addresses two primary research questions:

-   **RQ1:** What is the relative contribution of spatial features
    (player positioning, court geometry) versus temporal features
    (movement sequences, timing patterns) to the recognition of
    different tactical strategies?

-   **RQ2:** To what extent does self-supervised contrastive
    pre-training on unlabeled match footage improve few-shot tactical
    classification compared to randomly initialized and
    ImageNet-transferred baselines?

1.3 Proposed Solution

We propose a three-stage pipeline: (1) skeleton extraction from raw
video using pose estimation, (2) self-supervised representation learning
via contrastive objectives on unlabeled match data, and (3) few-shot
classification using prototypical networks on the small labeled set. The
approach is skeleton-based rather than pixel-based, providing
interpretability and robustness to visual variation (camera angle,
lighting, jersey color).

1.4 Scope and Boundaries

To maintain scientific rigor with the available data, we explicitly
scope the project as follows:

-   **In scope:** Five strategies that are distinguishable from skeleton
    and positional data alone: intercept, defensive, move to net, create
    depth, and passive.

-   **Out of scope:** Deception (requires biomechanical discrepancy data
    between preparation and execution phases), hesitation (requires
    sub-frame timing analysis beyond our temporal resolution), and
    seamlessly (a quality modifier, not a discrete strategy class).

-   **Rationale:** Attempting to classify unlearnable categories would
    inflate error rates and obscure the system's true capability on
    tractable strategies.

1.5 Contributions

This work makes the following contributions:

1.  First application of skeleton-based graph learning combined with
    few-shot meta-learning for tactical strategy recognition in racket
    sports, going beyond stroke classification (addressed by BST) to
    higher-level strategic reasoning.

2.  Systematic node feature engineering for tactical skeleton graphs,
    demonstrating that enriched features (kinematics, court context,
    joint angles) derived from raw coordinates substantially improve
    strategy discrimination over coordinates alone.

3.  Quantitative analysis of spatial versus temporal feature importance
    per strategy class via systematic ablation.

4.  Demonstration that self-supervised pre-training on unlabeled match
    footage significantly reduces label requirements for tactical
    understanding.

5.  Architectural comparison between ST-GCN structural priors and
    Transformer-based attention (following BST's paradigm) for the
    few-shot, strategy-level regime — establishing which inductive bias
    is better suited to tactical analysis.

6.  A reproducible, end-to-end pipeline from raw video input to tactical
    prediction with interpretable confidence scores.

2\. Literature Review

2.1 Skeleton-Based Action Recognition

Skeleton-based approaches model human motion as sequences of joint
coordinates rather than raw pixel data. ST-GCN (Yan et al., 2018)
introduced graph convolutions over spatial joint connections and
temporal sequences, achieving 81.5% top-1 accuracy on Kinetics-400 for
general action recognition. Subsequent work (2s-AGCN, MS-G3D, CTR-GCN)
improved spatial attention mechanisms and multi-scale temporal modeling.
However, these methods target coarse action categories (e.g., "playing
badminton" vs. "playing tennis") rather than fine-grained tactical
patterns within a single sport.

2.2 Self-Supervised Learning for Skeleton Sequences

Label-free representation learning for skeletons has advanced rapidly.
CrosSCLR (Li et al., 2021) introduced cross-view contrastive learning
across different skeleton augmentation views. AimCLR (Guo et al., 2022)
addressed augmentation sensitivity through extreme augmentation
strategies. 3s-HCN applied hierarchical contrastive learning at joint,
body-part, and sequence levels. These methods demonstrate that skeleton
embeddings pre-trained without labels can match or approach supervised
performance when fine-tuned, motivating our self-supervised pre-training
strategy.

2.3 Badminton Analytics

Existing computational badminton analysis focuses primarily on
shot-level classification rather than strategic reasoning. CoachAI (Wang
et al., 2022) applied ShuttleNet (ResNet+LSTM) to classify 22 shot types
on the ShuttleSet dataset, achieving 73.4% accuracy. FineBadminton (the
dataset we use) contributed fine-grained annotations including strategy
labels but did not report classification results for tactical
categories; its contribution was the annotation framework itself.
TrackNet-series work focused on shuttle trajectory detection for
broadcast analytics.

Most recently, BST (Chang, 2025; CVPR 2025 Workshop) introduced a
Transformer-based approach for skeleton-based stroke classification on
ShuttleSet, using both player skeleton sequences and shuttlecock
trajectory as inputs. BST outperforms previous methods on stroke type
classification (smash, clear, drop, etc.), establishing the current
state-of-the-art for skeleton-based badminton analysis. Concurrently,
RacketVision (Dong et al., 2025) advanced racket detection and tracking
in badminton video, providing a complementary signal for fine-grained
action understanding.

Critically, BST and RacketVision operate at the stroke type level — they
classify what shot was played, not why it was played. No published work
addresses automated tactical strategy classification (intercept,
defensive, create depth, etc.) in badminton. However, BST's success with
a Transformer architecture on skeleton-based badminton data weakens the
assumption that ST-GCN is the default choice for this domain, and
necessitates a direct architectural comparison (see Section 6.3).

2.4 Few-Shot and Meta-Learning in Sports

Prototypical Networks (Snell et al., 2017) remain a strong baseline for
few-shot classification, learning a metric space where classification is
performed by distance to class prototypes. Applications in sports have
been limited: few-shot methods have been explored for player
identification and rare event detection, but not for tactical pattern
recognition. The scarcity of tactical labels in sports datasets makes
this an ideal application domain for few-shot learning, yet it remains
unexplored.

2.5 Research Gap

Table 1 summarizes the positioning of our work relative to existing
methods. While BST (Chang, 2025) has closed the gap on skeleton-based
stroke classification using Transformers, the critical gap remains: no
published work addresses fine-grained tactical strategy classification in
badminton (or any racket sport). Our work goes beyond stroke
classification to tactical strategy recognition, and beyond full
supervision to few-shot learning. BST's success does, however,
necessitate that we compare ST-GCN's structural priors against
Transformer-based attention for the strategy-level, few-shot regime.

*Table 1: Positioning relative to existing work*

| Method | Task | Data Req. | Sport-Specific | Tactical? |
|---|---|---|---|---|
| CoachAI ShuttleNet | Shot type (22-cls) | Fully supervised | Yes (badminton) | No |
| BST (Chang, 2025) | Stroke type classif. | Fully supervised (skeleton + trajectory) | Yes (badminton) | No |
| RacketVision (Dong, 2025) | Racket detection | Fully supervised | Yes (badminton) | No |
| ST-GCN | Action recog. | Fully supervised | No (general) | No |
| CrosSCLR / AimCLR | Action recog. | Self-supervised | No (general) | No |
| Prototypical Nets | Image classif. | Few-shot | No (general) | No |
| **Ours (Proposed)** | **Tactical strategy** | **SSL + Few-shot** | **Yes (badminton)** | **Yes** |

3\. Problem Formulation

3.1 Formal Task Definition

We define the tactical strategy recognition task as an N-way K-shot
classification problem. Given a rally R consisting of a sequence of T
shots, where each shot s_t is represented by dual-player skeleton graphs
G_t = (V, E, X_t), the goal is to assign R to one of N = 5 tactical
strategy classes.

The skeleton graph G_t has V = 34 nodes (17 joints per player), with
edges E encoding both intra-player anatomical connections and
inter-player relational edges. The node feature matrix X_t contains
enriched per-joint features: court-relative 2D coordinates, velocity,
acceleration, court-relative distances, and optionally joint angles (see
Section 5.2 for the full feature engineering pipeline and layered feature
sets).

3.2 Few-Shot Formulation

The few-shot setting provides a support set S = {(R_i, y_i)} of K
labeled examples per class and a query set Q of unlabeled rallies to
classify. A prototypical network computes class prototypes c_k as the
mean embedding of support examples per class: c_k = (1/\|S_k\|) \*
sum(f_theta(R_i)) for all R_i in S_k, where f_theta is the ST-GCN
encoder. Classification assigns the query to the nearest prototype using
Euclidean distance in the embedding space.

3.3 Target Strategy Taxonomy

We classify five strategies, selected based on learnability from
skeleton and positional features:

*Table 2: Target strategy definitions and observable signals*

| Strategy | Definition | Observable Skeleton/Position Signals |
|---|---|---|
| **Create Depth** | Push opponent to rear court to open front court space | Rear-court shuttle landing, opponent in mid/front, controlled arm extension |
| **Intercept** | Early aggressive contact to reduce opponent reaction time | Forward court position, early contact timing, flat/forward arm trajectory |
| **Move to Net** | Sequential approach toward net to apply pressure | Progressive forward displacement across consecutive frames |
| **Passive** | Non-aggressive return with no spatial advantage gained | Neutral body posture, no significant court position change |
| **Defensive** | Reactive shot from disadvantaged position | Stretched/extended body posture, rear/lateral court position, upward arm trajectory |

Three strategies from FineBadminton are excluded: deception (requires
biomechanical discrepancy data unavailable from 2D skeletons),
hesitation (requires sub-frame timing precision beyond our temporal
resolution of \~30fps), and seamlessly (a quality modifier rather than a
discrete tactical category).

4\. Datasets and Data Acquisition

4.1 Dataset Overview

*Table 3: Datasets, availability, and acquisition requirements*

| Dataset | Size | What Is Provided | Acquisition Steps | Role in Pipeline |
|---|---|---|---|---|
| **FineBadminton** | 40 rallies, ~500 shots, ~12K frames | Frames at 20fps (jpg), strategy annotations, shot subtypes, quality scores (1–7) | 1. Request from authors 2. Run YOLOv8-Pose on frames 3. Extract skeletons | Few-shot classification (support + query sets) |
| **ShuttleSet** | 1,500 matches total; 50-match subset selected | CSV tracking data, 22 shot type labels, YouTube video links. No video files provided. | 1. Parse CSV, select 50 matches 2. Download via yt-dlp 3. Extract frames (30fps) 4. Run YOLOv8-Pose | SSL pre-training (skeleton extraction) + auxiliary shot-type task |

4.2 Optimized Acquisition Strategy

A key practical consideration is that the full ShuttleSet corpus (1,500
matches, \~100GB) is neither necessary nor feasible to download. We
select a 50-match subset (\~500--800 rallies, \~5K--8K shots) that
provides sufficient diversity for contrastive pre-training while
reducing download time from weeks to days. This subset is 5--10× the
size of the FineBadminton labeled set, which is adequate for
self-supervised learning to discover meaningful motion patterns.

**ShuttleSet match selection criteria:** Matches are filtered for high
video quality (1080p available), diverse players and tournaments (to
maximize playing style variation), and complete rally data in the CSV
(no missing shot entries). From the filtered pool, 50 matches are
sampled with a fixed random seed for reproducibility.

**FineBadminton acquisition:** The FineBadminton authors provide
pre-extracted frames at 20fps as jpg files. This eliminates the video
download and frame extraction step entirely. The 20fps rate is
sufficient for pose estimation (human motion is smooth at this rate),
and each shot window of \~0.8 seconds yields 16 frames---an ideal
temporal window size for the ST-GCN encoder.

4.3 Data Volume and Storage Requirements

Understanding exact data volumes at each stage is critical for resource
planning:

*Table 4: Data volumes through the pipeline*

| Pipeline Stage | FineBadminton | ShuttleSet (50 matches) | Notes |
|---|---|---|---|
| Raw frames (jpg) | ~12K frames, ~2–3GB | ~200K frames, ~20GB | FineBadminton: provided. ShuttleSet: extracted at 30fps via ffmpeg. |
| Downloaded videos (mp4) | N/A (frames provided) | ~10GB (50 matches) | Deletable after frame extraction to save storage. |
| Processed skeletons (npy/pkl) | ~100MB | ~1–2GB | Compact numerical arrays. Frames deletable after this stage. |
| Shot segments (T=16 frames each) | ~500 labeled shots | ~5K–8K unlabeled shots | Grouped using annotation timestamps (FB) or CSV shot timestamps (SS). |
| Model checkpoints | — | — | ~500MB for ST-GCN weights. |

**Storage optimization:** By deleting raw videos after frame extraction
(−10GB) and deleting frames after skeleton extraction (−20GB), the
minimal persistent footprint is \~5GB (FineBadminton frames + all
processed skeletons + checkpoints).

4.4 Few-Shot Data Split Strategy

Given the extremely small labeled set (40 rallies), we employ 5-fold
cross-validation to maximize data utilization and provide variance
estimates. Each fold uses 32 rallies for the support set (prototype
computation), 4 for validation (hyperparameter tuning, early stopping),
and 4 for testing (final evaluation). All results report mean and
standard deviation across folds.

Class distribution across the 5 target strategies is expected to be
imbalanced (intercept and defensive are more common than move to net).
We address this through balanced episodic sampling during meta-training:
each episode samples equal numbers of support and query examples per
class, regardless of overall class frequency.

4.5 Dataset Alignment Challenge

A critical clarification: ShuttleNet (the CoachAI video dataset,
referenced in earlier literature) and ShuttleSet (the structured CSV
tracking dataset) are separate resources from different research groups.
They do not share the same matches. The auxiliary shot-type prediction
task during pre-training uses ShuttleSet shot labels applied to
ShuttleSet's tracking data as a secondary signal, while the primary
self-supervised objective operates on skeleton sequences extracted from
ShuttleSet's YouTube videos. We treat the contrastive learning and
auxiliary supervision as complementary objectives on the same data
source (ShuttleSet videos + ShuttleSet CSVs).

5\. Methodology

5.1 Pipeline Overview

The system comprises three phases executed sequentially. Phase A
(representation pre-training) operates on unlabeled ShuttleSet data to
learn general motion features. Phase B (few-shot adaptation) uses the
small FineBadminton labeled set to map learned features to tactical
categories. Phase C (inference) applies the trained pipeline to new,
unseen footage — requiring automatic court detection, shot boundary
detection, and the full feature-to-prediction cascade. No manual
calibration or annotation is needed.

A critical distinction: Phases A and B rely on pre-annotated shot
boundaries (provided by FineBadminton timestamps and ShuttleSet CSV
data). Phase C must detect both court geometry and shot boundaries
automatically, since a new video has no annotations. We use Hough
transform + RANSAC-based court model fitting for automatic court
detection (see Section 5.6.1), and TrackNetV3 (pre-trained, open source)
for shuttle tracking and hit event detection, which also yields shuttle
trajectory as a bonus signal (see Section 5.6.2).

*Table 5: End-to-end pipeline stages with data flow*

| Stage | Component | Input | Output | Data Volume / Notes |
|---|---|---|---|---|
| A1 | Pose Estimator (YOLOv8-Pose + ViTPose) | Video frames (jpg) | 2D skeleton keypoints (17 joints × 2 players) | FB: ~12K frames @ 20fps. SS: ~200K frames @ 30fps. Kalman filtering applied for smoothing. |
| A2 | Graph Builder + Feature Engineering | Skeleton sequences | Spatio-temporal graph G = (V, E, X) with enriched node features (L0–L3) | 34 nodes (17 per player), intra + inter-player edges. Node features: coords + velocity + court context + joint angles (9–12 dim per node). Stored as npy. |
| A3 | Shot Segmentation | Skeleton graphs + timestamps | Fixed-length shot windows (T=16 frames) | FB: ~500 labeled shots (timestamps provided). SS: ~5K–8K unlabeled shots (timestamps from CSV). |
| A4 | Encoder (ST-GCN or Transformer) | Skeleton graph with enriched features | Motion embedding (d=256) | Pre-trained via contrastive learning on SS data. |
| A5 | SimCLR Contrastive Head | Augmented skeleton pairs | Invariant embedding space | NT-Xent loss. Auxiliary shot-type task weighted at 0.3. |
| B1 | Prototype Computation | FB labeled support embeddings | 5 class prototypes (mean vectors) | From 32 support rallies per fold. |
| B2 | Classifier (ProtoNet or k-NN) | Query embedding + prototypes / support set | Strategy label + confidence score | ProtoNet: Euclidean distance to centroids. k-NN: majority vote of nearest neighbors. |
| **C1** | **Frame Extraction (ffmpeg)** | **New video file (.mp4)** | **Frames at 30fps** | **Standard ffmpeg extraction.** |
| **C2** | **Court Detector (Hough + RANSAC)** | **Reference frame** | **Homography matrix H (pixel → court coords)** | **Automatic court line detection via Canny + Hough transform, matched against badminton court template via RANSAC. Computed once per video. No manual input.** |
| **C3** | **TrackNetV3 (pre-trained, frozen)** | **Consecutive frames** | **Shuttle (x,y) per frame + hit event timestamps** | **Pre-trained weights, no training needed. Hit events = sharp shuttle trajectory direction changes. Also provides shuttle trajectory as bonus signal.** |
| **C4** | **YOLOv8-Pose (pre-trained)** | **All frames** | **2D skeletons (17 joints × 2 players)** | **Same model as Phase A. Kalman filtering applied.** |
| **C5** | **Shot Segmentation** | **Skeletons + hit timestamps from C3** | **T=16 frame windows per shot** | **Windows centered on each hit event. Wrist velocity peaks from skeleton data used as secondary validation.** |
| **C6** | **Feature Engineering (L0–L3)** | **Shot skeleton windows + H from C2** | **Enriched node features in court-relative coords** | **Same feature engineering as training. Homography H converts pixel coords to court coords for L0 and L2 features.** |
| **C7** | **Encoder (Phase A weights)** | **Enriched skeleton graphs** | **256-dim embedding per shot** | **Frozen or lightly fine-tuned encoder from Phase A.** |
| **C8** | **Classifier (Phase B prototypes)** | **Embeddings + prototypes** | **Strategy label + confidence + shot type** | **ProtoNet for strategy. Aux head for shot type (smash/clear/drop/etc.).** |

5.2 Graph Construction and Node Feature Engineering

The dual-player spatio-temporal graph uses a unified graph with 34 nodes
(17 joints per player). Intra-player edges follow the standard COCO
skeleton topology (anatomical connections). Inter-player edges connect
corresponding joints between players (e.g., right wrist to right wrist)
to capture relative positioning. The adjacency matrix A combines three
sub-matrices: A_intra1 (player 1 skeleton), A_intra2 (player 2
skeleton), and A_inter (cross-player relations). Temporal edges connect
each joint to itself in adjacent frames with a configurable window size
w (default w=1, experiments with w=3).

5.2.1 Node Feature Layers

Raw skeleton coordinates alone (x, y per joint) are an impoverished
input representation. The strategy taxonomy (Table 2) requires the model
to perceive velocity ("quick forward lunge" for intercept), court
context ("rear-court position" for defensive), body configuration
("stretched posture" for defensive vs. "neutral posture" for passive),
and inter-player spatial relationships. Rather than forcing the encoder
to derive all of these from raw coordinates, we compute enriched node
features in layers of increasing richness:

*Table 2b: Node feature layers (cumulative)*

| Layer | Features per node per frame | Dim | What it adds |
|---|---|---|---|
| L0: Raw coordinates | [x, y] court-relative coordinates (via homography) | 2 | Position only. Model must learn everything else. |
| L1: + Kinematics | [x, y, vx, vy, ax, ay] velocity via finite difference; acceleration via second difference | 6 | Directly encodes "lunging forward" (intercept) vs. "stationary" (passive). Free to compute. |
| L2: + Court context | [..., dist_to_net, dist_to_center, dist_to_opponent_centroid] | 9 | Court-relative positional context. Directly encodes "forward court" (intercept) vs. "rear court" (defensive). Free to compute. |
| L3: + Joint angles | [..., elbow_angle, shoulder_angle, ...] (angles between connected joints) | 11–12 | Body configuration. Encodes "arm extended overhead" (defensive) vs. "arm flat forward" (intercept) vs. "neutral posture" (passive). Derived from coordinates. |
| L4: + Racket (stretch) | [..., racket_x, racket_y, racket_angle] as 18th node (requires RacketVision) | 14–15 | Racket state. Partially encodes shot direction — the main signal skeletons miss. Requires external model. |

Layers L0--L3 are all computable from the skeleton coordinates already
extracted in Stage A1 using simple numpy operations (finite differences,
Euclidean distances, arctan2). They require no additional data, models,
or processing time. Layer L4 requires running RacketVision on frames and
is treated as a stretch goal.

The graph topology (34 nodes, 3 adjacency types) remains unchanged
across all feature layers. Only the input channel dimension of the
encoder changes (from 2 to 6 to 9 to 12, etc.). This allows clean
ablation of input richness with zero architectural modification.

5.2.2 Design Rationale

The node feature layers are ordered by the strategy signals they encode:

-   **Position (L0):** "Where is the player?" --- necessary but
    insufficient for all strategies.

-   **Dynamics (L1):** "How is the player moving?" --- critical for
    intercept (rapid forward), move to net (progressive forward), and
    passive (minimal movement).

-   **Court context (L2):** "Where is the player relative to court
    landmarks and the opponent?" --- critical for defensive (rear court,
    far from opponent) and create depth (opponent pushed back).

-   **Body pose (L3):** "What shape is the player's body?" --- critical
    for defensive vs. passive disambiguation (stretched vs. neutral
    posture).

-   **Racket (L4):** "What direction is the shot going?" --- the main
    missing signal for create depth and intercept, partially recoverable
    from racket angle.

5.3 Self-Supervised Pre-Training (Phase A)

We apply a SimCLR-style contrastive objective to skeleton sequences
extracted from ShuttleSet videos. Data augmentation for skeleton
contrastive learning includes: (a) joint-level jittering (Gaussian noise
on coordinates), (b) temporal crop and resample, (c) spatial rotation
(random court-relative rotation), and (d) joint masking (randomly
zeroing 10--20% of joints). The NT-Xent loss trains the ST-GCN encoder
to produce similar embeddings for augmented views of the same rally and
dissimilar embeddings for different rallies.

An optional auxiliary task predicts shot type using ShuttleSet's
22-class labels from the CSV tracking data. This provides a
domain-specific supervisory signal without requiring tactical labels.
The auxiliary loss is weighted at 0.3 relative to the contrastive loss.

5.4 Few-Shot Classification (Phase B)

Phase B involves minimal learning. The frozen (or lightly fine-tuned)
encoder from Phase A maps labeled rallies from FineBadminton into the
embedding space. The core classifier is a Prototypical Network, which
operates in two steps:

**Step 1 --- Prototype computation:** For each of the 5 strategy
classes, compute the mean embedding (centroid) of all support examples
belonging to that class. With 32 support rallies per fold, each
prototype is the average of approximately 6--8 shot embeddings. This is
arithmetic mean, not a learned operation.

**Step 2 --- Distance-based classification:** For a query shot, compute
Euclidean distance from its embedding to each of the 5 prototypes.
Assign the query to the nearest prototype. The "training" in Phase B is
limited to optional light fine-tuning of encoder weights (10--20 epochs)
so that support embeddings cluster tighter around their prototypes.

During meta-training, we sample balanced 5-way K-shot episodes (K
ranging from 5 to 15 per class) with 5 query examples per class.

**Why ProtoNet works here:** The entire bet is that Phase A produced an
embedding space where shots with similar movement patterns are nearby.
If intercept shots share a distinctive motion signature (forward lunge,
early contact, flat arm), then the encoder --- trained to distinguish
movement patterns via contrastive learning --- should naturally group
them together. Prototypes merely label those clusters.

**Alternative: k-Nearest Neighbors (k-NN).** ProtoNet assumes each
class forms a single, roughly spherical cluster around its centroid. If
a class has sub-clusters (e.g., "defensive recovery to the left" and
"defensive recovery to the right"), the centroid falls in empty space
between them. k-NN (k=3 or k=5) classifies by majority vote of the k
nearest individual support examples, making no assumption about cluster
shape. We include k-NN as a comparison to validate the centroid
assumption (see Section 6.3, Step 4).

5.5 Confidence Estimation

Prediction confidence is derived from the softmax over negative
distances to prototypes. Additionally, we compute a margin score
(difference between the nearest and second-nearest prototype distances)
as a complementary uncertainty indicator. Predictions with margin below
a tunable threshold are flagged as low-confidence for human review.

5.6 Inference Pipeline (Phase C)

Phases A and B operate on datasets with pre-annotated shot boundaries:
FineBadminton provides shot timestamps in its annotations, and
ShuttleSet provides them in CSV tracking data. When the trained pipeline
is applied to a new, unseen video (Phase C), shot boundaries must be
detected automatically. This requires two additional pre-processing
components: shuttle-based shot detection and court calibration.

5.6.1 Shot Boundary Detection via TrackNetV3

We use TrackNetV3 (pre-trained, open source) to detect the shuttlecock
in each frame and track its trajectory. A "shot" (hit event) is
identified when the shuttle's trajectory exhibits a sharp direction
change, corresponding to racket-shuttle contact. TrackNetV3 was
developed specifically for badminton shuttle tracking in broadcast
footage and provides pre-trained weights that require no additional
training.

TrackNetV3 outputs two signals:

-   **Hit event timestamps:** Frame indices where a shot occurs. These
    replace the annotated timestamps used in Phases A–B, enabling
    automatic segmentation of the video into T=16 frame windows centered
    on each hit.

-   **Shuttle trajectory:** Per-frame (x, y) coordinates of the shuttle.
    This is a valuable bonus signal — shuttle trajectory is one of the
    main information sources that the skeleton-only pipeline lacks (see
    Section 9.2). The trajectory can optionally be integrated as a
    global graph-level feature or as a 35th node in the skeleton graph.

As a fallback validation signal, we also compute wrist velocity peaks
from the L1 kinematics features (already available from pose
estimation). When TrackNet and wrist-velocity peaks agree on a hit
event, confidence is high. When they disagree, the shot boundary is
flagged as uncertain.

5.6.2 Court Calibration

The L2 node features (dist_to_net, dist_to_center,
dist_to_opponent_centroid) require court-relative coordinates, which
depend on a homography transformation from pixel space to real-world
court space. For training data, court geometry is either provided in the
dataset metadata or can be estimated from consistent broadcast camera
angles. For new video in Phase C, the homography must be computed per
video.

We support two calibration modes:

-   **Automatic:** Detect court boundary lines using a Hough transform
    on edge-detected frames. Fit lines to the four court boundaries and
    compute the homography from the detected quadrilateral to the
    standard badminton court dimensions (13.4m × 6.1m for doubles,
    13.4m × 5.18m for singles). This works reliably for broadcast
    footage with clear court markings.

-   **Manual fallback:** The user clicks four court corners in a
    reference frame. The homography is computed from these four
    correspondences. This takes approximately 10 seconds of user effort
    and is robust to any camera angle.

The homography is computed once per video and applied to all subsequent
frames.

5.6.3 Full Phase C Sequence

For a new video, the inference pipeline executes the following steps in
order:

1. Extract frames at 30fps via ffmpeg.
2. Run TrackNetV3 to obtain shuttle trajectory and hit event timestamps.
3. Run court calibration (automatic or manual) to obtain the homography.
4. Run YOLOv8-Pose on all frames to extract dual-player skeletons.
5. Apply Kalman filtering for skeleton smoothing.
6. Segment into T=16 frame windows centered on each TrackNet hit event.
7. Compute enriched node features (L0–L3) using the court homography.
8. Run each shot window through the trained encoder to obtain 256-dim embeddings.
9. Classify via ProtoNet (strategy label + confidence) and auxiliary head (shot type).
10. Output per-shot results with confidence scores and low-margin flags.

Steps 2–4 can run in parallel (TrackNet and YOLOv8-Pose are independent
models operating on the same frames). Total inference time for a single
rally (~20 seconds of footage) is estimated at 30–60 seconds on a T4
GPU, dominated by pose estimation and shuttle tracking.

5.6 Inference Pipeline (Phase C)

Phases A and B operate on datasets where shot boundaries are
pre-annotated: FineBadminton provides shot timestamps in its annotation
files, and ShuttleSet provides them in its CSV tracking data. When the
trained pipeline is applied to a new, unannotated video, two additional
sub-problems must be solved automatically before the encoder and
classifier can run: (1) court detection for coordinate normalization,
and (2) shot boundary detection to segment continuous footage into
discrete shot windows.

5.6.1 Court Detection and Homography

All L2 features (dist_to_net, dist_to_center, dist_to_opponent) and the
court-relative L0 coordinates depend on a homography transform that maps
pixel-space joint positions to real-world court coordinates. During
training, this calibration is pre-computed for each dataset's known
camera setup. For inference on unseen footage, the court must be detected
automatically.

We use a court line detection approach based on classical computer vision:

1.  **Edge detection:** Apply Canny edge detection to a reference frame
    (first frame or median-composite frame to reduce player occlusion).

2.  **Line detection:** Run probabilistic Hough transform (cv2.HoughLinesP)
    to extract line segments. Filter by length and angle to retain only
    court boundary and service line candidates.

3.  **Court model fitting:** Match detected lines against a known
    badminton court template (13.4m × 6.1m with standard markings) using
    RANSAC-based homography estimation. The four outer court corners plus
    the net line provide sufficient correspondences for a robust
    homography.

4.  **Homography output:** A 3×3 perspective transform matrix H that maps
    any pixel coordinate (u, v) to court-relative meters (x, y) with
    origin at court center and y-axis along the net.

This approach works reliably on broadcast footage where court lines are
clearly visible. For amateur footage with partial court visibility, the
system falls back to a reduced line set (minimum 4 correspondences) or
prompts with a confidence warning if fewer than 4 lines are detected.
No manual calibration is required in the default pipeline.

An alternative approach using a learning-based court keypoint detector
(e.g., a small CNN trained on court images) could improve robustness on
low-quality footage and is noted as a possible enhancement.

5.6.2 Shot Boundary Detection (TrackNetV3)

A "shot" is the moment a player strikes the shuttlecock. Between shots,
players reposition. The pipeline requires discrete shot windows (T=16
frames centered on each hit event) to feed into the encoder.

We use TrackNetV3 (open-source, pre-trained weights available) for
shuttle tracking and hit detection:

1.  **Shuttle detection:** TrackNetV3 takes consecutive frames as input
    and outputs a per-pixel heatmap of shuttle position. It is
    specifically designed for small, fast-moving objects in badminton
    video and handles motion blur.

2.  **Trajectory extraction:** Shuttle (x, y) coordinates are extracted
    per frame from heatmap peaks, producing a continuous trajectory.

3.  **Hit event detection:** A shot (hit event) is detected when the
    shuttle trajectory exhibits a sharp direction change — the velocity
    vector reverses or changes by more than a configurable angle
    threshold (default: 90°). This is computed as peaks in the second
    derivative of shuttle position.

4.  **Shot windowing:** For each detected hit at frame f, a T=16 frame
    window [f−8, f+7] is extracted. The skeleton sequences within this
    window become the input to the encoder.

TrackNetV3 is used as a frozen, pre-trained model — no fine-tuning
is needed. It was trained on broadcast badminton footage similar to
ShuttleSet. As a secondary validation signal, wrist velocity peaks from
the already-extracted skeleton data (L1 kinematics) can confirm hit
events: a genuine shot produces both a shuttle direction change and a
rapid wrist deceleration.

5.6.3 Bonus: Shuttle Trajectory as a Feature

Running TrackNetV3 for shot detection also produces shuttle (x, y)
coordinates per frame as a byproduct. This partially addresses the
"no shuttle trajectory" limitation noted in Section 9.2. The shuttle
position can be incorporated as either a 35th graph node (with
court-relative coordinates as features) or as a global graph-level
feature vector. This integration is treated as an optional enhancement
during inference — the core strategy classification pipeline remains
skeleton-only to match the training regime, but the shuttle trajectory
can be logged alongside predictions for coaching analysis.

5.6.4 Full Inference Sequence

The complete Phase C pipeline for a new video:

| Step | Component | Input | Output |
|---|---|---|---|
| C1 | Frame extraction (ffmpeg) | Video file (.mp4) | Frames at 30fps |
| C2 | Court detector (Hough + RANSAC) | Reference frame | Homography matrix H |
| C3 | TrackNetV3 (pre-trained, frozen) | Consecutive frames | Shuttle (x,y) per frame + hit event timestamps |
| C4 | YOLOv8-Pose (pre-trained) | All frames | Skeleton keypoints (17 joints × 2 players) |
| C5 | Shot segmentation | Hit timestamps + skeletons | T=16 frame windows per shot |
| C6 | Feature engineering (L0–L3) | Skeleton windows + H | Enriched node features in court-relative coords |
| C7 | Encoder (Phase A trained weights) | Enriched skeleton graphs | 256-dim embedding per shot |
| C8 | Classifier (Phase B prototypes) | Embeddings + prototypes | Strategy label + confidence + shot type |

Steps C2, C3, and C4 use pre-trained off-the-shelf models with no
fine-tuning. Only the encoder (C7) and prototypes (C8) come from this
project's training. The shot type in C8 comes from the auxiliary
prediction head trained during Phase A's SSL pre-training.

6\. Experimental Design

6.1 Evaluation Metrics

All experiments report the following metrics, averaged across 5-fold
cross-validation with standard deviations:

-   **Primary:** Macro-F1 score (5-class), which equally weights each
    strategy class regardless of frequency.

-   **Secondary:** Per-class precision, recall, and F1 to identify
    strategy-specific strengths and weaknesses.

-   **Auxiliary:** Accuracy (for comparability with action recognition
    literature), confusion matrix, and embedding quality metrics
    (silhouette score, inter-class separation).

6.2 Baselines

We compare against trivial baselines to establish floor performance:

*Table 6: Trivial baselines*

| Baseline | Description | What It Tests |
|---|---|---|
| Random | Uniform random assignment (20% for 5 classes) | Floor performance |
| Majority class | Always predict most frequent strategy | Class imbalance severity |

All substantive comparisons (architecture, input, pre-training, and
few-shot method) are organized as ablation axes below.

6.3 Ablation Studies

We organize experiments into five sequential steps. Each step holds all
other variables constant and varies one axis. The steps are ordered so
that earlier decisions (what the model sees) are settled before later
ones (how the model processes it), ensuring fair comparisons.

**Step 1 --- Input Representation (what does the model see?)**

Hold encoder fixed at ST-GCN (default) with SSL + auxiliary
pre-training. Vary node features and graph structure to determine the
best input configuration.

*Table 6a: Input representation ablation*

| Variant | Node Features / Graph | What It Tests |
|---|---|---|
| L0: Raw coordinates only | [x, y] per node, 2-dim | Can the model learn everything from position alone? |
| L1: + Kinematics | [x, y, vx, vy, ax, ay], 6-dim | Does explicit velocity and acceleration help? (Expect: yes for intercept, move-to-net) |
| L2: + Court context | [..., dist_to_net, dist_to_center, dist_to_opponent], 9-dim | Does court-relative context help? (Expect: yes for defensive, create depth) |
| L3: + Joint angles | [..., elbow_angle, shoulder_angle], 11–12-dim | Does body pose encoding help? (Expect: yes for defensive vs. passive) |
| L4: + Racket 18th joint (stretch) | 18 nodes, +3 features on racket node | Does racket angle improve tactical discrimination? |
| Spatial-only (best features, no temporal edges) | Remove temporal edges, pool across time | Spatial contribution (RQ1) |
| Temporal-only (best features, no graph topology) | Remove graph edges, treat joints as independent series | Temporal contribution (RQ1) |
| Single-player (17 nodes) | Drop opponent skeleton | Is inter-player modeling needed? |
| Dual-player + inter-player edges | Default (34 nodes, 3 adj. types) | Reference configuration |

The feature layers (L0--L3) are cumulative and tested incrementally.
The best-performing feature set carries forward to Step 2. The
spatial-only vs. temporal-only comparison uses the best features to
cleanly answer RQ1 without confounding input richness with dimensional
decomposition.

**Step 2 --- Encoder Architecture (how does the model process it?)**

Hold input representation fixed at the best from Step 1. Hold
pre-training fixed at SimCLR + auxiliary. Swap only the encoder.

*Table 6b: Encoder architecture ablation*

| Encoder | How It Processes Skeleton Input | Inductive Bias |
|---|---|---|
| ST-GCN (default) | Graph convolutions over skeleton topology. Adjacency matrix encodes anatomical and inter-player structure. | "Nearby joints influence each other more than distant ones." |
| Transformer (BST-style) | Self-attention over joint tokens across spatial and temporal dimensions. No fixed graph topology. | "Let the data decide which joints relate." BST (Chang, 2025) showed this works for badminton stroke classif. |
| LSTM | Flatten joints to vector, process as time series. Bidirectional, 2-layer. | No spatial structure. Tests whether structured representations matter. |
| 1D-CNN | Temporal convolutions over flattened joint vectors. | Local temporal patterns only, no graph or attention. Cheap sanity check. |

All encoders receive the same enriched input and output a 256-dim
embedding. The Transformer encoder uses BST's architectural idea
(self-attention over skeleton sequences) but with our input (skeleton
only, no shuttle trajectory) and our pre-training regime. This is a
clean architecture comparison, not a direct comparison against BST's
published numbers (which use different input modalities, different
labels, and full supervision). LSTM and 1D-CNN are cheap sanity
baselines --- if either beats ST-GCN or Transformer, something is wrong
with the structured representations.

**Step 3 --- Pre-Training Regime (how is the encoder initialized?)**

Hold encoder and input fixed at the best from Steps 1--2. Vary
pre-training strategy to answer RQ2.

*Table 6c: Pre-training ablation*

| Pre-Training | Description | Evaluation |
|---|---|---|
| Random initialization | No pre-training. Encoder weights randomly initialized. | Few-shot F1 (ProtoNet) |
| SimCLR contrastive only | NT-Xent loss on augmented skeleton pairs from ShuttleSet. | Linear probe accuracy + few-shot F1 |
| SimCLR + auxiliary shot-type (default) | Contrastive + shot-type classification head (weight 0.3) using ShuttleSet CSV labels. | Linear probe accuracy + few-shot F1 |

**Step 4 --- Few-Shot Method (how do embeddings become strategy labels?)**

Hold encoder, input, and pre-training fixed at the best from Steps
1--3. Vary the Phase B classifier to test whether the centroid
assumption holds.

*Table 6d: Few-shot method ablation*

| Method | How It Classifies | What It Tests |
|---|---|---|
| Prototypical Network (default) | Distance to class mean (centroid). Almost no training. | Each class forms a single spherical cluster. If a class has sub-clusters, the centroid falls in empty space. |
| k-NN (k=3, k=5) | Majority vote of k nearest individual support examples. Zero training. | Classes can have irregular shapes. If k-NN beats ProtoNet, sub-clusters exist. |
| Linear probe | Logistic regression on frozen embeddings. | Linear separability. Also serves as a representation quality check. |

**Step 5 --- K-Shot Sensitivity**

Hold all components fixed at the best configuration from Steps 1--4.
Vary K from 1 to 15 shots per class. Identify the minimum supervision
needed to reach target performance and plot the learning curve (F1 vs.
K) to show diminishing returns.

6.4 Expected Results Template

The following tables present the structure for reporting results. Values
marked \[TBD\] will be populated after experiments. Hypothesized ranges
are provided for project planning purposes and are explicitly labeled as
estimates.

*Table 7: Main results --- encoder x pre-training (hypothesized ranges
for planning; actual TBD)*

| Encoder | Pre-training | Few-Shot | Macro-F1 | Accuracy |
|---|---|---|---|---|
| Random baseline | — | — | ~20% | ~20% |
| Majority class | — | — | [TBD] | [TBD] |
| ST-GCN + ProtoNet | None (random init) | 5-way 10-shot | [TBD] est. 35–45% | [TBD] |
| ST-GCN + Linear probe | SimCLR + Aux | — | [TBD] est. 50–60% | [TBD] |
| **ST-GCN + ProtoNet (proposed)** | **SimCLR + Aux** | **5-way 10-shot** | **[TBD] est. 65–75%** | **[TBD]** |
| Transformer + ProtoNet | SimCLR + Aux | 5-way 10-shot | [TBD] | [TBD] |
| LSTM + ProtoNet | None (random init) | 5-way 10-shot | [TBD] | [TBD] |
| 1D-CNN + ProtoNet | None (random init) | 5-way 10-shot | [TBD] | [TBD] |

*Table 8: Per-strategy F1 by feature dimension (RQ1 ablation; all
values TBD). Uses best encoder, best features, SSL + Aux pre-training.*

| Strategy | Spatial-Only | Temporal-Only | Full (S+T) | Dominant Dim. |
|---|---|---|---|---|
| Create Depth | [TBD] | [TBD] | [TBD] | [TBD] |
| Intercept | [TBD] | [TBD] | [TBD] | [TBD] |
| Move to Net | [TBD] | [TBD] | [TBD] | [TBD] |
| Passive | [TBD] | [TBD] | [TBD] | [TBD] |
| Defensive | [TBD] | [TBD] | [TBD] | [TBD] |

*Table 8b: Per-strategy F1 by input feature layer (Step 1 ablation; all
values TBD). Uses ST-GCN encoder, SSL + Aux pre-training, ProtoNet.*

| Strategy | L0 (x,y) | L1 (+vel, accel) | L2 (+court context) | L3 (+joint angles) | L4 (+racket, stretch) |
|---|---|---|---|---|---|
| Create Depth | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Intercept | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Move to Net | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Passive | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Defensive | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

6.5 Visualization Plan

-   **t-SNE / UMAP embedding plots:** Colored by strategy class,
    comparing random-init vs. SSL-pretrained embeddings. Expect tighter,
    more separated clusters after pre-training.

-   **Confusion matrix:** 5x5 matrix showing per-class error patterns.
    Expect defensive/passive confusion (both involve reactive body
    postures).

-   **Graph attention maps:** Visualize which joints and temporal frames
    receive highest attention per strategy class. Supports
    interpretability claims.

-   **K-shot learning curve:** F1 as a function of K (1 to 15), showing
    diminishing returns and minimum viable supervision.

7\. Implementation

7.1 Codebase Structure

*Table 9: Module responsibilities*

| File | Responsibility | Pipeline Stage |
|---|---|---|
| config.py | Hyperparameters, file paths, experiment settings | Global configuration |
| pose_extractor.py | YOLOv8-Pose + ViTPose skeleton extraction with Kalman filtering | A1 / C3: Pose Extraction |
| graph_builder.py | Dual-player spatio-temporal graph construction (34 nodes, 3 adjacency types) | A2: Graph Construction |
| feature_eng.py | Node feature enrichment: velocity, acceleration, court-relative distances, joint angles. Configurable feature layer selection. | A2 / C5: Feature Engineering (L0–L3 layers) |
| shot_detector.py | TrackNetV3 wrapper for shuttle tracking and hit event detection. Wrist-velocity fallback. Outputs shot timestamps. | C1: Shot Boundary Detection |
| court_detector.py | Automatic court line detection (Canny edge detection + probabilistic Hough transform) and RANSAC-based homography estimation against badminton court template. No manual input required. | C2: Court Detection |
| stgcn_model.py | ST-GCN backbone with configurable layers/channels and input dim | A4 / C5: Feature Encoder |
| transformer_enc.py | Transformer-based encoder (BST-style) for architecture ablation comparison | A4: Encoder (ablation variant) |
| simclr_loss.py | NT-Xent contrastive loss, projection head, augmentation pipeline | A5: Self-Supervised Learning |
| dataset.py | Data loading, episode sampling, fold splitting, storage management | Data Pipeline |
| proto_net.py | Prototypical network, prototype computation, distance metrics, confidence. Also k-NN variant. | B1–B2 / C6: Few-Shot Classification |
| train.py | Training loops for both SSL and few-shot stages | Training Pipeline |
| inference.py | End-to-end Phase C prediction on new video. Orchestrates shot detection, calibration, pose extraction, encoding, and classification. | C1–C6: Deployment / Demo |

7.2 Technical Stack

-   **Framework:** PyTorch 2.x with PyTorch Geometric for graph
    operations.

-   **Pose Estimation:** YOLOv8-Pose (real-time, person detection +
    keypoints) with ViTPose (high-accuracy 17-joint COCO format) for
    refinement.

-   **Shuttle Tracking:** TrackNetV3 (pre-trained) for shuttle detection,
    trajectory extraction, and hit event detection during Phase C
    inference. Open-source with pre-trained weights for badminton.

-   **Court Detection:** OpenCV-based automatic court line detection
    (Canny + probabilistic Hough transform) with RANSAC homography
    estimation against a standard badminton court template (13.4m ×
    6.1m). Fully automatic — no manual calibration required.

-   **Video Processing:** yt-dlp for ShuttleSet video download; ffmpeg
    for frame extraction at 30fps; frames stored as
    match_ID/rally_ID/frame_XXXX.jpg.

-   **Compute:** Google Colab Pro (T4/A100 GPU); estimated \~8 hours for
    SSL pre-training, \~1 hour for few-shot training, \~30–60 seconds
    per rally for Phase C inference.

-   **Reproducibility:** All random seeds fixed; config.py stores all
    hyperparameters; experiment tracking via Weights & Biases.

7.3 Data Directory Structure

Processed data follows a consistent directory hierarchy to ensure
traceability from raw frames to final predictions:

-   **FineBadminton:** data/finebadminton/rally_ID/frame_XXXX.jpg
    (provided frames), data/finebadminton/skeletons/rally_ID.npy
    (extracted skeletons).

-   **ShuttleSet:** data/shuttleset/videos/match_ID.mp4 (downloaded,
    deletable), data/shuttleset/frames/match_ID/rally_ID/frame_XXXX.jpg
    (extracted, deletable),
    data/shuttleset/skeletons/match_ID/rally_ID.npy (persistent).

-   **Outputs:** models/ (checkpoints), results/ (metrics, plots,
    confusion matrices per fold).

8\. Project Timeline

The project follows a 12-week timeline organized into four phases.
Timeline estimates account for the FineBadminton frames being
pre-provided (eliminating \~1 week of video extraction work).

*Table 10: 12-week execution timeline*

| Week | Phase | Tasks | Deliverables |
|---|---|---|---|
| 1–2 | **Data Acquisition** | Request FineBadminton frames from authors. Parse ShuttleSet CSV, select 50 matches, download via yt-dlp, extract frames at 30fps. Test YOLOv8-Pose on sample frames from both datasets. | FineBadminton frames verified. ShuttleSet 50-match subset downloaded and extracted. Pose estimation quality validated. |
| 3–4 | **Skeleton Extraction** | Batch-process all frames through YOLOv8-Pose. Apply Kalman filtering for temporal smoothing. Build dual-player spatio-temporal graphs. Segment shots (T=16 frames per shot). | ~12K FB skeleton frames + ~200K SS skeleton frames processed. Shot-level segments created. Storage optimized (delete raw frames). |
| 5–6 | **SSL Pre-Training** | Implement SimCLR contrastive learning on ShuttleSet skeletons. Train ST-GCN encoder. Add auxiliary shot-type prediction task. Run linear probe evaluation. | Pre-trained ST-GCN encoder. Linear probe accuracy reported. Embedding visualizations (t-SNE). |
| 7–8 | **Few-Shot Training** | Implement Prototypical Networks. Run 5-fold CV on FineBadminton. Execute all ablation studies (input representation, encoder architecture, pre-training, few-shot method, K-shot sensitivity). | Main results table populated. Ablation tables complete. Confusion matrices generated. |
| 9–10 | **Analysis & Visualization** | Generate all planned visualizations. Analyze per-strategy feature importance (RQ1). Quantify SSL benefit (RQ2). Compute confidence calibration. | All figures and tables finalized. RQ1 and RQ2 answered with quantitative evidence. |
| 11–12 | **Report & Demo** | Write final report. Build inference demo on custom video. (Stretch: test on amateur phone footage.) Prepare presentation. | Final report submitted. Working demo. (Stretch: cross-domain evaluation.) |

8.1 Week 1 Action Items

To de-risk the project early, Week 1 focuses on data access validation
and pipeline feasibility:

-   **Days 1--2 (FineBadminton):** Email authors requesting frame data.
    Verify what is received (frame count, resolution, format). Test
    YOLOv8-Pose on 10 sample frames. Confirm annotation format (JSON,
    CSV, etc.).

-   **Days 3--4 (ShuttleSet):** Clone CoachAI GitHub repo. Parse CSV to
    extract YouTube URLs. Apply selection criteria (1080p, diversity,
    completeness). Test-download 1--2 videos to verify availability.

-   **Days 5--7 (Pipeline Testing):** Extract frames from test downloads
    via ffmpeg. Run YOLOv8-Pose on samples from both datasets. Visualize
    skeleton quality and check for missing keypoints. Estimate full
    processing time for batch extraction.

9\. Limitations and Risk Analysis

9.1 Data Limitations

-   **Small test set variance:** With 4 test rallies per fold (\~50--65
    shots), a single misclassified rally can shift macro-F1 by 5--10
    percentage points. We mitigate this through 5-fold CV and reporting
    confidence intervals, but results should be interpreted with this
    variance in mind.

-   **Class imbalance:** Strategy distribution in FineBadminton is
    uneven. Balanced episodic sampling addresses this during training,
    but test set evaluation on natural distribution may yield different
    results than balanced evaluation. We report both.

-   **Annotation subjectivity:** Tactical strategies are inherently
    ambiguous; expert annotators may disagree on borderline cases. We
    rely on the existing FineBadminton annotations without
    inter-annotator agreement metrics.

9.2 Technical Limitations

-   **2D skeleton limitations:** We use 2D pose estimation, losing depth
    information that could help distinguish strategies (e.g., shuttle
    height for create depth). 3D pose lifting (e.g., MotionBERT) is a
    potential extension but adds complexity.

-   **Pose estimation errors:** Occlusion, fast motion blur, and
    overlapping players cause pose estimation failures. Error rates are
    higher for amateur footage with suboptimal camera angles.

-   **Domain gap:** Pre-training on broadcast footage and evaluating on
    broadcast footage does not validate generalization to amateur phone
    recordings. We acknowledge this as an unvalidated claim pending
    future data collection.

-   **Shuttle trajectory (partially addressed):** Several strategies
    (create depth, intercept) are partially defined by shuttle placement.
    TrackNetV3 is integrated in Phase C for shot boundary detection and
    also produces shuttle (x, y) trajectory per frame as a byproduct.
    However, this trajectory is not yet incorporated as an input feature
    to the encoder during Phases A–B (the encoder is trained on
    skeleton-only data). Integrating shuttle trajectory as a graph-level
    feature or 35th node during training is future work that would
    require re-running Phase A pre-training on skeleton + shuttle data.

-   **Court detection robustness:** Automatic court line detection
    (Canny + Hough transform + RANSAC) works reliably on broadcast
    footage with clear court markings and stable camera angles. On
    amateur footage with partial court visibility, oblique angles, or
    cluttered backgrounds, fewer than 4 court lines may be detected,
    degrading homography accuracy. The system reports a court detection
    confidence score (based on number of matched lines and reprojection
    error) and flags low-confidence calibrations. A learning-based court
    keypoint detector could improve robustness and is noted as future
    work.

-   **Frame rate considerations:** FineBadminton provides frames at
    20fps while ShuttleSet is processed at 30fps. While 20fps is
    sufficient for human motion capture (tested empirically in Week 1),
    the frame rate mismatch between pre-training and fine-tuning data
    requires temporal resampling or rate-agnostic shot segmentation.

9.3 Operational and Methodological Risks

*Table 11: Combined risk assessment and mitigation*

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| FineBadminton frame data request denied | Low | High — blocks few-shot pipeline | Polite email citing research/education use. Offer data agreement. Backup: manually extract 40 rallies from ShuttleSet broadcast footage. |
| ShuttleSet YouTube URLs dead (copyright takedowns) | Medium | Medium — reduces pre-training data | 50-match target provides redundancy; if >10 fail, select alternatives from remaining 1,450 matches. 40 successful downloads is sufficient minimum. |
| YOLOv8-Pose fails on 20fps FineBadminton frames | Very Low | Medium — degrades skeleton quality | Test on sample frames in Week 1. YOLOv8 performs well at 30fps; 20fps is adequate. Fallback: use ViTPose (more robust, slower). |
| SSL pre-training does not improve over random init (RQ2 negative) | Medium | Undermines core contribution; still publishable as negative result | Report honest comparison. Explore alternative SSL methods (BYOL, VICReg). Negative result is still a valid finding. |
| Class confusion between defensive and passive | High | Reduces macro-F1 significantly | Consider merging into "non-aggressive" super-class. Report both 4-class and 5-class results. |
| Insufficient data for rare strategy classes | High | Unreliable prototypes for underrepresented classes | Report per-class sample count. Drop classes with < 5 support examples from evaluation. Use data augmentation on support set. |
| Pose extraction fails on target video quality | Medium | Blocks entire skeleton pipeline | Fallback to ShuttleSet CSV tracking data (no skeleton needed). Report performance degradation without skeleton features. |
| Court detection fails on amateur footage | Medium | Degrades L0/L2 features — court-relative coords become inaccurate | Report court detection confidence score. System flags low-confidence homographies. Fallback: use pixel-space coordinates (loses court-relative features but pipeline still runs). Future: learning-based court keypoint detector for robustness. |
| TrackNetV3 fails to track shuttle | Low (broadcast) / Medium (amateur) | Blocks shot boundary detection in Phase C | Fallback: wrist velocity peak detection from skeleton data (cruder but no extra model). For training data, shot boundaries are pre-annotated so TrackNet failure only affects inference. |

10\. Success Criteria and Measurable Targets

We define three tiers of success to account for the uncertainty inherent
in a novel task with limited data:

*Table 12: Tiered success criteria*

| Tier | Criterion | Metric | Interpretation |
|---|---|---|---|
| **Minimum** | System produces above-chance predictions and SSL improves over random init | Macro-F1 > 40% AND SSL > random-init by > 10 pts | Validates approach feasibility; supports RQ2 |
| **Target** | Meaningful tactical discrimination with clear spatial/temporal insights | Macro-F1 65–75% AND per-strategy ablation shows differential feature importance | Addresses both RQs; demonstrates practical utility |
| **Stretch** | Near-supervised performance with interpretable and generalizable results | Macro-F1 > 75% AND works on custom-collected amateur footage | Publication-ready results; validates full vision |

11\. Future Work

-   **Multi-modal fusion:** TrackNetV3 is already integrated for Phase C
    inference (shot detection), producing shuttle trajectory as a
    byproduct. The next step is incorporating this trajectory as an
    encoder input feature during Phases A–B training — either as a 35th
    graph node (shuttle position per frame) or as a global graph-level
    feature. This would give the encoder access to shot direction and
    landing position, directly addressing the main signal gap for
    strategies like create depth and intercept. Court occupancy heatmaps
    are a further extension.

-   **3D pose lifting:** Replace 2D pose estimation with monocular 3D
    lifting (MotionBERT, MotionAGFormer) to recover depth information
    and improve discrimination of strategies involving vertical shuttle
    placement.

-   **Rally-level temporal modeling:** Current analysis operates at the
    shot level. Extending to rally-level sequence modeling (e.g.,
    Transformer over shot embeddings) could capture strategic arcs that
    span multiple shots.

-   **Active learning:** Use the confidence estimation module to
    identify the most informative unlabeled rallies for expert
    annotation, iteratively expanding the labeled set from 40 rallies
    with maximum efficiency.

-   **Cross-sport transfer:** Evaluate the pre-trained encoder on
    tennis, squash, and table tennis tactical analysis to test the
    generalizability of the learned representations.

-   **Domain adaptation for amateur footage:** Collect a small set of
    amateur phone-recorded matches and develop domain adaptation
    techniques to bridge the visual gap between broadcast and amateur
    footage.

-   **Learning-based court detection:** Replace the Hough transform +
    RANSAC court detection with a trained court keypoint detector (small
    CNN predicting court corner and line intersection positions). This
    would improve robustness on amateur footage with partial court
    visibility, non-standard markings, or oblique camera angles where
    classical line detection struggles.

12\. References

\[1\] Wang, W. Y., et al. (2022). ShuttleNet: Position-aware Fusion of
Rally Progress and Player Styles for Stroke Forecasting in Badminton.
Proceedings of the AAAI Conference on Artificial Intelligence.

\[2\] FineBadminton Dataset. Fine-grained badminton annotations with
strategy labels, shot subtypes, and quality scores. \[Dataset paper
reference TBD upon access\].

\[3\] Yan, S., Xiong, Y., & Lin, D. (2018). Spatial Temporal Graph
Convolutional Networks for Skeleton-Based Action Recognition.
Proceedings of the AAAI Conference on Artificial Intelligence.

\[4\] Snell, J., Swersky, K., & Zemel, R. (2017). Prototypical Networks
for Few-shot Learning. Advances in Neural Information Processing
Systems.

\[5\] Li, L., et al. (2021). CrosSCLR: Cross-View Contrastive Learning
for 3D Skeleton-Based Action Recognition. Proceedings of CVPR.

\[6\] Guo, T., et al. (2022). AimCLR: Contrastive Learning of
Skeleton-Based Action Recognition with Extreme Augmentations.
Proceedings of CVPR.

\[7\] Chen, T., et al. (2020). A Simple Framework for Contrastive
Learning of Visual Representations (SimCLR). Proceedings of ICML.

\[8\] Chang, W. (2025). BST: Badminton Stroke-type Transformer for
Skeleton-Based Stroke Classification. CVPR 2025 Workshop.

\[9\] Dong, Y., et al. (2025). RacketVision: Racket Detection and
Tracking in Badminton Video. \[Venue TBD\].

\[10\] Sun, Y., et al. (2023). TrackNetV3: Real-Time Shuttlecock
Tracking in Badminton. Proceedings of ACM Multimedia.

---

Appendix A: Design Reasoning Archive
=====================================

This appendix documents the key design decisions, trade-offs, and
reasoning that shaped the project's experimental design. These notes are
not part of the formal report but serve as a reference for understanding
why specific choices were made.

A.1 Why Skeleton-Based (Paradigm 1) Over Vision-Based (Paradigm 2)
-------------------------------------------------------------------

There are two fundamentally different ways to approach tactical strategy
recognition:

**Paradigm 1 (skeleton-based, chosen):** Video frames → pose estimation
→ skeleton coordinates → encoder (ST-GCN or Transformer) → embedding →
ProtoNet. The model never sees pixels. It works with 34 points moving
through 2D space over 16 frames — structured numerical data with known
graph topology.

**Paradigm 2 (vision-based, not chosen):** Video frames → visual
encoder (ViT, ResNet, VideoMAE) → embedding → ProtoNet. The model sees
raw pixels and must learn everything from the image.

Key trade-offs:

-   **Data efficiency:** Paradigm 1 is far more data-efficient.
    Structured skeleton input + graph priors means the model learns from
    less data. Paradigm 2 (vision) typically needs orders of magnitude
    more data for pre-training. With only 40 labeled rallies and ~5K
    unlabeled shots, skeleton-based is the pragmatic choice.

-   **Strategy signal alignment:** The strategy taxonomy (Table 2) is
    defined almost entirely in terms of skeletal and positional signals:
    "forward court position," "stretched body posture," "progressive
    forward displacement." The strategies were designed around what
    skeletons can capture.

-   **Research novelty:** Skeleton + few-shot + tactical strategy is
    novel. ViT applied to sports video is well-explored in existing
    literature.

-   **What skeletons miss:** Shuttle trajectory (where the shuttle
    lands), racket angle (shot direction), and visual context (court
    markings, environmental cues). These are acknowledged as limitations
    rather than disqualifying weaknesses.

**ViT as a cross-paradigm baseline (optional):** A pre-trained ViT
(ImageNet) with temporal pooling feeding into the same ProtoNet could
serve as a Paradigm 2 sanity check. If it outperforms the entire
skeleton pipeline, that would indicate the skeleton engineering was
unnecessary. If the skeleton pipeline wins, it validates the approach.
This is ~1 day of work (swap the feature extractor, keep everything
downstream identical) and strengthens the paper regardless of outcome.
However, it is not a required experiment.

A.2 Where BST Fits (and Where It Does Not)
-------------------------------------------

BST (Chang, 2025) is not a separate box in the pipeline. It is an
alternative answer to one question: "what encoder sits in the middle?"

The pipeline has an encoder slot. Currently it is filled by ST-GCN. The
architecture ablation (Step 2) tests: what if we put a Transformer
there instead?

"BST-style" means: take BST's architectural idea — a Transformer that
processes skeleton joint sequences using self-attention instead of graph
convolutions — and drop it into our pipeline with our input
(skeleton-only, no shuttle trajectory) and our training regime (SimCLR
pre-training → ProtoNet few-shot).

We are NOT running BST itself. BST the actual model takes skeleton +
shuttlecock trajectory, is fully supervised on stroke-type labels, and
solves a different task (stroke classification). We cannot directly
compare against BST's published numbers because the task, labels, data
regime, and inputs are all different.

What we can do: test whether BST's core design insight — "Transformers
work well for skeleton-based badminton understanding" — holds in the
few-shot, strategy-level regime.

A.3 How the Prototypical Network Works (and Why It Is Not Really Training)
--------------------------------------------------------------------------

Phase B involves minimal learning. A Prototypical Network is barely a
"network" — it is a nearest-neighbor lookup in the embedding space that
Phase A created.

**The literal procedure:**

Step 1: Take the 32 support rallies. Run each shot through the frozen
Phase A encoder to get a 256-dim embedding. For each strategy class,
average all embeddings belonging to that class. These averages are the
5 "prototypes." This is arithmetic mean. No gradient descent. No loss
function.

Step 2: Take a new query shot. Run it through the same encoder. Measure
Euclidean distance to each of the 5 prototypes. Assign to the nearest
one.

**Where does the actual learning happen?** Mostly in Phase A, implicitly.
The contrastive pre-training forced the encoder to put similar movements
near each other in embedding space. If "intercept" shots share a
distinctive motion signature (forward lunge, early contact, flat arm),
the encoder — trained to distinguish movement patterns — should
naturally group them together. Prototypes merely label those clusters.

**When ProtoNet fails:** If a class does not form a single tight
cluster, the average (prototype) falls in empty space between
sub-clusters. Example: "defensive recovery left" and "defensive recovery
right" might form two sub-clusters. The centroid falls in the middle
where no actual defensive shot exists.

**Alternatives considered:**

-   **k-NN (k=3 or k=5):** Classify by majority vote of k nearest
    support examples. No centroid assumption. Tests whether classes have
    sub-clusters. Cheap to implement, included as Step 4 ablation.

-   **Matching Networks:** Attention-weighted distance to every support
    example. Handles irregular class shapes but adds complexity.

-   **MAML:** Fine-tunes the entire encoder per episode using inner-loop
    gradient steps. Too complex for the small support set and adds
    overfitting risk.

-   **Simple linear probe:** Logistic regression on frozen embeddings.
    Tests linear separability. Included as a representation quality
    check.

ProtoNet was chosen because extreme simplicity is a strength when you
have ~6-8 support examples per class. There is almost nothing to
overfit.

A.4 Why Input Representation (Step 1) Comes Before Encoder Architecture (Step 2)
----------------------------------------------------------------------------------

The ablation axes must be ordered because earlier decisions affect later
comparisons. Input representation comes first because:

You need to decide what data goes into the encoder before you can
meaningfully compare which encoder is best. If you compare ST-GCN vs.
Transformer on raw (x, y) coordinates only and the Transformer wins,
you do not know if it would still win once you add velocity and
court-relative features — because those hand-crafted features might
give ST-GCN the contextual information that the Transformer was
learning from scratch via attention.

The correct logic:

1.  Fix the encoder (ST-GCN). Find the best input representation.
2.  Take the best input. Now compare encoders.
3.  Take the best (encoder + input). Now compare pre-training.
4.  Take the best (encoder + input + pre-training). Now compare
    few-shot methods.
5.  Run K-shot sensitivity on the final best configuration.

Each step varies one thing and holds everything else constant at the
best value found so far.

A.5 Why Enriched Node Features Matter More Than You Think
----------------------------------------------------------

The report originally specified only raw (x, y) coordinates as
node features — 2 features per joint per frame. This forces the encoder
to derive all tactical signals from position alone. But the strategy
taxonomy explicitly requires signals the model cannot efficiently learn
from coordinates:

-   **Velocity:** "Quick forward lunge" (intercept) vs. "stationary"
    (passive). The model can learn velocity as a finite difference
    between adjacent frames, but this wastes network capacity on what is
    a trivial computation.

-   **Court context:** "Forward court position" (intercept) vs. "rear
    court position" (defensive). Raw coordinates encode this, but
    dist_to_net makes it explicit and removes dependence on the
    homography calibration quality.

-   **Body configuration:** "Stretched posture" (defensive) vs. "neutral
    posture" (passive). Joint angles encode this directly; raw
    coordinates require the model to learn trigonometry.

-   **Inter-player relationship:** "Opponent pushed back" (create depth).
    dist_to_opponent_centroid makes this a single feature rather than
    something the model must compute from 34 separate (x, y) pairs.

Layers L0–L3 are all free to compute (a few lines of numpy on existing
skeleton data), require no new data or models, and directly encode the
signals the strategy taxonomy says matter. The ST-GCN architecture
handles enriched features with zero modification — only the input
channel dimension changes.

A.6 Strategy Signal Mapping to Pipeline Components
----------------------------------------------------

This table maps each signal type from the strategy taxonomy to what the
model actually receives and which pipeline component provides it:

| Signal Type | What the Model Receives | Pipeline Component |
|---|---|---|
| Spatial signals (court position, player locations) | (x,y) court-relative coords + dist_to_net, dist_to_center (L2) | Node features (L0) + court context (L2) |
| Temporal signals (timing, sequence patterns) | How features change across T=16 frames; velocity/accel (L1) | Temporal edges in graph + kinematics (L1) |
| Movement patterns (lunge, stretch, recovery) | Skeleton shape changing over time: joint angles (L3) + velocity (L1) | Learned by encoder from L1 + L3 features |
| Player state / context (offensive, under pressure) | Relative positioning between two skeletons: dist_to_opponent (L2) | Inter-player edges + court context (L2) |
| Shot/intent signals (trajectory, direction) | NOT directly available. Partially inferable from arm kinematics + racket angle | Known limitation. Racket (L4, stretch) helps. Shuttle trajectory = future work. |

The last row is the critical gap: shot trajectory signals ("deep
controlled trajectory" for create depth, "flat aggressive trajectory"
for intercept) are not directly available in a skeleton-only pipeline.
The model can only infer shot direction from arm kinematics and (if
available) racket angle. This is why create depth may be the hardest
strategy to classify, and why the RacketVision 18th-joint stretch goal
and eventual TrackNet shuttle integration are high-value additions.

A.7 Full Strategy Taxonomy Including Excluded Categories
---------------------------------------------------------

The complete FineBadminton taxonomy includes 8 categories. Only 5 are
in scope for this project:

| Strategy | Key Signals | In Scope? |
|---|---|---|
| Create depth | Rear-court landing; opponent in mid/front; controlled execution; deep trajectory | Yes |
| Intercept | Forward court; early timing; quick forward lunge; flat/aggressive trajectory | Yes |
| Move to net | Progressive forward positioning; sequential across multiple shots; net-oriented | Yes |
| Passive | No spatial gain; standard/late timing; minimal repositioning; non-aggressive | Yes |
| Defensive | Rear/stretched position; reactive timing; recovery movement; high lift | Yes |
| Deception | Spatial expectation vs. actual mismatch; timing disguise; deceptive body mechanics | No — requires biomechanical data |
| Hesitation | Normal setup; delayed contact; micro-stall before execution | No — requires sub-frame timing |
| Seamlessly | Smooth transitions; fluid movement; consistent shot flow | No — quality modifier, not strategy |

A.8 Ablation Step Summary Diagram
----------------------------------

```
STEP 1 — INPUT REPRESENTATION (fix encoder = ST-GCN, SSL+Aux)
├── L0: Raw [x,y] only                              ← current baseline
├── L1: + velocity, acceleration [x,y,vx,vy,ax,ay]  ← free to compute
├── L2: + court-relative distances                   ← free to compute
├── L3: + joint angles                               ← free to compute
├── Full enriched features (L0–L3)                   ← best representation
├── Spatial-only vs. temporal-only (RQ1)             ← on best features
├── Single-player vs. dual-player                    ← inter-player value
└── + Racket 18th joint, L4 (stretch)                ← needs RacketVision

STEP 2 — ENCODER ARCHITECTURE (fix input = best from Step 1, SSL+Aux)
├── ST-GCN          ← graph convolutions, structural priors
├── Transformer     ← self-attention, BST-style
├── LSTM            ← sequential baseline (sanity check)
└── 1D-CNN          ← convolutional baseline (sanity check)

STEP 3 — PRE-TRAINING (fix encoder + input = best from Steps 1–2)
├── Random initialization
├── SimCLR contrastive only
└── SimCLR + auxiliary shot-type (default)

STEP 4 — FEW-SHOT METHOD (fix encoder + input + pre-training = best)
├── ProtoNet (centroid distance)
├── k-NN, k=3 and k=5 (individual example distance)
└── Linear probe (logistic regression, representation quality check)

STEP 5 — K-SHOT SENSITIVITY (final best configuration)
└── K = 1, 3, 5, 8, 10, 15
```

Each step varies one axis. The steps are ordered so that earlier
decisions (what the model sees) are settled before later ones (how the
model processes it). Results from each step carry forward as the fixed
value for subsequent steps.

A.9 The Full Pipeline as One Picture
-------------------------------------

```
ShuttleSet (5K+ unlabeled shots)
         │
         ▼
   ┌─────────────┐
   │  YOLOv8-Pose │ ──→ Raw skeleton coords (17 joints × 2 players)
   └─────────────┘
         │
         ▼
   ┌──────────────────┐
   │  Feature Eng.     │ ──→ Enriched node features (L0–L3):
   │  (feature_eng.py) │     [x, y, vx, vy, ax, ay, dist_net,
   └──────────────────┘      dist_center, dist_opp, angles...]
         │
         ▼
   ┌─────────────────────────────────────────┐
   │         ENCODER (the variable)           │
   │                                          │
   │  ST-GCN (default)  ◄── graph priors     │   Phase A:
   │       vs.                                │   SimCLR contrastive
   │  Transformer        ◄── BST's insight   │   + aux shot-type
   │       vs.                                │   pre-training
   │  LSTM / 1D-CNN      ◄── sanity check    │
   │                                          │
   └─────────────────────────────────────────┘
         │
         │  256-dim embedding per shot
         ▼
   ┌─────────────────────────────────────────┐
   │      CLASSIFIER (the variable)           │
   │                                          │   Phase B:
   │  ProtoNet: distance to class centroids   │   Few-shot on 40
   │       vs.                                │   FineBadminton
   │  k-NN: nearest individual examples       │   labeled rallies
   │       vs.                                │
   │  Linear probe: logistic regression       │
   │                                          │
   │  → strategy label + confidence score     │
   └─────────────────────────────────────────┘
```

Phase A teaches the encoder what badminton movement looks like by
training on thousands of unlabeled shots. Phase B shows it 40 examples
of "this movement pattern = this strategy" and asks it to generalize.
The ablations test which combination of input features, encoder
architecture, pre-training, and classifier makes that generalization
work best.

A.10 Inference on New Video — The Shot Detection Gap
-----------------------------------------------------

During training (Phases A–B), shot boundaries are handed to us:
FineBadminton provides annotated shot timestamps, ShuttleSet provides
them in CSV tracking data. When someone uploads a new video (Phase C),
none of that exists. The video is just a continuous stream of frames.

This creates a chicken-and-egg problem: the pipeline needs discrete shot
windows (T=16 frames each) to feed the encoder, but generating those
windows requires knowing when each shot occurs.

**Options considered for shot detection:**

1. **TrackNetV3 (chosen):** Pre-trained shuttle tracker for badminton.
   Detects the shuttlecock in each frame, tracks its trajectory, and
   identifies hit events from sharp trajectory direction changes. This is
   the gold standard approach used by ShuttleSet's own annotation
   pipeline. Open source with pre-trained weights — zero training needed.
   Bonus: shuttle trajectory (x, y per frame) comes free, partially
   addressing the main signal gap in the skeleton-only approach.

2. **Pose-based wrist velocity peaks:** Since YOLOv8-Pose already runs
   on every frame, a shot strike produces a distinctive wrist acceleration
   spike detectable from L1 kinematics. Requires no extra model but is
   cruder — repositioning movements and feints produce false positives.
   Used as a fallback/validation signal alongside TrackNet.

3. **Audio-based detection:** Racket-shuttle contact produces a
   distinctive sound. A simple audio peak detector works well in clean
   audio conditions. Rejected as primary method because it requires video
   with audio (not always available) and is fragile to crowd noise and
   commentary.

4. **Combined approach (recommended):** TrackNetV3 as primary, wrist
   velocity as secondary validation. When both agree on a hit event,
   confidence is high. When they disagree, the shot boundary is flagged
   as uncertain.

**Why TrackNet is a double win:** It solves the shot detection problem
AND gives us shuttle trajectory for free. The shuttle trajectory is one
of the biggest information gaps in the skeleton-only approach — strategies
like "create depth" are partially defined by where the shuttle lands,
which skeletons alone cannot capture. With TrackNet already running for
shot detection, shuttle (x, y) per frame is available to optionally
integrate as a graph-level feature or 35th node. This integration is
future work for Phases A–B (would require re-training the encoder on
skeleton + shuttle data), but the trajectory is immediately available
for display in the Phase C inference UI.

**Court detection for new video:** The L2 features (dist_to_net,
dist_to_center) require a homography from pixel coordinates to court
coordinates. For training data this is pre-calibrated or inferable from
consistent broadcast camera angles. For new video, automatic court
detection is required.

The approach uses classical computer vision in a three-step pipeline:

1. **Edge detection:** Canny edge detection on a reference frame
   (median-composite of first N frames to reduce player occlusion).
2. **Line detection:** Probabilistic Hough transform (cv2.HoughLinesP)
   extracts line segments, filtered by length and angle to retain only
   court boundary and service line candidates.
3. **Court model fitting:** RANSAC-based homography estimation matches
   detected lines against a known badminton court template (13.4m ×
   6.1m). The four outer court corners plus the net line provide
   sufficient correspondences for a robust homography.

This works reliably on broadcast footage (clear markings, top-down or
elevated camera). For amateur footage with partial court visibility,
the system uses whatever lines are detected (minimum 4 correspondences)
and reports a confidence score based on number of matched lines and
reprojection error. Low confidence triggers a warning in the UI.

A learning-based court keypoint detector (small CNN trained on court
images) could improve robustness on difficult footage and is noted as
future work.

**Phase C inference timing estimate:** For a single rally (~20 seconds
at 30fps = ~600 frames):
- Frame extraction: ~2 seconds
- Court detection (Hough + RANSAC): ~1 second (runs once per video)
- TrackNetV3: ~10 seconds (GPU-bound)
- YOLOv8-Pose: ~15 seconds (GPU-bound, runs in parallel with TrackNet)
- Feature engineering + encoding + classification: ~3 seconds
- Total: ~30 seconds on T4 GPU

TrackNet and YOLOv8-Pose can run in parallel since they are independent
models operating on the same frames. The bottleneck is pose estimation.
