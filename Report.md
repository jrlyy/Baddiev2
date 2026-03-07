**Few-Shot Tactical Strategy Recognition in Badminton**

Using Self-Supervised Skeleton-Based Spatio-Temporal Graph Learning

Project Report

\[Author Name\] \| \[Institution\] \| \[Date\]

---

Abstract
--------

This project develops an end-to-end pipeline for recognizing fine-grained
tactical strategies in badminton from raw match footage. The system extracts
dual-player skeleton sequences from video, learns generalizable spatio-temporal
representations through self-supervised contrastive pre-training on unlabeled
data, and classifies tactical patterns using few-shot prototypical networks
with only 40 expert-labeled rallies. We target five classifiable strategies
(intercept, defensive, move to net, create depth, passive) and investigate
two core research questions: (1) the relative contribution of spatial versus
temporal features to strategy recognition, and (2) the effectiveness of
self-supervised pre-training in reducing label dependency.

---

1\. Introduction
----------------

Tactical analysis in badminton remains a manually intensive process. Coaches
review match footage frame-by-frame to identify strategic patterns such as
interceptions, defensive formations, and net approaches. This process is
time-prohibitive, subjective, and inaccessible to amateur and
semi-professional players who lack dedicated analyst support.

Automated approaches face a fundamental bottleneck: fine-grained tactical
annotations require domain expertise and are expensive to produce at scale. In
badminton, a hitting player's strategy is reactive and anticipatory — it is
influenced not only by their own intentions but also by the opponent's
positioning, movement, and state. The FineBadminton dataset, the only
publicly available resource with expert-level strategy labels, contains just
40 annotated rallies. Fully supervised deep learning is infeasible at this
scale.

We target five classifiable strategies that are distinguishable from skeleton
and positional data alone: **intercept**, **defensive**, **move to net**,
**create depth**, and **passive**. The following table defines each strategy
and the observable signals that distinguish it:

*Table 1: Target strategy definitions and observable skeleton/position signals*

| Strategy | Goal | Hitter Signals | Opponent Signals | Shot Type |
|---|---|---|---|---|
| **Intercept** | Cut off shuttle early; offensive play | Early racket contact; forward court position; fast forward step | Opponent mid-court/forward; low lateral movement; upright pose | Flat / aggressive |
| **Defensive** | Protect position; recover control | Reactive contact; rear-court position; stretched body | Opponent forward/stretched; off-balance; weight leaning forward/back | High lift / clear / block |
| **Move to Net** | Drive / drop / approach | Sequential forward steps; forward racket motion | Opponent gradually moving backward; mid-court; upright stance | Advance toward net; offensive positioning |
| **Create Depth** | Deep clear / push back opponent | Rear-court contact; shot directed deep; forward/backward step | Opponent mid/front court; moving forward; leaning slightly forward | Push opponent to rear court; gain spatial control |
| **Passive** | Recovery play; minimal gain | Neutral placement; low speed; minimal forward/backward movement | Opponent mid/rear court; stable stance; minimal movement | Any non-aggressive / neutral shot |

**Out of scope:** Deception (requires biomechanical discrepancy data between
preparation and execution phases — not captured in 2D skeletons), hesitation
(requires sub-frame timing analysis beyond our temporal resolution), seamlessly
(a quality modifier, not a discrete strategy class), and "a high net early
shot" (insufficient samples for reliable classification). Shots with these
labels are excluded from the training set.

**Research questions:**

- **RQ1:** What is the relative contribution of spatial features (player
  positioning, court geometry) versus temporal features (movement sequences,
  timing patterns) to the recognition of different tactical strategies?

- **RQ2:** To what extent does self-supervised contrastive pre-training on
  unlabeled match footage improve few-shot tactical classification compared
  to randomly initialized baselines?

---

2\. Literature Review
---------------------

### 2.1 Skeleton-Based Action Recognition

Skeleton-based approaches model human motion as sequences of joint coordinates
rather than raw pixel data. ST-GCN (Yan et al., 2018) introduced graph
convolutions over spatial joint connections and temporal sequences, achieving
81.5% top-1 accuracy on Kinetics-400 for general action recognition. Subsequent
work (2s-AGCN, MS-G3D, CTR-GCN) improved spatial attention mechanisms and
multi-scale temporal modeling. However, these methods target coarse action
categories (e.g., "playing badminton" vs. "playing tennis") rather than
fine-grained tactical patterns within a single sport.

### 2.2 Self-Supervised Learning for Skeleton Sequences

Label-free representation learning for skeletons has advanced rapidly.
CrosSCLR (Li et al., 2021) introduced cross-view contrastive learning across
different skeleton augmentation views. AimCLR (Guo et al., 2022) addressed
augmentation sensitivity through extreme augmentation strategies. 3s-HCN
applied hierarchical contrastive learning at joint, body-part, and sequence
levels. These methods demonstrate that skeleton embeddings pre-trained without
labels can match or approach supervised performance when fine-tuned, motivating
our self-supervised pre-training strategy.

### 2.3 Badminton Analytics

Existing computational badminton analysis focuses primarily on shot-level
classification rather than strategic reasoning. CoachAI (Wang et al., 2022)
applied ShuttleNet (ResNet+LSTM) to classify 22 shot types on the ShuttleSet
dataset, achieving 73.4% accuracy. FineBadminton contributed fine-grained
annotations including strategy labels but did not report classification results
for tactical categories — its contribution was the annotation framework itself.

Most recently, BST (Chang, 2025; CVPR 2025 Workshop) introduced a
Transformer-based approach for skeleton-based stroke classification on
ShuttleSet, using both player skeleton sequences and shuttlecock trajectory as
inputs. BST outperforms previous methods on stroke type classification (smash,
clear, drop, etc.), establishing the current state-of-the-art for
skeleton-based badminton analysis. Concurrently, RacketVision (Dong et al.,
2025) advanced racket detection and tracking in badminton video, providing a
complementary signal for fine-grained action understanding.

Critically, BST and RacketVision operate at the stroke type level — they
classify *what* shot was played, not *why* it was played. No published work
addresses automated tactical strategy classification in badminton.

### 2.4 Few-Shot and Meta-Learning in Sports

Prototypical Networks (Snell et al., 2017) remain a strong baseline for
few-shot classification, learning a metric space where classification is
performed by distance to class prototypes. Applications in sports have been
limited: few-shot methods have been explored for player identification and rare
event detection, but not for tactical pattern recognition. The scarcity of
tactical labels makes this an ideal application domain.

### 2.5 Research Gap

*Table 2: Positioning relative to existing work*

| Method | Task | Data Regime | Tactical? |
|---|---|---|---|
| CoachAI ShuttleNet | Shot type (22-cls) | Fully supervised | No |
| BST (Chang, 2025) | Stroke type classification | Fully supervised (skeleton + trajectory) | No |
| RacketVision (Dong, 2025) | Racket detection | Fully supervised | No |
| ST-GCN | General action recognition | Fully supervised | No |
| CrosSCLR / AimCLR | Action recognition | Self-supervised | No |
| Prototypical Networks | Image classification | Few-shot | No |
| **Ours (proposed)** | **Tactical strategy** | **SSL + few-shot** | **Yes** |

---

3\. Proposed Solution
---------------------

We propose a three-stage pipeline:

1. **Skeleton extraction** from raw video using pose estimation (YOLOv8-Pose),
   producing dual-player joint sequences.
2. **Self-supervised representation learning** via SimCLR contrastive objectives
   on unlabeled ShuttleSet match data, learning generalizable motion embeddings.
3. **Few-shot classification** using prototypical networks on the small
   FineBadminton labeled set.

The approach is skeleton-based rather than pixel-based, providing
interpretability and robustness to visual variation (camera angle, lighting,
jersey color). Skeleton inputs also align directly with the strategy taxonomy:
all five target strategies are defined in terms of joint positions, velocities,
and inter-player geometry — precisely what the skeleton graph captures.

---

4\. Datasets & Data Acquisition
--------------------------------

Both datasets feature professional broadcast matches. For the purpose of this
project, we classify tactical strategy **per shot**. The rally and match
labels are not relevant to the classification task; they appear in the demo UI
for ease of navigation convention.

**Scoring context.** Under modern BWF rally scoring, each rally gives 1 point
regardless of who served. Up to 177 rallies are possible in a 3-set match;
professional matches typically have 120–150 rallies. Each rally consists of
multiple shots (strokes), each labeled with the hitting player's strategy.

### 4.1 FineBadminton — Labeled Dataset

FineBadminton is our fine-grained labeled dataset used for few-shot
classification (Phase B).

**Frame Extraction.** The dataset comes with pre-extracted frames at 20 FPS,
organized by rally ID. No video download is required.

**Scale.** 40 rallies, 414 annotated shots. After excluding 59 shots with
out-of-scope strategy labels (deception ×14, high_net_early ×36, hesitation ×8,
seamlessly ×1), we have **355 shots with strategy labels**, of which **296 have
successfully extracted skeletons** used in training.

**Key annotation fields per shot:**

| Field | Description |
|---|---|
| `hit_frame` | Exact contact frame (absolute, no missing values) |
| `start_frame` / `end_frame` | Shot window boundaries |
| `hitter` | `"top"` or `"bottom"` court player |
| `hit_type` | Stroke type (12 classes) |
| `subtype` | Sub-classification (e.g., "flat lift", "short serve") |
| `quality` | Annotator quality score 1–7 |
| `ball_area` | Court zone of shot (9 zones) |
| `strategies` | Tactical strategy labels (our classification target) |
| `shot_characteristics` | "cross-court", "straight", "over head", etc. |
| `player_actions` | "forehand", "backhand", "turnaround" |

**Strategy label distribution (N=355 shots, multi-label):**

| Strategy | Count | % of shots |
|---|---|---|
| passive | 137 | 33.1% |
| intercept | 120 | 29.0% |
| create depth | 61 | 14.7% |
| defensive | 61 | 14.7% |
| move to net | 59 | 14.3% |
| (excluded: high_net_early) | 44 | — |
| (excluded: hesitation) | 27 | — |
| (excluded: deception) | 21 | — |
| (excluded: seamlessly) | 25 | — |

Note: `move_to_net` has only 18 samples after windowing and skeleton extraction
(6.1% of training set) — this is the most data-scarce class and underperforms
accordingly in baseline experiments (see §8).

### 4.2 ShuttleSet — SSL Pre-Training Dataset

ShuttleSet is our large-scale unlabeled dataset used for self-supervised
pre-training (Phase A).

**Frame Extraction.** The dataset provides annotations and YouTube video links
only — no raw frames. We manually downloaded videos via yt-dlp and extracted
frames at 30 FPS via ffmpeg. Of 44 matches in `match.csv`: **25 YouTube links
successfully processed**, 18 had broken links (content removed), and 1 had no
URL.

**Scale.** The 25 processed matches yield **21,191 labeled shot records** across
**963 rallies**, covering **19 shot types**. This is approximately 71× the
size of the FineBadminton labeled set, providing adequate diversity for
self-supervised pre-training.

**Coverage.** 13 tournaments, 17 unique elite players (including Viktor
Axelsen, Kento Momota, Carolina Marín, Chou Tien Chen), 2018–2021. Average
848 strokes per match, average rally length 22.0 strokes.

**Key annotation fields (from per-match JSON pipeline output):**

| Field | Description |
|---|---|
| `frame_num` | Absolute frame index of the hit event |
| `type` | Shot type label (19 classes, Chinese) |
| `player` | `"A"` or `"B"` |
| `player_location_x/y` | Hitter pixel position (used for hitter-first ordering) |
| `opponent_location_x/y` | Opponent pixel position |
| `hit_x/y`, `landing_x/y` | Shuttle contact and landing coordinates |
| `backhand`, `hit_height` | Additional shot context (partially missing) |

---

5\. Dataset Pre-processing
--------------------------

From the frames we extract player skeletons, shuttlecock trajectories, and
player court positioning. The pipeline outputs `(2, 16, 34)` skeleton tensors
per shot — the direct input to the encoder.

### 5.1 Skeleton Extraction

**Pose Extraction.** Skeleton keypoints are extracted using YOLOv8-Pose
(yolov8s-pose), which performs joint person detection and 17-joint COCO
keypoint estimation in a single forward pass. For each frame, all detected
persons are scored by their mean keypoint confidence; the top-2 detections
are retained as the two players. Temporal smoothing is applied via an
exponential filter (α=0.7) across the keypoint sequence for stability. Kalman
filtering is a planned upgrade for improved handling of occlusion frames.

**Quality audit — umpire contamination.** A systematic audit of the extracted
FineBadminton skeletons revealed a significant contamination issue. YOLOv8-Pose
selects the top-2 persons by mean keypoint confidence per frame with no spatial
constraint. The chair umpire is consistently detected with high confidence
because they are stationary, upright, and well-lit — occupying a seat at the
left edge of the court at approximately x ≈ 102px on a 1280px-wide frame.
In frames where fewer than 2 players are confidently detected, the pipeline
falls back to the umpire as the second "player" and forward-fills those frames.

**Grounding DINO-guided extraction (adopted solution).** A GDINO-guided
approach uses Grounding DINO (grounding-dino-tiny) with the text prompt
`"player"` to produce court-region bounding boxes before YOLOv8 keypoint
estimation. Only YOLO detections with IoU ≥ 0.25 against a GDINO bounding box
are accepted as valid players. If fewer than 2 valid players pass this filter,
the pipeline falls back to plain YOLOv8 top-2 selection. This approach
substantially reduces umpire contamination in the extracted skeletons.

The FineBadminton skeletons used in all experiments were extracted with the
GDINO-guided pipeline. All 40 rallies (10,620 frames) were successfully
extracted.

### 5.2 Player Ordering and Hitter Assignment

**Player ordering (Y-sort).** After selecting the top-2 detections, both
players are sorted by their mean Y keypoint centroid in image coordinates.
Since image Y increases downward:

- Player 0 (joints 0–16) = smaller mean Y = top-court player
- Player 1 (joints 17–33) = larger mean Y = bottom-court player

This assignment is made purely from image geometry — no annotations are
consulted. The resulting `(2, T, 34)` skeleton arrays saved to disk always
follow this Y-sort convention.

**Hitter ordering.** For the few-shot classifier to generalize across rallies
where either player may be the hitter, we always place the hitting player at
nodes 0–16. This is done by conditionally swapping the two 17-joint halves:

- **FineBadminton (at dataset load time):** If `hitter == "bottom"`, nodes
  0–16 and 17–33 are swapped after windowing. If `"top"`, no swap is needed
  (the Y-sorted player 0 is already the hitter). The per-rally `.npy` files on
  disk are always raw Y-sorted and are never modified.
- **ShuttleSet (at extraction time):** The annotated `player_location_y` (pixel
  Y of the hitter) is compared against each player's skeleton centroid Y. The
  closer player is identified as the hitter; if this is player 1, the halves
  are swapped before saving the per-shot `.npy`. Saved files are already
  hitter-first ordered.

Result: for both datasets, every model input has the same semantics — nodes
0–16 are the hitting player, nodes 17–33 are the opponent.

### 5.3 Shot Segmentation (Temporal Windowing)

Each shot sample is a T=16 frame window centered on the hit frame, spanning
±8 frames (≈0.8 s at 20fps). This captures the preparation phase before
contact (footwork direction, racket windup) and the follow-through after
contact (net approach confirmation, opponent repositioning). Frames before the
hit are critical — strategy is committed to during preparation, not at
contact. See Appendix B.13 for full rationale.

For FineBadminton, windowing happens at dataset load time from the per-rally
`.npy` files using the annotated `hit_frame`. For ShuttleSet, windowing
happens at extraction time; per-shot `.npy` files are already shape `(2,16,34)`.

### 5.4 Shuttlecock Trajectory

Shuttle trajectories are extracted using **TrackNetV4** (`tracknet-series-pytorch`,
local clone). V4 adds a learnable `MotionPrompt` layer (inter-frame difference
attention) over the V2 U-Net baseline, which is particularly effective for a
fast-moving object like a shuttlecock moving 50–100 pixels per frame.

Outputs: per-rally `.npy` files of shape `(T, 3)` — columns `[x, y, visible]`,
frame-aligned with the skeleton arrays. These are stored in
`datasets_preprocessing/finebadminton_shuttles/` and visualized in the demo UI
as a yellow dot + trail overlay. Shuttle trajectory is **not yet incorporated
as an encoder input** during Phases A–B; integration as a 35th graph node is
the highest-priority near-term extension.

### 5.5 Player Court Position

Accurate court-relative coordinates are needed for the L2 node features
(dist_to_net, dist_to_center, dist_to_opponent). Two approaches are available:

**Option 1 — Player-span normalization (no external calibration required).**
Normalize Y coordinates using the two players' centroid positions:

```python
p0_cy = skeleton[1, :, :17].mean(axis=-1, keepdims=True)   # (T, 1)
p1_cy = skeleton[1, :, 17:].mean(axis=-1, keepdims=True)
span  = (p1_cy - p0_cy).clip(min=1)
y_norm = (skeleton[1] - (p0_cy + p1_cy) / 2) / span        # y=0 ≈ net
```

This maps P0 centroid → y = −0.5 (top-court baseline), P1 centroid →
y = +0.5 (bottom-court baseline), net → y ≈ 0. It is fully self-contained per
shot and requires no external model. Less accurate than homography (assumes
player centroids span half-courts evenly) but a large improvement over raw
pixels.

**Option 2 — Full homography (most accurate; required for Phase C inference).**
A 3×3 perspective transform H maps pixel coordinates (u, v) to court-relative
meters (x, y) with origin at court center. For FineBadminton (fixed broadcast
camera), H is computed once per resolution group. For ShuttleSet, H is computed
per match. For Phase C (new unseen video), H is detected automatically via
Canny edge detection + probabilistic Hough transform + RANSAC fitting against
a standard badminton court template (13.4m × 6.1m for singles). Manual fallback
allows clicking four court corners in a reference frame (≈10 seconds of effort).

---

6\. Model Training
------------------

### 6.1 Few-Shot Training Setup

Given the extremely small labeled set (40 rallies), we employ **5-fold
cross-validation** to maximize data utilization and provide variance estimates.
Each fold uses:

- **32 rallies** for the support set (prototype computation)
- **4 rallies** for validation (hyperparameter tuning, early stopping)
- **4 rallies** for testing (final evaluation)

All results report mean ± standard deviation across folds. Class distribution
is imbalanced (intercept and passive are more common than move_to_net). We
address this through balanced episodic sampling during meta-training: each
episode samples equal numbers of support and query examples per class
regardless of overall class frequency.

**Phase B baseline (no SSL pre-training, completed).** Setup: L2 features
(9-dim), ST-GCN encoder (3.1M params) randomly initialized, episodic
meta-training (50 epochs × 100 episodes, 5-way 10-shot), fine-tune encoder
during Phase B. Data: 296 labeled shots across 40 GDINO-guided rallies. Run
on Colab T4 GPU.

**Overall result: Macro-F1 = 37.0% ± 3.3%**

| Fold | Train loss (ep.50) | Train acc | Best Val-F1 | Test Macro-F1 |
|---|---|---|---|---|
| 1 | 0.049 | 0.983 | 0.519 | 0.409 |
| 2 | 0.041 | 0.986 | 0.337 | 0.375 |
| 3 | 0.034 | 0.990 | 0.326 | 0.333 |
| 4 | 0.040 | 0.988 | 0.486 | 0.331 |
| 5 | 0.022 | 0.991 | 0.396 | 0.399 |

**Per-class F1 (mean ± std across 5 folds):**

| Strategy | Samples | Mean F1 | Std | Observation |
|---|---|---|---|---|
| create_depth | 61 | 0.581 | 0.087 | Best-performing class despite moderate count |
| intercept | 108 | 0.522 | 0.074 | Most common class, consistent across folds |
| defensive | 59 | 0.417 | 0.085 | Moderate performance |
| passive | 50 | 0.290 | 0.130 | High variance; confused with intercept/defensive |
| **move_to_net** | **18** | **0.036** | **0.073** | **Near-zero in 4/5 folds — insufficient samples** |

**Key observations:**

1. Result is within the expected 35–45% range for random-init condition,
   confirming the experimental design is sound.
2. **Severe episodic overfitting.** Training accuracy reaches 98–99% while
   validation F1 peaks at epoch 10 and degrades thereafter. SSL pre-training
   is expected to be the primary fix.
3. **`move_to_net` is effectively unlearnable at this scale** (18 samples,
   6.1% of dataset). ProtoNet prototype is noise-dominated.
4. **`create_depth` outperforms `intercept`** despite fewer samples (61 vs 108),
   suggesting create_depth has more distinctive skeleton signatures (rear-court
   positioning, extended arm overhead).
5. **ProtoNet > Linear probe > k-NN** (relative, from single-fold comparison).
   The centroid assumption holds — the embedding space (even random-init) creates
   roughly spherical per-class clusters.

### 6.2 SSL Pre-Training (Phase A)

We apply a SimCLR-style contrastive objective to skeleton sequences extracted
from ShuttleSet videos. Data augmentation for skeleton contrastive learning
includes: (a) joint-level jittering (Gaussian noise on coordinates),
(b) temporal crop and resample, (c) spatial rotation (random court-relative
rotation), and (d) joint masking (randomly zeroing 10–20% of joints). The
NT-Xent loss trains the ST-GCN encoder to produce similar embeddings for
augmented views of the same shot and dissimilar embeddings for different shots.

An auxiliary task predicts shot type using ShuttleSet's 19-class labels from
the CSV tracking data. This provides a domain-specific supervisory signal
without requiring tactical labels. The auxiliary loss is weighted at 0.3
relative to the contrastive loss.

---

7\. Architecture & Pipeline
----------------------------

### 7.1 End-to-End Pipeline

The system comprises three phases executed sequentially:

- **Phase A (SSL pre-training):** Unlabeled ShuttleSet data → encoder learns
  general badminton motion features via contrastive learning.
- **Phase B (few-shot adaptation):** Small FineBadminton labeled set → ProtoNet
  maps learned features to 5 tactical strategy classes.
- **Phase C (inference):** New unseen video → automatic court detection + shot
  boundary detection + full feature-to-prediction cascade.

*Table 3: End-to-end pipeline stages*

| Stage | Component | Input | Output |
|---|---|---|---|
| A1 | YOLOv8-Pose (GDINO-guided) | Video frames | 2D skeletons (17 joints × 2 players), Y-sorted, exponentially smoothed |
| A2 | Graph Builder + Feature Engineering | Skeleton sequences | Spatio-temporal graph G=(V,E,X) with enriched node features (L0–L3) |
| A3 | Shot Segmentation | Skeleton graphs + timestamps | T=16 frame windows per shot, hitter-first ordered |
| A4 | ST-GCN Encoder | Skeleton graph | 256-dim motion embedding |
| A5 | SimCLR Head | Augmented skeleton pairs | NT-Xent contrastive loss + auxiliary shot-type loss |
| B1 | Prototype Computation | FB labeled support embeddings | 5 class prototypes (mean vectors, one per strategy) |
| B2 | ProtoNet Classifier | Query embedding + prototypes | Strategy label + confidence score |
| C1 | Frame Extraction (ffmpeg) | New video (.mp4) | Frames at 30fps |
| C2 | Court Detector (Hough + RANSAC) | Reference frame | Homography H (pixel → court coords) |
| C3 | TrackNetV3 (pre-trained, frozen) | Consecutive frames | Shuttle (x,y) per frame + hit event timestamps |
| C4 | YOLOv8-Pose | All frames | Skeleton keypoints (17 joints × 2 players) |
| C5 | Shot Segmentation | Hit timestamps + skeletons | T=16 frame windows per shot |
| C6 | Feature Engineering (L0–L3) | Skeleton windows + H | Enriched node features in court-relative coords |
| C7 | Encoder (Phase A weights) | Enriched skeleton graphs | 256-dim embedding per shot |
| C8 | ProtoNet (Phase B prototypes) | Embeddings + prototypes | Strategy label + confidence + shot type |

### 7.2 Graph Construction

The dual-player spatio-temporal graph uses a unified graph with **34 nodes**
(17 joints per player). Intra-player edges follow the standard COCO skeleton
topology (anatomical connections). Inter-player edges connect corresponding
joints between players (e.g., right wrist to right wrist) to capture relative
positioning. The adjacency matrix A combines three sub-matrices: A_intra1
(player 1 skeleton), A_intra2 (player 2 skeleton), and A_inter (cross-player
relations). Temporal edges connect each joint to itself in adjacent frames
(default window w=1, ablation with w=3).

### 7.3 Node Feature Engineering

Raw skeleton coordinates alone are an impoverished input. The strategy
taxonomy requires the model to perceive velocity, court context, body
configuration, and inter-player spatial relationships. We compute enriched
node features in cumulative layers:

*Table 4: Node feature layers (cumulative)*

| Layer | Features per node | Dim | Signal encoded |
|---|---|---|---|
| L0: Raw coordinates | [x, y] court-relative via homography | 2 | Position only |
| L1: + Kinematics | [x, y, vx, vy, ax, ay] finite differences | 6 | Velocity encodes "lunging forward" (intercept) vs. "stationary" (passive) |
| L2: + Court context | [..., dist_to_net, dist_to_center, dist_to_opponent] | 9 | Court position encodes "forward court" (intercept) vs. "rear court" (defensive) |
| L3: + Joint angles | [..., elbow_angle, knee_angle, shoulder_angle] | 12 | Body pose encodes "arm extended" (defensive) vs. "arm forward flat" (intercept) |
| L4: + Racket (stretch) | [..., racket_x, racket_y, racket_angle] as 18th node | 14–15 | Shot direction — requires RacketVision (stretch goal) |

Layers L0–L3 are computable from existing skeleton coordinates with simple
numpy operations (finite differences, Euclidean distances, arctan2). They
require no additional data or models. Layer L4 requires running RacketVision
on frames and is treated as a stretch goal.

### 7.4 ST-GCN Encoder

The ST-GCN encoder applies graph convolutions alternating between spatial
graph operations (over the 34-node skeleton topology) and temporal convolutions
(over the T=16 frame sequence). 9 ST-GCN blocks produce a final 256-dim
embedding per shot. The architecture is configurable via `DataConfig` and
`STGCNConfig` in `src/config.py`. Input channel dimension changes from 2 to 6
to 9 to 12 across feature layers L0–L3 with zero architectural modification.

An alternative Transformer encoder (BST-style) uses self-attention over joint
tokens across spatial and temporal dimensions without a fixed graph topology,
enabling a direct architectural comparison (RQ1, see §8).

### 7.5 Prototypical Network Classifier

Phase B is near-parameter-free. The ProtoNet operates in two steps:

1. **Prototype computation:** For each of the 5 strategy classes, compute the
   mean embedding (centroid) of all support examples in that class. With 32
   support rallies per fold, each prototype averages approximately 6–8 shot
   embeddings. No gradient descent, no loss function.
2. **Distance-based classification:** Compute Euclidean distance from a query
   shot's embedding to each of the 5 prototypes. Assign to the nearest.
   Confidence is derived from softmax over negative distances; margin score
   (nearest vs. second-nearest distance) flags low-confidence predictions.

---

8\. Results & Key Findings
--------------------------

### 8.1 Main Results Table

*Table 5: Main results — encoder × pre-training (values filled as experiments complete)*

| Encoder | Pre-training | Classifier | Macro-F1 | Notes |
|---|---|---|---|---|
| Random baseline | — | — | ~20.0% | Theoretical floor (uniform 5-class) |
| Majority class | — | — | \[TBD\] | Always predict "passive" |
| **ST-GCN + ProtoNet** | **None (random init)** | **5-way 10-shot** | **37.0% ± 3.3%** | **Completed — L2 features, GDINO skeletons** |
| ST-GCN + Linear probe | SimCLR + Aux | — | \[TBD\] est. 50–60% | Awaiting SSL pre-training |
| **ST-GCN + ProtoNet** | **SimCLR + Aux** | **5-way 10-shot** | **\[TBD\] est. 65–75%** | **Primary target** |
| Transformer + ProtoNet | SimCLR + Aux | 5-way 10-shot | \[TBD\] | Architecture ablation |
| LSTM + ProtoNet | None (random init) | 5-way 10-shot | \[TBD\] | Sanity check |
| 1D-CNN + ProtoNet | None (random init) | 5-way 10-shot | \[TBD\] | Sanity check |

### 8.2 Per-Strategy F1 by Feature Layer (Step 1 Ablation)

*Table 6: Per-strategy F1 across input feature layers*

| Strategy | L0 (x,y) | L1 (+vel,accel) | L2 (+court ctx) — random init | L2 — SSL+Aux | L3 (+angles) |
|---|---|---|---|---|---|
| create_depth | \[TBD\] | \[TBD\] | **0.581** | \[TBD\] | \[TBD\] |
| intercept | \[TBD\] | \[TBD\] | **0.522** | \[TBD\] | \[TBD\] |
| move_to_net | \[TBD\] | \[TBD\] | **0.036** | \[TBD\] | \[TBD\] |
| passive | \[TBD\] | \[TBD\] | **0.290** | \[TBD\] | \[TBD\] |
| defensive | \[TBD\] | \[TBD\] | **0.417** | \[TBD\] | \[TBD\] |
| **Macro-F1** | | | **0.369 ± 0.033** | \[TBD\] | \[TBD\] |

### 8.3 Spatial vs. Temporal Contribution (RQ1)

*Table 7: Per-strategy F1 by feature dimension (uses best encoder, SSL+Aux pre-training)*

| Strategy | Spatial-Only | Temporal-Only | Full (S+T) | Dominant Dim. |
|---|---|---|---|---|
| create_depth | \[TBD\] | \[TBD\] | \[TBD\] | \[TBD\] |
| intercept | \[TBD\] | \[TBD\] | \[TBD\] | \[TBD\] |
| move_to_net | \[TBD\] | \[TBD\] | \[TBD\] | \[TBD\] |
| passive | \[TBD\] | \[TBD\] | \[TBD\] | \[TBD\] |
| defensive | \[TBD\] | \[TBD\] | \[TBD\] | \[TBD\] |

### 8.4 Ablation Plan Summary

Five ablation steps are run sequentially, each varying one axis while holding
all others fixed at the best value found so far:

```
STEP 1 — INPUT REPRESENTATION (fix encoder = ST-GCN, SSL+Aux)
├── L0: Raw [x,y] only
├── L1: + velocity, acceleration
├── L2: + court-relative distances        ← random-init baseline complete
├── L3: + joint angles
├── Spatial-only vs. temporal-only (RQ1)
├── Single-player vs. dual-player
└── + Racket 18th joint, L4 (stretch)

STEP 2 — ENCODER ARCHITECTURE (fix input = best from Step 1, SSL+Aux)
├── ST-GCN          ← graph convolutions, structural priors
├── Transformer     ← self-attention, BST-style
├── LSTM            ← sequential baseline
└── 1D-CNN          ← convolutional baseline

STEP 3 — PRE-TRAINING (fix encoder + input = best from Steps 1–2)
├── Random initialization
├── SimCLR contrastive only
└── SimCLR + auxiliary shot-type (default)

STEP 4 — FEW-SHOT METHOD (fix encoder + input + pre-training = best)
├── ProtoNet (centroid distance)
├── k-NN, k=3 and k=5
└── Linear probe (logistic regression)

STEP 5 — K-SHOT SENSITIVITY (final best configuration)
└── K = 1, 3, 5, 8, 10, 15
```

### 8.5 Planned Visualizations

- **t-SNE / UMAP embedding plots:** colored by strategy class, comparing
  random-init vs. SSL-pretrained embeddings.
- **Confusion matrix:** 5×5 matrix showing per-class error patterns. Expect
  defensive/passive confusion (both involve reactive body postures).
- **K-shot learning curve:** F1 as a function of K (1–15), showing diminishing
  returns and minimum viable supervision.
- **Graph attention maps:** which joints and temporal frames receive highest
  attention per strategy class.

---

9\. Implementation
------------------

### 9.1 Codebase Structure

*Table 8: Module responsibilities*

| File | Responsibility | Pipeline Stage |
|---|---|---|
| `src/config.py` | Hyperparameters, file paths, experiment settings | Global configuration |
| `src/data/pose_extractor.py` | YOLOv8s-Pose skeleton extraction (GDINO-guided); top-2 by keypoint confidence, Y-sorted; exponential smoothing (α=0.7) | A1 / C4 |
| `src/data/graph_builder.py` | Dual-player spatio-temporal graph (34 nodes, 3 adjacency types) | A2 |
| `src/data/dataset.py` | Data loading, shot windowing, hitter-first ordering, episode sampling, 5-fold CV | Data Pipeline |
| `src/models/stgcn_model.py` | ST-GCN backbone (9 blocks, configurable input dim) | A4 / C7 |
| `src/models/transformer_encoder.py` | BST-style Transformer encoder (architecture ablation) | A4 (ablation) |
| `src/models/simclr_loss.py` | NT-Xent contrastive loss, projection head, augmentation pipeline | A5 |
| `src/models/proto_net.py` | ProtoNet + k-NN variant, confidence/margin scoring | B1–B2 / C8 |
| `src/inference.py` | Phase C end-to-end inference on new video | C1–C8 |
| `notebooks/01_EDA_finebadminton.ipynb` | FineBadminton exploratory analysis | EDA |
| `notebooks/01_EDA_shuttleset.ipynb` | ShuttleSet exploratory analysis | EDA |
| `notebooks/02_skeleton_extraction_finebadminton.ipynb` | Batch GDINO+YOLO extraction for 40 FB rallies → `finebadminton_skeletons/` ✅ | A1 |
| `notebooks/02_skeleton_extraction_shuttleset.ipynb` | Batch extraction for SS shots → `ShuttleSet/skeletons/` 🔄 | A1 |
| `notebooks/02_shuttlecock_tracking_finebadminton.ipynb` | TrackNetV4 shuttle trajectory extraction → `finebadminton_shuttles/` ✅ | Preprocessing |
| `notebooks/03_ssl_pretraining.ipynb` | SimCLR pre-training on ShuttleSet → `models/ssl_pretrained.pt` | A4–A5 |
| `notebooks/04_fewshot_training.ipynb` | ProtoNet 5-fold CV → `models/fewshot_final.pt` | B1–B2 |
| `notebooks/05_ablations.ipynb` | All ablation configurations | Experiments |
| `notebooks/06_analysis_and_plots.ipynb` | t-SNE, confusion matrix, K-shot curves | Analysis |
| `badminton_server.py` | HTTP demo server (port 7860) serving FB frames, skeleton overlays, shuttle trajectories | Demo Backend |
| `badminton_pipeline_demo.html` | Standalone React+Babel demo UI — Match → Rally → Shot hierarchy, real frame canvas with skeleton + shuttle overlay | Demo Frontend |

### 9.2 Technical Stack

- **Framework:** PyTorch 2.x with PyTorch Geometric for graph operations.
- **Pose Estimation:** YOLOv8s-Pose — combined person detection + 17-joint
  COCO keypoint estimation in a single forward pass. Top-2 detections by mean
  keypoint confidence. Temporal smoothing via exponential filter (α=0.7).
- **Player Filtering:** Grounding DINO (grounding-dino-tiny) with text prompt
  `"player"` for umpire-rejection during FineBadminton extraction.
- **Shuttle Tracking (preprocessing):** TrackNetV4 (`tracknet-series-pytorch`,
  local clone) for per-frame shuttle position on FineBadminton. Pre-trained
  weights downloaded from GitHub Releases v1.0.1.
- **Shuttle Tracking (inference):** TrackNetV3 (pre-trained, frozen) for
  real-time shuttle detection and hit event timestamps on unseen video.
- **Court Detection:** OpenCV Canny + probabilistic Hough + RANSAC
  homography estimation against a 13.4m × 6.1m badminton court template.
- **Video Processing:** yt-dlp for ShuttleSet download; ffmpeg for frame
  extraction at 30fps.
- **Compute:** Google Colab Pro (T4/A100 GPU). Estimated ~8 hours for SSL
  pre-training, ~1 hour for few-shot training, ~30–60 seconds per rally
  for Phase C inference.
- **Reproducibility:** All random seeds fixed. `config.py` stores all
  hyperparameters. Experiment tracking via Weights & Biases.

### 9.3 Data Directory Structure

```
Baddiev2/
├── Datasets/                                   # Raw datasets (provided/downloaded)
│   ├── FineBadminton-dataset/dataset/
│   │   ├── image/                              # 10,620 .jpg frames (all 40 rallies)
│   │   └── transformed_combined_rounds_output_en_evals_translated.json
│   └── finebadminton_skeletons/                # (legacy path — not used)
│
├── datasets/                                   # Cloned repos and tools
│   ├── ShuttleSet/set/match.csv                # ShuttleSet CSV annotations
│   └── tracknet-series-pytorch/                # TrackNetV4 implementation
│       └── checkpoints/
│           ├── tracknet-v4_best-model.pth
│           └── tracknet-v2_best-model.pth
│
├── datasets_preprocessing/                     # All extracted/processed data
│   ├── finebadminton_skeletons/                # 40 per-rally .npy (2, T, 34) ✅
│   │   └── 0011_001.npy  … 0030_004.npy       # T range: 74–652 frames, avg 265
│   ├── finebadminton_shuttles/                 # 40 per-rally .npy (T, 3) ✅
│   └── ShuttleSet/
│       ├── frames/                             # Extracted video frames (deletable)
│       └── skeletons/                          # Per-shot .npy (2, 16, 34) 🔄
│
├── models/                                     # Saved model checkpoints
│   ├── ssl_pretrained.pt                       # Phase A output
│   └── fewshot_final.pt                        # Phase B output
│
├── src/                                        # All reusable Python source code
│   ├── config.py
│   ├── inference.py
│   └── data/, models/
│
├── notebooks/                                  # All Jupyter notebooks
│
├── badminton_server.py                         # Demo HTTP server (port 7860)
└── badminton_pipeline_demo.html                # Standalone React+Babel demo UI
```

**Key conventions:**
- `Datasets/` (capital D): raw provided data, never modified
- `datasets/` (lowercase): cloned third-party repos and tools
- `datasets_preprocessing/`: all outputs of extraction notebooks
- Per-rally skeleton `.npy` shape: `(2, T, 34)` — axis 0 = coord channel (X/Y),
  axis 1 = frame, axis 2 = joint (0–16 = player 0, 17–33 = player 1)
- Per-rally shuttle `.npy` shape: `(T, 3)` — columns = [x, y, visible];
  frame-aligned with skeleton array

---

10\. Limitations & Future Work
-------------------------------

### 10.1 Current Limitations

- **Small test set variance:** With 4 test rallies per fold (~50–65 shots),
  a single misclassified rally can shift macro-F1 by 5–10 percentage points.
  Mitigated by 5-fold CV and confidence intervals, but results should be
  interpreted accordingly.

- **Class imbalance:** `move_to_net` has only 18 training samples (6.1%),
  making its prototype noise-dominated and near-unlearnable at this scale.

- **2D skeleton limitations:** We use 2D pose estimation, losing depth
  information that could help distinguish strategies (e.g., shuttle height for
  create_depth). 3D pose lifting (MotionBERT) is a potential extension.

- **No shuttle trajectory in encoder:** Several strategies (create_depth,
  intercept) are partially defined by shuttle placement. TrackNetV4 trajectories
  are extracted and visualized in the demo but not yet incorporated as encoder
  input during Phases A–B.

- **Pose estimation errors:** Occlusion, fast motion blur, and overlapping
  players cause pose estimation failures. The GDINO filter substantially reduces
  umpire contamination but does not eliminate all errors.

- **Domain gap:** Pre-training and evaluation both use broadcast footage.
  Generalization to amateur phone recordings is an unvalidated claim.

- **Frame rate mismatch:** FineBadminton at ~20fps, ShuttleSet at 30fps.
  Temporal resampling or rate-agnostic segmentation is needed for consistent
  window semantics across datasets.

### 10.2 Future Work

- **Shuttle trajectory as encoder input (highest priority):** Integrate the
  extracted TrackNetV4 trajectories as a 35th graph node or global graph-level
  feature during Phases A–B training. Directly addresses the main signal gap
  for create_depth and intercept.

- **3D pose lifting:** Replace 2D pose estimation with monocular 3D lifting
  (MotionBERT, MotionAGFormer) to recover depth and improve vertical shuttle
  trajectory discrimination.

- **Rally-level temporal modeling:** Extend from shot-level to rally-level
  sequence modeling (Transformer over shot embeddings) to capture multi-shot
  strategic arcs.

- **Active learning:** Use the confidence estimation module to identify the
  most informative unlabeled rallies for expert annotation, iteratively
  expanding the labeled set from 40 rallies with maximum efficiency.

- **Learning-based court detection:** Replace Hough + RANSAC with a trained
  court keypoint detector for robustness on amateur footage with partial court
  visibility or oblique angles.

- **Cross-sport transfer:** Evaluate the pre-trained encoder on tennis, squash,
  and table tennis tactical analysis.

---

11\. References
---------------

\[1\] Wang, W. Y., et al. (2022). ShuttleNet: Position-aware Fusion of
Rally Progress and Player Styles for Stroke Forecasting in Badminton.
*Proceedings of the AAAI Conference on Artificial Intelligence.*

\[2\] FineBadminton Dataset. Fine-grained badminton annotations with strategy
labels, shot subtypes, and quality scores. \[Dataset paper reference TBD upon
access\].

\[3\] Yan, S., Xiong, Y., & Lin, D. (2018). Spatial Temporal Graph
Convolutional Networks for Skeleton-Based Action Recognition. *Proceedings of
the AAAI Conference on Artificial Intelligence.*

\[4\] Snell, J., Swersky, K., & Zemel, R. (2017). Prototypical Networks for
Few-shot Learning. *Advances in Neural Information Processing Systems.*

\[5\] Li, L., et al. (2021). CrosSCLR: Cross-View Contrastive Learning for
3D Skeleton-Based Action Recognition. *Proceedings of CVPR.*

\[6\] Guo, T., et al. (2022). AimCLR: Contrastive Learning of Skeleton-Based
Action Recognition with Extreme Augmentations. *Proceedings of CVPR.*

\[7\] Chen, T., et al. (2020). A Simple Framework for Contrastive Learning of
Visual Representations (SimCLR). *Proceedings of ICML.*

\[8\] Chang, W. (2025). BST: Badminton Stroke-type Transformer for
Skeleton-Based Stroke Classification. *CVPR 2025 Workshop.*

\[9\] Dong, Y., et al. (2025). RacketVision: Racket Detection and Tracking
in Badminton Video. \[Venue TBD\].

\[10\] Sun, Y., et al. (2023). TrackNetV3: Real-Time Shuttlecock Tracking in
Badminton. *Proceedings of ACM Multimedia.*

---

Appendix A: Exploratory Data Analysis
======================================

This appendix reports factual statistics and annotation structure for both
datasets, as produced by `notebooks/01_EDA_finebadminton.ipynb` and
`notebooks/01_EDA_shuttleset.ipynb`.

### A.1 FineBadminton Dataset

**Source and acquisition.** Frames are pre-extracted JPEGs organized by rally
ID (`{rally_id}_{abs_frame}.jpg`). Annotations are in two JSON files; the
pipeline uses the English-translated version
(`transformed_combined_rounds_output_en_evals_translated.json`).

**Per-shot annotation fields:**

| Field | Type | Notes |
|---|---|---|
| `hit_frame` | int | Exact contact frame, no missing values |
| `start_frame` / `end_frame` | int | Shot window boundaries |
| `hitter` | str | `"top"` or `"bottom"` (1 missing across all shots) |
| `player` | str | Player name |
| `hit_type` | str | Stroke type (12 classes) |
| `subtype` | list[str] | Sub-classification |
| `quality` | int | Annotator quality score 1–7 |
| `ball_area` | str | 9 court zones (left/mid/right × front/mid/back) |
| `player_actions` | list[str] | forehand / backhand / turnaround |
| `shot_characteristics` | list[str] | cross-court / straight / over head / body hit |
| `strategies` | list[str] | Tactical strategy labels (355/414 shots) |

**Scale:**

| Statistic | Value |
|---|---|
| Total rallies | 40 |
| Total hitting events | 414 |
| Shots with strategy labels | 355 (85.7%) |
| Shots used in training | 296 |
| Total frames | 10,620 JPEG images |
| Rally frame length | min 73 · max 651 · mean 264 |
| Shots per rally | min 3 · max 27 · mean 10.3 |
| FPS range | ~24.6–25.0 fps |
| Resolution mix | 1280×720, 1912×1080, 1920×1080 |

**Shot type distribution (hit_type, N=414):**

| Shot type | Count | % |
|---|---|---|
| push shot | 84 | 20.3% |
| kill | 70 | 16.9% |
| net shot | 44 | 10.6% |
| block | 43 | 10.4% |
| serve | 40 | 9.7% |
| drive | 34 | 8.2% |
| clear | 31 | 7.5% |
| drop shot | 23 | 5.6% |
| cross-court net shot | 20 | 4.8% |
| net lift | 16 | 3.9% |
| net kill | 8 | 1.9% |
| (unlabelled) | 1 | 0.2% |

**Quality score distribution (scale 1–7):**

| Score | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| Count | 11 | 27 | 70 | 136 | 108 | 32 | 30 |
| % | 2.7 | 6.5 | 16.9 | 32.9 | 26.1 | 7.7 | 7.2 |

**Ball area distribution (court zone at contact, N=413):**

```
mid front court    85  (20.6%)
left front court   52  (12.6%)
mid court          47  (11.4%)
right front court  46  (11.1%)
right mid court    41   (9.9%)
right back court   38   (9.2%)
left mid court     37   (9.0%)
left back court    35   (8.5%)
mid back court     32   (7.7%)
```

Front-court shots dominate (44.3%), consistent with the high frequency of net
shots, push shots, and blocks in elite men's singles.

**Shot characteristics and player actions (multi-label):**

| Characteristic | Count | Action | Count |
|---|---|---|---|
| cross-court | 241 | forehand | 254 |
| straight | 169 | backhand | 158 |
| over head | 96 | turnaround | 8 |
| body hit | 16 | | |

**Hitter balance:** top-court hitter 206 shots (49.8%), bottom-court hitter
207 shots (50.0%), 1 missing. Near-perfect balance eliminates systematic bias.

---

### A.2 ShuttleSet Dataset

**Top-level structure (match.csv fields):** `id`, `video` (match name),
`tournament`, `round`, `year`/`month`/`day`, `set`, `duration` (minutes),
`winner`/`loser`, `downcourt`, `url`.

**Scale:**

| Statistic | Value |
|---|---|
| Total matches in match.csv | 44 |
| Successfully downloaded and processed | 25 (56.8%) |
| Failed / broken YouTube links | 18 |
| Missing URL | 1 |
| Total stroke records | 21,191 |
| Total rallies | 963 |
| Unique shot types | 19 |
| Unique players | 17 |
| Tournaments covered | 13 |
| Avg strokes per match | 848 (min 312 · max 1,644) |
| Avg rally length | 22.0 strokes (min 1 · max 83) |
| Class imbalance (most/least common) | 98× |

**Shot type distribution (N=21,191):**

| Shot type (English) | Chinese | Count | % |
|---|---|---|---|
| Drop (Net) | 放小球 | 3,823 | 18.0% |
| Lift | 挑球 | 3,159 | 14.9% |
| Block | 擋小球 | 2,145 | 10.1% |
| Push | 推球 | 1,686 | 8.0% |
| Clear | 長球 | 1,609 | 7.6% |
| Smash | 殺球 | 1,405 | 6.6% |
| Short Serve | 發短球 | 1,328 | 6.3% |
| Slice/Cut | 切球 | 1,208 | 5.7% |
| Tap Smash | 點扣 | 1,013 | 4.8% |
| Cross-Net | 勾球 | 842 | 4.0% |
| (過度切球) | 過度切球 | 787 | 3.7% |
| (Unknown) | 未知球種 | 615 | 2.9% |
| Drive | 平球 | 423 | 2.0% |
| (撲球) | 撲球 | 280 | 1.3% |
| (後場抽平球) | 後場抽平球 | 253 | 1.2% |
| (防守回抽) | 防守回抽 | 238 | 1.1% |
| Defensive Lift | 防守回挑 | 174 | 0.8% |
| Long Serve | 發長球 | 164 | 0.8% |
| (小平球) | 小平球 | 39 | 0.2% |

The 98× imbalance is handled via weighted sampling in the auxiliary shot-type
task; it has no effect on the primary contrastive objective.

---

### A.3 Dataset Comparison

| Property | FineBadminton | ShuttleSet |
|---|---|---|
| **Primary role** | Few-shot classification (labeled) | SSL pre-training (unlabeled) |
| **Strategy labels** | Yes — 5-class per shot | No |
| **Shot type labels** | 12 classes | 19 classes (Chinese) |
| **Frames provided?** | Yes (JPEG, ~20fps) | No — download + extract |
| **Scale (shots)** | 414 annotated / 296 in training | 21,191 shots, 963 rallies |
| **Video variety** | 11 matches, 1 camera angle | 25 matches, 13 tournaments, 17 players |
| **Skeleton storage** | Per-rally `.npy` `(2, T_full, 34)` | Per-shot `.npy` `(2, 16, 34)` |
| **Hitter ordering** | At dataset load time | At extraction time |
| **Class imbalance** | Moderate | Severe (98×) |

---

Appendix B: Design Reasoning Archive
======================================

This appendix documents key design decisions, trade-offs, and reasoning that
shaped the project's experimental design.

B.1 Why Skeleton-Based (Paradigm 1) Over Vision-Based (Paradigm 2)
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

-   **Strategy signal alignment:** The strategy taxonomy (Table 1) is
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

B.2 Where BST Fits (and Where It Does Not)
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

B.3 How the Prototypical Network Works (and Why It Is Not Really Training)
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

B.4 Why Input Representation (Step 1) Comes Before Encoder Architecture (Step 2)
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

B.5 Why Enriched Node Features Matter More Than You Think
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
signals the strategy taxonomy says matter.

B.6 Strategy Signal Mapping to Pipeline Components
----------------------------------------------------

| Signal Type | What the Model Receives | Pipeline Component |
|---|---|---|
| Spatial signals (court position) | (x,y) court-relative coords + dist_to_net, dist_to_center (L2) | Node features (L0) + court context (L2) |
| Temporal signals (timing, sequence) | How features change across T=16 frames; velocity/accel (L1) | Temporal edges in graph + kinematics (L1) |
| Movement patterns (lunge, stretch) | Skeleton shape changing over time: joint angles (L3) + velocity (L1) | Learned by encoder from L1 + L3 features |
| Player state / context | Relative positioning between two skeletons: dist_to_opponent (L2) | Inter-player edges + court context (L2) |
| Shot/intent signals (trajectory) | NOT directly available. Partially inferable from arm kinematics + racket angle | Known limitation. Racket (L4) helps. Shuttle trajectory = future work. |

The last row is the critical gap: shot trajectory signals ("deep
controlled trajectory" for create depth, "flat aggressive trajectory"
for intercept) are not directly available in a skeleton-only pipeline.

B.7 Full Strategy Taxonomy Including Excluded Categories
---------------------------------------------------------

| Strategy | Key Signals | In Scope? |
|---|---|---|
| Create depth | Rear-court landing; opponent in mid/front; controlled execution | Yes |
| Intercept | Forward court; early timing; quick forward lunge; flat/aggressive | Yes |
| Move to net | Progressive forward positioning; sequential across multiple shots | Yes |
| Passive | No spatial gain; standard/late timing; minimal repositioning | Yes |
| Defensive | Rear/stretched position; reactive timing; recovery movement | Yes |
| Deception | Spatial expectation vs. actual mismatch; timing disguise | No — requires biomechanical data |
| Hesitation | Normal setup; delayed contact; micro-stall before execution | No — requires sub-frame timing |
| Seamlessly | Smooth transitions; fluid movement; consistent shot flow | No — quality modifier |

B.8 Ablation Step Summary Diagram
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

B.9 The Full Pipeline as One Picture
-------------------------------------

```
ShuttleSet (21,191 unlabeled shots)
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

B.10 Inference on New Video — The Shot Detection Gap
-----------------------------------------------------

During training (Phases A–B), shot boundaries are handed to us:
FineBadminton provides annotated shot timestamps, ShuttleSet provides
them in CSV tracking data. When someone uploads a new video (Phase C),
none of that exists. The video is just a continuous stream of frames.

**Options considered for shot detection:**

1. **TrackNetV3 (chosen):** Pre-trained shuttle tracker for badminton.
   Detects the shuttlecock in each frame, tracks its trajectory, and
   identifies hit events from sharp trajectory direction changes. Open
   source with pre-trained weights — zero training needed. Bonus: shuttle
   trajectory (x, y per frame) comes free.

2. **Pose-based wrist velocity peaks:** YOLOv8-Pose already runs on every
   frame, so a shot strike produces a distinctive wrist acceleration spike
   detectable from L1 kinematics. Used as a fallback/validation signal.

3. **Audio-based detection:** Rejected as primary method — fragile to crowd
   noise and requires video with audio.

4. **Combined approach (recommended):** TrackNetV3 as primary, wrist
   velocity as secondary validation. When both agree, confidence is high.
   When they disagree, the shot boundary is flagged as uncertain.

**Phase C inference timing estimate** (single rally, ~20 seconds, T4 GPU):

| Step | Time |
|---|---|
| Frame extraction (ffmpeg) | ~2 s |
| Court detection (Hough + RANSAC, once per video) | ~1 s |
| TrackNetV3 (GPU-bound) | ~10 s |
| YOLOv8-Pose (GPU-bound, runs in parallel with TrackNet) | ~15 s |
| Feature engineering + encoding + classification | ~3 s |
| **Total** | **~30 s** |

B.11 Hitter-First Skeleton Ordering — The Core Data Convention
--------------------------------------------------------------

**The problem:** The model must always know which player is the hitter so
it learns strategy signals from the correct skeleton.

**Initial bug — X-sort instead of Y-sort:** The original implementation
sorted players by X position (left vs. right on screen). This is incorrect
because broadcasts can flip orientation, and FineBadminton annotation uses
"top"/"bottom" (court halves), not "left"/"right".

**The fix — Y-sort by court position:**

```
player 0 (nodes 0–16)  = top court   (smaller Y = top of image)
player 1 (nodes 17–33) = bottom court (larger Y = bottom of image)
```

This maps cleanly to the FineBadminton annotation field:
`"hitter": "top"` → hitter is already at nodes 0–16, no swap needed.
`"hitter": "bottom"` → swap both halves so hitter moves to nodes 0–16.

**FineBadminton implementation (`dataset.py`):** After extracting the
T=16 frame window centred on the hit frame, `_reorder_hitter_first()` is
called. If `"bottom"`, nodes 0–16 and 17–33 are swapped in-place.

**ShuttleSet implementation (notebook 02):** `player_location_y` (pixel Y
of the hitting player) is compared against each player's skeleton centroid
Y. The closer player is swapped to nodes 0–16 if needed. Applied at
extraction time — saved `.npy` files are already hitter-first ordered.

B.12 Extraction Granularity — Why FineBadminton is Per-Rally and ShuttleSet is Per-Shot
----------------------------------------------------------------------------------------

**FineBadminton:** Frames have rally identity encoded in the filename
(`0011_001_00001.jpg`). The natural unit is to load all frames for one
rally, run YOLOv8-Pose, and save one `.npy` covering the full rally
duration. Windowing to T=16 happens at dataset load time.

**ShuttleSet:** Frames are flat sequential integers within a match directory
(no rally/shot identity in filename). The only way to locate a specific shot
is to look up `frame_num` from the JSON record. Since per-shot lookup is
required anyway, extracting one `.npy` per shot (16 frames centred on
`frame_num`) is equally straightforward and avoids loading the entire match
into memory.

| Dataset | Skeleton granularity | Shape | Hitter reordering |
|---|---|---|---|
| FineBadminton | Per rally | `(2, T_full, 34)` | At load time |
| ShuttleSet | Per shot | `(2, 16, 34)` | At extraction time |

Both produce the same model input: `(2, 16, 34)`, hitter at nodes 0–16.

B.13 Shot Segmentation Design — Why a Window Around the Hit, Not Just the Hit Frame
------------------------------------------------------------------------------------

**Answer:** A temporal window centred on the hit frame is strongly preferred.
The implementation uses T=16 frames (±8 at 20fps ≈ 0.8 s).

**Why frames BEFORE the hit are critical:**

Tactical strategy is committed to during preparation, not at contact:

- **Footwork direction:** A player moving to the net before striking reveals
  "move to net" strategy in their approach, not in the swing.
- **Preparation pose:** Shoulder/racket windup angle signals whether the
  decision was deliberate.
- **Opponent positioning:** Where the opponent is *before* contact determines
  which strategy is rational. A baseline gap invites intercept; an opponent at
  the net invites create_depth.

**Why frames AFTER the hit are also included:**

Post-contact frames confirm whether the executed shot matches the strategy
(e.g., a "move to net" stroke should result in the hitter approaching the net).

**Stroke-level samples, not frame-level samples:**

Each sample is one hit event (one stroke), not one frame. The dataset is
structured at the stroke level — hitter roles, labels, and T=16 windows are
all defined per stroke.

**Dynamic role assignment per stroke:**

| Stroke | Hit frame | Hitter | Nodes 0–16 | Nodes 17–33 |
|---|---|---|---|---|
| Stroke 1 | Frame 10 | Player A | Player A skeleton | Player B skeleton |
| Stroke 2 | Frame 24 | Player B | Player B skeleton | Player A skeleton |
| Stroke 3 | Frame 38 | Player A | Player A skeleton | Player B skeleton |

**Frame overlap between samples is intentional and correct:**

Because consecutive strokes produce overlapping windows, the same frame may
appear in two samples. This is standard practice in temporal action segmentation.
The two samples have *different labels* and *different hitter assignments*, so
the overlap does not inflate metrics.

```python
# dataset.py — _extract_shot_window()
half = self.shot_window // 2          # 8 frames
start = max(0, rel_hit - half)        # 8 frames before hit
end   = start + self.shot_window      # 8 frames after hit (T=16 total)
segment = full_skeleton[:, start:end, :]
segment = _reorder_hitter_first(segment, hitter)
```
