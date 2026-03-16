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
representations through supervised contrastive pre-training on shot-type labeled
data, and classifies tactical patterns using few-shot prototypical networks
with only 40 expert-labeled rallies. We target five classifiable strategies
(intercept, defensive, move to net, create depth, passive) and investigate
two core research questions: (1) the relative contribution of spatial versus
temporal features to strategy recognition, and (2) the effectiveness of
self-supervised pre-training in reducing label dependency. Pre-training uses
20 ShuttleSet matches (~17,000 shots) with shot-type supervision via SupCon
(Supervised Contrastive Learning); few-shot classification is evaluated on
the held-out FineBadminton set. The system additionally outputs shot-type
predictions alongside strategy at inference.

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

1. **Skeleton extraction** from raw video using GDINO-guided YOLOv8-Pose,
   with spatial court-region masking to eliminate non-player detections
   (chair umpires, ball kids), producing dual-player joint sequences.
2. **Supervised contrastive pre-training (SupCon)** on 20 ShuttleSet matches
   (15 train / 5 validation) using shot-type labels. Positive pairs are shots
   of the same type; negative pairs are shots of different types. This forces
   the encoder to learn shot mechanics rather than player identity, and
   provides richer supervisory signal than self-supervised SimCLR alone.
3. **Few-shot strategy classification** using prototypical networks on the
   small FineBadminton labeled set (30 train rallies / 10 test rallies).
4. **Dual inference output:** strategy prediction (5 classes, confidence +
   margin) and shot-type prediction (17 classes, logistic head trained on
   frozen encoder) — enabling expert cross-validation of high-confidence
   strategy predictions.

The approach is skeleton-based rather than pixel-based, providing
interpretability and robustness to visual variation (camera angle, lighting,
jersey color). Skeleton inputs also align directly with the strategy taxonomy:
all five target strategies are defined in terms of joint positions, velocities,
and inter-player geometry — precisely what the skeleton graph captures.

### 3.1 Architecture Overview

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PHASE A — SSL PRE-TRAINING  (ShuttleSet, ~17,000 shots, no strategy labels)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Rally video frames
        │
        ▼
  YOLOv8 / GDINO ──► skeleton.npy  (2, T, 34)   x,y of 34 joints
        │
        ▼
  Feature Engineering ──► L2: position + velocity + court distances  (9, T, 34)
        │
        ├──── augment view 1 ──► ST-GCN encoder ──► projection head ──┐
        │                                                               ├──► SimCLR / SupCon loss
        └──── augment view 2 ──► ST-GCN encoder ──► projection head ──┘
                                      │
                                      ▼
                              ssl_pretrained_L2.pt   ← saved weights


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PHASE B — FEW-SHOT TRAINING  (FineBadminton, 40 labelled rallies)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  skeleton.npy  (2, T, 34)
        │
        ▼
  Feature Engineering (L2)  →  (9, T, 34)
        │
        ▼
  ST-GCN encoder  ◄── load ssl_pretrained_L2.pt
        │
        ▼
  embedding  (256,)
        │
        ▼
  Prototypical Network
   ┌─────────────────────────────────────────────────────┐
   │  support set (k labelled shots per strategy)        │
   │  → compute class prototype (mean embedding)         │
   │                                                     │
   │  query shot → nearest prototype → predicted strategy│
   └─────────────────────────────────────────────────────┘
        │
        ▼
  fewshot_L2.pt   ← saved weights + prototypes


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 INFERENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  New video
        │
        ▼
  skeleton → Feature Eng (L2) → ST-GCN → embedding (256,)
                                               │
                              ┌────────────────┴───────────────┐
                              ▼                                 ▼
                     ProtoNet                          shot_type_clf
                  strategy + confidence              shot type + confidence
             (intercept / defensive /              (smash / drop / clear / ...)
              net / depth / passive)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ABLATION  — answers RQ1 and RQ2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  RQ1 — which input features matter?  (Step 1a, random init, ST-GCN)
  ┌──────────────────────────────────────────┐
  │  L0 (x,y only)          → score          │
  │  L1 (+ velocity)        → score          │  ← compare
  │  L2 (+ court distances) → score          │
  │  L3 (+ joint angles)    → score          │
  └──────────────────────────────────────────┘

  RQ2 — does pre-training help?  (Step 3, best layer from RQ1, ST-GCN)
  ┌──────────────────────────────────────────┐
  │  Random init            → score          │
  │  SimCLR (self-supervised) → score        │  ← compare
  │  SupCon (shot-type labels) → score       │
  └──────────────────────────────────────────┘
```

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

ShuttleSet is our large-scale dataset used for supervised contrastive
pre-training (Phase A). Unlike FineBadminton, it carries **shot-type labels**
(19 classes) but no tactical strategy labels.

**Frame Extraction.** The dataset provides CSV annotations and YouTube video
links only. We downloaded videos via yt-dlp and extracted frames using a
streaming pipeline (frame stride = 4, every 4th frame extracted + all annotated
hit frames). Of 44 matches in `match.csv`: 38 are usable broadcast-angle
matches. **20 matches were selected for skeleton extraction** due to Colab GPU
time constraints. 4 matches were excluded earlier due to non-standard broadcast
camera angles (close-up or side-on views):
- An Se Young vs Pornpawee Chochuwong (TOYOTA Thailand Open 2021 QF)
- CHEN Long vs CHOU Tien Chen (Denmark Open 2019 QF)
- CHOU Tien Chen vs Jonatan Christie (Sudirman Cup 2019 QF)
- CHOU Tien Chen vs NG Ka Long Angus (Sudirman Cup 2019 Group Stage)

**Scale (20 matches, estimated):**

| Statistic | Value |
|---|---|
| Matches extracted | 20 (of 38 usable) |
| Total stroke records | ~17,000 (est.) |
| Total rallies | ~590 (est.) |
| Unique shot types | 17 (unified vocabulary, see §4.3) |
| Avg strokes per match | ~847 |

*Exact figures updated after extraction completes.*

**Train/Validation split (match-level):**

| Split | Matches | Strokes (est.) | Role |
|---|---|---|---|
| Train | 15 | ~12,700 | SupCon pre-training |
| Validation | 5 | ~4,200 | Shot-type linear probe (proxy metric) |
| **Total** | **20** | **~17,000** | — |

Split at match level to prevent frame leakage, with greedy balancing to
minimise KL divergence between train and validation shot-type distributions.

**Validation role:** The held-out validation set is used to evaluate
shot-type classification accuracy as a proxy for representation quality —
if the encoder separates shot types well, it likely separates strategies too.
In practice (Task 7a), 8 matches with GDINO skeletons were available,
yielding 4,823 train / 3,455 val shots via rally-level splits (see §6.2.6).
The same encoder is then loaded for expert-verified strategy evaluation on
selected SS validation shots (qualitative; see Section 8.3).

**Key annotation fields (from per-match CSV files):**

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
`"badminton player ."` to produce court-region bounding boxes before YOLOv8
keypoint estimation. Only YOLO detections with IoU ≥ 0.25 against a GDINO
bounding box are accepted as valid players. If fewer than 2 valid players pass
this filter, the pipeline falls back to plain YOLOv8 top-2 selection.
Additionally, a **spatial court-region mask** filters all detections (both
GDINO priors and YOLO candidates) to the central 76% of the frame width
(x: 12%–88%) and enforces a minimum bounding box height of 10% of frame
height. This eliminates chair umpires (seated at x ≈ 5–10%), line judges, and
ball kids from the candidate set before IoU matching. The fallback (plain top-2
YOLO when GDINO provides fewer than 2 priors) applies the same spatial filter,
preventing contamination even when GDINO fails. This approach substantially
reduces umpire contamination in the extracted skeletons.

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
as a yellow dot + trail overlay.

Shuttle trajectory is optionally incorporated as **virtual node 34** in the
dual-player graph (35 nodes total). When `use_shuttle=True`, the shuttle's
`(x, y)` position is appended before feature engineering so that the homography
transform applies to it, making all distance/velocity features camera-invariant.
The shuttle node is connected to both players' wrist joints (nodes 9, 10 for P1;
26, 27 for P2) in the ST-GCN graph. This is evaluated as **Step 6** in the
ablation study (skeleton-only 34-node vs. skeleton+shuttle 35-node graph).

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

**Perspective elevation error and mitigation.** A homography is defined on the
court floor plane (z = 0), so it maps floor-level points exactly. However,
skeleton joints are elevated above the floor — the torso by ≈1 m and the head
by ≈2 m. Under a typical broadcast camera (elevated ≈10 m, tilted ≈15–20° down),
elevated joints on the far-side player (top court) appear shifted *toward* the
camera in image space, causing the homography to project their body centroid past
the net into the opponent's half. The near-side player is less affected because
perspective displacement pulls their joints *away* from the net. Three mitigations
were evaluated in notebook 07: (A) ankle-only positioning, using only joints 15
and 16 (left/right ankle) which lie on the floor plane — this is the adopted
approach and is sufficient to eliminate net-crossing artefacts at negligible
cost; (B) hip-midpoint positioning (joints 11–12), which reduces error relative
to the full-body centroid; and (C) depth-corrected positioning using Depth
Anything V2 Small to estimate per-joint depth, shifting each joint's image
coordinate downward to its floor shadow before applying the homography. Approach
A is adopted as the default because ankle detections are reliable and the
floor-plane assumption is exactly satisfied; approach C is available as an
optional refinement when full-body projections are needed for visualisation.

---

6\. Model Training
------------------

### 6.1 Few-Shot Training Setup

Given the extremely small labeled set (40 rallies), the evaluation protocol
uses **rally-level 5-fold cross-validation** over all 40 rallies (~8 test
rallies per fold, ~24 train, ~8 val for checkpoint selection).

**Why rally-level (not shot-level) splitting:** Shots within the same rally
share the same skeleton file and the same two players across consecutive
frames. Shot-level k-fold places shots from the same rally in both train and
test (empirically 25–28 of 40 rallies appear in both splits every fold),
meaning the model has already seen those players' movement patterns at test
time. Rally-level splitting ensures every test shot comes from a rally whose
players were never seen during training, giving an honest generalisation
estimate. All 40 rallies contribute to evaluation via rotation across folds.

**Split design (per fold):**
- **~24 rallies** — episodic ProtoNet training
- **~8 rallies** — validation (checkpoint selection only)
- **~8 rallies** — test (reported metrics)

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

### 6.2 SSL Pre-Training (Phase A) — Contrastive Representation Learning

Pre-training on ShuttleSet's ~7,000 skeleton shots gives the ST-GCN encoder
exposure to diverse badminton motion patterns before it sees any strategy
labels. We train two contrastive variants — **SimCLR** (fully self-supervised)
and **SupCon** (shot-type supervised) — and compare both against random
initialisation in the Step 3 ablation (§8.5).

#### 6.2.1 Shared Architecture & Augmentation Pipeline

Both methods share the same encoder (ST-GCN, 3.08M params), projection head
(MLP: 256 → 256 → 128 with BatchNorm + ReLU), and augmentation pipeline.
Two independently augmented "views" of each skeleton sequence are generated
per training step. The augmentations are designed to preserve the semantic
content of the shot while varying surface-level appearance:

| Augmentation | Parameters | Purpose |
|---|---|---|
| Joint jittering | Gaussian noise σ=0.02 | Simulate pose estimation noise, prevent overfitting to exact coordinates |
| Speed perturbation | ±20% temporal resampling | Invariance to shot speed variation across players and match contexts |
| Spatial rotation | ±10° court-relative | Invariance to minor camera angle differences across matches |
| Joint masking | 15% of joints zeroed | Forces the encoder to reconstruct information from partial observations |

These augmentation parameters were tuned conservatively — earlier experiments
with stronger augmentations (σ=0.06, 40% masking, ±25° rotation) degraded
convergence, likely because aggressive perturbation destroys the court-relative
spatial relationships encoded in L2 features.

**Training data:** 7 ShuttleSet matches with GDINO-extracted skeletons
(~38,200 frames, ~7,050 shots after shot segmentation with T=32 frame windows).
The encoder uses L2 features (9-dim: x, y + velocity + court context).

**Shared hyperparameters:** AdamW optimiser (lr=3e-4, weight_decay=1e-4),
batch size 64, temperature τ=0.2, early stopping with patience=10 epochs.

#### 6.2.2 SimCLR — Self-Supervised (NT-Xent Loss)

SimCLR (Chen et al., 2020) requires no labels. For each skeleton sequence in
the batch, the two augmented views form a single positive pair; all other
2(B-1) samples serve as negatives. The NT-Xent loss is:

```
L_NT-Xent = -log [ exp(sim(z_i, z_j)/τ) / Σ_{k≠i} exp(sim(z_i, z_k)/τ) ]
```

where z_i, z_j are the L2-normalised projections of the two views and
sim(·,·) is cosine similarity.

**Key characteristic:** Each anchor has exactly **1 positive** (its own
augmentation twin). This means the contrastive signal is invariant to semantic
content — the encoder learns motion-general features (e.g. "similar skeleton
poses should embed nearby") without any shot-type or strategy awareness.

**Training results:** SimCLR used all 7 available matches (no split filtering,
since no labels are needed), yielding 7,051 training samples and 110 batches
per epoch. Training converged at epoch 93 (early stopped after patience=10)
with a final NT-Xent loss of 0.640. The loss curve shows smooth convergence
from 0.698 (epoch 5) with no signs of instability.

**Checkpoint:** `models/ssl_pretrained_simclr_L2.pt`

**Linear probe on FB strategy labels (sanity check):** Macro-F1 = 0.107 ± 0.002
(5-fold CV with frozen encoder + logistic regression on 296 FineBadminton
shots). This is below the random baseline of 0.20, indicating that SimCLR's
self-supervised objective alone — without any semantic grouping — does not
produce features that linearly separate strategy classes. This does not
preclude benefit after episodic fine-tuning (evaluated in Step 3 ablation).

#### 6.2.3 SupCon — Supervised Contrastive Learning

SupCon (Khosla et al., 2020) extends SimCLR by using ShuttleSet's shot-type
labels to define positive pairs. Rather than pairing only augmented views of
the *same* sequence, SupCon treats **all shots of the same type** (across
different players and matches) as positives. This forces the encoder to
cluster by shot mechanics (smash trajectory, drop placement, lob arc) rather
than by player identity or match context.

**Training objective:**

For a batch of skeleton sequences {x_i} with shot-type labels y_i:

```
L_SupCon = Σ_i [ -1/|P(i)| · Σ_{p∈P(i)} log [ exp(z_i·z_p/τ) / Σ_{a≠i} exp(z_i·z_a/τ) ] ]
```

where P(i) = {j : y_j = y_i, j ≠ i} is the set of same-type positives in
the batch, z_i is the L2-normalised projection head output, and τ=0.2.

**Key characteristic:** Each anchor has **multiple positives** — all
batch members sharing its shot-type label. In a batch of 64 with ~10 shot
types represented, each anchor averages ~7 positives (7x more gradient signal
per step than SimCLR's single positive). This richer training signal enables
the encoder to learn that structurally similar movements (e.g. all smashes,
regardless of player or match) should cluster together.

**Training data:** 5 train-split matches (~4,900 shots with valid shot-type
labels), using only matches with both GDINO skeletons and matched CSV
annotations (76 batches per epoch).

**Training results:** SupCon converged at epoch 60 (early stopped) with a
final loss of 2.516. The loss curve shows steeper initial descent than SimCLR
(from 4.14 at epoch 5 to 2.52 at epoch 50), consistent with the richer
per-step gradient signal from multiple positives. Note: SupCon loss values are
not directly comparable to NT-Xent values due to the different normalisation
(averaging over |P(i)| positives vs. a single positive).

**Checkpoint:** `models/ssl_pretrained_supcon_L2.pt`

#### 6.2.4 SimCLR vs SupCon — Conceptual Comparison

| Aspect | SimCLR (NT-Xent) | SupCon |
|---|---|---|
| **Labels required** | None | Shot-type labels (from ShuttleSet CSVs) |
| **Positive definition** | Augmentation twin of same sample | All samples with same shot-type label |
| **Positives per anchor** | 1 (constant) | ~7 on average (scales with class frequency) |
| **What the encoder learns** | General motion similarity | Shot-type-aware motion clustering |
| **Training data** | All 7 matches (7,051 shots) | 5 train-split matches (~4,900 shots) |
| **Convergence** | 93 epochs, loss 0.640 | 60 epochs, loss 2.516 |
| **Risk** | May learn irrelevant invariances (player identity) | Depends on shot-type label quality |

**Why SupCon is expected to outperform SimCLR:** Shot types partially correlate
with tactical strategies — a smash is more likely to occur during an
"intercept" strategy, while a lob is more associated with "defensive" play.
By forcing the encoder to group by shot mechanics, SupCon pre-structures the
embedding space in a way that is more directly useful for downstream strategy
classification. An earlier design used SimCLR + an auxiliary shot-type
classification head, but this required separate loss weighting and created
competing objectives. SupCon integrates shot-type supervision directly into
the contrastive loss, producing semantically structured embeddings in a single
unified objective.

#### 6.2.5 Diagnostic Visualisations

Both notebooks include diagnostic visualisations run before training:

- **Augmentation pipeline (SimCLR):** Multi-frame ghost trail showing each
  augmentation in isolation (jitter, speed, rotation, masking) and the
  combined "View A / View B" pair that the encoder actually sees. Confirms
  that augmentations preserve recognisable player poses while introducing
  sufficient variation.

- **Positive-pair structure (SupCon):** Side-by-side 2B x 2B similarity
  matrices for SimCLR vs SupCon, showing how many cells are "pull together"
  (positive) vs "push apart" (negative). SupCon's matrix is visibly denser
  in the positive region.

- **Training signal richness:** Bar chart comparing positives-per-anchor
  between SimCLR (constant 1) and SupCon (~7 average), quantifying the
  gradient signal advantage.

- **Intra-class skeleton consistency (SupCon):** Grid of skeleton stick
  figures grouped by shot type, testing the core SupCon hypothesis — that
  same-type shots share similar poses. Visual inspection confirms that
  smashes, clears, and drops show recognisable within-class consistency.

These visualisations are saved to `results/ssl_*_viz_*.png`.

#### 6.2.6 SupCon Shot-Type Proxy Evaluation (Task 7a)

To evaluate whether SupCon pre-training produces semantically meaningful
embeddings, we train a logistic regression classifier on frozen SupCon
encoder outputs and evaluate shot-type classification on the ShuttleSet
validation split.

**Experimental setup:**

| Parameter | Value |
|---|---|
| **Encoder** | ST-GCN (frozen, SupCon checkpoint `ssl_pretrained_supcon_L2.pt`) |
| **Feature layer** | L2 (9-dim: x, y, velocity, court context) |
| **Embedding dim** | 256 (backbone output, no projection head) |
| **Classifier** | Logistic regression (scikit-learn, via joblib) |
| **Train split** | 4,823 shots from 8 ShuttleSet matches (whole-match skeletons) |
| **Val split** | 3,455 shots from the same 8 matches (disjoint rallies) |
| **Classes** | 15 of 17 unified shot types present (classes 14 `net_shot` and 15 `smash_defense` absent) |
| **Augmentations** | None (frozen embeddings, no augmentation at eval time) |
| **Shot window** | 16 frames (default `config.shot_window`) |
| **Metric** | Macro-F1 (class-balanced), per-class precision/recall/F1 |

**Results:** SS Val macro-F1 = **0.534**, accuracy = **0.63**.

*Table 3a: Per-class shot-type classification on ShuttleSet validation set (SupCon L2 embeddings)*

| ID | Shot Type | Precision | Recall | F1 | Support | Assessment |
|---|---|---|---|---|---|---|
| 0 | `short_serve` | 0.81 | 0.89 | **0.85** | 240 | Strong — distinctive compact serve posture |
| 5 | `clear` | 0.89 | 0.89 | **0.89** | 314 | Best — full-body overhead arc well captured |
| 7 | `net_drop` | 0.66 | 0.87 | **0.75** | 563 | Good recall, precision diluted by net-area confusion |
| 11 | `lob_lift` | 0.78 | 0.66 | **0.71** | 513 | Good precision, ~1/3 missed (confused with clears) |
| 1 | `long_serve` | 0.61 | 0.76 | **0.67** | 37 | Decent despite tiny support — biomechanically distinct |
| 10 | `block` | 0.64 | 0.53 | **0.58** | 455 | Large class; confused with net_drop/transition |
| 8 | `transition` | 0.55 | 0.61 | **0.58** | 160 | Inherently ambiguous — transitional shots overlap many types |
| 16 | `push` | 0.50 | 0.63 | **0.56** | 276 | Recall OK but many false positives from other mid-court shots |
| 2 | `smash` | 0.69 | 0.44 | **0.54** | 245 | Good precision, low recall — smashes misclassified as drives/clears |
| 9 | `drive` | 0.39 | 0.38 | **0.39** | 152 | Flat shots overlap with blocks, pushes, smashes |
| 3 | `tap_smash` | 0.30 | 0.53 | **0.38** | 151 | Many false positives — confused with smash/push_rush |
| 12 | `defensive_lift` | 0.37 | 0.39 | **0.38** | 18 | Too few samples; confused with lob_lift |
| 6 | `slice_drop` | 0.54 | 0.27 | **0.36** | 182 | Very low recall — classified as net_drop or transition |
| 4 | `push_rush` | 0.29 | 0.16 | **0.20** | 38 | Rare + fast action = indistinguishable from push/tap_smash |
| 13 | `cross_net` | 0.30 | 0.12 | **0.17** | 111 | Worst — subtle wrist motion invisible in skeleton space |

**Confusion clusters identified:**

- **Net area:** `net_drop` ↔ `cross_net` ↔ `slice_drop` ↔ `block` — these
  shots differ primarily in racket angle and wrist orientation, which 2D
  skeleton keypoints cannot resolve.
- **Power shots:** `smash` ↔ `tap_smash` ↔ `push_rush` — similar overhead
  body posture with different contact timing and racket speed.
- **Lifts:** `lob_lift` ↔ `defensive_lift` ↔ `clear` — similar upward
  trajectories with different court positions and intent.

**Interpretation:** The 0.534 macro-F1 confirms that SupCon embeddings
capture meaningful shot-type structure — shots with distinctive full-body
motion signatures (serves, clears, net drops) are well-separated, while
shots differentiated by racket/wrist subtleties remain confounded. This is a
fundamental limitation of skeleton-only features, not a failure of the
contrastive objective. The result serves as the baseline for comparison
against SimCLR embeddings (Task 7b) to quantify the benefit of shot-type
supervision in the contrastive loss.

**Saved artefact:** `models/shot_type_clf_supcon_L2.joblib`

#### 6.2.7 Limitations & Known Issues

- **SSL checkpoints exist only for L2 features** (9-dim). The Step 1 ablation
  (feature engineering) uses random init for L0/L1/L3, meaning the L2
  advantage in Step 1 may partially reflect SSL pre-training rather than
  feature engineering alone — a known limitation of the sequential ablation
  design (see §8.2).

- **Proxy evaluation — resolved:** The SS val-split linear probe previously
  failed due to a split-matching issue (`shuttleset_split.json` match names
  not aligning with GDINO skeleton directory names) and a path casing issue.
  These were resolved in Task 7a (§6.2.6) by using whole-match skeletons
  with rally-level train/val splits, yielding the 0.534 macro-F1 result above.

- **Training data scale:** Only 7–8 of 20+ SS matches had complete GDINO
  skeleton extractions at training time. Expanding to all ShuttleSet matches
  would increase training data ~3x and likely improve both methods.

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
| A5 | SupCon Head | Augmented skeleton pairs + shot-type labels | Supervised contrastive loss (same-type = positive) |
| A6 | Shot-Type Classifier | Frozen encoder embeddings (SS train) | Logistic head saved for inference |
| B1 | Prototype Computation | FB labeled support embeddings | 5 class prototypes (mean vectors, one per strategy) |
| B2 | ProtoNet Classifier | Query embedding + prototypes | Strategy label + confidence score |
| C1 | Frame Extraction (ffmpeg) | New video (.mp4) | Frames at 30fps |
| C2 | Court Detector (Hough + RANSAC) | Reference frame | Homography H (pixel → court coords) |
| C3 | TrackNetV3 (pre-trained, frozen) | Consecutive frames | Shuttle (x,y) per frame + hit event timestamps |
| C4 | YOLOv8-Pose | All frames | Skeleton keypoints (17 joints × 2 players) |
| C5 | Shot Segmentation | Hit timestamps + skeletons | T=16 frame windows per shot |
| C6 | Feature Engineering (L0–L3) | Skeleton windows + H | Enriched node features in court-relative coords |
| C7 | Encoder (Phase A weights) | Enriched skeleton graphs | 256-dim embedding per shot |
| C8 | ProtoNet (Phase B prototypes) | Embeddings + prototypes | Strategy label + confidence score + margin |
| C9 | Shot-Type Head (Phase A weights) | Same embedding | Shot-type label + confidence (17 classes) |

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

An alternative Transformer encoder (BST-style) is implemented for future
encoder comparison but is not part of the current ablation scope (see §10.2).

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
| Random baseline | — | uniform | ~20.0% | Theoretical floor (uniform 5-class) |
| Majority class | — | always "intercept" | ~15.6% | Always predict majority class |
| **ST-GCN + ProtoNet** | **None (random init)** | **5-way 10-shot** | **36.9% ± 3.3%** | **✅ Done — L2, GDINO, episodic fine-tune** |
| ST-GCN + ProtoNet | SupCon (shot type, L2) | 5-way 10-shot (frozen enc) | 55.9% ± 8.9% | ✅ Done (preliminary — frozen enc, not fine-tuned) |
| **ST-GCN + ProtoNet** | **SupCon (shot type, L2)** | **5-way 10-shot (fine-tuned)** | **\[TBD\]** | **Primary target — RQ2** |

### 8.2 Per-Strategy F1 by Feature Layer (Step 1a)

*Table 6: Per-strategy F1 across input feature layers (Step 1a ablation — pending Colab run)*

| Strategy | L0 (x,y) | L1 (+vel,accel) | L2 (+court ctx) — random init | L3 (+angles) |
|---|---|---|---|---|
| create_depth | \[TBD\] | \[TBD\] | **0.581** | \[TBD\] |
| intercept | \[TBD\] | \[TBD\] | **0.522** | \[TBD\] |
| move_to_net | \[TBD\] | \[TBD\] | **0.036** | \[TBD\] |
| passive | \[TBD\] | \[TBD\] | **0.290** | \[TBD\] |
| defensive | \[TBD\] | \[TBD\] | **0.417** | \[TBD\] |
| **Macro-F1** | \[TBD\] | \[TBD\] | **0.369 ± 0.033** | \[TBD\] |

> L2 is the only layer with an SSL checkpoint; L0/L1/L3 comparisons use random init. This means the L2 advantage may partially reflect SSL pre-training rather than feature engineering alone — a known limitation of the sequential ablation design.

### 8.3 Expert-Verified Strategy Evaluation (SS Validation Set)

As a qualitative cross-dataset evaluation, the trained system is applied to
the 5 SS validation matches. For each shot the model outputs both a
**strategy prediction** (5 classes, confidence score) and a **shot-type
prediction** (17 classes). A domain expert reviews shots where strategy
confidence exceeds a threshold (e.g. ≥ 0.75), confirming or rejecting each
prediction. This produces:

- **Precision at threshold:** fraction of high-confidence strategy predictions
  confirmed by expert
- **Shot-type accuracy:** automatic metric (ground truth from SS CSVs), used
  as a sanity check that the encoder is functioning correctly
- **Qualitative examples:** rally visualisations showing skeleton overlaid with
  predicted strategy and shot type per frame, reviewed for coherence

This evaluation is intentionally qualitative — ShuttleSet has no strategy
labels — but provides meaningful evidence that the representation generalises
across datasets. High shot-type accuracy (proxy) combined with expert-confirmed
strategy predictions constitutes a sufficient cross-dataset validity check
for a few-shot system.

### 8.4 Graph Structure Contribution (Step 1b)

*Table 7: Macro-F1 by graph topology — does the opponent skeleton matter?*

| Variant | Nodes | Inter-player edges | Macro-F1 |
|---------|------:|-------------------|----------|
| full_dual | 34 | Yes | \[TBD\] |
| no_inter_edges | 34 | No | \[TBD\] |
| single_player | 17 | — | \[TBD\] |

> RQ1 (spatial vs. temporal contribution) is addressed indirectly by comparing L0 (position only) vs. L1 (+ velocity/acceleration) in Step 1a, rather than by disabling spatial/temporal conv layers, which would require distinct model variants not currently implemented.

### 8.5 Ablation Plan Summary

Six ablation steps are run sequentially, each varying one axis while holding
all others fixed at the best value found so far. Steps 1a and 1b are separated
because feature engineering and graph topology answer distinct questions.

```
STEP 1a — FEATURE ENGINEERING ⭐ (fix encoder=ST-GCN, graph=full dual-player)
├── L0: Raw [x, y] joint coordinates only                (2-dim)
├── L1: + velocity, acceleration                         (6-dim)
├── L2: + dist_to_net, dist_to_center, dist_to_opponent  (9-dim) ← baseline done (37.0%)
└── L3: + elbow, shoulder, knee angles                   (12-dim)
    Note: SSL checkpoint exists only for L2; L0/L1/L3 use random init.
    Note: spatial_only/temporal_only excluded — not implementable without
          a separate model variant; RQ1 is addressed by L0 vs L1 comparison.

STEP 1b — GRAPH STRUCTURE (fix feature_layer = best from Step 1a)
├── full_dual:      34 nodes, inter-player edges = Yes   ← default
├── no_inter_edges: 34 nodes, inter-player edges = No
└── single_player:  17 nodes (hitter only), inter-player = N/A
    Question: does the opponent skeleton contribute to strategy recognition?

STEP 2 — PRE-TRAINING REGIME  ⭐ RQ2 (fix encoder=ST-GCN, input = best from Step 1)
├── random_init:    no pre-training                       ← baseline: 36.9%
├── simclr:         SimCLR (self-supervised, no labels)   ← checkpoint: ssl_pretrained_simclr_L2.pt
└── supcon:         SupCon (shot-type labels as positives) ← checkpoint: ssl_pretrained_supcon_L2.pt
    RQ2: does pre-training help, and does shot-type supervision (SupCon) beat self-supervision (SimCLR)?

STEP 3 — FEW-SHOT CLASSIFIER (fix all above = best)
├── ProtoNet        ← nearest centroid (preliminary: 55.9% ± 8.9%)
├── k-NN (k=3)      ← (preliminary: 53.5% ± 11.7%)
├── k-NN (k=5)      ← (preliminary: 54.5% ± 9.4%)
└── Linear probe    ← logistic regression (preliminary: 51.5% ± 9.5%)

STEP 4 — K-SHOT SENSITIVITY (final best configuration)
└── K = 1, 3, 5, 8, 10   (capped at 10 — move_to_net has ~11 train samples/fold)
```

### 8.6 Planned Visualizations

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
| `src/models/stgcn_model.py` | ST-GCN backbone (9 blocks, configurable input dim); 3.08M params | A4 / C7 |
| `src/models/transformer_encoder.py` | BST-style Transformer encoder (4 layers, 8 heads); future encoder comparison | Future work |
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
  `"badminton player ."` for umpire-rejection during FineBadminton extraction.
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

- **Encoder architecture comparison (ST-GCN vs Transformer):** The Transformer
  encoder is implemented but not evaluated. A direct comparison against ST-GCN
  on the same few-shot task would answer whether graph-structural priors help
  or whether attention-based models can match performance without them. LSTM
  and 1D-CNN baselines would also contextualize the ST-GCN choice.

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

\[11\] Khosla, P., et al. (2020). Supervised Contrastive Learning. *Advances in Neural Information Processing Systems (NeurIPS).*

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
| Excluded (wrong broadcast angle) | 4 |
| Usable broadcast-angle matches | 38 (of 44) |
| Matches with skeleton extracted | 20 |
| Total stroke records (20 matches) | ~17,000 (est.) |
| Total rallies (20 matches) | ~590 (est.) |
| Unique shot types (unified vocab) | 17 |
| Avg strokes per match | ~847 |
| Class imbalance (most/least common) | ~98× |

*Full 38-match statistics (for reference only — subset of 20 used in experiments):*

| Statistic | 38-match total |
|---|---|
| Total stroke records | 32,203 |
| Extracted frames | 216,112 JPEG |
| Unique players | 26 |
| Tournaments covered | 13 |

**Shot type distribution (N=32,203, full 38-match reference):**

| Shot type (English) | Chinese | Count | % |
|---|---|---|---|
| Net Drop | 放小球 | 5,562 | 17.3% |
| Lift | 挑球 | 4,756 | 14.8% |
| Net Block | 擋小球 | 3,267 | 10.1% |
| Clear | 長球 | 2,551 | 7.9% |
| Push | 推球 | 2,519 | 7.8% |
| Smash | 殺球 | 2,237 | 6.9% |
| Slice | 切球 | 1,879 | 5.8% |
| Short Serve | 發短球 | 1,722 | 5.3% |
| Tap Smash | 點扣 | 1,522 | 4.7% |
| Unknown | 未知球種 | 1,273 | 4.0% |
| Cross-Net | 勾球 | 1,209 | 3.8% |
| Trans. Slice | 過度切球 | 1,193 | 3.7% |
| Drive | 平球 | 630 | 2.0% |
| Rush | 撲球 | 453 | 1.4% |
| BG Drive | 後場抽平球 | 394 | 1.2% |
| Long Serve | 發長球 | 354 | 1.1% |
| Def. Drive | 防守回抽 | 354 | 1.1% |
| Def. Lift | 防守回挑 | 271 | 0.8% |
| Half Smash | 小平球 | 57 | 0.2% |

The 98× imbalance is handled via weighted sampling in the auxiliary shot-type
task; it has no effect on the primary contrastive objective.

**Full match list (SS01–SS38, alphabetical; TR=train, VA=val/test):**

| ID | Split | Match |
|---|---|---|
| SS01 | TR | An Se Young vs Ratchanok Intanon — YONEX Thailand Open 2021 QF |
| SS02 | TR | Anders Antonsen vs Jonatan Christie — Indonesia Masters 2020 QF |
| SS03 | VA | Anders Antonsen vs Sameer Verma — TOYOTA Thailand Open 2021 QF |
| SS04 | TR | Anders Antonsen vs Viktor Axelsen — HSBC BWF World Tour Finals 2020 Finals |
| SS05 | TR | Anthony Sinisuka Ginting vs Anders Antonsen — Indonesia Masters 2020 Final |
| SS06 | TR | Anthony Sinisuka Ginting vs Viktor Axelsen — Indonesia Masters 2020 SF |
| SS07 | TR | Anthony Sinisuka Ginting vs Rasmus Gemke — YONEX Thailand Open 2021 QF |
| SS08 | TR | CHEN Long vs CHOU Tien Chen — World Tour Finals Group Stage |
| SS09 | TR | CHOU Tien Chen vs Anders Antonsen — Fuzhou Open 2019 SF |
| SS10 | VA | CHOU Tien Chen vs Jonatan Christie — Indonesia Open 2019 QF |
| SS11 | TR | Carolina Marin vs An Se Young — HSBC BWF World Tour Finals 2020 QF |
| SS12 | TR | Carolina Marin vs An Se Young — TOYOTA Thailand Open 2021 SF |
| SS13 | TR | Carolina Marin vs Neslihan Yigit — TOYOTA Thailand Open 2021 QF |
| SS14 | TR | Carolina Marin vs Pornpawee Chochuwong — HSBC BWF World Tour Finals 2020 SF |
| SS15 | TR | Carolina Marin vs Supanida Katethong — YONEX Thailand Open 2021 QF |
| SS16 | TR | Evgeniya Kosetskaya vs Michelle Li — HSBC BWF World Tour Finals 2020 QF |
| SS17 | TR | Hans-Kristian Solberg Vittinghus vs Anders Antonsen — TOYOTA Thailand Open 2021 SF |
| SS18 | VA | Hans-Kristian Solberg Vittinghus vs Lee Cheuk Yu — TOYOTA Thailand Open 2021 QF |
| SS19 | VA | Kento Momota vs CHOU Tien Chen — Denmark Open 2018 Finals |
| SS20 | TR | Kento Momota vs CHOU Tien Chen — Fuzhou Open 2018 Finals |
| SS21 | VA | Kento Momota vs CHOU Tien Chen — Fuzhou Open 2019 Finals |
| SS22 | TR | Kento Momota vs CHOU Tien Chen — Korea Open 2019 Final |
| SS23 | TR | Kento Momota vs CHOU Tien Chen — Malaysia Open 2018 QF |
| SS24 | TR | Kento Momota vs Viktor Axelsen — Malaysia Masters 2020 Finals |
| SS25 | TR | Mia Blichfeldt vs Busanan Ongbamrungphan — YONEX Thailand Open 2021 QF |
| SS26 | TR | NG Ka Long Angus vs Jonatan Christie — Malaysia Masters 2020 QF |
| SS27 | VA | Ng Ka Long Angus vs Kidambi Srikanth — HSBC BWF World Tour Finals 2020 QF |
| SS28 | TR | Ng Ka Long Angus vs Lee Cheuk Yiu — YONEX Thailand Open 2021 QF |
| SS29 | TR | Pusarla V. Sindhu vs Pornpawee Chochuwong — HSBC BWF World Tour Finals 2020 QF |
| SS30 | TR | Ratchanok Intanon vs Pusarla V. Sindhu — TOYOTA Thailand Open 2021 QF |
| SS31 | TR | Viktor Axelsen vs SHI Yu Qi — All England Open 2020 QF |
| SS32 | TR | Viktor Axelsen vs CHEN Long — Malaysia Masters 2020 QF |
| SS33 | TR | Viktor Axelsen vs NG Ka Long Angus — Malaysia Masters 2020 SF |
| SS34 | TR | Viktor Axelsen vs Anthony Sinisuka Ginting — YONEX Thailand Open 2021 SF |
| SS35 | TR | Viktor Axelsen vs Hans-Kristian Solberg Vittinghus — TOYOTA Thailand Open 2021 Finals |
| SS36 | TR | Viktor Axelsen vs Jonatan Christie — YONEX Thailand Open 2021 QF |
| SS37 | TR | Viktor Axelsen vs Liew Daren — TOYOTA Thailand Open 2021 QF |
| SS38 | TR | Viktor Axelsen vs Ng Ka Long Angus — YONEX Thailand Open 2021 Finals |

*4 excluded matches (non-standard camera angle, moved to `others_dataset_excluded/`): An Se Young vs Pornpawee Chochuwong (TOYOTA Thailand 2021 QF), CHEN Long vs CHOU Tien Chen (Denmark Open 2019 QF), CHOU Tien Chen vs Jonatan Christie (Sudirman Cup 2019 QF), CHOU Tien Chen vs NG Ka Long Angus (Sudirman Cup 2019 Group Stage).*

---

### A.3 Dataset Comparison

| Property | FineBadminton | ShuttleSet |
|---|---|---|
| **Primary role** | Few-shot classification (labeled) | SSL pre-training (unlabeled) |
| **Strategy labels** | Yes — 5-class per shot | No |
| **Shot type labels** | 12 classes | 19 classes (Chinese) |
| **Frames provided?** | Yes (JPEG, ~20fps) | No — download + extract |
| **Scale (shots)** | 414 annotated / 296 in training | ~17,000 shots (20 matches extracted) |
| **Video variety** | 11 matches, 1 camera angle | 20 matches extracted (38 usable, 26 players) |
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

*B.12 removed — content covered in §5.3.*

*B.13 removed — content merged into §5.3.*
