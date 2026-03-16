# Few-Shot Tactical Strategy Recognition in Badminton Using Self-Supervised Skeleton-Based Spatio-Temporal Graph Learning

\[Author Name\] | \[Institution\] | \[Date\]

---

## Abstract

This project develops an end-to-end pipeline for recognising fine-grained tactical strategies in badminton from raw match footage. The system extracts dual-player skeleton sequences from video, learns generalisable spatio-temporal representations through contrastive pre-training on unlabelled match data, and classifies tactical patterns using few-shot prototypical networks with only 40 expert-labelled rallies. We target five classifiable strategies --- intercept, defensive, move to net, create depth, and passive --- and investigate two research questions: (RQ1) the relative contribution of spatial versus temporal input features to strategy recognition, and (RQ2) the effectiveness of self-supervised and supervised contrastive pre-training in reducing label dependency. Pre-training uses 11 ShuttleSet matches (~8,900 shots); few-shot classification is evaluated on the held-out FineBadminton set. The system additionally outputs shot-type predictions alongside strategy at inference, enabling expert cross-validation.

---

## 1. Introduction

Tactical analysis in badminton remains a manually intensive process. Coaches review match footage frame-by-frame to identify strategic patterns such as interceptions, defensive formations, and net approaches. This process is time-prohibitive, subjective, and inaccessible to amateur and semi-professional players who lack dedicated analyst support.

Automated approaches face a fundamental bottleneck: fine-grained tactical annotations require domain expertise and are expensive to produce at scale. In badminton, a hitting player's strategy is reactive and anticipatory --- it is influenced not only by their own intentions but also by the opponent's positioning, movement, and state.

The FineBadminton dataset, the only publicly available resource with expert-level strategy labels, contains just 40 annotated rallies. Fully supervised deep learning is infeasible at this scale.

We target five classifiable strategies that are distinguishable from skeleton and positional data alone: **intercept**, **defensive**, **move to net**, **create depth**, and **passive**. Table 1 defines each strategy and the observable signals that distinguish it.

*Table 1: Target strategy definitions and observable skeleton/position signals.*

| Strategy | Goal | Hitter Signals | Opponent Signals | Typical Shot Types |
|---|---|---|---|---|
| **Intercept** | Cut off shuttle early; offensive play | Early racket contact; forward court position; fast forward step | Mid-court/forward; low lateral movement; upright pose | Flat / aggressive |
| **Defensive** | Protect position; recover control | Reactive contact; rear-court position; stretched body | Forward/stretched; off-balance; weight leaning | High lift / clear / block |
| **Move to Net** | Drive / drop / approach | Sequential forward steps; forward racket motion | Gradually moving backward; mid-court; upright stance | Net shot; offensive positioning |
| **Create Depth** | Deep clear / push opponent back | Rear-court contact; directed deep; controlled step | Mid/front court; moving forward; leaning slightly forward | Push; clear |
| **Passive** | Recovery play; minimal gain | Neutral placement; low speed; minimal repositioning | Mid/rear court; stable stance; minimal movement | Any non-aggressive shot |

**Out of scope.** Deception (requires biomechanical discrepancy data between preparation and execution phases not captured in 2D skeletons), hesitation (requires sub-frame timing analysis beyond our temporal resolution), seamlessly (a quality modifier, not a discrete strategy class), and "a high net early shot" (insufficient samples). Shots with these labels are excluded from the training set.

**Research questions:**

- **RQ1:** What is the relative contribution of spatial features (player positioning, court geometry) versus temporal features (movement sequences, velocity) to strategy recognition?
- **RQ2:** To what extent does contrastive pre-training on unlabelled match footage improve few-shot tactical classification compared to randomly initialised baselines?

---

## 2. Literature Review

### 2.1 Skeleton-Based Action Recognition

Skeleton-based approaches model human motion as sequences of joint coordinates rather than raw pixel data. ST-GCN \[3\] introduced graph convolutions over spatial joint connections and temporal sequences, achieving 81.5% top-1 accuracy on Kinetics-400 for general action recognition. Subsequent work (2s-AGCN, MS-G3D, CTR-GCN) improved spatial attention and multi-scale temporal modelling. However, these methods target coarse action categories (e.g., "playing badminton" vs. "playing tennis") rather than fine-grained tactical patterns within a single sport.

### 2.2 Self-Supervised Learning for Skeleton Sequences

Label-free representation learning for skeletons has advanced rapidly. CrosSCLR \[5\] introduced cross-view contrastive learning across different skeleton augmentation views. AimCLR \[6\] addressed augmentation sensitivity through extreme augmentation strategies. These methods demonstrate that skeleton embeddings pre-trained without labels can match or approach supervised performance when fine-tuned, motivating our self-supervised pre-training strategy.

### 2.3 Badminton Analytics

Existing computational badminton analysis focuses primarily on shot-level classification rather than strategic reasoning. CoachAI \[1\] applied ShuttleNet (ResNet+LSTM) to classify 22 shot types on ShuttleSet, achieving 73.4% accuracy. FineBadminton contributed fine-grained annotations including strategy labels but did not report classification results for tactical categories.

Most recently, BST \[8\] (CVPR 2025 Workshop) introduced a Transformer-based approach for skeleton-based stroke classification on ShuttleSet, using both player skeleton sequences and shuttlecock trajectory as inputs. Concurrently, RacketVision \[9\] advanced racket detection and tracking in badminton video. Critically, both operate at the stroke type level --- they classify *what* shot was played, not *why* it was played. No published work addresses automated tactical strategy classification in badminton.

### 2.4 Few-Shot and Meta-Learning

Prototypical Networks \[4\] remain a strong baseline for few-shot classification, learning a metric space where classification is performed by distance to class prototypes. Applications in sports have been limited: few-shot methods have been explored for player identification and rare event detection, but not for tactical pattern recognition.

### 2.5 Research Gap

*Table 2: Positioning relative to existing work.*

| Method | Task | Data Regime | Tactical? |
|---|---|---|---|
| CoachAI ShuttleNet | Shot type (22-cls) | Fully supervised | No |
| BST (Chang, 2025) | Stroke type | Fully supervised (skeleton + trajectory) | No |
| RacketVision (Dong, 2025) | Racket detection | Fully supervised | No |
| ST-GCN | General action recognition | Fully supervised | No |
| CrosSCLR / AimCLR | Action recognition | Self-supervised | No |
| Prototypical Networks | Image classification | Few-shot | No |
| **Ours** | **Tactical strategy** | **SSL + few-shot** | **Yes** |

---

## 3. Proposed Solution

We propose a three-stage pipeline:

1. **Skeleton extraction** from raw video using GDINO-guided YOLOv8-Pose, with spatial court-region masking to eliminate non-player detections (chair umpires, ball kids), producing dual-player joint sequences.
2. **Encoder pre-training** via contrastive objectives on unlabelled ShuttleSet data:
   - *SimCLR* (self-supervised): augmentation-pair contrastive learning with NT-Xent loss, requiring no labels.
   - *SupCon* (supervised contrastive): uses ShuttleSet shot-type labels to define positive pairs. All shots of the same type are pulled together regardless of player or match, forcing the encoder to learn shot mechanics rather than player identity.
3. **Few-shot strategy classification** using prototypical networks on the small FineBadminton labelled set (30 train rallies / 10 held-out rallies).
4. **Dual inference output:** strategy prediction (5 classes, confidence + margin) and shot-type prediction (17 classes, logistic head trained on frozen encoder) --- enabling expert cross-validation of high-confidence strategy predictions.

### 3.1 Why Skeleton-Based Over Vision-Based?

The approach is skeleton-based rather than pixel-based for three reasons. First, **data efficiency**: structured skeleton input with graph priors means the model learns from far less data than a vision encoder would require. With only 40 labelled rallies and ~8,900 unlabelled shots, skeleton-based is the pragmatic choice. Second, **signal alignment**: all five target strategies are defined in terms of joint positions, velocities, and inter-player geometry --- precisely what the skeleton graph captures. Third, **interpretability**: skeleton inputs provide directly inspectable representations (joint angles, court distances) that vision embeddings cannot offer.

Badminton strategy is driven by spatial relationships between joints (elbow angle, opponent distance). ST-GCN captures these via explicit graph topology, while a Transformer treats all joints equally through attention. This structural prior acts as a regulariser on the small dataset.

---

## 4. Datasets and Data Acquisition

Both datasets feature professional broadcast matches. Under modern BWF rally scoring, each rally awards one point regardless of who served. Up to 177 rallies are possible in a 3-set match; professional matches typically contain 120--150 rallies. For this project, we classify tactical strategy **per shot** (stroke).

### 4.1 FineBadminton --- Labelled Dataset

FineBadminton is the fine-grained labelled dataset used for few-shot classification (Phase B). The published paper describes ~3,000 matches, but the publicly available demo comprises 11 matches with 40 rallies.

**Frame extraction.** The dataset provides pre-extracted frames at ~20 FPS organised by rally ID. No video download is required.

**Scale.** 40 rallies, 414 annotated shots. After excluding 59 shots with out-of-scope strategy labels (deception x14, high\_net\_early x36, hesitation x8, seamlessly x1), 355 shots with strategy labels remain. Of these, 296 have successfully extracted skeletons and are used in training.

*Table 3: Key annotation fields per shot.*

| Field | Description |
|---|---|
| `hit_frame` | Exact contact frame (absolute, no missing values) |
| `start_frame` / `end_frame` | Shot window boundaries |
| `hitter` | `"top"` or `"bottom"` court player |
| `hit_type` | Stroke type (12 classes) |
| `subtype` | Sub-classification (e.g., "flat lift", "short serve") |
| `quality` | Annotator quality score 1--7 |
| `ball_area` | Court zone of shot (9 zones) |
| `strategies` | Tactical strategy labels (classification target) |

*Table 4: Strategy label distribution (N=355 shots).*

| Strategy | Count | % of Shots |
|---|---|---|
| passive | 137 | 33.1% |
| intercept | 120 | 29.0% |
| create depth | 61 | 14.7% |
| defensive | 61 | 14.7% |
| move to net | 59 | 14.3% |

Note: `move_to_net` has only 18 samples after windowing and skeleton extraction (6.1% of training set) --- this is the most data-scarce class.

**Strategy--shot type co-occurrence.** Initial co-occurrence analysis reveals meaningful correlations between strategy labels and shot types. For example, "create depth" is almost exclusively associated with push shots (47/61 occurrences), while "intercept" is dominated by kills (66/120). These correlations motivate the use of shot-type labels as a proxy supervision signal during pre-training.

<!-- Figure: results/eda_shot_strategy_heatmap.png --- Strategy vs shot type co-occurrence heatmap -->
<!-- Figure: results/eda_shot_strategy_prob.png --- Conditional probability of strategy given shot type -->

### 4.2 ShuttleSet --- Pre-Training Dataset

ShuttleSet is the large-scale dataset used for contrastive pre-training (Phase A). Unlike FineBadminton, it carries **shot-type labels** (19 classes) but no tactical strategy labels.

**Frame extraction.** The dataset provides only CSV annotations and YouTube video links. Videos were downloaded via yt-dlp and frames extracted using a streaming pipeline with a stride of every 4 frames plus all annotated hit frames, yielding approximately 10 FPS effective rate. This balances storage efficiency with sufficient temporal resolution for player tracking.

**Scale.** Of 44 matches in `match.csv`, 4 were excluded due to non-standard broadcast camera angles (side-court or extreme distance). Of the 38 usable matches, 20 were selected for frame extraction and skeleton processing.

*Table 5: ShuttleSet scale (20 extracted matches).*

| Statistic | Value |
|---|---|
| Matches extracted | 20 (of 38 usable) |
| Total stroke records | ~17,000 |
| Total rallies | ~590 |
| Unique shot types | 17 (unified vocabulary) |

**Train/Test/Held-out split (match-level).** The 11 matches with GDINO-extracted skeletons are split at match level to prevent frame leakage:

*Table 6: ShuttleSet data split.*

| Split | Matches | Shots | Role |
|---|---|---|---|
| Train | 8 | 6,149 | SupCon/SimCLR pre-training |
| Test | 1 | 1,127 | SSL checkpoint selection (monitoring only) |
| Held-out | 2 | 1,675 | Strategy prediction + expert verification |
| **Total** | **11** | **8,951** | --- |

The held-out matches (Anthony Sinisuka Ginting vs Lee Zii Jia; CHEN Long vs CHOU Tien Chen) feature different players and tournaments from the training set, providing a meaningful cross-dataset evaluation target.

### 4.3 Shot Type and Shot Duration Analysis

Exploratory analysis of ShuttleSet reveals a strong relationship between shot type and shot duration (time between the current hit and the next), which reflects the physics of badminton and implicitly encodes attacking versus defensive intent.

*Table 7: Shot duration by category.*

| Category | Mean Duration | Median | Interpretation |
|---|---|---|---|
| Attack (smash, tap smash, push kill) | 0.43s | 0.40s | Fastest --- opponent has least reaction time |
| Serve | 0.68s | 0.60s | Short serves quick (0.62s), long serves slower (0.96s) |
| Net play (drop, block, cross-court net) | 0.74s | 0.73s | Mid-range, consistent |
| Defence (defensive return) | 0.75s | 0.75s | Similar to net play |
| Rear court (clear, slice, drop) | 0.86s | 0.83s | Slower --- shuttle travels full court length |
| Transition (lift, push, flat drive) | 0.91s | 0.97s | Slowest --- lifts (1.15s) give most recovery time |

Statistical significance: one-way ANOVA F = 493, p ~ 0; eta-squared = 0.57 (large effect size --- shot type explains ~57% of variance in duration). All pairwise comparisons significant at p < 0.001.

*Table 8: Strategy vs duration and quality (FineBadminton).*

| Strategy | N | Mean Duration | Median | Mean Quality |
|---|---|---|---|---|
| intercept | 120 | 0.84s | 0.80s | 5.22 |
| defensive | 61 | 0.92s | 0.84s | 4.05 |
| passive | 137 | 1.14s | 1.04s | 3.21 |
| move to net | 59 | 1.17s | 1.16s | 4.31 |
| create depth | 61 | 1.32s | 1.24s | 3.66 |

**Key findings.** Fast strategies (~0.8--0.9s: intercept, defensive) are reactive, pressure-driven plays. Slow strategies (~1.2--1.3s: move to net, create depth) are positional plays relying on push shots and clears with longer flight times. Passive shots (1.14s) have the lowest quality score (3.21), indicating forced responses under pressure rather than strategic choices. This temporal structure suggests that shot duration could serve as an auxiliary input feature for strategy classification.

---

## 5. Dataset Pre-Processing

From raw video frames we extract three types of information: player skeletons, shuttlecock trajectories, and court homographies. A manual annotation quality assurance tool was developed to verify extraction quality.

### 5.1 Skeleton Extraction Pipeline

**Pose extraction.** Skeleton keypoints are extracted using YOLOv8-Pose (yolov8s-pose), which performs joint person detection and 17-joint COCO keypoint estimation in a single forward pass. For each frame, all detected persons are scored by mean keypoint confidence; the top-2 detections are retained as the two players.

**Umpire contamination.** A systematic quality audit of the extracted FineBadminton skeletons revealed a significant contamination issue. YOLOv8-Pose imposes no spatial constraint on detected persons. The chair umpire is consistently detected with high confidence because they are stationary, upright, and well-lit --- occupying a seat at approximately x ~ 102 px on a 1280 px-wide frame. In some cases, audience members were also detected as players.

<!-- Figure: Side-by-side comparison of YOLOv8-only vs GDINO-guided extraction showing umpire contamination -->

**Grounding DINO-guided extraction (adopted solution).** A GDINO-guided approach uses Grounding DINO (grounding-dino-tiny) with the text prompt `"badminton player ."` to produce court-region bounding boxes before YOLOv8 keypoint estimation. Only YOLO detections with IoU >= 0.25 against a GDINO bounding box are accepted as valid players. If fewer than 2 valid players pass this filter, the pipeline falls back to plain YOLOv8 top-2 selection.

Because Grounding DINO is not spatially aware, additional filtering was required. A **spatial court-region mask** excludes detections outside the estimated court boundaries:

*Table 9: Court-region mask parameters (pixel coordinates on 1920x1080 frames).*

| Parameter | Value | Fraction |
|---|---|---|
| X\_MIN | 450 | 0.234 |
| X\_MAX | 1450 | 0.755 |
| Y\_MIN | 200 | 0.185 |
| Y\_MAX | 1000 | 0.926 |

This eliminates chair umpires (seated at x ~ 5--10%), line judges, and ball kids from the candidate set before IoU matching. The FineBadminton skeletons used in all experiments were extracted with the GDINO-guided pipeline. All 40 rallies (10,620 frames) were successfully extracted.

### 5.2 Player Ordering and Hitter Assignment

**Player ordering (Y-sort).** After selecting the top-2 detections, players are sorted by their mean Y keypoint centroid in image coordinates (Y increases downward):

- Player 0 (joints 0--16) = smaller mean Y = top-court player (further from camera)
- Player 1 (joints 17--33) = larger mean Y = bottom-court player (closer to camera)

This assignment is made purely from image geometry --- no annotations are consulted. The resulting `(2, T, 34)` skeleton arrays on disk always follow this Y-sort convention.

Note: Player 0 (top court) appears faint and tightly clustered in pixel space because they are further from the camera. This is expected behaviour and does not indicate an extraction error.

**Hitter ordering.** For the classifier to generalise across rallies where either player may be the hitter, the hitting player is always placed at nodes 0--16 at dataset load time. This is achieved by conditionally swapping the two 17-joint halves via `_reorder_hitter_first()`:

- **FineBadminton:** Each shot annotation specifies the hitter's court side as "top" or "bottom". If `hitter == "bottom"`, nodes 0--16 and 17--33 are swapped after windowing.
- **ShuttleSet:** The `player` column (A or B) identifies the hitter. For each rally, the median `player_location_y` for players A and B determines court side --- the player with the smaller median Y is on the top court. This per-rally median approach yields the expected 50/50 top--bottom split across all matches.

The graph structure (inter-player edges in `graph_builder.py`) can therefore learn hitter-specific versus opponent-specific spatial patterns.

**Left-handed players.** Some players (e.g., Kento Momota) are left-handed and hold the racket in their left hand, producing slightly different poses for the same shot type. This was not accounted for in the current project and represents a potential source of noise.

### 5.3 Shot Segmentation (Temporal Windowing)

Each shot sample is a T = 32 frame window centred on the hit frame. At 20 FPS this spans approximately 1.6 seconds, covering the 75th percentile of shot-to-shot intervals.

**Why frames before the hit are critical:** Tactical strategy is committed to during preparation, not at contact. Footwork direction reveals "move to net" strategy in the approach, not in the swing. Shoulder/racket windup angle signals whether the shot was deliberate. Opponent positioning before contact determines which strategy is rational.

**Why frames after the hit are included:** Post-contact frames confirm whether the executed shot matches the strategy (e.g., a "move to net" stroke should result in the hitter approaching the net).

Because consecutive strokes produce overlapping windows, the same frame may appear in two samples. This is standard practice in temporal action segmentation. The two samples have different labels and different hitter assignments, so the overlap does not inflate metrics.

**Frame rate mismatch and temporal coverage.** FineBadminton frames are at ~20 FPS; ShuttleSet frames are extracted at stride-4 (~10 FPS effective). A T = 32 window therefore covers ~1.6 s of real time for FineBadminton but ~3.2 s for ShuttleSet. This means the ShuttleSet encoder sees a wider temporal context per sample during pre-training than what the FineBadminton classifier sees at evaluation. Our hypothesis is that this mismatch is **not harmful** --- the SSL pre-training benefits from seeing longer movement arcs, and the few-shot classifier adapts to FineBadminton's narrower window during episodic fine-tuning. The speed perturbation augmentation (+/- 20%) further mitigates this by exposing the encoder to a range of temporal scales during pre-training. However, resampling ShuttleSet to match FineBadminton's effective frame rate (or vice versa) would eliminate this confound and is listed as future work.

### 5.4 Shuttlecock Trajectory

Shuttle trajectories are extracted using **TrackNetV4** \[10\] (`tracknet-series-pytorch`, local clone). V4 adds a learnable MotionPrompt layer (inter-frame difference attention) over the V2 U-Net baseline, which is effective for a fast-moving object like a shuttlecock covering 50--100 pixels per frame.

Outputs: per-rally `.npy` files of shape `(T, 3)` --- columns `[x, y, visible]`, frame-aligned with the skeleton arrays. These are stored in `datasets_preprocessing/finebadminton_shuttles/` and visualised in the demo UI as a yellow dot with a trail overlay.

Shuttle trajectory is optionally incorporated as **virtual node 34** in the dual-player graph (35 nodes total). When `use_shuttle=True`, the shuttle's (x, y) position is appended before feature engineering so that the homography transform applies to it, making all distance/velocity features camera-invariant. The shuttle node is connected to both players' wrist joints (nodes 9, 10 for P1; 26, 27 for P2) in the ST-GCN graph. This is evaluated as Step 6 in the ablation study.

**ShuttleSet shuttle extraction status.** Shuttle trajectories have been extracted for FineBadminton (all 40 rallies). For ShuttleSet, 8 partial `.npy` files exist in `datasets_preprocessing/shuttleset_shuttles/` but a complete extraction notebook does not yet exist. A dedicated notebook (`02B_Shuttlecock Tracking/shuttlecock_tracking_shuttleset.ipynb`) needs to be created, mirroring the FineBadminton shuttle extraction pipeline. This is a prerequisite for the Step 6 shuttle ablation on ShuttleSet data.

### 5.5 Court Homography

The camera views badminton courts from an elevated angle, producing perspective distortion: a player moving one metre near the camera (bottom court) covers more pixels than a player moving one metre far from the camera (top court). A 3x3 homography matrix H maps pixel coordinates (u, v) to court-relative metres (x, y) with origin at court centre.

- **FineBadminton** (fixed broadcast camera): H is computed once per resolution group.
- **ShuttleSet**: H is computed per match using Roboflow-detected court keypoints.

<!-- Figure: Example of incorrect ShuttleSet homography annotation requiring manual correction -->

**At training time**, the feature engineering pipeline applies the homography as the first step:

1. Load raw `.npy` skeleton (pixel coordinates)
2. `FeatureEngineer.compute()` applies homography (pixel to court metres)
3. Compute velocity, acceleration (L1)
4. Compute dist-to-net, dist-to-centre, dist-to-opponent (L2)
5. Compute joint angles (L3)
6. Output (9, T, 35) tensor feeds into the ST-GCN encoder

**Perspective elevation error.** A homography is defined on the court floor plane (z = 0). Skeleton joints are elevated above the floor (torso by ~1 m, head by ~2 m), causing projected positions to shift toward the camera. Three mitigations were evaluated: (A) ankle-only positioning using joints 15 and 16 which lie on the floor plane (adopted), (B) hip-midpoint positioning, and (C) depth-corrected positioning using Depth Anything V2. Approach A is adopted as the default because ankle detections are reliable and the floor-plane assumption is exactly satisfied.

---

## 6. Feature Engineering and Graph Construction

### 6.1 Node Feature Layers

Raw skeleton coordinates alone are an impoverished input. The strategy taxonomy requires the model to perceive velocity, court context, body configuration, and inter-player spatial relationships. We compute enriched node features in cumulative layers:

*Table 10: Node feature layers (cumulative).*

| Layer | Features Added | Dim | Signal Encoded |
|---|---|---|---|
| L0 | [x, y] court-relative via homography | 2 | Position only |
| L1 | + [vx, vy, ax, ay] finite differences | 6 | Velocity encodes "lunging forward" (intercept) vs. "stationary" (passive) |
| L2 | + [dist\_net, dist\_centre, dist\_opp] | 9 | Court position encodes "forward court" (intercept) vs. "rear court" (defensive) |
| L3 | + [elbow\_L, elbow\_R, knee\_L] angles | 12 | Body pose encodes "arm extended" (defensive) vs. "arm forward flat" (intercept) |

**Feature pipeline diagram.**

```
  Video Frames              Video Frames              Court Image
       |                         |                         |
       v                         v                         v
  GDINO + YOLOv8-Pose      TrackNetV4              Roboflow Keypoints
       |                         |                         |
       v                         v                         v
  .npy (2, T, 34)          .npy (T, 3)             H_img_to_court_m
  [x,y] per joint          [x, y, vis]                .npy (3x3)
  x 17 joints x 2 players  per frame               pixel -> metres
       |                         |                         |
       |    if use_shuttle=True  |                         |
       |    append as node 34    |                         |
       v                         v                         |
  (2, T, 35) or (2, T, 34)                                |
       |                                                   |
       v                                                   |
  FeatureEngineer.compute() <------------------------------+
       |
       |  1. Homography (pixel -> court metres)
       |  2. L0: [x, y]                                    ->  2 dims
       |  3. L1: + [vx, vy, ax, ay]                        ->  6 dims
       |  4. L2: + [dist_net, dist_centre, dist_opp]       ->  9 dims (default)
       |  5. L3: + [elbow_L, elbow_R, knee_L]              -> 12 dims
       v
  Output: (9, T, V) tensor -> ST-GCN Encoder -> 256-dim embedding
```

### 6.2 Graph Construction

The dual-player spatio-temporal graph uses **34 nodes** (17 COCO joints per player). Intra-player edges follow the standard COCO skeleton topology (anatomical connections). Inter-player edges connect corresponding joints between players (e.g., right wrist to right wrist) to capture relative positioning. The adjacency matrix is partitioned into three subsets following Yan et al. \[3\]: identity (self-loops), centripetal (toward root), and centrifugal (away from root).

```
  Player 0 (top court)       Player 1 (bottom court)     Shuttle (optional)
  nodes 0-16                 nodes 17-33                  node 34
  +----------+               +----------+                 +---+
  | COCO 17  |<-- inter ---->| COCO 17  |                 | o |
  | skeleton |   player      | skeleton |                 +-+-+
  | edges    |   edges       | edges    |                   |
  +-----+----+               +-----+----+            wrist edges
        |                          |               (nodes 9,10,26,27)
        +--------------------------+--------------------+

  Adjacency partitioned -> (3, V, V): identity / inward / outward
```

When `use_shuttle=True`, the graph expands to 35 nodes. The shuttle node receives the same feature engineering pipeline as skeleton joints (homography, velocity, court context).

### 6.3 ST-GCN Encoder

The ST-GCN encoder applies graph convolutions alternating between spatial operations (over the skeleton topology) and temporal convolutions (over the T = 32 frame sequence). Each of the 9 ST-GCN blocks performs:

1. **Spatial convolution** --- captures relationships between joints within a single frame (e.g., how the elbow relates to the shoulder).
2. **Temporal convolution** --- slides a 9-frame kernel across time to capture how joint relationships change (e.g., the arm extending into a smash).

The time dimension is progressively compressed (32 -> 16 -> 8 -> 4) via strided convolutions, analogous to how image CNNs downsample spatial resolution. At the end, global average pooling produces a single **256-dimensional embedding** per shot.

*Table 11: ST-GCN architecture summary.*

| Parameter | Value |
|---|---|
| Input channels | 9 (L2 features) |
| Number of nodes | 34 (or 35 with shuttle) |
| Number of layers | 9 |
| Base channels | 64 |
| Embedding dimension | 256 |
| Temporal kernel size | 9 |
| Dropout | 0.3 |
| Total parameters | ~3.08M |

---

## 7. Encoder Pre-Training (Phase A)

Pre-training on ShuttleSet's ~6,100 skeleton shots (train split) gives the ST-GCN encoder exposure to diverse badminton motion patterns before it sees any strategy labels. We train two contrastive variants --- **SimCLR** (fully self-supervised) and **SupCon** (shot-type supervised) --- and compare both against random initialisation in the ablation study (Section 9).

### 7.1 Shared Architecture and Augmentation Pipeline

Both methods share the same encoder (ST-GCN, 3.08M params), projection head (MLP: 256 -> 256 -> 128 with BatchNorm + ReLU), and augmentation pipeline. Two independently augmented "views" of each skeleton sequence are generated per training step:

*Table 12: Skeleton augmentation parameters.*

| Augmentation | Parameters | Purpose |
|---|---|---|
| Joint jittering | Gaussian noise sigma = 0.02 | Simulate pose estimation noise |
| Speed perturbation | +/- 20% temporal resampling | Invariance to shot speed variation |
| Spatial rotation | +/- 15 degrees court-relative | Invariance to minor camera angle differences |
| Joint masking | 5--15% of joints zeroed | Force reconstruction from partial observations |

Shared hyperparameters: AdamW optimiser (lr = 1e-3, weight decay = 1e-5), batch size 64, temperature tau = 0.07, early stopping with patience = 10 epochs.

### 7.2 SimCLR --- Self-Supervised (NT-Xent Loss)

SimCLR \[7\] requires no labels. For each skeleton sequence in the batch, the two augmented views form a single positive pair; all other 2(B-1) samples serve as negatives. The NT-Xent loss is:

```
L_NT-Xent = -log [ exp(sim(z_i, z_j) / tau) / sum_{k != i} exp(sim(z_i, z_k) / tau) ]
```

where z\_i, z\_j are the L2-normalised projections of the two views and sim(.,.) is cosine similarity.

**Key characteristic:** Each anchor has exactly **1 positive** (its own augmentation twin). The contrastive signal is invariant to semantic content --- the encoder learns motion-general features without shot-type or strategy awareness.

**Training results:** SimCLR used all available matches (no split filtering required since no labels are needed). Training converged at epoch 93 (early stopped) with a final NT-Xent loss of 0.640.

### 7.3 SupCon --- Supervised Contrastive Learning

SupCon \[11\] extends SimCLR by using ShuttleSet's shot-type labels to define positive pairs. Rather than pairing only augmented views of the same sequence, SupCon treats **all shots of the same type** (across different players and matches) as positives:

```
L_SupCon = sum_i [ -1/|P(i)| * sum_{p in P(i)} log [ exp(z_i . z_p / tau) / sum_{a != i} exp(z_i . z_a / tau) ] ]
```

where P(i) = {j : y\_j = y\_i, j != i} is the set of same-type positives.

**Key characteristic:** Each anchor has **multiple positives** (~7 on average in a batch of 64 with ~10 shot types represented). This richer gradient signal enables the encoder to learn that structurally similar movements (e.g., all smashes) should cluster together regardless of player identity.

**Training results:** SupCon converged at epoch 60 (early stopped) with a final loss of 2.516. The loss curve shows steeper initial descent than SimCLR, consistent with the richer per-step gradient signal.

*Table 13: SimCLR vs SupCon comparison.*

| Aspect | SimCLR (NT-Xent) | SupCon |
|---|---|---|
| Labels required | None | Shot-type labels |
| Positives per anchor | 1 (constant) | ~7 (mean) |
| What encoder learns | General motion similarity | Shot-type-aware clustering |
| Training data | All matches | Train-split matches only |
| Convergence | 93 epochs, loss 0.640 | 60 epochs, loss 2.516 |
| Risk | May learn irrelevant invariances | Depends on label quality |

**Why SupCon is expected to outperform SimCLR.** Shot types partially correlate with tactical strategies --- smashes are more likely during "intercept," while lobs are associated with "defensive" play. By forcing the encoder to group by shot mechanics, SupCon pre-structures the embedding space in a way that is more directly useful for downstream strategy classification.

### 7.4 Linear Probe Evaluation (Sanity Check)

To assess whether the frozen pre-trained encoder produces useful representations, we fit a logistic regression classifier on the frozen embeddings and evaluate on FineBadminton strategy labels (5-fold stratified CV). This is a **post-hoc evaluation only** --- the encoder weights are never updated.

*Table 14: Linear probe vs ProtoNet --- role clarification.*

| Aspect | Linear Probe (Phase A eval) | ProtoNet (Phase B training) |
|---|---|---|
| Purpose | Sanity check: "do frozen features linearly separate classes?" | Downstream task: "classify with few labelled examples" |
| Encoder | Frozen | Fine-tunable via episodic backprop |
| Classifier | Logistic regression | Distance to class prototypes |
| Training regime | Standard supervised, full labelled set | Episodic: k-shot support -> query classification |
| Data split | Sample-level stratified k-fold | Rally-level splits |

SimCLR linear probe: macro-F1 = 0.107 +/- 0.002 (below random baseline of 0.20). This weak result does not preclude benefit after episodic fine-tuning, because ProtoNet uses distance-based metric learning which is better suited to the low-data regime than linear separation.

### 7.5 SupCon Shot-Type Proxy Evaluation

To evaluate whether SupCon pre-training produces semantically meaningful embeddings beyond strategy, we evaluate shot-type classification on the ShuttleSet validation split.

*Table 15: Shot-type classification on ShuttleSet validation (SupCon L2 embeddings, frozen encoder + logistic regression).*

| Shot Type | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `clear` | 0.89 | 0.89 | **0.89** | 314 |
| `short_serve` | 0.81 | 0.89 | **0.85** | 240 |
| `net_drop` | 0.66 | 0.87 | **0.75** | 563 |
| `lob_lift` | 0.78 | 0.66 | **0.71** | 513 |
| `long_serve` | 0.61 | 0.76 | **0.67** | 37 |
| `block` | 0.64 | 0.53 | **0.58** | 455 |
| `transition` | 0.55 | 0.61 | **0.58** | 160 |
| `push` | 0.50 | 0.63 | **0.56** | 276 |
| `smash` | 0.69 | 0.44 | **0.54** | 245 |
| `drive` | 0.39 | 0.38 | **0.39** | 152 |
| `tap_smash` | 0.30 | 0.53 | **0.38** | 151 |
| `defensive_lift` | 0.37 | 0.39 | **0.38** | 18 |
| `slice_drop` | 0.54 | 0.27 | **0.36** | 182 |
| `push_rush` | 0.29 | 0.16 | **0.20** | 38 |
| `cross_net` | 0.30 | 0.12 | **0.17** | 111 |
| **Macro-F1** | | | **0.534** | 3,455 |

**Confusion clusters:** Net area shots (`net_drop` <-> `cross_net` <-> `block`) differ primarily in racket angle and wrist orientation, which 2D skeleton keypoints cannot resolve. Power shots (`smash` <-> `tap_smash`) share similar overhead posture. This is a fundamental limitation of skeleton-only features, not a failure of the contrastive objective.

---

## 8. Few-Shot Training (Phase B)

### 8.1 Prototypical Network Classifier

ProtoNet asks: "given only k labelled examples of each strategy, can we classify new ones?" It operates in two steps:

1. **Prototype computation:** For each of the 5 strategy classes, compute the mean embedding (centroid) of all support examples. No gradient descent, no loss function --- just arithmetic mean.
2. **Distance-based classification:** Compute Euclidean distance from a query shot's embedding to each of the 5 prototypes. Assign to the nearest. Confidence is derived from softmax over negative distances; margin score (nearest vs. second-nearest distance) flags low-confidence predictions.

**Episodic training.** Rather than standard batch training, each batch simulates a few-shot task: pick N-way classes, provide k-shot support examples, classify query examples. The encoder is fine-tuned during these episodes (not frozen), with an **auxiliary hit-type head** (weight = 0.3) trained alongside the prototype loss to provide additional gradient signal.

### 8.2 Evaluation Protocol

Given the extremely small labelled set (40 rallies), the evaluation protocol uses **rally-level splits** to prevent data leakage --- all shots from a given rally appear in only one partition.

**Split design:**

- **30 rallies** (~241 shots): used for 5-fold cross-validation. Per fold: ~18 rallies train (episodic ProtoNet), ~6 rallies validation (checkpoint selection), ~6 rallies test (reported metrics).
- **10 rallies** (~55 shots): completely held out for final generalisation evaluation. Never seen during any CV fold.

All results report mean +/- standard deviation across 5 folds. Class distribution is imbalanced; this is addressed through balanced episodic sampling: each episode samples equal numbers of support and query examples per class regardless of overall frequency.

**Why macro-F1?** We report macro-F1 (unweighted average of per-class F1 scores) because it treats all 5 strategy classes equally regardless of frequency. With a skewed distribution where `passive` and `intercept` dominate, accuracy would reward a model that ignores minority classes. Macro-F1 penalises this behaviour, making it the appropriate metric for imbalanced few-shot classification. A random baseline scores ~0.20.

### 8.3 Baseline Results (No Pre-Training)

*Table 16: Phase B baseline --- random initialisation, L2 features, 5-way 10-shot ProtoNet, 5-fold CV on 30 training rallies.*

| Fold | Train Loss (ep.50) | Train Acc | Best Val-F1 | Test Macro-F1 |
|---|---|---|---|---|
| 1 | 0.049 | 0.983 | 0.519 | 0.409 |
| 2 | 0.041 | 0.986 | 0.337 | 0.375 |
| 3 | 0.034 | 0.990 | 0.326 | 0.333 |
| 4 | 0.040 | 0.988 | 0.486 | 0.331 |
| 5 | 0.022 | 0.991 | 0.396 | 0.399 |
| **Mean** | | | | **0.369 +/- 0.033** |

*Table 17: Per-class F1 (mean +/- std across 5 folds, random init baseline).*

| Strategy | Samples | Mean F1 | Std | Observation |
|---|---|---|---|---|
| create\_depth | 61 | 0.581 | 0.087 | Best-performing despite moderate count |
| intercept | 108 | 0.522 | 0.074 | Most common class, consistent across folds |
| defensive | 59 | 0.417 | 0.085 | Moderate performance |
| passive | 50 | 0.290 | 0.130 | High variance; confused with intercept/defensive |
| **move\_to\_net** | **18** | **0.036** | **0.073** | **Near-zero in 4/5 folds --- insufficient samples** |

<!-- Figure: results/fewshot_confusion_matrix.png --- 5x5 confusion matrix -->
<!-- Figure: results/fewshot_training_curves.png --- Training loss and validation F1 curves -->

**Key observations:**

1. **Severe episodic overfitting.** Training accuracy reaches 98--99% while validation F1 peaks around epoch 10 and degrades thereafter. SSL pre-training is expected to be the primary mitigation.
2. **`move_to_net` is effectively unlearnable at this scale** (18 samples, 6.1% of dataset). The prototype is noise-dominated.
3. **`create_depth` outperforms `intercept`** despite fewer samples (61 vs 108), suggesting create\_depth has more distinctive skeleton signatures (rear-court positioning, extended arm overhead).

---

## 9. Ablation Studies

Six ablation steps are run sequentially, each varying one axis while holding all others fixed at the best value found so far. All results use 5-fold stratified cross-validation on the 30 training rallies. The primary metric is macro-F1.

### 9.1 Step 1 --- Feature Engineering: Which Input Signals Matter?

We test four progressively richer feature layers while fixing the encoder (ST-GCN), graph (full dual-player, 34 nodes), and classifier (episodic ProtoNet with fine-tuning).

*Table 18: Feature engineering ablation (Step 1). SSL checkpoint exists only for L2; others use random init.*

| Variant | Layer | Dim | Init | Macro-F1 | +/- Std |
|---|---|---|---|---|---|
| L0\_raw\_xy | L0 | 2 | Random | \[TBD\] | \[TBD\] |
| L1\_kinematics | L1 | 6 | Random | \[TBD\] | \[TBD\] |
| **L2\_court\_ctx** | **L2** | **9** | **SSL** | **0.369** | **0.033** |
| L3\_joint\_angles | L3 | 12 | Random | \[TBD\] | \[TBD\] |

Note: The L2 advantage may partially reflect SSL pre-training rather than feature engineering alone, since SSL checkpoints exist only for L2. This is a known limitation of the sequential ablation design.

<!-- Figure: results/ablation_feature_layers.png --- Bar chart of feature layer comparison -->

### 9.2 Step 2 --- Encoder Architecture: Does Graph Structure Help?

We compare ST-GCN (graph convolutions with skeleton structural priors) against a BST-style Transformer (self-attention over all joints, no built-in skeleton structure). SSL weights exist only for ST-GCN; the Transformer uses random initialisation.

*Table 19: Encoder architecture ablation (Step 2).*

| Encoder | Params | Structural Prior | Macro-F1 | +/- Std |
|---|---|---|---|---|
| ST-GCN | 3.08M | Skeleton topology | \[TBD\] | \[TBD\] |
| Transformer | ~2.1M | None (attention) | \[TBD\] | \[TBD\] |

### 9.3 Step 3 --- Pre-Training Regime: Does SSL Transfer? (RQ2)

This step directly addresses **RQ2**. We compare three initialisation strategies using the best feature layer and encoder from Steps 1--2.

*Table 20: Pre-training regime ablation (Step 3 --- RQ2).*

| Variant | Labels Used | Macro-F1 | +/- Std |
|---|---|---|---|
| Random init | None | 0.369 | 0.033 |
| SimCLR | None (self-supervised) | \[TBD\] | \[TBD\] |
| SupCon | Shot-type labels | \[TBD\] | \[TBD\] |

Preliminary result with SupCon (frozen encoder, not fine-tuned): macro-F1 = 0.559 +/- 0.089 --- a substantial improvement over random init, suggesting pre-training is highly beneficial even without encoder fine-tuning.

### 9.4 Step 4 --- Classifier Comparison

The encoder is trained once with episodic ProtoNet, frozen, and then different classifiers are evaluated on the same embeddings.

*Table 21: Classifier comparison (Step 4, from fewshot\_results.json).*

| Method | Macro-F1 | +/- Std |
|---|---|---|
| Linear probe | 0.362 | 0.039 |
| k-NN (k = 3) | 0.328 | 0.052 |
| k-NN (k = 5) | 0.308 | 0.050 |
| ProtoNet | 0.206 | 0.036 |

Note: These results use SSL pre-trained embeddings from a single-match checkpoint (`ssl_pretrained_L2_1matchonly.pt`), which likely underestimates the benefit of more extensive pre-training. The linear probe outperforming ProtoNet here suggests the embedding space may not yet form tight per-class clusters, which more diverse pre-training data would improve.

### 9.5 Step 5 --- K-Shot Sensitivity

We sweep K = 1, 3, 5, 8, 10 support shots per class to determine the minimum viable annotation effort. The query count is capped so that K + n\_query <= 11 (limited by the minimum class count of `move_to_net`).

*Table 22: K-shot sensitivity (Step 5).*

| K | n\_query | Macro-F1 | +/- Std |
|---|---|---|---|
| 1 | 5 | \[TBD\] | \[TBD\] |
| 3 | 5 | \[TBD\] | \[TBD\] |
| 5 | 5 | \[TBD\] | \[TBD\] |
| 8 | 3 | \[TBD\] | \[TBD\] |
| 10 | 1 | \[TBD\] | \[TBD\] |

<!-- Figure: results/ablation_kshot_curve.png --- K-shot learning curve -->

### 9.6 Step 6 --- Shuttlecock Trajectory

We test whether adding the shuttlecock as a 35th graph node improves strategy recognition. SSL weights were trained on the 34-node graph, so the skeleton+shuttle variant uses random initialisation for the shuttle-related parameters.

*Table 23: Shuttlecock ablation (Step 6).*

| Variant | Nodes | Macro-F1 | +/- Std |
|---|---|---|---|
| Skeleton only | 34 | \[TBD\] | \[TBD\] |
| Skeleton + shuttle | 35 | \[TBD\] | \[TBD\] |

<!-- Figure: results/ablation_shuttle.png --- Shuttle ablation bar chart -->

---

## 10. Architecture and Pipeline Summary

### 10.1 End-to-End Pipeline

```
Phase A: SSL Pre-Training (ShuttleSet, ~6,100 train shots)
  Option A: SimCLR (self-supervised, no labels)
  Option B: SupCon (shot-type labels as positives)  <-- stronger
  -> saves ssl_pretrained_{method}_L2.pt

Phase B: Few-Shot Training (FineBadminton, 30 train rallies)
  Load pre-trained encoder (fine-tuned during episodic training)
  ProtoNet episodes over 5 strategy classes
  Auxiliary hit-type head (weight=0.3)
  -> saves fewshot_L2.pt

Evaluation:
  Task 1: FB held-out (10 rallies)
    -> Strategy classification (ground truth exists)
    -> Shot-type classification (ground truth exists)
  Task 2: SS held-out (2 matches)
    -> Shot-type classification (ground truth exists)
    -> Strategy prediction (confidence + expert verification)
```

*Table 24: Complete pipeline stages.*

| Stage | Component | Input | Output |
|---|---|---|---|
| A1 | YOLOv8-Pose (GDINO-guided) | Video frames | 2D skeletons (2, T, 34) |
| A2 | Graph Builder + Feature Eng. | Skeleton sequences | Enriched graph (9, T, 34) |
| A3 | Shot Segmentation | Skeletons + timestamps | T = 32 frame windows, hitter-first |
| A4 | ST-GCN Encoder | Skeleton graph | 256-dim embedding |
| A5 | Contrastive Head | Augmented pairs +/- labels | SimCLR or SupCon loss |
| A6 | Shot-Type Classifier | Frozen embeddings | Logistic head (17 classes) |
| B1 | Prototype Computation | FB support embeddings | 5 class prototypes |
| B2 | ProtoNet Classifier | Query + prototypes | Strategy + confidence |
| C1 | Court Detector | Reference frame | Homography H |
| C2 | TrackNetV3 (frozen) | Consecutive frames | Shuttle position + hit timestamps |
| C3 | Full Pipeline | New video | Strategy + shot-type per shot |

### 10.2 Codebase Structure

*Table 25: Module responsibilities.*

| File | Responsibility |
|---|---|
| `src/config.py` | All hyperparameters, paths, label mappings |
| `src/data/pose_extractor.py` | YOLOv8 + GDINO skeleton extraction |
| `src/data/graph_builder.py` | Dual-player graph (34/35 nodes, 3 adjacency partitions) |
| `src/data/dataset.py` | Data loading, windowing, hitter ordering, episodic sampling |
| `src/data/feature_eng.py` | L0--L3 feature computation, homography transform |
| `src/models/stgcn_model.py` | ST-GCN encoder (9 blocks, 3.08M params) |
| `src/models/transformer_encoder.py` | BST-style Transformer (4 layers, 8 heads) |
| `src/models/simclr_loss.py` | NT-Xent loss, projection head, augmentations |
| `src/models/proto_net.py` | ProtoNet + k-NN, confidence/margin scoring |
| `src/inference.py` | Phase C end-to-end inference |
| `notebooks/05_ablations.ipynb` | All 6 ablation steps |
| `badminton_server.py` | Demo HTTP server (port 7860) |
| `badminton_pipeline_demo.html` | Interactive React demo UI |

---

## 11. Limitations and Discussion

1. **Small test set variance.** With ~50--65 shots per test fold, a single misclassified rally can shift macro-F1 by 5--10 percentage points. Mitigated by 5-fold CV but results should be interpreted with this variance in mind.

2. **Class imbalance.** `move_to_net` has only 18 training samples (6.1%), making its prototype noise-dominated and near-unlearnable at this scale.

3. **2D skeleton limitations.** We use 2D pose estimation, losing depth information that could help distinguish strategies (e.g., shuttle height for create\_depth). 3D pose lifting (MotionBERT) is a potential extension.

4. **No shuttle trajectory in encoder (default).** Several strategies (create\_depth, intercept) are partially defined by shuttle placement. TrackNetV4 trajectories are extracted and visualised in the demo but only optionally incorporated during training (Step 6 ablation).

5. **Pose estimation errors.** Occlusion, fast motion blur, and overlapping players cause pose estimation failures. GDINO substantially reduces umpire contamination but does not eliminate all errors.

6. **Domain gap.** Pre-training and evaluation both use broadcast footage. Generalisation to amateur phone recordings is unvalidated.

7. **Frame rate mismatch.** FineBadminton at ~20 FPS, ShuttleSet at ~10 FPS effective (stride-4). A T = 32 window covers ~1.6 s for FB but ~3.2 s for SS. Speed perturbation augmentation partially mitigates this, but resampling to a common rate would eliminate the confound entirely.

8. **Potential dataset overlap.** FineBadminton and ShuttleSet both feature elite broadcast matches. Some players may appear in both datasets, creating an indirect leakage path through player-specific movement patterns.

---

## 12. Future Work

1. **Shuttle trajectory as encoder input** (highest priority). Integrate TrackNetV4 trajectories as a 35th graph node during Phases A--B training. Requires creating a ShuttleSet shuttle extraction notebook first. Directly addresses the signal gap for create\_depth and intercept.

2. **Encoder architecture comparison.** The Transformer encoder is implemented but not fully evaluated. A direct ST-GCN vs Transformer comparison would clarify whether graph-structural priors help or whether attention can compensate.

3. **3D pose lifting.** Replace 2D pose with monocular 3D lifting (MotionBERT, MotionAGFormer) to recover depth and improve vertical trajectory discrimination.

4. **Rally-level temporal modelling.** Extend from shot-level to rally-level sequence modelling (Transformer over shot embeddings) to capture multi-shot strategic arcs.

5. **Active learning.** Use the confidence estimation module to identify the most informative unlabelled rallies for expert annotation, iteratively expanding the labelled set.

6. **Cross-sport transfer.** Evaluate the pre-trained encoder on tennis, squash, and table tennis tactical analysis.

---

## References

\[1\] Wang, W. Y., et al. (2022). ShuttleNet: Position-aware Fusion of Rally Progress and Player Styles for Stroke Forecasting in Badminton. *Proceedings of AAAI.*

\[2\] FineBadminton Dataset. Fine-grained badminton annotations with strategy labels. \[Dataset paper reference TBD\].

\[3\] Yan, S., Xiong, Y., and Lin, D. (2018). Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition. *Proceedings of AAAI.*

\[4\] Snell, J., Swersky, K., and Zemel, R. (2017). Prototypical Networks for Few-shot Learning. *Advances in NeurIPS.*

\[5\] Li, L., et al. (2021). CrosSCLR: Cross-View Contrastive Learning for 3D Skeleton-Based Action Recognition. *Proceedings of CVPR.*

\[6\] Guo, T., et al. (2022). AimCLR: Contrastive Learning of Skeleton-Based Action Recognition with Extreme Augmentations. *Proceedings of CVPR.*

\[7\] Chen, T., et al. (2020). A Simple Framework for Contrastive Learning of Visual Representations (SimCLR). *Proceedings of ICML.*

\[8\] Chang, W. (2025). BST: Badminton Stroke-type Transformer for Skeleton-Based Stroke Classification. *CVPR 2025 Workshop.*

\[9\] Dong, Y., et al. (2025). RacketVision: Racket Detection and Tracking in Badminton Video. \[Venue TBD\].

\[10\] Sun, Y., et al. (2023). TrackNetV3: Real-Time Shuttlecock Tracking in Badminton. *Proceedings of ACM Multimedia.*

\[11\] Khosla, P., et al. (2020). Supervised Contrastive Learning. *Advances in NeurIPS.*

---

## Appendix A: Unified Shot Type Vocabulary

Both datasets use different shot type taxonomies. We define a 17-class unified vocabulary to enable cross-dataset training:

*Table A1: Unified shot type mapping.*

| ID | Unified Label | ShuttleSet (Chinese) | FineBadminton (English) |
|---|---|---|---|
| 0 | `short_serve` | fa duan qiu | short serve |
| 1 | `long_serve` | fa chang qiu | high serve, flick serve |
| 2 | `smash` | sha qiu | jump/full/common/slice/stick smash |
| 3 | `tap_smash` | dian kou | net kill |
| 4 | `push_rush` | tui pu qiu, pu qiu | --- |
| 5 | `clear` | chang qiu | attacking clear |
| 6 | `slice_drop` | qie qiu | slice drop shot |
| 7 | `net_drop` | fang xiao qiu | stop drop shot, blocked drop shot |
| 8 | `transition` | guo du qiu, guo du qie qiu | --- |
| 9 | `drive` | ping qiu, hou chang chou ping qiu, etc. | high/flat drive |
| 10 | `block` | dang xiao qiu | high block |
| 11 | `lob_lift` | tiao qiu | flat lift, high lift |
| 12 | `defensive_lift` | fang shou hui tiao | --- |
| 13 | `cross_net` | gou qiu | cross-court net shot |
| 14 | `net_shot` | wang qian qiu | spinning net |
| 15 | `smash_defense` | jie sha fang shou | --- |
| 16 | `push` | tui qiu | push shot |

*Table A2: ShuttleSet shot type distribution (20 extracted matches, N ~ 17,000).*

| Shot Type | English | Count | % |
|---|---|---|---|
| fang xiao qiu | Net Drop | 3,823 | 18.0% |
| tiao qiu | Lift | 3,159 | 14.9% |
| dang xiao qiu | Block | 2,145 | 10.1% |
| tui qiu | Push | 1,686 | 8.0% |
| chang qiu | Clear | 1,609 | 7.6% |
| sha qiu | Smash | 1,405 | 6.6% |
| fa duan qiu | Short Serve | 1,328 | 6.3% |
| qie qiu | Slice/Cut | 1,208 | 5.7% |
| dian kou | Tap Smash | 1,013 | 4.8% |
| gou qiu | Cross-Net | 842 | 4.0% |
| guo du qie qiu | Transitional Slice | 787 | 3.7% |
| wei zhi qiu zhong | Unknown | 615 | 2.9% |
| ping qiu | Drive | 423 | 2.0% |
| pu qiu | Rush | 280 | 1.3% |
| hou chang chou ping qiu | Backcourt Drive | 253 | 1.2% |
| fang shou hui chou | Defensive Drive | 238 | 1.1% |
| fang shou hui tiao | Defensive Lift | 174 | 0.8% |
| fa chang qiu | Long Serve | 164 | 0.8% |
| xiao ping qiu | Half Smash | 39 | 0.2% |

## Appendix B: ShuttleSet Data Split

*Table B1: ShuttleSet match-level split (11 matches with GDINO skeletons).*

| Split | Match | Shots |
|---|---|---|
| **Train** | Anders Antonsen vs Viktor Axelsen (HSBC BWF WTF 2020 Finals) | 895 |
| Train | Anthony Sinisuka Ginting vs Anders Antonsen (Indonesia Masters 2020 Final) | 1,424 |
| Train | Anthony Sinisuka Ginting vs Viktor Axelsen (Indonesia Masters 2020 SF) | 506 |
| Train | CHOU Tien Chen vs Anders Antonsen (Fuzhou Open 2019 SF) | 704 |
| Train | Ng Ka Long Angus vs Lee Cheuk Yiu (YONEX Thailand Open 2021 QF) | 829 |
| Train | Viktor Axelsen vs SHI Yu Qi (All England Open 2020 QF) | 312 |
| Train | Viktor Axelsen vs CHEN Long (Malaysia Masters 2020 QF) | 910 |
| Train | Viktor Axelsen vs NG Ka Long Angus (Malaysia Masters 2020 SF) | 569 |
| **Test** | Anthony Sinisuka Ginting vs Rasmus Gemke (YONEX Thailand Open 2021 QF) | 1,127 |
| **Held-out** | Anthony Sinisuka Ginting vs Lee Zii Jia (HSBC BWF WTF 2020 QF) | 815 |
| Held-out | CHEN Long vs CHOU Tien Chen (World Tour Finals Group Stage) | 860 |

Rationale: held-out matches feature different player pairings and tournaments from the training set. All train matches are men's singles broadcast. The test match is used for SSL checkpoint selection only (monitoring loss, never for strategy evaluation).

## Appendix C: Hyperparameter Summary

*Table C1: Complete hyperparameter settings.*

| Component | Parameter | Value |
|---|---|---|
| **Data** | Shot window (T) | 32 frames |
| | FineBadminton FPS | ~20 |
| | ShuttleSet effective FPS | ~10 (stride-4) |
| | Number of folds | 5 |
| | Random seed | 42 |
| **ST-GCN** | Input channels | 9 (L2) |
| | Nodes | 34 |
| | Layers | 9 |
| | Base channels | 64 |
| | Embedding dim | 256 |
| | Temporal kernel | 9 |
| | Dropout | 0.3 |
| **SSL** | Temperature tau | 0.07 |
| | Projection dim | 128 |
| | Auxiliary weight | 0.3 |
| | Batch size | 64 |
| | Learning rate | 1e-3 |
| | Epochs | 100 (early stopping, patience=10) |
| **ProtoNet** | N-way | 5 |
| | K-shot | 10 |
| | N-query | 5 |
| | Distance | Euclidean |
| | Episodes/epoch | 100 |
| | Epochs | 50 |
| | Learning rate | 1e-4 |
| | Encoder LR scale | 0.1 |
| **Augmentation** | Jitter sigma | 0.02 |
| | Speed range | +/- 20% |
| | Rotation | +/- 15 degrees |
| | Mask ratio | 0.15 |