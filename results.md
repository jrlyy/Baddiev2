# Experiment Results

## Overview

| Phase | Status | Macro-F1 |
|---|---|---|
| Phase B — Random init + ProtoNet (baseline) | ✅ Done | 36.9% ± 3.3% |
| Phase A — SSL pre-training (SimCLR + Aux on ShuttleSet) | ✅ Done | — (checkpoint: `ssl_pretrained_L2.pt`) |
| Phase A→B — SSL init + ProtoNet | ⏳ Pending (ablation notebook) | TBD |
| Step 1a — Feature layer ablation (L0–L3) | ⏳ Pending | TBD |
| Step 1b — Graph structure ablation | ⏳ Pending | TBD |
| Step 2 — Encoder architecture ablation | ⏳ Pending | TBD |
| Step 3 — Pre-training regime ablation (RQ2) | ⏳ Pending | TBD |
| Step 4 — Few-shot classifier ablation | ✅ Done (preliminary) | ProtoNet best (55.9%) |
| Step 5 — K-shot sensitivity | ⏳ Pending | TBD |

---

## Phase B — Random Init + ProtoNet (5-fold CV)

### Data

| Setting | Value |
|---|---|
| Skeleton source | FineBadminton GDINO v2 |
| Dataset | FineBadminton labeled (296 shots, 5 strategy classes) |
| Matches | 11 (6 MS, 5 WS) across 40 rallies |
| Encoder init | Random (no SSL pre-training) |

**Class distribution:**

| Class | Count | % |
|---|---:|---:|
| intercept | 108 | 36.5% |
| create_depth | 61 | 20.6% |
| defensive | 59 | 19.9% |
| passive | 50 | 16.9% |
| move_to_net | 18 | 6.1% |

### Model Architecture

| Component | Setting |
|---|---|
| Encoder | ST-GCN |
| Feature layer | L2 (joint coords + bone vectors + relative angles, 9 features/node) |
| Input shape | (B, 9, 16, 34) |
| Num nodes | 34 (17 joints × 2 players) |
| ST-GCN layers | 9 |
| Base channels | 64 |
| Embedding dim | 256 |
| Temporal kernel | 9 |
| Dropout | 0.3 |
| Inter-player edges | Yes |
| Total parameters | 3,083,199 |

### Training Config

| Hyperparameter | Value |
|---|---|
| Evaluation | 5-way, 5-fold stratified CV |
| Split granularity | Shot-level — all 11 matches present in every fold |
| Fold sizes | Train ≈177, Val ≈60, Test ≈59 shots |
| N-way | 5 |
| K-shot | 10 |
| N-query | 5 |
| Distance | Euclidean |
| Episodes per epoch | 100 |
| Epochs | 50 |
| Fine-tune encoder | True |
| Optimizer | Adam |

> **Split note:** Stratification is by strategy label (StratifiedKFold). Because the split is shot-level, train and test share the same players and match contexts — an optimistic evaluation. Match-level leave-one-out would be cleaner but noisy (~27 test shots per fold).

### Results

**Overall Macro-F1 = 36.9% ± 3.3%**

| Fold | Macro-F1 | intercept | defensive | move_to_net | create_depth | passive |
|------|----------:|----------:|----------:|------------:|-------------:|--------:|
| 1    | 40.9%    | 42.1%    | 50.0%    | 18.2%      | 50.0%       | 44.4%  |
| 2    | 37.5%    | 47.4%    | 48.3%    | 0.0%       | 66.7%       | 25.0%  |
| 3    | 33.3%    | 50.0%    | 38.5%    | 0.0%       | 69.2%       | 8.7%   |
| 4    | 33.1%    | 62.2%    | 27.3%    | 0.0%       | 47.6%       | 28.6%  |
| 5    | 39.9%    | 59.5%    | 44.4%    | 0.0%       | 57.1%       | 38.5%  |
| **Mean** | **36.9%** | **52.2%** | **41.7%** | **3.6%** | **58.1%** | **29.0%** |
| **Std**  | **3.3%**  | **8.4%**  | **8.5%**  | **7.3%**  | **9.1%**   | **13.1%** |

> `move_to_net` is severely underrepresented (18/296 shots = 6%) and is effectively unlearnable without pre-training.

### Training Curves

Per-epoch metrics (every 10 epochs; best val-F1 checkpoint used for test evaluation):

| Epoch | F1 Loss | F2 Loss | F3 Loss | F4 Loss | F5 Loss | F1 Acc | F2 Acc | F3 Acc | F4 Acc | F5 Acc |
|------:|--------:|--------:|--------:|--------:|--------:|-------:|-------:|-------:|-------:|-------:|
| 10    | 0.462   | 0.491   | 0.367   | 0.374   | 0.374   | 0.822  | 0.821  | 0.864  | 0.861  | 0.856  |
| 20    | 0.195   | 0.202   | 0.129   | 0.198   | 0.117   | 0.928  | 0.935  | 0.958  | 0.923  | 0.962  |
| 30    | 0.097   | 0.096   | 0.081   | 0.128   | 0.062   | 0.970  | 0.966  | 0.969  | 0.957  | 0.981  |
| 40    | 0.066   | 0.068   | 0.043   | 0.081   | 0.034   | 0.977  | 0.976  | 0.986  | 0.972  | 0.989  |
| 50    | 0.049   | 0.041   | 0.034   | 0.040   | 0.022   | 0.983  | 0.986  | 0.990  | 0.988  | 0.991  |

Validation Macro-F1 per fold:

| Fold | Best Val-F1 | At epoch |
|------|------------:|---------:|
| 1    | 51.9%       | 10       |
| 2    | 33.7%       | 20       |
| 3    | 32.6%       | 10       |
| 4    | 48.6%       | 30       |
| 5    | 39.6%       | 40       |

> Episodic train accuracy converges near 99% by epoch 50 (strong in-episode fitting), while validation F1 peaks early (epochs 10–30) and then stagnates — generalization is limited at 177 training shots.

### Confusion Matrix

Key patterns from averaged 5-fold confusion matrix:
- `intercept` (majority class, 36.5%) is the most reliably predicted
- `create_depth` well-separated (F1 ~58%), second-best
- `move_to_net` (6.1% of shots) almost never predicted correctly (F1 ~4%)
- `passive` has high variance across folds (F1 range 8.7%–44.4%)

### Classifier Comparison (Frozen Encoder, SSL-init weights from notebook 04)

Same 5-fold splits, embeddings from SSL pre-trained encoder (not episodically fine-tuned):

| Classifier | Macro-F1 | Std |
|---|---:|---:|
| **ProtoNet (nearest centroid)** | **55.9%** | **8.9%** |
| k-NN (k=5) | 54.5% | 9.4% |
| k-NN (k=3) | 53.5% | 11.7% |
| Linear probe (logistic) | 51.5% | 9.5% |

> These numbers use a single SSL-pretrained encoder across all folds (not per-fold fine-tuned checkpoints) — not directly comparable to the 36.9% training-loop result. ProtoNet is consistently the best classifier.

### Saved Artefacts

| File | Description |
|---|---|
| `models/fewshot_L2.pt` | Best-fold encoder + prototypes |
| `results/fewshot_results.json` | Full CV results + classifier comparison |
| `results/fewshot_training_curves.png` | Loss / accuracy / val-F1 per fold |
| `results/fewshot_confusion_matrix.png` | Averaged 5-fold confusion matrix |

---

## Phase A — SSL Pre-training (SimCLR + Aux on ShuttleSet)

### Data

| Setting | Value |
|---|---|
| Match | Kento MOMOTA vs CHOU Tien Chen, Fuzhou Open 2019 Finals |
| Skeleton source | Per-rally GDINO-guided extraction (`shuttleset_skeletons_gdino/`) |
| Skeleton format | `r{rally:04d}.npy` shape `(2, T_rally, 34)` |
| Shot windows | 1,644 shots sliced at hit_frame ± 8 frames (T=16) |
| Shots with aux label | 1,495 / 1,644 (91%) |

### Training Config

| Hyperparameter | Value |
|---|---|
| Epochs | 100 |
| Batch size | 64 |
| Optimiser | AdamW, lr=1e-3, wd=1e-5 |
| NT-Xent temperature | 0.07 |
| Auxiliary weight | 0.3 |
| Augmentations | jitter σ=0.01, joint mask 15%, temporal crop 80%, rotation ±15° |
| Device | Apple MPS (M-series), ~57 min total |

### Training Curves

| Epoch | Total Loss | CL Loss | Aux Loss |
|---|---|---|---|
| 10 | 1.1023 | 0.4978 | 2.0150 |
| 20 | 0.9587 | 0.3787 | 1.9331 |
| 30 | 0.8969 | 0.3324 | 1.8816 |
| 40 | 0.8566 | 0.3038 | 1.8427 |
| 50 | 0.7802 | 0.2472 | 1.7767 |
| 60 | 0.7908 | 0.2607 | 1.7670 |
| 70 | 0.7454 | 0.2348 | 1.7019 |
| 80 | 0.7551 | 0.2463 | 1.6962 |
| 100 | 0.7052 (total) | — | — |

Random-chance CL loss reference: ~4.1 (log(63), batch=64, τ=0.07)
Random-chance Aux loss reference: ~2.83 (−log(1/17))

CL loss dropped ~53% from epoch 10 to 100 — well below random baseline. Checkpoint saved: `models/ssl_pretrained_L2.pt`.

---

## Ablation Experiments (Notebook 05 — Pending Colab Run)

Sequential ablation: each step fixes the best setting from the previous step.

### Step 1a — Feature Engineering

**Question:** Which feature layer carries the most strategy-relevant information?

| Variant | Layer | Dim | Features added | Init | Result |
|---------|-------|----:|----------------|------|--------|
| L0_raw_xy | L0 | 2 | x, y joint coords | random | TBD |
| L1_kinematics | L1 | 6 | + velocity, acceleration | random | TBD |
| L2_court_ctx | L2 | 9 | + dist_to_net, dist_to_center, dist_to_opponent | SSL | TBD |
| L3_joint_angles | L3 | 12 | + elbow, shoulder, knee angles | random | TBD |

> Note: SSL checkpoint only exists for L2. L0/L1/L3 use random init, making this a mixed comparison. The L2 advantage may include SSL benefit — a known limitation.

### Step 1b — Graph Structure

**Question:** Does the opponent skeleton matter?

| Variant | Nodes | Inter-player edges | Result |
|---------|------:|-------------------|--------|
| full_dual | 34 | Yes | TBD |
| no_inter_edges | 34 | No | TBD |
| single_player | 17 | — | TBD |

> `spatial_only` and `temporal_only` variants are excluded — the ST-GCN architecture does not support disabling its graph conv or temporal conv layers without a separate model variant.

### Step 2 — Encoder Architecture

**Question:** Does the graph inductive bias of ST-GCN outperform generic sequence models?

| Encoder | Type | Params | Result |
|---------|------|-------:|--------|
| ST-GCN | Spatial-temporal GCN, 9 blocks | 3.08M | TBD |
| Transformer | BST-style self-attention, 4 layers | ~1.5M | TBD |
| LSTM | Bidirectional, 2 layers | 2.86M | TBD |
| 1D-CNN | 3 temporal conv blocks | 0.48M | TBD |

### Step 3 — Pre-Training Regime (RQ2)

**Question:** Does SSL pre-training on unlabeled ShuttleSet data improve few-shot accuracy?

| Variant | SSL corpus | Result |
|---------|-----------|--------|
| random_init | None | TBD (baseline: 36.9%) |
| ssl_plus_aux | 1,644 shots (Momota/Chou) | TBD |

> A `ssl_contrastive_only` (no aux task) variant would require a separate pre-training run with `auxiliary_weight=0`. Skipped unless time permits.

### Step 4 — Few-Shot Classifier

**Question:** Is ProtoNet the best classifier for this embedding space?

| Classifier | Macro-F1 (preliminary) | Std |
|---|---:|---:|
| **ProtoNet** | **55.9%** | **8.9%** |
| k-NN (k=5) | 54.5% | 9.4% |
| k-NN (k=3) | 53.5% | 11.7% |
| Linear probe | 51.5% | 9.5% |

> Preliminary result from notebook 04 using SSL-init frozen encoder (not episodically fine-tuned). Step 4 in notebook 05 will re-run with the best encoder from Steps 1–3.

### Step 5 — K-Shot Sensitivity

**Question:** How many labeled support shots are needed?

| K | n_query | Result |
|---|---------|--------|
| 1 | 5 | TBD |
| 3 | 5 | TBD |
| 5 | 5 | TBD |
| 8 | 3 | TBD |
| 10 | 1 | TBD |

> K capped at 10 (n_query ≥ 1): `move_to_net` has only ~11 training samples per fold.

---

## Data Split Limitation

The current shot-level 5-fold CV is the only feasible approach at 296 labeled shots. For context:
- Match-level leave-one-out = 11 folds × ~27 test shots → very noisy per-class estimates
- A clean train/test match split (e.g. 8 train / 3 test matches) would give ~216 train / ~80 test shots — feasible but only a single evaluation point