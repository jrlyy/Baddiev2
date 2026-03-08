# Experiment Results

## Phase B — Random Init + ProtoNet (5-fold CV)

### Data

| Setting | Value |
|---|---|
| Skeleton source | FineBadminton GDINO v2 |
| Dataset | FineBadminton labeled (296 shots, 5 strategy classes) |
| Matches | 11 (6 MS, 5 WS) across 40 rallies |
| Encoder init | **Random** (no SSL pre-training — `ssl_pretrained_L2.pt` not available on Colab run) |

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
| Split granularity | **Shot-level** — all 11 matches present in every fold |
| Fold sizes | Train ≈177, Val ≈60, Test ≈59 shots |
| N-way | 5 |
| K-shot | 10 |
| N-query | 5 |
| Distance | Euclidean |
| Episodes per epoch | 100 |
| Epochs | 50 |
| Fine-tune encoder | True |
| Optimizer | Adam |

> **Split note:** Stratification is by strategy label (StratifiedKFold), so class proportions are balanced across folds. However, because the split is shot-level, train and test share the same players and match contexts. A match-level leave-one-out would give a cleaner but noisier evaluation (~27 test shots per fold).

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
| **Std** | **3.3%** | **8.4%** | **8.5%** | **7.3%** | **9.1%** | **13.1%** |

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

Validation Macro-F1 per fold (best checkpoint epoch in parentheses):

| Fold | Best Val-F1 | At epoch |
|------|------------:|---------:|
| 1    | 51.9%       | 10       |
| 2    | 33.7%       | 20       |
| 3    | 32.6%       | 10       |
| 4    | 48.6%       | 30       |
| 5    | 39.6%       | 40       |

Training curves saved to: `results/fewshot_training_curves.png`

> Episodic train accuracy converges near 99% by epoch 50 (strong in-episode fitting), while validation F1 peaks early (epochs 10–30) and then stagnates — generalization is limited at 177 training shots.

### Confusion Matrix

Averaged confusion matrix across 5 folds saved to: `results/fewshot_confusion_matrix.png`

Key patterns:
- `intercept` (majority class, 36.5%) is the most reliably predicted
- `create_depth` well-separated (F1 ~58%), second-best
- `move_to_net` (6.1% of shots) almost never predicted correctly (F1 ~4%)
- `passive` has high variance across folds (F1 range 8.7%–44.4%)

### Classifier Comparison (Frozen Encoder, Last-Fold Weights)

Same 5-fold splits, embeddings from the last fold's trained encoder:

| Classifier | Macro-F1 | Std |
|---|---:|---:|
| **ProtoNet (nearest centroid)** | **55.9%** | **8.9%** |
| k-NN (k=5) | 54.5% | 9.4% |
| k-NN (k=3) | 53.5% | 11.7% |
| Linear probe (logistic) | 51.5% | 9.5% |

> These numbers are higher than the 36.9% training-loop result because they use a single final encoder across all folds rather than per-fold best checkpoints — not directly comparable. ProtoNet is the best classifier.

### Saved Artefacts

| File | Description |
|---|---|
| `models/fewshot_L2.pt` | Best-fold encoder + prototypes |
| `results/fewshot_results.json` | Full CV results + classifier comparison |
| `results/fewshot_training_curves.png` | Loss / accuracy / val-F1 per fold |
| `results/fewshot_confusion_matrix.png` | Averaged 5-fold confusion matrix |

---

## Phase A: SSL Pre-training — SimCLR on ShuttleSet

### Data

| Setting | Value |
|---|---|
| Match | Kento MOMOTA vs CHOU Tien Chen, Fuzhou Open 2019 Finals |
| Skeleton source | Per-rally GDINO-guided extraction (`shuttleset_skeletons_gdino/`) |
| Skeleton format | `r{rally:04d}.npy` shape `(2, T_rally, 34)` |
| Shot windows | 1,644 shots sliced at hit_frame ± 8 frames (T=16) |
| Auxiliary labels | 1,495 / 1,644 shots have a mapped shot type (149 unmapped → skipped in aux loss) |
| Shots with aux label | 1,495 / 1,644 (91%) |

### Model Architecture

| Component | Setting |
|---|---|
| Encoder | ST-GCN |
| Feature layer | L2 (9 features/node) |
| Input shape | (B, 9, 16, 34) |
| Num nodes | 34 (17 joints × 2 players) |
| ST-GCN layers | 9 |
| Base channels | 64 |
| Embedding dim | 256 |
| Temporal kernel | 9 |
| Dropout | 0.3 |
| Inter-player edges | Yes |
| Projection head | Linear(256→256→128) |
| Aux head | Linear(256→18 shot types) |

### Training Config

| Hyperparameter | Value |
|---|---|
| Epochs | 100 |
| Batch size | 64 (25 batches/epoch) |
| Optimiser | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-5 |
| NT-Xent temperature | 0.07 |
| Auxiliary weight | 0.3 |
| Jitter std | 0.01 |
| Joint mask ratio | 0.15 |
| Temporal crop ratio | 0.8 |
| Rotation range | ±15° |
| Device | Apple MPS (M-series) |
| Time per epoch | ~35–42s |
| Total time | ~57 min |

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
| 100 | TBD | TBD | TBD |

Random-chance CL loss reference: ~4.1 (log(63) with batch=64, τ=0.07)
Random-chance Aux loss reference: ~2.83 (−log(1/17) for 17 classes)

### Observations

- CL loss dropped ~53% from epoch 10 to 80 (0.498 → 0.235), well below random baseline, indicating strong contrastive structure
- Aux loss still declining at epoch 80 (not plateaued), shot-type signal continues improving
- Minor epoch-to-epoch fluctuation (e.g. epoch 60 > epoch 50) is normal given small batch count (25/epoch)
- Weights saved to `models/ssl_pretrained_L2.pt`

### Limitations

- SSL corpus is a single match (1,644 shots). Representations may be biased toward the Momota/Chou playing style
- GINTING/AXELSEN match (506 shots) skeleton extraction still pending — adding it would increase corpus to ~2,150 shots

---

## Phase A→B: SSL Pre-training + ProtoNet (pending)

To be filled after running notebook 04 on Colab with `ssl_pretrained_L2.pt` as encoder init.

| Setting | Value |
|---|---|
| Encoder init | SSL pretrained on SS (1 match, 1,644 shots) |
| SSL checkpoint | `models/ssl_pretrained_L2.pt` |
| Feature layer | L2 |
| Dataset | FineBadminton labeled (296 shots, 5 strategy classes) |
| Split | Same 5-fold shot-level CV as baseline (seed=42) |

**Result: Macro-F1 = TBD** (baseline to beat: 37.0% ± 3.3%)

---

## Planned Experiments (Next Steps)

### RQ2 — SSL Pre-training Ablation

| Variant | SSL corpus | Expected |
|---------|-----------|----------|
| No pre-training (done) | — | 37.0% |
| SSL on ss01 only | 1,644 shots (Momota/Chou) | TBD |
| SSL on ss01 + ss02 | ~2,150 shots (+ Ginting/Axelsen) | TBD |

### RQ1 — Feature Layer Ablation

| Feature | Dims | Components |
|---------|-----:|-----------|
| L0 | 2 | (x, y) joint coords |
| L1 | 6 | + velocity + acceleration |
| L2 (current) | 9 | + dist_to_net, dist_to_center, dist_to_opponent |
| L3 | 12 | + elbow/shoulder/knee angles |
| L4 (planned) | 15 | + shuttle_dist, shuttle_dx, shuttle_dy |

### Architecture Ablation

| Encoder | Desc |
|---------|------|
| ST-GCN (default) | 9-block spatial-temporal GCN |
| Transformer | BST-style frame-level attention |
| LSTM | Sequence baseline |
| CNN-1D | Temporal baseline (no graph) |

### Data split limitation

The current shot-level 5-fold CV is the only feasible approach at 296 labeled shots. For context:
- Match-level leave-one-out = 11 folds × ~27 test shots → very noisy per-class estimates
- A clean train/test match split (e.g. 8 train / 3 test matches) would give ~216 train / ~80 test shots — feasible but only a single evaluation point. Could be used as a final held-out test after CV tuning.
