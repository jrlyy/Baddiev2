# Run 4 Ablation Results

Sequential ablation: A → B → C → D. Each group's winner becomes the baseline for the next group.

---

## Group A: Skeleton Representation

Baseline: dual-player L2 (9-dim), fixed T=32, no shuttle, linear head.

| # | Ablation | Acc | Macro-F1 | Top-3 | Top-5 | Epochs |
|---|----------|-----|----------|-------|-------|--------|
| A1 | Dual-player, L2 (9-dim) | 0.6060 | **0.5461** | 0.8611 | 0.9121 | 72 ✓ |
| A2 | Dual-player, L3 (12-dim) | 0.5526 | 0.5216 | 0.8955 | 0.9502 | 28 |
| A3 | Single-player, L2 (9-dim) | 0.5759 | 0.5037 | 0.8679 | 0.9379 | 35 |
| A4 | Single-player, L3 (12-dim) | 0.5710 | 0.5158 | 0.8654 | 0.9373 | 42 |
| A5 | Dual-player, L2 + hitter (10-dim) | **0.6257** | 0.5440 | 0.8820 | 0.9459 | 51 |
| A6 | Dual-player, L3 + hitter (13-dim) | 0.5655 | 0.4978 | 0.8832 | 0.9428 | 26 |

**Winner: A1** — dual-player L2, no hitter channel. Higher accuracy from hitter (A5) doesn't translate to better F1.

---

## Group B: Temporal Window

Baseline: A1 config (dual-player L2, fixed T=32).

| # | Ablation | Acc | Macro-F1 | Top-3 | Top-5 | Epochs |
|---|----------|-----|----------|-------|-------|--------|
| B1 | Fixed T=32 (A1 repeat) | 0.6060 | **0.5461** | 0.8611 | 0.9121 | 72 ✓ |
| B2 | Variable window (hit-frame boundaries) | 0.5575 | 0.4758 | 0.8377 | 0.9201 | 34 |

**Winner: B1** — fixed window. Variable window hurts across all metrics.

---

## Group C: Shuttle Fusion

Baseline: A1 + B1 (dual-player L2, fixed T=32, no shuttle).

| # | Ablation | Acc | Macro-F1 | Top-3 | Top-5 | Epochs |
|---|----------|-----|----------|-------|-------|--------|
| C1 | No shuttle (B1 repeat) | 0.6060 | 0.5461 | 0.8611 | 0.9121 | 72 |
| C2 | + Shuttle (graph node) | 0.5962 | 0.5292 | 0.8814 | 0.9490 | 26 |
| C3 | + Shuttle (cross-attention) | **0.6189** | **0.5929** | **0.9060** | **0.9607** | 27 ✓ |

**Winner: C3** — shuttle cross-attention. +0.047 F1 over no-shuttle baseline. Graph node fusion hurts.

---

## Group D: Classifier Head

Baseline: C3 (dual-player L2, fixed T=32, shuttle cross-attn, linear head).

| # | Ablation | Acc | Macro-F1 | Top-3 | Top-5 | Epochs |
|---|----------|-----|----------|-------|-------|--------|
| D1 | Linear (C3 repeat) | 0.6189 | **0.5929** | 0.9060 | 0.9607 | 72 ✓ |
| D2 | MLP-1 (256→128→17) | 0.6263 | 0.5759 | 0.9023 | 0.9478 | 33 |
| D3 | MLP-2 (256→128→64→17) | **0.6281** | 0.5716 | **0.9133** | **0.9668** | 24 |

**Winner: D1** — linear head. MLP variants improve accuracy slightly but drop macro-F1 by ~0.02.

---

## Final Best Config (Run 4)

> Dual-player L2 skeleton · Fixed T=32 window · Shuttle cross-attention · Linear classifier head

| Metric | Value |
|--------|-------|
| Test Accuracy | 0.6189 |
| Test Macro-F1 | 0.5929 |
| Top-3 Accuracy | 0.9060 |
| Top-5 Accuracy | 0.9607 |

---

## Per-Class F1 — Final Config (C3 / D1)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| short_serve | 0.82 | 0.99 | 0.90 | 133 |
| long_serve | 0.80 | 1.00 | 0.89 | 4 |
| clear | 0.97 | 0.88 | 0.92 | 129 |
| net_drop | 0.86 | 0.68 | 0.76 | 296 |
| block | 0.82 | 0.55 | 0.65 | 163 |
| transition | 0.57 | 0.60 | 0.59 | 88 |
| lob_lift | 0.47 | 0.86 | 0.61 | 137 |
| slice_drop | 0.41 | 0.56 | 0.48 | 61 |
| cross_net | 0.33 | 0.81 | 0.47 | 63 |
| push | 0.62 | 0.40 | 0.49 | 226 |
| tap_smash | 0.33 | 0.60 | 0.43 | 75 |
| smash | 0.65 | 0.29 | 0.40 | 107 |
| push_rush | 0.40 | 0.55 | 0.46 | 31 |
| drive | 0.65 | 0.23 | 0.34 | 106 |
| defensive_lift | 0.45 | 0.62 | 0.53 | 8 |

---

## Diagnostic Notes

- **Class imbalance**: 28.7x ratio (largest 2753 vs smallest 96 samples)
- **Shuttle coverage**: 87.8% of samples have trajectory data
- **Overfitting**: Train loss → 0.0077, val F1 → 0.5442, test F1 → 0.5461 (gap OK)
- **Top confusion pairs**:
  - push → lob_lift: 163 samples (72% of push misclassified)
  - smash → tap_smash: 39 (36%)
  - transition → slice_drop: 33 (38%)
  - tap_smash → smash: 28 (37%)
  - net_drop → lob_lift: 22 (7%)
- **Worst classes by F1**: push (0.29 on A1), drive (0.29), defensive_lift (0.30), push_rush (0.34), tap_smash (0.40)

---

# Run 5 Ablation Results

Run 5 inserts two new groups into the Run 4 cascade: **A+** (feature representation) and **B+** (pooling strategy), then re-runs shuttle fusion as **C+**.

> **Note:** Run 5 uses an expanded **18-class** mapping (vs 15 in Run 4), so scores are not directly comparable across runs. The macro-F1 values are lower partly because there are more (and harder) classes.

Cascade: **A1 (run4 winner) → A+ → B+ → C+**

---

## Group A+: Feature Representation

Baseline: A7 = A1 reproduction on 18-class mapping (dual-player L2, fixed T=32, no bones, no bbox norm).

| # | Ablation | Acc | Macro-F1 | Top-3 | Top-5 | Epochs | Status |
|---|----------|-----|----------|-------|-------|--------|--------|
| A7 | Baseline (L2, no bones, no bbox) | 0.5882 | 0.4338 | 0.8513 | 0.9090 | 53 | ✓ |
| A8 | + Bone vectors (11-dim) | **0.6398** | **0.4904** | **0.8937** | **0.9490** | 33 ✓ | ✓ |
| A9 | + BBox normalization only | 0.5759 | 0.4487 | 0.8826 | 0.9299 | 45 | ✓ |
| A10 | + Bones + BBox norm | — | — | — | — | — | incomplete |

**Partial winner: A8** — bone vectors help (+0.057 F1 over baseline). BBox norm alone (A9) gives marginal F1 gain (+0.015) but hurts accuracy. A10 not yet complete.

---

## Group B+: Pooling Strategy

*Pending A+ winner. Will test attention pooling vs mean pooling on top of A+ winner config.*

| # | Ablation | Acc | Macro-F1 | Top-3 | Top-5 | Epochs | Status |
|---|----------|-----|----------|-------|-------|--------|--------|
| B3 | Attention pooling + A+ winner | — | — | — | — | — | not started |
| B4 | Max pooling + A+ winner | — | — | — | — | — | not started |

---

## Group C+: Full Cascade

*Pending B+ winner. Will re-run shuttle cross-attention with A+ + B+ improvements.*

| # | Ablation | Acc | Macro-F1 | Top-3 | Top-5 | Epochs | Status |
|---|----------|-----|----------|-------|-------|--------|--------|
| C4 | A+ winner + B+ winner + shuttle cross-attn | — | — | — | — | — | not started |

---

## Per-Class F1 — Run 5 Completed Variants

| Class | A7 baseline | A8 + bones | A9 + bbox norm |
|-------|-------------|------------|----------------|
| net shot | 0.80 | **0.82** | 0.78 |
| return net | 0.65 | **0.72** | 0.62 |
| smash | 0.49 | 0.28 | **0.42** |
| wrist smash | 0.40 | 0.46 | 0.33 |
| lob | 0.54 | **0.61** | 0.55 |
| defensive return lob | 0.15 | **0.21** | 0.33 |
| clear | 0.83 | **0.91** | 0.89 |
| drive | 0.10 | **0.27** | 0.20 |
| driven flight | 0.00 | 0.00 | 0.00 |
| back-court drive | 0.24 | **0.24** | 0.15 |
| drop | 0.45 | **0.59** | 0.41 |
| passive drop | 0.53 | **0.68** | 0.50 |
| push | 0.25 | **0.45** | 0.37 |
| rush | 0.46 | **0.58** | 0.36 |
| defensive return drive | 0.06 | 0.06 | **0.29** |
| cross-court net shot | 0.31 | **0.37** | 0.43 |
| short service | **0.96** | 0.91 | 0.89 |
| long service | 0.60 | **0.67** | 0.57 |