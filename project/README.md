# Baddiev2 — Badminton Shot Type Classification

Skeleton-based shot-type classification for men's singles badminton. Trains an ST-GCN encoder on the [ShuttleSet](https://github.com/wywyWang/CoachAI-Projects) dataset and evaluates cross-dataset on [FineBadminton](https://github.com/JenJenChung/FineBadminton). A local QA/inspection tool is included for visually browsing matches, rallies, skeletons, shuttle trajectories, and model predictions.

---

## 1. Project at a glance

**Task.** Classify each badminton shot into one of **15 shot types** (`net shot`, `smash`, `clear`, `drive`, `drop`, `lob`, `short service`, …) from skeleton + shuttle features. Vocabulary is defined in [src/config.py](src/config.py) under `SHOT_TYPES`.

**Data.** Two datasets, used differently:

| Dataset | Role | Active size | Where |
|---|---|---|---|
| **ShuttleSet (SS)** | Train + held-out eval | 19 train + 2 held-out matches, 17 754 shots, men's singles, right-handed | `datasets_preprocessing/shuttleset_*` |
| **FineBadminton (FB)** | Cross-dataset generalization test only | 40 rallies, 413 shots | `Datasets/FineBadminton-dataset/` |

The full inventory (active, excluded, exclusion reasons, FB → SS label mapping, benchmark comparison) is in [data_table.md](data_table.md).

**Pipeline.**

```
video → ffmpeg sparse frames → GroundingDINO bbox → YOLOv8-Pose skeleton (.npy)
                                                 ↘
                                                  TrackNetV4 shuttle (.npy)
                                                 ↗
                                  hit-window segment → feature eng. (L0–L3)
                                  → ST-GCN encoder → (optional) shuttle cross-attn
                                  → MLP head → 15-class softmax
```

**Best result so far.** Run 6, single-player L3 + shuttle cross-attention: **Macro-F1 = 0.6516** on the held-out matches. Full ablation grid in [results/ablations_run6/](results/ablations_run6/) and analysis in [notebooks/04_Analysis/04_analysis_run6v2.ipynb](notebooks/04_Analysis/04_analysis_run6v2.ipynb).

---

## 2. Repository layout

```
.
├── src/                                 # Pipeline code
│   ├── config.py                        # SHOT_TYPES, FEATURE_DIMS, paths
│   ├── data/
│   │   ├── pose_extractor.py            # GDINO + YOLOv8-Pose runner
│   │   ├── graph_builder.py             # COCO-17 → graph adjacency
│   │   ├── feature_eng.py               # L0/L1/L2/L3 feature layers
│   │   └── dataset.py                   # PyTorch Dataset for shot segments
│   ├── models/
│   │   ├── stgcn_model.py               # ST-GCN backbone
│   │   ├── shuttle_cross_attn.py        # Shuttle ↔ skeleton cross-attention
│   │   └── transformer_encoder.py       # Temporal transformer (ablation B5)
│   └── inference_shot_type.py           # Loads any run4/run6 checkpoint, runs on FB
│
├── notebooks/
│   ├── 01_EDA/                          # Dataset stats, class distribution
│   ├── 02A_Skeleton Extraction/         # GDINO + YOLOv8-Pose extraction
│   ├── 02B_Shuttlecock Tracking/        # TrackNetV4 shuttle detection
│   ├── 03_Training/                     # Ablation training (run6 = current)
│   ├── 04_Analysis/                     # Confusion matrices, curves, t-SNE
│   └── 07_court_detection_homography.ipynb
│
├── datasets_preprocessing/              # Extracted skeletons + shuttles + frames
├── datasets/                            # Raw ShuttleSet CSVs, TrackNet code
├── models/run6/                         # Run 6 checkpoints (.pt per ablation)
├── results/ablations_run6/              # Per-ablation metrics JSON + figures
├── badminton_server.py                  # Local server for QA/inspection tool
├── badminton_pipeline_demo.html         # QA tool front-end (React via Babel)
├── badminton_ui.jsx                     # Standalone mockup of the inference UI
└── data_table.md                        # Authoritative dataset inventory
```

---

## 3. Local QA/inspection tool

The tool is a single-file React app served by a small Python HTTP server. It lets you browse every shot in ShuttleSet and FineBadminton, scrub through rallies frame-by-frame, overlay skeletons and shuttle trajectories on the actual broadcast frame, and (for FB) compare model predictions against ground-truth labels.

### 3.1 Run it

```bash
python badminton_server.py            # http://localhost:7860
python badminton_server.py 8080       # custom port
```

The server (see [badminton_server.py](badminton_server.py)) reads directly from the on-disk preprocessed data; no database, no rebuild step. Endpoints live under `/api/...` (FB) and `/api/ss/...` (ShuttleSet) — full list at the top of `do_GET` in the same file.

Open `http://localhost:7860/` in a browser. The HTML demo loads from disk and talks to the same server.

### 3.2 What's in each tab

| Tab | Purpose | Key things to QA |
|---|---|---|
| **🏸 FineBadminton** *(inference)* | Browse FB rallies, run cross-dataset inference, compare predicted vs. ground-truth shot type | Top-K predictions, FB ↔ SS label mapping, low-margin shots flagged for review |
| **📊 ShuttleSet** *(training)* | Browse training/held-out matches and rallies | Per-shot skeleton overlay, shuttle hit-frame position, split filter (train / held_out / unused), shot-type label, rally scrubber |
| **🗺 Match Analysis** | Court-level pipeline visualization, list of SS matches still awaiting GDINO extraction | Pipeline stage status, court homography sanity check |

### 3.3 Typical QA workflows

**Spot-check skeleton quality on a match.**
1. ShuttleSet tab → pick a match from the sidebar.
2. Pick a rally → the rally scrubber appears at the bottom.
3. Toggle the **Skeleton** mode (`original` / `gdino` / `gdino-fallback`) to compare YOLOv8 top-2 vs. GDINO-guided extraction. GDINO eliminates the ~14% chair-umpire contamination from the top-2 baseline.
4. Scrub frame-by-frame — the orange-outlined player is the *hitter* per annotation; skeletons should follow them across hit-window frames.

**Verify hit-frame and hitter assignment.**
- The mini court (top-right) shows P0 (top-court) and P1 (bottom-court) in their two assigned colours, with the hitter ringed in orange.
- For FB rallies, *unlabelled hit windows* show as ticks on the scrubber so you can confirm the dataset doesn't miss hits.

**Inspect shuttle trajectories.**
- ShuttleSet shuttle data is per-rally dense (from TrackNetV4) — open a rally to see the full trajectory overlaid. Format: `{match}_s{set}r{rally}.npy` of shape `(T, 3)` = `[x, y, visible]` + a matching `_frames.npy` of absolute frame numbers.
- FB shuttle is sparse (hit-frame only).

**Run cross-dataset inference on FB.**
1. FineBadminton tab → click **Run inference** (calls `/api/fb/infer_all`, runs the current checkpoint over all 413 FB shots with TTA=5, caches to `results/fb_inference.json`).
2. For each rally, the prediction panel lists Top-K classes with confidence bars, and the FB ground-truth label badge.
3. Shots where the *margin* (top-1 minus top-2 probability) is below 15% are flagged with **⚠ Low margin** — review these first when looking for label noise or hard cases.

### 3.4 Switching which checkpoint the server serves

`badminton_server.py` looks for a checkpoint path passed to `inference_shot_type.ShotTypePredictor`. The current run-6 best is `models/run6/C3_shuttle_xattn.pt`. To swap, edit the path in the server file (search for `ShotTypePredictor(`) and restart. Inference auto-detects `num_classes`, `in_channels`, pooling, and cross-attention from the checkpoint weights — no config changes needed for run4 ↔ run6.

### 3.5 Note on `badminton_ui.jsx`

This is a *standalone mockup* of the inference UI, not the live tool. It uses fake data (`generateMockRally`) and was used to iterate on the look-and-feel before wiring up real endpoints. The shipped UI is the one inside [badminton_pipeline_demo.html](badminton_pipeline_demo.html). Treat the JSX as design source; the HTML is the source of truth.

---

## 4. Reproducing results

### Environment

```bash
# Python 3.10+, install deps used by the notebooks and src/
pip install torch numpy pandas opencv-python ultralytics matplotlib seaborn jupyter
# GroundingDINO + TrackNetV4 are set up inside their respective notebooks
```

### Train a model

Open [notebooks/03_Training/03d_shot_type_ablations_run6.ipynb](notebooks/03_Training/03d_shot_type_ablations_run6.ipynb) — it runs the full A/B/C/D ablation grid (skeleton variants, window sizes, shuttle fusion modes, head depth) and writes checkpoints to `models/run6/` and per-ablation metrics to `results/ablations_run6/`.

### Inference on FineBadminton

```python
from src.inference_shot_type import ShotTypePredictor

predictor = ShotTypePredictor("models/run6/C3_shuttle_xattn.pt")
results = predictor.run_fb_inference(
    json_path="Datasets/FineBadminton-dataset/dataset/transformed_combined_rounds_output_en_evals_translated.json",
    skel_dir="datasets_preprocessing/finebadminton_skeletons_gdino_v2",
    shuttle_dir="datasets_preprocessing/finebadminton_shuttles",
    img_dir="Datasets/FineBadminton-dataset/dataset/image",
)
```

Each result dict has `rally_id`, `hit_frame`, `hitter`, `predicted`, `confidence`, `top5`, `fb_label`. The QA tool reads the cached `results/fb_inference.json` directly.

### Analyse a run

[notebooks/04_Analysis/04_analysis_run6v2.ipynb](notebooks/04_Analysis/04_analysis_run6v2.ipynb) loads `results/ablations_run6/*.json` and produces the per-class heatmap, training curves, confusion matrix, t-SNE, and cross-attention visualisations seen in the report.

---

## 5. Conventions worth knowing

- **Skeleton tensor layout:** `.npy` of shape `(2, T, 34)` — `dim 0` = `[x, y]`, `dim 1` = frame, `dim 2` = 34 joints (17 per player, COCO format). **P0 = top-court player** (smaller image-Y), **P1 = bottom-court**.
- **Rally UID format:** `s{set}r{rally}`. Used in shuttle filenames (`{match}_s1r3.npy`) and the `/api/ss/rally/{match}/{uid}` endpoint.
- **Hitter encoding:** annotation says `"top"` or `"bottom"` — these map to P0 / P1 respectively. The QA tool rings the hitter in orange.
- **Why two skeleton sources:** `original/` = YOLOv8 top-2 by confidence (picks chair umpire ~14% of the time). `gdino/` = GroundingDINO bbox prompt `"badminton player ."` → YOLOv8-Pose inside that bbox; eliminates umpire contamination. Always prefer `gdino` for analysis.
- **Frame resolutions:** ShuttleSet is **1920×1080**, FineBadminton is **1280×720**. Per-match homographies (`H_img_to_court_m.npy`) live in `datasets_preprocessing/court_homographies/`.

---

## 6. Where to look when something is off

| Symptom | First place to check |
|---|---|
| Skeletons look wrong on one match | QA tool → switch to `original` mode; if `original` is fine and `gdino` is wrong, the GDINO bbox prompt missed. Re-run [notebooks/02A_Skeleton Extraction/reextract_bad_frames.ipynb](notebooks/02A_Skeleton%20Extraction/reextract_bad_frames.ipynb). |
| Court overlay misaligned | Re-run [notebooks/07_court_detection_homography.ipynb](notebooks/07_court_detection_homography.ipynb); the server picks up `H_img_to_court_m.npy` and `ss_per_match_H.npy` on restart. |
| FB inference gives garbage | Check checkpoint vocab matches FB label map: `_VOCAB` and `_FB_LABEL_MAP` in [src/inference_shot_type.py](src/inference_shot_type.py). Run 4 = 17 cls, Run 6 = 15 cls. |
| QA tab says "Server offline" | `python badminton_server.py` not running, or wrong port — the HTML hard-codes `SERVER = http://localhost:7860`. |
