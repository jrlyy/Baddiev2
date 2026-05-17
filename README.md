# Baddiev2

Badminton shot-type classification from broadcast video — supervised ST-GCN
on player skeletons with shuttle cross-attention.

This repository is organised into three top-level sections:

## 1. [`project/`](project/) — The project

All code, data, notebooks, results, and figures.

- [`project/src/`](project/src/) — model, data, and training code
- [`project/notebooks/`](project/notebooks/) — EDA, extraction, training, analysis
- [`project/datasets_preprocessing/`](project/datasets_preprocessing/) — extracted skeletons, shuttles, court homographies
- [`project/results/`](project/results/) — ablation outputs (run6, run7)
- [`project/figures/`](project/figures/) — report/slide figures
- [`project/models/`](project/models/) — trained checkpoints
- [`project/badminton_server.py`](project/badminton_server.py) — visualization UI server (port 7860)
- [`project/README.md`](project/README.md) — full project documentation

Run from inside `project/`:

```bash
cd project
python badminton_server.py
```

## 2. [`report/`](report/) — Final report

A single PDF file containing the final written report.

## 3. [`presentation/`](presentation/) — Final recorded presentation

A single MP4 file containing the recorded presentation.