# Run 6 — training & inference pipeline

Training and inference share the same preprocessing front-end. The two paths
diverge only after feature engineering: training fits the model and emits a
checkpoint; inference loads that checkpoint and predicts. The checkpoint is the
single artifact that links them.

```mermaid
flowchart TD
    %% ---------- Shared preprocessing front-end ----------
    subgraph PRE["Shared preprocessing"]
        direction TB
        VID["Match video<br/>(ShuttleSet)"] --> FR["Frame extraction<br/>1920×1080 JPEG q90"]
        FR --> POSE["Player detection + pose<br/>GroundingDINO 'badminton player' → keypoints"]
        FR --> SHUT["Shuttle tracking<br/>TrackNetV4"]
        POSE --> SKEL["Skeletons (2, T, 34)<br/>P0 top / P1 bottom court"]
        SHUT --> TRAJ["Shuttle trajectories<br/>dense per-rally (T, 3)"]
    end

    %% ---------- Sample construction ----------
    SKEL --> WIN["Window extraction<br/>±16 frames around hit → T=32"]
    TRAJ --> WIN
    META["Shot metadata<br/>shuttleset_outputs/*.json"] --> WIN
    WIN --> FEAT["Feature engineering<br/>single hitter V=17 · L3 14-dim"]

    %% ---------- Training path ----------
    subgraph TRAIN["Training path"]
        direction TB
        FEAT --> SPLIT["Train / val / held-out split<br/>19 train + 2 held-out matches"]
        SPLIT --> AUG["Augmentation<br/>(train split only)"]
        AUG --> MODEL["Model: ST-GCN encoder + mean pool<br/>+ shuttle cross-attention + FC head"]
        MODEL --> LOOP["Training loop<br/>cross-entropy · early stop on val macro-F1"]
        LOOP --> CKPT[("Checkpoint<br/>C3_shuttle_xattn.pt")]
    end

    %% ---------- Inference path ----------
    subgraph INFER["Inference path"]
        direction TB
        NEWVID["New rally video"] -.same front-end.-> PRE
        FEAT2["Feature engineering<br/>single hitter V=17 · L3 14-dim"] --> LOAD["Load checkpoint<br/>auto-detect in_ch / classes / pooling"]
        LOAD --> FWD["Forward pass<br/>encoder → cross-attn → FC"]
        FWD --> SOFT["Softmax"]
        SOFT --> PRED["Shot-type prediction<br/>top-1 + top-5"]
    end

    FEAT --> FEAT2
    CKPT --> LOAD

    %% ---------- Styling ----------
    classDef pre   fill:#EEEDFE,stroke:#534AB7,color:#26215C;
    classDef train fill:#E1F5EE,stroke:#0F6E56,color:#04342C;
    classDef infer fill:#FAECE7,stroke:#993C1D,color:#4A1B0C;
    classDef art   fill:#FDE8AA,stroke:#C07A15,color:#41280A;

    class VID,FR,POSE,SHUT,SKEL,TRAJ,WIN,FEAT,META pre;
    class SPLIT,AUG,MODEL,LOOP train;
    class NEWVID,FEAT2,LOAD,FWD,SOFT,PRED infer;
    class CKPT art;
```

**How the two paths link**
- **Shared front-end** — both paths run the identical frame → pose → skeleton /
  shuttle extraction and the same window + feature-engineering steps
  (single hitter, V=17, L3 14-dim). Inference must not drift from training here.
- **Checkpoint hand-off** — training's only output that inference consumes is
  `C3_shuttle_xattn.pt`. The inference loader auto-detects architecture
  (`in_channels`, `num_classes`, pooling, cross-attention) from the weights.
- **Train-only steps** — the train/val/held-out split and augmentation exist
  only on the training path; inference sees neither.