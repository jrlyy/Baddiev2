Project Structure to be updated if needed 

Baddiev2/
├── src/                          # All reusable Python code
│   ├── config.py                 # Central config with dataclasses for all hyperparams
│   ├── inference.py              # CLI: video → strategy predictions
│   ├── data/
│   │   ├── pose_extractor.py     # YOLOv8-Pose dual-player skeleton extraction
│   │   ├── graph_builder.py      # 34-node adjacency matrix (3 partitions for ST-GCN)
│   │   └── dataset.py            # FineBadminton + ShuttleSet loaders, episodic sampler
│   └── models/
│       ├── stgcn_model.py        # ST-GCN encoder (9 blocks, spatial+temporal conv)
│       ├── transformer_encoder.py # BST-style transformer (for architecture ablation)
│       ├── proto_net.py           # Prototypical network with confidence/margin scoring
│       └── simclr_loss.py         # NT-Xent loss, projection head, skeleton augmentations
│
├── notebooks/                     # Interactive work (train here, analyze here)
│   ├── 01_data_exploration.ipynb  # Inspect frames, annotations, test pose extraction
│   ├── 02_skeleton_extraction.ipynb # Batch skeleton extraction for both datasets
│   ├── 03_ssl_pretraining.ipynb   # SimCLR training → saves ssl_pretrained.pt
│   ├── 04_fewshot_training.ipynb  # ProtoNet 5-fold CV → saves fewshot_final.pt
│   ├── 05_ablations.ipynb         # All 6 ablation configs (RQ1, RQ2, architecture, etc.)
│   └── 06_analysis_and_plots.ipynb # t-SNE, confusion matrix, K-shot curves, bar charts
│
├── Datasets/                      # Your existing data (unchanged)
│   ├── extraction_pipeline.ipynb  # Your existing ShuttleSet downloader
│   ├── FineBadminton-dataset/
│   └── ShuttleSet/
├── models/                        # Saved checkpoints (created by notebooks)
└── results/                       # Saved metrics and plots


As far as possible train in a notebook with configurable ablations and save weights to .pt/.pth files and use them in inference.py. Just keep the model definitions in .py files so you're importing the same class everywhere.


Whenever changes are made to the notebooks or code, check if the Report md should updated as a source of truth