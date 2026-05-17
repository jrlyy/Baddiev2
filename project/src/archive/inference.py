"""
End-to-end inference pipeline: raw video → tactical strategy prediction.

Usage:
    python -m src.inference --video path/to/match.mp4 --model models/fewshot_final.pt
"""
import argparse
import json
import torch
import numpy as np
import cv2
from pathlib import Path

from .config import (
    MODELS_DIR, STRATEGY_CLASSES, IDX_TO_STRATEGY,
    SHOT_TYPES,
    STGCNConfig, TransformerConfig, DataConfig, AblationConfig,
)
from .data.pose_extractor import PoseExtractor
from .data.graph_builder import GraphBuilder
from .models.stgcn_model import STGCN
from .models.transformer_encoder import SkeletonTransformer
from .models.proto_net import PrototypicalNetwork


class TacticalPredictor:
    """
    Full inference pipeline from video frames to strategy predictions.

    Loads a trained model (encoder + prototypes) and processes
    new video to produce per-shot tactical strategy predictions
    with confidence scores.
    """

    def __init__(self, checkpoint_path, device=None):
        """
        Args:
            checkpoint_path: path to saved checkpoint (.pt) containing
                encoder weights, prototypes, and config
            device: torch device (auto-detected if None)
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = checkpoint.get("config", None)
        self.prototypes = checkpoint["prototypes"].to(self.device)

        # Build encoder
        ablation = self.config.ablation if self.config else AblationConfig()
        graph_builder = GraphBuilder(
            use_inter_player=ablation.use_inter_player,
            single_player=ablation.single_player,
        )
        adjacency = graph_builder.build_adjacency().to(self.device)

        if ablation.encoder == "transformer":
            t_cfg = self.config.transformer if self.config else TransformerConfig()
            encoder = SkeletonTransformer(
                in_channels=t_cfg.in_channels,
                d_model=t_cfg.d_model,
                nhead=t_cfg.nhead,
                num_layers=t_cfg.num_layers,
                embedding_dim=t_cfg.embedding_dim,
            )
        else:
            s_cfg = self.config.stgcn if self.config else STGCNConfig()
            encoder = STGCN(
                in_channels=s_cfg.in_channels,
                num_nodes=s_cfg.num_nodes,
                adjacency=adjacency,
                embedding_dim=s_cfg.embedding_dim,
            )

        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        encoder.to(self.device)
        encoder.eval()

        self.proto_net = PrototypicalNetwork(encoder)
        self.pose_extractor = PoseExtractor()

        d_cfg = self.config.data if self.config else DataConfig()
        self.shot_window = d_cfg.shot_window

        # Load shot-type classifier if available (trained in nb03 §7a)
        self.shot_clf = None
        feature_layer = getattr(self.config.ablation, "feature_layer", "L2") if self.config else "L2"
        for method in ("supcon", "simclr"):
            candidate = Path(checkpoint_path).parent / f"shot_type_clf_{method}_{feature_layer}.joblib"
            if candidate.exists():
                import joblib
                self.shot_clf = joblib.load(candidate)
                print(f"Shot-type classifier loaded: {candidate.name}")
                break

    def predict_from_frames(self, frames):
        """
        Predict tactical strategy from a list of video frames.

        Args:
            frames: list of numpy arrays (H, W, 3) BGR — one shot's worth

        Returns:
            dict with keys: strategy, confidence, margin, all_probs
        """
        # Extract skeleton
        skeleton = self.pose_extractor.extract_sequence(frames)  # (C, T, V)
        x = torch.tensor(skeleton, dtype=torch.float32).unsqueeze(0)  # (1, C, T, V)

        # Pad/crop to shot_window
        C, T, V = x.shape[1], x.shape[2], x.shape[3]
        if T < self.shot_window:
            pad = torch.zeros(1, C, self.shot_window - T, V)
            x = torch.cat([x, pad], dim=2)
        elif T > self.shot_window:
            x = x[:, :, :self.shot_window, :]

        x = x.to(self.device)

        # Predict
        with torch.no_grad():
            predictions, confidences, margins = self.proto_net.predict(
                x, self.prototypes
            )

        pred_idx = predictions[0].item()
        with torch.no_grad():
            emb = self.proto_net.encoder(x)
            dists = self.proto_net.compute_distances(emb, self.prototypes)

        result = {
            "strategy": IDX_TO_STRATEGY[pred_idx],
            "confidence": confidences[0].item(),
            "margin": margins[0].item(),
            "all_probs": {
                IDX_TO_STRATEGY[i]: p.item()
                for i, p in enumerate(torch.softmax(-dists, dim=1)[0])
            },
        }

        # Shot-type prediction (if classifier was loaded)
        if self.shot_clf is not None:
            emb_np = emb.cpu().numpy()
            shot_probs = self.shot_clf.predict_proba(emb_np)[0]
            shot_idx = int(shot_probs.argmax())
            # Map classifier class index back to shot type name
            clf_class = self.shot_clf.classes_[shot_idx]
            result["shot_type"] = SHOT_TYPES[clf_class]
            result["shot_type_confidence"] = float(shot_probs[shot_idx])

        return result

    def predict_from_video(self, video_path, shot_timestamps=None):
        """
        Process a full video and predict strategy for each shot.

        Args:
            video_path: path to video file
            shot_timestamps: list of (start_frame, end_frame) tuples
                            If None, uses sliding window

        Returns:
            list of prediction dicts
        """
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if shot_timestamps is None:
            # Sliding window with stride = shot_window
            shot_timestamps = [
                (i, min(i + self.shot_window, total_frames))
                for i in range(0, total_frames, self.shot_window)
            ]

        results = []
        for start, end in shot_timestamps:
            frames = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for _ in range(end - start):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)

            if len(frames) == 0:
                continue

            pred = self.predict_from_frames(frames)
            pred["start_frame"] = start
            pred["end_frame"] = end
            pred["start_time"] = start / fps
            pred["end_time"] = end / fps
            results.append(pred)

        cap.release()
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Tactical strategy prediction from badminton video"
    )
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument(
        "--model",
        type=str,
        default=str(MODELS_DIR / "fewshot_final.pt"),
        help="Path to model checkpoint",
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Flag predictions below this confidence",
    )
    args = parser.parse_args()

    predictor = TacticalPredictor(args.model)
    results = predictor.predict_from_video(args.video)

    # Print results
    for r in results:
        flag = " [LOW CONFIDENCE]" if r["confidence"] < args.confidence_threshold else ""
        shot_str = f"  [{r['shot_type']} {r['shot_type_confidence']:.0%}]" if "shot_type" in r else ""
        print(
            f"  [{r['start_time']:.1f}s - {r['end_time']:.1f}s] "
            f"{r['strategy']:>15s}  conf={r['confidence']:.2f}  "
            f"margin={r['margin']:.2f}{shot_str}{flag}"
        )

    # Save
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved {len(results)} predictions to {output_path}")


if __name__ == "__main__":
    main()
