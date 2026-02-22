"""
Pose extraction from video frames using YOLOv8-Pose.

Extracts 17-joint COCO keypoints for two players per frame,
with optional Kalman filtering for temporal smoothing.
"""
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional

from ..config import NUM_JOINTS, JOINT_DIM, PoseConfig


class PoseExtractor:
    """
    Extract dual-player skeleton keypoints from badminton frames.

    Uses YOLOv8-Pose for person detection + keypoint estimation,
    selecting the top-2 detections per frame as the two players.
    """

    def __init__(self, config=None):
        self.config = config or PoseConfig()
        self.model = None

    def load_model(self):
        """Lazy-load the pose estimation model."""
        if self.model is not None:
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(f"{self.config.model}.pt")
            print(f"[INFO] Loaded {self.config.model}")
        except ImportError:
            raise ImportError(
                "ultralytics not installed. Run: pip install ultralytics"
            )

    def extract_frame(self, frame) -> Optional[np.ndarray]:
        """
        Extract dual-player keypoints from a single frame.

        Args:
            frame: numpy array (H, W, 3) BGR image

        Returns:
            keypoints: (2, 17, 2) array of (x, y) for 2 players,
                      or None if fewer than 2 people detected
        """
        self.load_model()

        results = self.model(frame, verbose=False)
        if not results or len(results) == 0:
            return None

        result = results[0]
        if result.keypoints is None or len(result.keypoints) < 2:
            return None

        # Get keypoints and confidence for all detections
        kpts = result.keypoints.xy.cpu().numpy()   # (N, 17, 2)
        confs = result.keypoints.conf.cpu().numpy()  # (N, 17)

        # Select top-2 by mean keypoint confidence
        mean_confs = confs.mean(axis=1)
        top2_idx = np.argsort(mean_confs)[-2:]

        player_kpts = kpts[top2_idx]  # (2, 17, 2)

        # Sort by x-position (left player = player 0)
        center_x = player_kpts[:, :, 0].mean(axis=1)
        sort_idx = np.argsort(center_x)
        player_kpts = player_kpts[sort_idx]

        return player_kpts

    def extract_sequence(self, frames) -> np.ndarray:
        """
        Extract skeleton sequence from a list of frames.

        Args:
            frames: list of numpy arrays (H, W, 3)

        Returns:
            skeleton: (2, T, 17, 2) — channels-last format
                      Reshaped to (C, T, V) = (2, T, 34) for the model
        """
        T = len(frames)
        skeletons = np.zeros((T, 2, NUM_JOINTS, JOINT_DIM), dtype=np.float32)

        for t, frame in enumerate(frames):
            kpts = self.extract_frame(frame)
            if kpts is not None:
                skeletons[t] = kpts
            elif t > 0:
                # Forward-fill from previous frame on detection failure
                skeletons[t] = skeletons[t - 1]

        if self.config.kalman_smoothing:
            skeletons = self._apply_kalman(skeletons)

        # Reshape to model input format: (C, T, V)
        # C = 2 (x, y), T = num_frames, V = 34 (17 joints x 2 players)
        # From (T, 2_players, 17_joints, 2_xy)
        # → (T, 34, 2) → (2, T, 34)
        skeletons = skeletons.reshape(T, -1, JOINT_DIM)  # (T, 34, 2)
        skeletons = skeletons.transpose(2, 0, 1)          # (2, T, 34)

        return skeletons

    @staticmethod
    def _apply_kalman(skeletons):
        """
        Simple exponential smoothing as Kalman filter approximation.

        Args:
            skeletons: (T, 2, 17, 2)

        Returns:
            smoothed: (T, 2, 17, 2)
        """
        alpha = 0.7  # smoothing factor (higher = less smoothing)
        smoothed = skeletons.copy()
        for t in range(1, len(smoothed)):
            smoothed[t] = alpha * skeletons[t] + (1 - alpha) * smoothed[t - 1]
        return smoothed

    def process_directory(self, frames_dir, output_path, batch_size=None):
        """
        Process all frames in a directory and save skeleton as .npy.

        Args:
            frames_dir: directory containing frame images
            output_path: path to save .npy file
            batch_size: not used yet, for future batch processing
        """
        import cv2

        frames_dir = Path(frames_dir)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load frames in order
        frame_files = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
        if not frame_files:
            print(f"[WARN] No frames found in {frames_dir}")
            return

        frames = [cv2.imread(str(f)) for f in frame_files]
        skeleton = self.extract_sequence(frames)
        np.save(output_path, skeleton)

        print(f"[INFO] Saved skeleton {skeleton.shape} → {output_path}")
