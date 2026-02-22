#This notebook extracts the frames of interest (stroke frames + temporal neighbors) from the ShuttleSet videos, based on the annotations provided in the set CSV files from youtube videos. It processes each match sequentially, downloading the video, extracting frames, and saving structured JSON outputs for each stroke. The pipeline is designed to be memory efficient by streaming through the video frames rather than loading entire videos into memory.

import subprocess
import os
from pathlib import Path
import pandas as pd
import cv2
import json
import csv

# =========================
# CONFIG
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent
TMP_VIDEO = SCRIPT_DIR / "tmp_video.mp4"
SHUTTLESET_ROOT = SCRIPT_DIR / "ShuttleSet"
MATCHES_CSV = SHUTTLESET_ROOT / "set" / "match.csv"
ANNOT_ROOT = SHUTTLESET_ROOT / "set"
OUTPUT_ROOT = SCRIPT_DIR / "outputs"
FRAMES_ROOT = SCRIPT_DIR / "frames"

TEMPORAL_OFFSET = 10  # frames before/after the stroke frame

OUTPUT_ROOT.mkdir(exist_ok=True)
FRAMES_ROOT.mkdir(exist_ok=True)

SHOT_TYPES = [
    '發短球', '發長球', '推撲球', '殺球', '過渡球', '防守回挑',
    '切球', '接殺防守', '長球', '平球', '擋小球', '挑球',
    '放小球', '勾球', '網前球', '點扣', '推球', '未知'
]

# =========================
# CORE FUNCTIONS
# =========================

def download_video(url: str):
    """Download one video to a temp file"""
    if TMP_VIDEO.exists():
        os.remove(TMP_VIDEO)
    print("[INFO] Downloading video...")
    subprocess.run([
        "yt-dlp",
        "-f", "bv*[ext=mp4]/bv*",
        "-o", str(TMP_VIDEO),
        url
    ], check=True)


def load_all_set_annotations(match_dir: Path):
    """Load annotations from ALL sets (set1.csv, set2.csv, set3.csv, ...)"""
    all_dfs = []
    for set_file in sorted(match_dir.glob("set*.csv")):
        df = pd.read_csv(set_file)
        df["_set_file"] = set_file.name
        all_dfs.append(df)
        print(f"  Loaded {set_file.name}: {len(df)} strokes")

    if not all_dfs:
        return None

    combined = pd.concat(all_dfs, ignore_index=True)

    # Filter to rows with valid frame_num and shot type
    combined = combined[
        combined["frame_num"].notna() &
        combined["type"].notna() &
        (combined["type"] != "")
    ].copy()
    combined["frame_num"] = combined["frame_num"].astype(int)

    return combined


def collect_required_frames(annotations_df, offset=TEMPORAL_OFFSET):
    """Collect all unique frame numbers needed (stroke + temporal neighbors)"""
    stroke_frames = annotations_df["frame_num"].unique()
    all_frames = set()
    for f in stroke_frames:
        all_frames.add(f - offset)
        all_frames.add(f)
        all_frames.add(f + offset)
    return sorted(f for f in all_frames if f >= 0)


def save_frame(frame, frame_path: Path):
    """Save frame locally as PNG"""
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(frame_path), frame)


def process_frame(frame, frame_id, frame_path):
    """
    PLACEHOLDER — replace with your real model:
    - YOLO player detection
    - pose estimation
    - hit context features
    """
    save_frame(frame, frame_path)
    return {
        "frame_id": int(frame_id),
        "height": frame.shape[0],
        "width": frame.shape[1],
        "frame_path": str(frame_path)
    }


def process_match(video_url: str, match_id: str):
    print(f"\n=== Processing match: {match_id} ===")

    match_dir = ANNOT_ROOT / match_id
    if not match_dir.exists():
        print(f"[SKIP] Annotation dir not found: {match_dir}")
        return

    # 1. Load ALL set annotations
    annotations_df = load_all_set_annotations(match_dir)
    if annotations_df is None or annotations_df.empty:
        print(f"[SKIP] No valid annotations for {match_id}")
        return

    # Shot type distribution for this match
    type_counts = annotations_df["type"].value_counts()
    print(f"  Total strokes: {len(annotations_df)}, types: {len(type_counts)}")
    for shot, count in type_counts.head(5).items():
        print(f"    {shot}: {count}")

    # 2. Download video
    download_video(video_url)

    # 3. Collect all frames needed (stroke frames + temporal neighbors)
    all_frame_nums = collect_required_frames(annotations_df)
    print(f"[INFO] Unique frames to extract: {len(all_frame_nums)}")

    cap = cv2.VideoCapture(str(TMP_VIDEO))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 4. Extract and save frames (streaming, sequential seek)
    match_frames_dir = FRAMES_ROOT / match_id
    extracted = 0
    for fn in all_frame_nums:
        if fn < 0 or fn >= total_video_frames:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if not ret:
            print(f"[WARN] Failed to read frame {fn}")
            continue

        frame_path = match_frames_dir / f"frame_{fn:06d}.png"
        process_frame(frame, fn, frame_path)
        extracted += 1

    cap.release()
    print(f"[INFO] Extracted {extracted}/{len(all_frame_nums)} frames")

    # 5. Build per-stroke records with temporal frame paths
    stroke_records = []
    for _, row in annotations_df.iterrows():
        fn = int(row["frame_num"])
        prev_fn = fn - TEMPORAL_OFFSET
        next_fn = fn + TEMPORAL_OFFSET

        record = {
            "match_id": match_id,
            "set_file": row.get("_set_file", ""),
            "rally": int(row["rally"]) if pd.notna(row.get("rally")) else None,
            "ball_round": int(row["ball_round"]) if pd.notna(row.get("ball_round")) else None,
            "frame_num": fn,
            "type": row["type"],
            "player": row.get("player", ""),
            "frame_prev": str(match_frames_dir / f"frame_{prev_fn:06d}.png"),
            "frame_curr": str(match_frames_dir / f"frame_{fn:06d}.png"),
            "frame_next": str(match_frames_dir / f"frame_{next_fn:06d}.png"),
            # Court position features from annotations
            "hit_area": row.get("hit_area"),
            "hit_x": row.get("hit_x"),
            "hit_y": row.get("hit_y"),
            "landing_area": row.get("landing_area"),
            "landing_x": row.get("landing_x"),
            "landing_y": row.get("landing_y"),
            "player_location_x": row.get("player_location_x"),
            "player_location_y": row.get("player_location_y"),
            "opponent_location_x": row.get("opponent_location_x"),
            "opponent_location_y": row.get("opponent_location_y"),
            "backhand": row.get("backhand"),
            "hit_height": row.get("hit_height"),
        }
        stroke_records.append(record)

    # 6. Save structured output
    out_path = OUTPUT_ROOT / f"{match_id}.json"
    with open(out_path, "w") as f:
        json.dump(stroke_records, f, indent=2, ensure_ascii=False)

    # 7. Cleanup video
    if TMP_VIDEO.exists():
        os.remove(TMP_VIDEO)

    print(f"[DONE] {match_id}: {len(stroke_records)} stroke records saved")


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    print("=== STREAMING PIPELINE STARTED ===")

    if not MATCHES_CSV.exists():
        raise FileNotFoundError(f"match.csv not found at {MATCHES_CSV}")

    summary = {"total_matches": 0, "total_strokes": 0, "shot_distribution": {}}

    with open(MATCHES_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            match_id = row["video"].strip()
            url = row["url"].strip()

            if not url:
                print(f"[SKIP] No URL for {match_id}")
                continue

            process_match(url, match_id)
            summary["total_matches"] += 1

            # Accumulate stats from saved output
            out_file = OUTPUT_ROOT / f"{match_id}.json"
            if out_file.exists():
                with open(out_file) as jf:
                    records = json.load(jf)
                    summary["total_strokes"] += len(records)
                    for r in records:
                        shot = r["type"]
                        summary["shot_distribution"][shot] = (
                            summary["shot_distribution"].get(shot, 0) + 1
                        )

    # Save overall summary
    with open(OUTPUT_ROOT / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== ALL MATCHES PROCESSED ===")
    print(f"  Matches: {summary['total_matches']}")
    print(f"  Total strokes: {summary['total_strokes']}")
    print(f"  Shot types: {len(summary['shot_distribution'])}")
