# Extracts full consecutive rally frames from ShuttleSet videos, matching the
# FineBadminton structure where every frame in a rally is available and shots
# are indexed by rally_t = frame_num - rally_frame_start.
#
# Key differences from old pipeline:
#   - Extracts ALL frames in each rally's span (not just ±10 sparse frames)
#   - Streaming: seeks once per rally then reads consecutively (fast)
#   - Shot records include rally_t, rally_frame_start, rally_frame_end
#   - frame_prev/frame_next use offset=1 (consecutive, correct for TrackNet)
#   - Output paths match config.py: shuttleset_frames/, shuttleset_outputs/

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
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TMP_VIDEO    = SCRIPT_DIR / "tmp_video.mp4"

# Annotation CSVs — matches config.py SS_CSV_ROOT
SHUTTLESET_ROOT = PROJECT_ROOT / "datasets" / "ShuttleSet"
MATCHES_CSV     = SHUTTLESET_ROOT / "set" / "match.csv"
ANNOT_ROOT      = SHUTTLESET_ROOT / "set"

# Output dirs — match config.py SS_FRAMES / SS_OUTPUTS
OUTPUT_ROOT = SCRIPT_DIR / "shuttleset_outputs"
FRAMES_ROOT = SCRIPT_DIR / "shuttleset_frames"

# Frames of context padding before the first shot and after the last shot in
# each rally (≈0.5 s at 30 fps).  Keeps the same constant name so downstream
# code that imports it stays unchanged.
RALLY_CONTEXT = 15

# ── MVP / partial-run controls ──────────────────────────────────────────────
# Set MVP_MODE = True to extract only a few rallies for a quick sanity-check.
# Flip to False (or pass --full on the CLI) for the full Colab run.
MVP_MODE        = True
MVP_MATCH_LIMIT = 1   # how many matches to process
MVP_RALLY_LIMIT = 2   # how many rallies per match (first N by rally id)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
FRAMES_ROOT.mkdir(parents=True, exist_ok=True)

SHOT_TYPES = [
    '發短球', '發長球', '推撲球', '殺球', '過渡球', '防守回挑',
    '切球', '接殺防守', '長球', '平球', '擋小球', '挑球',
    '放小球', '勾球', '網前球', '點扣', '推球', '未知'
]


# =========================
# CORE FUNCTIONS
# =========================

def download_video(url: str):
    """Download one video to a temp file."""
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
    """Load annotations from ALL sets (set1.csv, set2.csv, set3.csv, ...)."""
    all_dfs = []
    for set_file in sorted(match_dir.glob("set*.csv")):
        df = pd.read_csv(set_file)
        df["_set_file"] = set_file.name
        all_dfs.append(df)
        print(f"  Loaded {set_file.name}: {len(df)} strokes")

    if not all_dfs:
        return None

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined[
        combined["frame_num"].notna() &
        combined["type"].notna() &
        (combined["type"] != "")
    ].copy()
    combined["frame_num"] = combined["frame_num"].astype(int)
    return combined


def compute_rally_ranges(annotations_df, context=RALLY_CONTEXT):
    """
    For each (set_file, rally) group compute the full consecutive frame range
    [first_shot_frame - context, last_shot_frame + context].

    Returns:
        dict mapping (set_file, rally) -> {"start": int, "end": int}
    """
    ranges = {}
    for (set_file, rally), group in annotations_df.groupby(["_set_file", "rally"]):
        first = int(group["frame_num"].min())
        last  = int(group["frame_num"].max())
        ranges[(set_file, rally)] = {
            "start": max(0, first - context),
            "end":   last + context,
        }
    return ranges


def extract_rally_frames(cap, rally_ranges, match_frames_dir, total_video_frames):
    """
    For each rally range, seek once to the start then read frames sequentially.
    Much faster than seeking to each sparse frame individually.

    Returns the total number of frames written.
    """
    match_frames_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0

    # Sort by start frame so we seek forward through the video monotonically
    for rally_key, rng in sorted(rally_ranges.items(), key=lambda x: x[1]["start"]):
        start_fn = rng["start"]
        end_fn   = min(rng["end"], total_video_frames - 1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_fn)
        current_fn = start_fn

        while current_fn <= end_fn:
            ret, frame = cap.read()
            if not ret:
                print(f"[WARN] Failed to read frame {current_fn} in rally {rally_key}")
                break

            frame_path = match_frames_dir / f"frame_{current_fn:06d}.jpg"
            if not frame_path.exists():   # skip already-extracted frames on re-run
                cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                extracted += 1

            current_fn += 1

    return extracted


def process_match(video_url: str, match_id: str, rally_limit: int = None):
    """
    Download, extract consecutive rally frames, build shot records, save JSON.

    Args:
        video_url:   YouTube URL for the match video.
        match_id:    Match identifier string (used for dir names and JSON file).
        rally_limit: If set, only process the first N rallies (MVP mode).
    """
    print(f"\n=== Processing match: {match_id} ===")

    match_dir = ANNOT_ROOT / match_id
    if not match_dir.exists():
        print(f"[SKIP] Annotation dir not found: {match_dir}")
        return

    annotations_df = load_all_set_annotations(match_dir)
    if annotations_df is None or annotations_df.empty:
        print(f"[SKIP] No valid annotations for {match_id}")
        return

    # Optionally restrict to first N rallies for MVP
    if rally_limit is not None:
        rally_keys = (
            annotations_df
            .groupby(["_set_file", "rally"])
            .size()
            .reset_index()[["_set_file", "rally"]]
        )
        keep = set(
            zip(rally_keys["_set_file"].iloc[:rally_limit],
                rally_keys["rally"].iloc[:rally_limit])
        )
        annotations_df = annotations_df[
            annotations_df.apply(lambda r: (r["_set_file"], r["rally"]) in keep, axis=1)
        ].copy()
        print(f"  MVP: restricted to {rally_limit} rallies → {len(annotations_df)} strokes")

    type_counts = annotations_df["type"].value_counts()
    print(f"  Total strokes: {len(annotations_df)}, types: {len(type_counts)}")

    # Compute per-rally frame ranges
    rally_ranges = compute_rally_ranges(annotations_df)
    total_frames_needed = sum(
        r["end"] - r["start"] + 1 for r in rally_ranges.values()
    )
    print(f"  Rallies: {len(rally_ranges)}, frames to extract: {total_frames_needed}")

    # Download and extract
    download_video(video_url)
    cap = cv2.VideoCapture(str(TMP_VIDEO))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Video: {total_video_frames} frames @ {fps:.1f} fps")

    match_frames_dir = FRAMES_ROOT / match_id
    extracted = extract_rally_frames(cap, rally_ranges, match_frames_dir, total_video_frames)
    cap.release()
    print(f"  Extracted {extracted} frames → {match_frames_dir}")

    # Build per-stroke records
    # rally_t = frame_num - rally_frame_start  (temporal index within the rally,
    #           equivalent to FineBadminton's frame index within the rally clip)
    stroke_records = []
    for _, row in annotations_df.iterrows():
        set_file = row["_set_file"]
        rally    = int(row["rally"]) if pd.notna(row.get("rally")) else None
        fn       = int(row["frame_num"])

        rng = rally_ranges.get((set_file, rally), {})
        rally_start = rng.get("start", fn)
        rally_end   = rng.get("end",   fn)

        record = {
            "match_id":         match_id,
            "set_file":         set_file,
            "rally":            rally,
            "ball_round":       int(row["ball_round"]) if pd.notna(row.get("ball_round")) else None,
            "frame_num":        fn,
            "rally_t":          fn - rally_start,
            "rally_frame_start": rally_start,
            "rally_frame_end":   rally_end,
            "type":             row["type"],
            "player":           row.get("player", ""),
            "frame_prev":       str(match_frames_dir / f"frame_{fn - 1:06d}.jpg"),
            "frame_curr":       str(match_frames_dir / f"frame_{fn:06d}.jpg"),
            "frame_next":       str(match_frames_dir / f"frame_{fn + 1:06d}.jpg"),
            "hit_area":             row.get("hit_area"),
            "hit_x":                row.get("hit_x"),
            "hit_y":                row.get("hit_y"),
            "landing_area":         row.get("landing_area"),
            "landing_x":            row.get("landing_x"),
            "landing_y":            row.get("landing_y"),
            "player_location_x":    row.get("player_location_x"),
            "player_location_y":    row.get("player_location_y"),
            "opponent_location_x":  row.get("opponent_location_x"),
            "opponent_location_y":  row.get("opponent_location_y"),
            "backhand":             row.get("backhand"),
            "hit_height":           row.get("hit_height"),
        }
        stroke_records.append(record)

    out_path = OUTPUT_ROOT / f"{match_id}.json"
    with open(out_path, "w") as f:
        json.dump(stroke_records, f, indent=2, ensure_ascii=False)

    if TMP_VIDEO.exists():
        os.remove(TMP_VIDEO)

    print(f"[DONE] {match_id}: {len(stroke_records)} stroke records → {out_path}")


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Process all matches (overrides MVP_MODE)")
    parser.add_argument("--match", type=str, default=None,
                        help="Process only this match ID (exact string from match.csv video column)")
    args = parser.parse_args()

    full_run = args.full or not MVP_MODE

    if not MATCHES_CSV.exists():
        raise FileNotFoundError(f"match.csv not found at {MATCHES_CSV}")

    print("=== SHUTTLESET STREAMING PIPELINE ===")
    if args.match:
        print(f"Mode: SINGLE MATCH ({args.match})")
    elif full_run:
        print("Mode: FULL")
    else:
        print(f"Mode: MVP (matches={MVP_MATCH_LIMIT}, rallies/match={MVP_RALLY_LIMIT})")
    print(f"Frames → {FRAMES_ROOT}")
    print(f"JSON   → {OUTPUT_ROOT}")

    summary = {"total_matches": 0, "total_strokes": 0, "shot_distribution": {}}
    matches_processed = 0

    with open(MATCHES_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            match_id = row["video"].strip()
            url      = row["url"].strip()

            if not url:
                print(f"[SKIP] No URL for {match_id}")
                continue

            if args.match and match_id != args.match:
                continue

            rally_limit = None if (full_run or args.match) else MVP_RALLY_LIMIT
            process_match(url, match_id, rally_limit=rally_limit)
            matches_processed += 1
            summary["total_matches"] += 1

            out_file = OUTPUT_ROOT / f"{match_id}.json"
            if out_file.exists():
                records = json.loads(out_file.read_text())
                summary["total_strokes"] += len(records)
                for r in records:
                    shot = r["type"]
                    summary["shot_distribution"][shot] = (
                        summary["shot_distribution"].get(shot, 0) + 1
                    )

            if args.match:
                break  # single-match mode — stop after processing it

            if not full_run and matches_processed >= MVP_MATCH_LIMIT:
                print(f"\n[MVP] Reached {MVP_MATCH_LIMIT} match limit — stopping.")
                break

    with open(OUTPUT_ROOT / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== DONE ===")
    print(f"  Matches : {summary['total_matches']}")
    print(f"  Strokes : {summary['total_strokes']}")
    print(f"  Types   : {len(summary['shot_distribution'])}")