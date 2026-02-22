"""
Quick visual verification that extracted frames match ShuttleSet annotations.
Samples a few strokes, draws annotation info on the frames, and saves a grid.

Usage:
    python Datasets/verify_extraction.py
    # Or for a specific match:
    python Datasets/verify_extraction.py Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2019_Finals
"""

import sys
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "outputs"
VERIFY_DIR = SCRIPT_DIR / "verification"
VERIFY_DIR.mkdir(exist_ok=True)

NUM_SAMPLES = 6  # strokes to verify

# Try to load a font that supports CJK characters
def get_cjk_font(size=24):
    """Find a system font that supports Chinese characters"""
    font_paths = [
        # macOS
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        # Linux
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for fp in font_paths:
        if Path(fp).exists():
            return ImageFont.truetype(fp, size)
    # Fallback to default (won't render CJK but won't crash)
    return ImageFont.load_default()


FONT_LARGE = get_cjk_font(28)
FONT_MEDIUM = get_cjk_font(22)
FONT_SMALL = get_cjk_font(18)


def draw_annotation(frame, record):
    """Draw annotation overlay on a full-resolution frame copy using PIL for CJK text"""
    # Convert BGR (OpenCV) -> RGB (PIL)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    h, w = frame.shape[:2]

    # Compute approximate timestamp from frame number (assume 30fps)
    frame_num = record['frame_num']
    total_secs = frame_num / 30.0
    mins, secs = divmod(int(total_secs), 60)
    hours, mins = divmod(mins, 60)
    timestamp = f"{hours}:{mins:02d}:{secs:02d}" if hours else f"{mins}:{secs:02d}"

    # Set, rally, time, shot type
    set_name = record.get('set_file', '').replace('.csv', '')
    line1 = f"{set_name} | Rally {record['rally']} Round {record['ball_round']} | {timestamp}"
    draw.text((10, 10), line1, font=FONT_MEDIUM, fill=(255, 255, 255))

    # Shot type + player
    line2 = f"{record['type']} | Player {record['player']}"
    draw.text((10, 40), line2, font=FONT_LARGE, fill=(0, 255, 0))

    # Hit position marker (coordinates are in original video pixel space)
    hit_x = record.get("hit_x")
    hit_y = record.get("hit_y")
    if hit_x and hit_y:
        try:
            cx, cy = int(float(hit_x)), int(float(hit_y))
            if not (np.isnan(float(hit_x)) or np.isnan(float(hit_y))):
                r = 15
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 0, 0), width=3)
                draw.text((cx + 20, cy - 10), "HIT", font=FONT_SMALL, fill=(255, 0, 0))
        except (ValueError, TypeError):
            pass

    # Landing position marker
    land_x = record.get("landing_x")
    land_y = record.get("landing_y")
    if land_x and land_y:
        try:
            lx, ly = int(float(land_x)), int(float(land_y))
            if not (np.isnan(float(land_x)) or np.isnan(float(land_y))):
                r = 15
                draw.ellipse([lx - r, ly - r, lx + r, ly + r], outline=(0, 100, 255), width=3)
                draw.text((lx + 20, ly - 10), "LAND", font=FONT_SMALL, fill=(0, 100, 255))
        except (ValueError, TypeError):
            pass

    # Frame number
    draw.text((10, h - 30), f"frame {record['frame_num']}", font=FONT_SMALL, fill=(200, 200, 200))

    # Convert back to BGR (OpenCV)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def verify_match(match_id: str):
    """Load stroke records, sample a few, build a visual verification grid"""
    json_path = OUTPUT_ROOT / f"{match_id}.json"
    if not json_path.exists():
        print(f"[ERROR] No output found: {json_path}")
        return False

    with open(json_path) as f:
        records = json.load(f)

    print(f"\nVerifying: {match_id}")
    print(f"  Total stroke records: {len(records)}")

    # Take first N strokes in chronological order
    sorted_records = sorted(records, key=lambda r: r["frame_num"])
    samples = sorted_records[:NUM_SAMPLES]

    # Build verification grid: 3 columns (prev, curr, next) x N rows
    rows = []
    missing = 0
    for record in samples:
        triplet = []
        for key, label in [("frame_prev", "PREV"), ("frame_curr", "STROKE"), ("frame_next", "NEXT")]:
            fpath = Path(record[key])
            if fpath.exists():
                img = cv2.imread(str(fpath))
                # Draw annotations on full-res frame BEFORE resizing
                if key == "frame_curr":
                    img = draw_annotation(img, record)
                # Add column label at full res
                cv2.putText(img, label, (img.shape[1] - 120, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
                triplet.append(img)
            else:
                missing += 1
                print(f"  [MISSING] {fpath}")
                placeholder = np.full((360, 640, 3), 50, dtype=np.uint8)
                cv2.putText(placeholder, f"MISSING: {fpath.name}", (20, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                triplet.append(placeholder)

        # Resize all to same height AFTER drawing annotations
        target_h = 480
        resized = []
        for img in triplet:
            scale = target_h / img.shape[0]
            resized.append(cv2.resize(img, (int(img.shape[1] * scale), target_h)))

        rows.append(np.hstack(resized))

    # Ensure all rows same width
    max_w = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
            r = np.hstack([r, pad])
        padded.append(r)

    grid = np.vstack(padded)

    out_path = VERIFY_DIR / f"verify_{match_id}.png"
    cv2.imwrite(str(out_path), grid)
    print(f"  Saved verification grid: {out_path}")
    print(f"  Missing frames: {missing}")
    print(f"  Sampled shot types: {[s['type'] for s in samples]}")

    return missing == 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        match_ids = [sys.argv[1]]
    else:
        match_ids = [p.stem for p in sorted(OUTPUT_ROOT.glob("*.json"))
                     if p.stem != "pipeline_summary"]

    if not match_ids:
        print("No processed matches found in outputs/. Run streaming_pipeline.py first.")
        sys.exit(1)

    all_ok = True
    for mid in match_ids:
        ok = verify_match(mid)
        if not ok:
            all_ok = False

    if all_ok:
        print("\n=== ALL VERIFIED OK ===")
    else:
        print("\n=== SOME FRAMES MISSING — check output above ===")
