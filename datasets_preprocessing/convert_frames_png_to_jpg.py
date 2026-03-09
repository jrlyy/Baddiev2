# One-time conversion: ShuttleSet PNG frames → JPEG (quality 90)
# Deletes PNGs after successful conversion and updates frame_prev/curr/next
# in shuttleset_outputs/*.json to reflect the new .jpg extension.
#
# Run from any directory:
#   python datasets_preprocessing/convert_frames_png_to_jpg.py
#
# The server already serves .jpg first and falls back to .png, so the UI
# works correctly both before and after this conversion.

import json
import cv2
from pathlib import Path
from tqdm import tqdm

SCRIPT_DIR  = Path(__file__).resolve().parent
FRAMES_ROOT = SCRIPT_DIR / "shuttleset_frames"
OUTPUT_ROOT = SCRIPT_DIR / "shuttleset_outputs"
QUALITY     = 90   # JPEG quality (90 is a good balance of size vs fidelity)

# ── 1. Convert frames ────────────────────────────────────────────────────────
png_files = sorted(FRAMES_ROOT.rglob("*.png"))
print(f"Found {len(png_files):,} PNG files to convert")

errors = []
for png_path in tqdm(png_files, desc="Converting PNG → JPG"):
    jpg_path = png_path.with_suffix(".jpg")
    if jpg_path.exists():
        png_path.unlink()   # JPG already exists, just clean up the PNG
        continue
    img = cv2.imread(str(png_path))
    if img is None:
        errors.append(str(png_path))
        continue
    ok = cv2.imwrite(str(jpg_path), img, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
    if ok:
        png_path.unlink()
    else:
        errors.append(str(png_path))

if errors:
    print(f"\n[WARN] Failed to convert {len(errors)} files:")
    for e in errors[:10]:
        print(f"  {e}")
else:
    print(f"\nAll PNGs converted and deleted.")

# ── 2. Update JSONs (frame_prev / frame_curr / frame_next paths) ─────────────
json_files = [f for f in OUTPUT_ROOT.glob("*.json") if f.name != "pipeline_summary.json"]
print(f"\nUpdating {len(json_files)} JSON files...")

for jf in json_files:
    records = json.loads(jf.read_text())
    changed = False
    for r in records:
        for key in ("frame_prev", "frame_curr", "frame_next"):
            val = r.get(key, "")
            if val and val.endswith(".png"):
                r[key] = val[:-4] + ".jpg"
                changed = True
    if changed:
        jf.write_text(json.dumps(records, indent=2, ensure_ascii=False))
        print(f"  Updated {jf.name}")

print("\nDone.")