"""Precompute a per-region color descriptor from actual crop pixels.

Motivation: CLIP-family crop similarity is dominated by garment *shape*, so a
red jacket can score almost as high as a blue one for the clause "blue jacket"
(documented in the report as the #1 limitation). Pixel hue, by contrast, is a
direct and cheap signal. For every annotated garment region we compute the
fraction of its pixels falling into each named color bucket (HSV-based). The
retriever then uses this as a soft, per-clause color-consistency gate.

Output: data/features/region_colors.npy   shape (n_regions, n_colors), float32
        data/features/region_colors_meta.json  {"colors": [...]}  column order

Row order matches data/features/region_mapping.csv (region_index).
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from src.indexer.encoders import norm_path

FEATURE_DIR = PROJECT_ROOT / "data" / "features"
REGION_MAPPING = FEATURE_DIR / "region_mapping.csv"
OUT_NPY = FEATURE_DIR / "region_colors.npy"
OUT_META = FEATURE_DIR / "region_colors_meta.json"

# Canonical color buckets. Order defines the output column order.
COLORS = [
    "red", "orange", "yellow", "green", "blue", "purple", "pink",
    "brown", "white", "gray", "black",
]
COLOR_INDEX = {c: i for i, c in enumerate(COLORS)}

# Aliases the retriever may query with -> canonical bucket(s).
COLOR_ALIASES = {
    "grey": "gray",
    "navy": "blue",
    "beige": "brown",   # low-sat tan; closest bucket
    "cream": "white",
}


def classify_hsv_vec(h, s, v):
    """Vectorized: h in [0,360), s in [0,1], v in [0,1] arrays -> color-name
    index array (int) into COLORS. Same thresholds as the scalar reference."""
    idx = np.full(h.shape, COLOR_INDEX["red"], dtype=np.int16)
    # Chromatic hue buckets (applied first, overwritten by achromatic below).
    chroma = [
        (COLOR_INDEX["red"], (h < 12) | (h >= 348)),
        (COLOR_INDEX["brown"], (h >= 12) & (h < 40) & (v < 0.55)),
        (COLOR_INDEX["orange"], (h >= 12) & (h < 40) & (v >= 0.55)),
        (COLOR_INDEX["yellow"], (h >= 40) & (h < 70)),
        (COLOR_INDEX["green"], (h >= 70) & (h < 165)),
        (COLOR_INDEX["blue"], (h >= 165) & (h < 255)),
        (COLOR_INDEX["purple"], (h >= 255) & (h < 290)),
        (COLOR_INDEX["pink"], (h >= 290) & (h < 348) & (v > 0.55)),
        (COLOR_INDEX["red"], (h >= 290) & (h < 348) & (v <= 0.55)),
    ]
    for ci, mask in chroma:
        idx[mask] = ci
    # Achromatic pixels: decided by value/saturation, not hue.
    achroma = s < 0.18
    idx[achroma & (v > 0.72)] = COLOR_INDEX["white"]
    idx[achroma & (v <= 0.72)] = COLOR_INDEX["gray"]
    idx[v < 0.20] = COLOR_INDEX["black"]
    return idx


def rgb_to_hsv_array(arr):
    """arr: (N,3) uint8 RGB -> (h[0,360), s[0,1], v[0,1]) each (N,)."""
    rgb = arr.astype(np.float32) / 255.0
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    mx = np.max(rgb, axis=1)
    mn = np.min(rgb, axis=1)
    diff = mx - mn
    v = mx
    s = np.where(mx > 1e-6, diff / np.maximum(mx, 1e-6), 0.0)
    h = np.zeros_like(mx)
    mask = diff > 1e-6
    # per-channel hue
    rc = np.where((mx == r) & mask)
    gc = np.where((mx == g) & mask)
    bc = np.where((mx == b) & mask)
    h[rc] = (60 * ((g[rc] - b[rc]) / diff[rc]) + 360) % 360
    h[gc] = (60 * ((b[gc] - r[gc]) / diff[gc]) + 120) % 360
    h[bc] = (60 * ((r[bc] - g[bc]) / diff[bc]) + 240) % 360
    return h, s, v


def color_fractions(crop):
    """Return length-len(COLORS) vector of pixel fractions per color bucket."""
    small = crop.resize((48, 48))
    arr = np.asarray(small, dtype=np.uint8).reshape(-1, 3)
    h, s, v = rgb_to_hsv_array(arr)
    labels = classify_hsv_vec(h, s, v)
    counts = np.bincount(labels, minlength=len(COLORS)).astype(np.float32)
    total = counts.sum()
    return counts / total if total else counts


def clamp(v, lo, hi):
    return max(lo, min(v, hi))


def crop_region(image, row):
    w, hgt = image.size
    x, y = float(row["bbox_x"]), float(row["bbox_y"])
    bw, bh = float(row["bbox_width"]), float(row["bbox_height"])
    left = clamp(int(round(x)), 0, w)
    top = clamp(int(round(y)), 0, hgt)
    right = clamp(int(round(x + bw)), 0, w)
    bottom = clamp(int(round(y + bh)), 0, hgt)
    if right <= left or bottom <= top:
        return None
    return image.crop((left, top, right, bottom)).convert("RGB")


def main():
    mapping = pd.read_csv(REGION_MAPPING)
    print(f"Computing color descriptors for {len(mapping)} regions...")
    out = np.zeros((len(mapping), len(COLORS)), dtype=np.float32)

    cur_path, cur_img = None, None
    for pos, row in mapping.iterrows():
        p = str(PROJECT_ROOT / norm_path(row["image_path"]))
        if p != cur_path:
            if cur_img is not None:
                cur_img.close()
            cur_img = Image.open(p).convert("RGB")
            cur_path = p
        crop = crop_region(cur_img, row)
        if crop is not None:
            out[int(row["region_index"])] = color_fractions(crop)
        if (pos + 1) % 500 == 0:
            print(f"  {pos + 1}/{len(mapping)}")
    if cur_img is not None:
        cur_img.close()

    np.save(OUT_NPY, out)
    with open(OUT_META, "w") as f:
        json.dump({"colors": COLORS, "aliases": COLOR_ALIASES}, f, indent=2)
    print(f"Saved {OUT_NPY}  shape={out.shape}")
    print(f"Saved {OUT_META}")


if __name__ == "__main__":
    main()
