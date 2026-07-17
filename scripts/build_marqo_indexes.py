"""Build global + region indexes with Marqo-FashionSigLIP.

Marqo-FashionSigLIP is a frozen, fashion-adapted SigLIP encoder that the Marqo
team report as a large improvement over FashionCLIP on fashion retrieval
benchmarks. We use it exactly like FashionCLIP here (drop-in, zero-shot): whole
images -> global_marqo.*, annotated garment crops -> region_marqo.*. Row order
matches the existing manifest / region_mapping so all downstream logic is reused.
"""
from pathlib import Path
import sys

import faiss
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from src.indexer.encoders import norm_path
from src.retriever.retriever_v2 import MarqoEncoder

FEATURE_DIR = PROJECT_ROOT / "data" / "features"
MANIFEST = PROJECT_ROOT / "data" / "metadata" / "manifest.csv"
REGION_MAPPING = FEATURE_DIR / "region_mapping.csv"
BATCH = 16


def clamp(v, lo, hi):
    return max(lo, min(v, hi))


def crop_region(image, row):
    w, h = image.size
    x, y = float(row["bbox_x"]), float(row["bbox_y"])
    bw, bh = float(row["bbox_width"]), float(row["bbox_height"])
    left, top = clamp(int(round(x)), 0, w), clamp(int(round(y)), 0, h)
    right, bottom = clamp(int(round(x + bw)), 0, w), clamp(int(round(y + bh)), 0, h)
    if right <= left or bottom <= top:
        return None
    return image.crop((left, top, right, bottom)).convert("RGB")


def build_global(encoder):
    manifest = pd.read_csv(MANIFEST)
    paths = manifest["image_path"].tolist()
    embs = []
    for s in range(0, len(paths), BATCH):
        batch = [PROJECT_ROOT / norm_path(p) for p in paths[s:s + BATCH]]
        embs.append(encoder.encode_images(batch))
        print(f"  global {min(s + BATCH, len(paths))}/{len(paths)}")
    embs = np.vstack(embs).astype(np.float32)
    np.save(FEATURE_DIR / "global_marqo.npy", embs)
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    faiss.write_index(idx, str(FEATURE_DIR / "global_marqo.faiss"))
    print(f"global_marqo: {embs.shape}")


def build_region(encoder):
    mapping = pd.read_csv(REGION_MAPPING).sort_values("region_index").reset_index(drop=True)
    embs = np.zeros((len(mapping), 0), dtype=np.float32)
    out = None
    cur_path, cur_img = None, None
    batch_crops, batch_rows = [], []

    def flush():
        nonlocal out, batch_crops, batch_rows
        if not batch_crops:
            return
        e = encoder.encode_images(batch_crops)
        if out is None:
            out = np.zeros((len(mapping), e.shape[1]), dtype=np.float32)
        for ri, vec in zip(batch_rows, e):
            out[ri] = vec
        batch_crops, batch_rows = [], []

    for pos, row in mapping.iterrows():
        p = str(PROJECT_ROOT / norm_path(row["image_path"]))
        if p != cur_path:
            if cur_img is not None:
                cur_img.close()
            cur_img = Image.open(p).convert("RGB")
            cur_path = p
        crop = crop_region(cur_img, row)
        if crop is None:
            crop = Image.new("RGB", (16, 16))  # keep row alignment
        batch_crops.append(crop)
        batch_rows.append(int(row["region_index"]))
        if len(batch_crops) >= BATCH:
            flush()
        if (pos + 1) % 500 == 0:
            print(f"  region {pos + 1}/{len(mapping)}")
    flush()
    if cur_img is not None:
        cur_img.close()
    np.save(FEATURE_DIR / "region_marqo.npy", out)
    idx = faiss.IndexFlatIP(out.shape[1])
    idx.add(out)
    faiss.write_index(idx, str(FEATURE_DIR / "region_marqo.faiss"))
    print(f"region_marqo: {out.shape}")


def main():
    enc = MarqoEncoder()
    print("=== global ===")
    build_global(enc)
    print("=== region ===")
    build_region(enc)
    print("MARQO_BUILD_DONE")


if __name__ == "__main__":
    main()
