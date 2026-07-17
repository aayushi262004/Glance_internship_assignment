from pathlib import Path
import sys

import faiss
import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.indexer.encoders import FashionCLIPEncoder, norm_path


REGIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "garment_regions.csv"
)

FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
)

REGION_EMBEDDINGS_PATH = (
    FEATURE_DIR
    / "region_fashionclip.npy"
)

REGION_INDEX_PATH = (
    FEATURE_DIR
    / "region_fashionclip.faiss"
)

REGION_MAPPING_PATH = (
    FEATURE_DIR
    / "region_mapping.csv"
)


# -------------------------------------------------------
# Categories useful for localized fashion retrieval
# -------------------------------------------------------

INDEXED_CATEGORY_IDS = set(range(0, 27))

BATCH_SIZE = 32


def clamp(value, minimum, maximum):
    return max(
        minimum,
        min(value, maximum),
    )


def crop_region(image, row):
    """
    Crop a Fashionpedia annotated apparel instance.
    bbox format: [x, y, width, height]
    """

    image_width, image_height = image.size

    x = float(row["bbox_x"])
    y = float(row["bbox_y"])
    width = float(row["bbox_width"])
    height = float(row["bbox_height"])

    left = clamp(
        int(round(x)),
        0,
        image_width,
    )

    top = clamp(
        int(round(y)),
        0,
        image_height,
    )

    right = clamp(
        int(round(x + width)),
        0,
        image_width,
    )

    bottom = clamp(
        int(round(y + height)),
        0,
        image_height,
    )

    if right <= left or bottom <= top:
        return None

    return image.crop(
        (left, top, right, bottom)
    ).convert("RGB")


def main():
    FEATURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    regions = pd.read_csv(
        REGIONS_PATH
    )

    print(
        f"Loaded {len(regions)} annotated regions."
    )

    # --------------------------------------------------
    # Keep garment + wearable/accessory regions.
    # Remove low-level garment details for primary MaxSim.
    # --------------------------------------------------

    regions = regions[
        regions["category_id"].isin(
            INDEXED_CATEGORY_IDS
        )
    ].copy()

    regions = regions.reset_index(
        drop=True
    )

    print(
        f"Indexing {len(regions)} major fashion regions."
    )

    print("\nIndexed categories:")
    print(
        regions["category_name"]
        .value_counts()
    )

    encoder = FashionCLIPEncoder()

    all_embeddings = []
    valid_mapping_rows = []

    crop_batch = []
    mapping_batch = []

    current_image_path = None
    current_image = None

    total_regions = len(regions)

    def flush_batch():
        nonlocal crop_batch
        nonlocal mapping_batch

        if not crop_batch:
            return

        embeddings = encoder.encode_images(
            crop_batch
        )

        all_embeddings.append(
            embeddings
        )

        valid_mapping_rows.extend(
            mapping_batch
        )

        crop_batch = []
        mapping_batch = []

    for position, row in regions.iterrows():
        image_path = (
            PROJECT_ROOT
            / norm_path(row["image_path"])
        )

        image_path_string = str(image_path)

        if image_path_string != current_image_path:
            if current_image is not None:
                current_image.close()

            current_image = Image.open(
                image_path
            ).convert("RGB")

            current_image_path = (
                image_path_string
            )

        crop = crop_region(
            current_image,
            row,
        )

        if crop is None:
            continue

        crop_batch.append(crop)

        mapping_batch.append(
            {
                "region_index": len(
                    valid_mapping_rows
                ) + len(mapping_batch),
                "image_id": row["image_id"],
                "image_path": row["image_path"],
                "annotation_id": row["annotation_id"],
                "category_id": int(
                    row["category_id"]
                ),
                "category_name": row[
                    "category_name"
                ],
                "bbox_x": row["bbox_x"],
                "bbox_y": row["bbox_y"],
                "bbox_width": row["bbox_width"],
                "bbox_height": row["bbox_height"],
                "attribute_ids": row[
                    "attribute_ids"
                ],
            }
        )

        if len(crop_batch) >= BATCH_SIZE:
            flush_batch()

        processed = position + 1

        if processed % 250 == 0:
            print(
                f"Processed "
                f"{processed}/{total_regions}"
            )

    flush_batch()

    if current_image is not None:
        current_image.close()

    if not all_embeddings:
        raise RuntimeError(
            "No valid region embeddings were generated."
        )

    embeddings = np.vstack(
        all_embeddings
    ).astype(np.float32)

    mapping = pd.DataFrame(
        valid_mapping_rows
    )

    if len(embeddings) != len(mapping):
        raise RuntimeError(
            "Embedding and region mapping counts differ."
        )

    print(
        "\nRegion embedding shape:",
        embeddings.shape,
    )

    np.save(
        REGION_EMBEDDINGS_PATH,
        embeddings,
    )

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(
        embeddings
    )

    faiss.write_index(
        index,
        str(REGION_INDEX_PATH),
    )

    mapping.to_csv(
        REGION_MAPPING_PATH,
        index=False,
    )

    print(
        f"Region index contains "
        f"{index.ntotal} vectors."
    )

    print(
        f"Embeddings: "
        f"{REGION_EMBEDDINGS_PATH}"
    )

    print(
        f"FAISS index: "
        f"{REGION_INDEX_PATH}"
    )

    print(
        f"Mapping: "
        f"{REGION_MAPPING_PATH}"
    )

    print(
        "\nFashion region indexing complete."
    )


if __name__ == "__main__":
    main()