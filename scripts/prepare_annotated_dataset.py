from pathlib import Path
import json
import random
import shutil

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ANNOTATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "instances_attributes_val2020.json"
)

RAW_DIR = PROJECT_ROOT / "data" / "raw"

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "manifest.csv"
)

REGIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "garment_regions.csv"
)

TARGET_IMAGES = 1000
SEED = 42


def find_source_images():
    """
    Search likely locations for the original val/test images.
    """

    search_roots = [
        Path.home() / "Downloads",
        Path("C:/Users/aayus/Downloads"),
    ]

    image_lookup = {}

    print("Searching for source Fashionpedia images...")

    for root in search_roots:
        if not root.exists():
            continue

        print(f"Scanning: {root}")

        for extension in ("*.jpg", "*.jpeg", "*.png"):
            for path in root.rglob(extension):
                image_lookup.setdefault(
                    path.name,
                    path,
                )

    print(
        f"Found {len(image_lookup)} unique image filenames."
    )

    return image_lookup


def main():
    random.seed(SEED)

    print("Loading Fashionpedia annotations...")

    with open(
        ANNOTATION_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    images = data["images"]
    annotations = data["annotations"]
    categories = data["categories"]

    print(f"Annotated images: {len(images)}")
    print(f"Annotations: {len(annotations)}")
    print(f"Categories: {len(categories)}")

    category_map = {
        category["id"]: category["name"]
        for category in categories
    }

    image_info = {
        image["id"]: image
        for image in images
    }

    source_lookup = find_source_images()

    available_images = []

    for image in images:
        filename = image["file_name"]

        if filename in source_lookup:
            available_images.append(image)

    print(
        "Annotated images available locally:",
        len(available_images),
    )

    if not available_images:
        raise RuntimeError(
            "No annotated validation images were found locally. "
            "Make sure val_test2020.zip is extracted in Downloads."
        )

    if len(available_images) < TARGET_IMAGES:
        print(
            f"WARNING: only {len(available_images)} "
            f"annotated images are locally available."
        )

    random.shuffle(available_images)

    selected_images = available_images[
        :min(TARGET_IMAGES, len(available_images))
    ]

    selected_ids = {
        image["id"]
        for image in selected_images
    }

    print(
        f"Selected {len(selected_images)} annotated images."
    )

    # --------------------------------------------------
    # Backup current corpus
    # --------------------------------------------------

    backup_dir = (
        PROJECT_ROOT
        / "data"
        / "raw_baseline_v1"
    )

    if RAW_DIR.exists() and any(RAW_DIR.iterdir()):
        if not backup_dir.exists():
            print(
                "Backing up current baseline corpus..."
            )

            shutil.copytree(
                RAW_DIR,
                backup_dir,
            )

            print(
                f"Baseline backup: {backup_dir}"
            )

    # --------------------------------------------------
    # Rebuild raw directory
    # --------------------------------------------------

    if RAW_DIR.exists():
        shutil.rmtree(RAW_DIR)

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_rows = []

    original_to_new = {}

    print("Copying selected images...")

    for index, image in enumerate(selected_images):
        original_filename = image["file_name"]

        new_image_id = f"img_{index:04d}"
        new_filename = f"{new_image_id}.jpg"

        source_path = source_lookup[
            original_filename
        ]

        destination_path = (
            RAW_DIR / new_filename
        )

        shutil.copy2(
            source_path,
            destination_path,
        )

        original_to_new[
            image["id"]
        ] = {
            "image_id": new_image_id,
            "image_path": str(
                destination_path.relative_to(
                    PROJECT_ROOT
                )
            ),
            "original_filename": original_filename,
        }

        manifest_rows.append(
            {
                "image_id": new_image_id,
                "image_path": str(
                    destination_path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "fashionpedia_image_id": image["id"],
                "original_filename": original_filename,
                "width": image["width"],
                "height": image["height"],
                "split": "validation",
            }
        )

    manifest = pd.DataFrame(
        manifest_rows
    )

    manifest.to_csv(
        MANIFEST_PATH,
        index=False,
    )

    # --------------------------------------------------
    # Save apparel region metadata
    # --------------------------------------------------

    region_rows = []

    for annotation in annotations:
        image_id = annotation["image_id"]

        if image_id not in selected_ids:
            continue

        bbox = annotation.get("bbox")

        if not bbox or len(bbox) != 4:
            continue

        x, y, width, height = bbox

        if width <= 1 or height <= 1:
            continue

        mapping = original_to_new[
            image_id
        ]

        category_id = annotation[
            "category_id"
        ]

        region_rows.append(
            {
                "image_id": mapping["image_id"],
                "image_path": mapping["image_path"],
                "fashionpedia_image_id": image_id,
                "annotation_id": annotation["id"],
                "category_id": category_id,
                "category_name": category_map.get(
                    category_id,
                    "unknown",
                ),
                "bbox_x": x,
                "bbox_y": y,
                "bbox_width": width,
                "bbox_height": height,
                "attribute_ids": json.dumps(
                    annotation.get(
                        "attribute_ids",
                        [],
                    )
                ),
            }
        )

    regions = pd.DataFrame(
        region_rows
    )

    regions.to_csv(
        REGIONS_PATH,
        index=False,
    )

    print("\nDataset preparation complete.")
    print(
        f"Images: {len(manifest)}"
    )
    print(
        f"Garment/apparel regions: {len(regions)}"
    )
    print(
        f"Manifest: {MANIFEST_PATH}"
    )
    print(
        f"Regions: {REGIONS_PATH}"
    )

    print("\nTop region categories:")
    print(
        regions["category_name"]
        .value_counts()
        .head(20)
    )


if __name__ == "__main__":
    main()