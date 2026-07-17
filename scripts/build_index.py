from pathlib import Path
import sys

import faiss
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT))


from src.indexer.encoders import CLIPEncoder, norm_path


MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "manifest.csv"
)

FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
)

BATCH_SIZE = 32


def main():
    FEATURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = pd.read_csv(MANIFEST_PATH)

    print(
        f"Loaded {len(manifest)} images "
        f"from manifest"
    )

    encoder = CLIPEncoder()

    all_embeddings = []

    image_paths = manifest[
        "image_path"
    ].tolist()

    for start in range(
        0,
        len(image_paths),
        BATCH_SIZE,
    ):
        end = min(
            start + BATCH_SIZE,
            len(image_paths),
        )

        batch_paths = [
            PROJECT_ROOT / norm_path(path)
            for path in image_paths[start:end]
        ]

        batch_embeddings = (
            encoder.encode_images(batch_paths)
        )

        all_embeddings.append(
            batch_embeddings
        )

        print(
            f"Encoded {end}/{len(image_paths)}"
        )

    embeddings = np.vstack(
        all_embeddings
    ).astype(np.float32)

    print(
        "Embedding shape:",
        embeddings.shape,
    )

    np.save(
        FEATURE_DIR / "global_clip.npy",
        embeddings,
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    faiss.write_index(
        index,
        str(
            FEATURE_DIR
            / "global_clip.faiss"
        ),
    )

    manifest[
        [
            "image_id",
            "image_path",
            "original_filename",
        ]
    ].to_csv(
        FEATURE_DIR / "image_mapping.csv",
        index=False,
    )

    print(
        f"FAISS index contains "
        f"{index.ntotal} vectors"
    )

    print("Indexing complete.")


if __name__ == "__main__":
    main()