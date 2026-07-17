from pathlib import Path
import sys

import faiss
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT))


from src.indexer.encoders import FashionCLIPEncoder, norm_path


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

BATCH_SIZE = 16


def main():
    FEATURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    print(
        f"Loaded {len(manifest)} images"
    )

    encoder = FashionCLIPEncoder()

    image_paths = manifest[
        "image_path"
    ].tolist()

    all_embeddings = []

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

        embeddings = encoder.encode_images(
            batch_paths
        )

        all_embeddings.append(
            embeddings
        )

        print(
            f"Encoded {end}/{len(image_paths)}"
        )

    embeddings = np.vstack(
        all_embeddings
    ).astype(np.float32)

    print(
        "Fashion embedding shape:",
        embeddings.shape,
    )

    np.save(
        FEATURE_DIR
        / "global_fashionclip.npy",
        embeddings,
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    faiss.write_index(
        index,
        str(
            FEATURE_DIR
            / "global_fashionclip.faiss"
        ),
    )

    print(
        f"FashionCLIP index contains "
        f"{index.ntotal} vectors"
    )

    print(
        "FashionCLIP indexing complete."
    )


if __name__ == "__main__":
    main()