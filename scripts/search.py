from pathlib import Path
import sys

import faiss
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT))


from src.indexer.encoders import CLIPEncoder

FEATURE_DIR = PROJECT_ROOT / "data" / "features"

TOP_K = 20


def main():
    index = faiss.read_index(str(FEATURE_DIR / "global_clip.faiss"))

    mapping = pd.read_csv(FEATURE_DIR / "image_mapping.csv")

    encoder = CLIPEncoder()

    query = input("Enter search query: ").strip()

    if not query:
        raise ValueError("Query cannot be empty.")

    query_embedding = encoder.encode_texts([query])

    scores, indices = index.search(
        query_embedding,
        TOP_K,
    )

    print("\nTop results:\n")

    results = []

    for rank, (
        score,
        index_position,
    ) in enumerate(
        zip(scores[0], indices[0]),
        start=1,
    ):
        row = mapping.iloc[index_position]

        image_path = PROJECT_ROOT / row["image_path"]

        results.append(
            (
                rank,
                float(score),
                row["image_id"],
                image_path,
            )
        )

        print(f"{rank}. " f"{row['image_id']} | " f"score={score:.4f}")

        print(f"   {row['image_path']}")

    fig, axes = plt.subplots(
        4,
        5,
        figsize=(18, 14),
    )

    axes = axes.flatten()

    for axis, result in zip(
        axes,
        results,
    ):
        (
            rank,
            score,
            image_id,
            image_path,
        ) = result

        image = Image.open(image_path).convert("RGB")

        axis.imshow(image)

        axis.set_title(f"Rank {rank}\n" f"{image_id}\n" f"{score:.3f}")

        axis.axis("off")

    plt.suptitle(
        f"Pure CLIP context audit\n{query}",
        fontsize=18,
    )

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
