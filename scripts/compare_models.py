from pathlib import Path
import sys

import faiss
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT))


from src.indexer.encoders import (
    CLIPEncoder,
    FashionCLIPEncoder,
)


FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "model_comparison"
)

TOP_K = 5


def search(
    index,
    encoder,
    query,
):
    query_embedding = (
        encoder.encode_texts([query])
    )

    scores, indices = index.search(
        query_embedding,
        TOP_K,
    )

    return scores[0], indices[0]


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    mapping = pd.read_csv(
        FEATURE_DIR
        / "image_mapping.csv"
    )

    clip_index = faiss.read_index(
        str(
            FEATURE_DIR
            / "global_clip.faiss"
        )
    )

    fashion_index = faiss.read_index(
        str(
            FEATURE_DIR
            / "global_fashionclip.faiss"
        )
    )

    clip_encoder = CLIPEncoder()

    fashion_encoder = (
        FashionCLIPEncoder()
    )

    query = input(
        "Enter search query: "
    ).strip()

    clip_scores, clip_indices = search(
        clip_index,
        clip_encoder,
        query,
    )

    fashion_scores, fashion_indices = search(
        fashion_index,
        fashion_encoder,
        query,
    )

    fig, axes = plt.subplots(
        2,
        TOP_K,
        figsize=(18, 9),
    )

    results = [
        (
            "Generic CLIP",
            clip_scores,
            clip_indices,
        ),
        (
            "FashionCLIP",
            fashion_scores,
            fashion_indices,
        ),
    ]

    for row_index, (
        model_name,
        scores,
        indices,
    ) in enumerate(results):

        for rank, (
            score,
            index_position,
        ) in enumerate(
            zip(scores, indices)
        ):
            row = mapping.iloc[
                index_position
            ]

            image_path = (
                PROJECT_ROOT
                / row["image_path"]
            )

            image = Image.open(
                image_path
            ).convert("RGB")

            axes[
                row_index,
                rank,
            ].imshow(image)

            axes[
                row_index,
                rank,
            ].axis("off")

            axes[
                row_index,
                rank,
            ].set_title(
                f"Rank {rank + 1}\n"
                f"{score:.3f}\n"
                f"{row['image_id']}"
            )

        axes[
            row_index,
            0,
        ].set_ylabel(
            model_name,
            fontsize=14,
        )

    fig.suptitle(
        query,
        fontsize=18,
    )

    plt.tight_layout()

    safe_query = "".join(
        char if char.isalnum()
        else "_"
        for char in query
    )

    output_path = (
        OUTPUT_DIR
        / f"{safe_query[:80]}.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    print(
        f"Comparison saved to: "
        f"{output_path}"
    )

    plt.show()


if __name__ == "__main__":
    main()