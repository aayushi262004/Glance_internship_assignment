from pathlib import Path
import sys

import faiss
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT))


from src.indexer.encoders import CLIPEncoder


FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "baseline"
)

TOP_K = 5


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    index = faiss.read_index(
        str(
            FEATURE_DIR
            / "global_clip.faiss"
        )
    )

    mapping = pd.read_csv(
        FEATURE_DIR
        / "image_mapping.csv"
    )

    encoder = CLIPEncoder()

    query = input(
        "Enter search query: "
    ).strip()

    if not query:
        raise ValueError(
            "Query cannot be empty."
        )

    query_embedding = (
        encoder.encode_texts([query])
    )

    scores, indices = index.search(
        query_embedding,
        TOP_K,
    )

    fig, axes = plt.subplots(
        1,
        TOP_K,
        figsize=(18, 5),
    )

    for rank, (
        score,
        index_position,
    ) in enumerate(
        zip(scores[0], indices[0])
    ):
        row = mapping.iloc[index_position]

        image_path = (
            PROJECT_ROOT
            / row["image_path"]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        axes[rank].imshow(image)

        axes[rank].axis("off")

        axes[rank].set_title(
            f"Rank {rank + 1}\n"
            f"Score: {score:.4f}\n"
            f"{row['image_id']}"
        )

    fig.suptitle(
        query,
        fontsize=16,
    )

    plt.tight_layout()

    safe_query = "".join(
        character
        if character.isalnum()
        else "_"
        for character in query
    )

    output_path = (
        OUTPUT_DIR
        / f"{safe_query[:80]}.png"
    )

    plt.savefig(
        output_path,
        bbox_inches="tight",
        dpi=150,
    )

    print(
        f"Result grid saved to: "
        f"{output_path}"
    )

    plt.show()


if __name__ == "__main__":
    main()