from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

AUDIT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "dataset_audit"
    / "semantic_audit.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "dataset_audit"
    / "grids"
)

TOP_N = 12


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_df = pd.read_csv(AUDIT_PATH)

    concepts = audit_df["concept"].unique()

    for concept in concepts:
        concept_df = (
            audit_df[
                audit_df["concept"] == concept
            ]
            .sort_values("rank")
            .head(TOP_N)
        )

        fig, axes = plt.subplots(
            3,
            4,
            figsize=(14, 11),
        )

        axes = axes.flatten()

        for axis in axes:
            axis.axis("off")

        for axis, (_, row) in zip(
            axes,
            concept_df.iterrows(),
        ):
            image_path = (
                PROJECT_ROOT
                / row["image_path"]
            )

            image = Image.open(
                image_path
            ).convert("RGB")

            axis.imshow(image)

            axis.set_title(
                f"Rank {row['rank']}\n"
                f"Score: {row['score']:.3f}\n"
                f"{row['image_id']}",
                fontsize=9,
            )

            axis.axis("off")

        fig.suptitle(
            f"Dataset Audit — {concept}",
            fontsize=18,
        )

        plt.tight_layout()

        output_path = (
            OUTPUT_DIR
            / f"{concept}.png"
        )

        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(
            f"Saved: {output_path}"
        )

    print(
        "\nDataset audit visualization complete."
    )


if __name__ == "__main__":
    main()