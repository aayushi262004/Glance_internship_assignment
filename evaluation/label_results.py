from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULT_FILE = PROJECT_ROOT / "evaluation" / "retrieval_results.csv"

TOP_K = 5


def show_query_results(query_results):
    query = query_results.iloc[0]["query"]

    fig, axes = plt.subplots(
        1,
        TOP_K,
        figsize=(18, 5),
    )

    axes = list(axes)

    for axis, (_, result) in zip(
        axes,
        query_results.iterrows(),
    ):
        image_path = PROJECT_ROOT / result["image_path"]

        image = Image.open(image_path).convert("RGB")

        axis.imshow(image)

        axis.set_title(
            f"Rank {int(result['rank'])}\n"
            f"{result['image_id']}\n"
            f"Score: {result['final_score']:.3f}"
        )

        axis.axis("off")

    for axis in axes[len(query_results) :]:
        axis.axis("off")

    fig.suptitle(
        query,
        fontsize=18,
    )

    plt.tight_layout()
    plt.show()


def get_relevance_labels(expected_count):
    while True:
        labels_text = input(
            "\nEnter relevance labels "
            "(2 = relevant, 1 = partial, 0 = irrelevant)\n"
            f"Enter {expected_count} values: "
        ).strip()

        try:
            labels = [int(value) for value in labels_text.split()]
        except ValueError:
            print("Invalid input. Use only 0, 1, or 2.")
            continue

        if len(labels) != expected_count:
            print(f"Expected {expected_count} labels, " f"received {len(labels)}.")
            continue

        if any(label not in {0, 1, 2} for label in labels):
            print("Labels must only be 0, 1, or 2.")
            continue

        return labels


def main():
    results = pd.read_csv(RESULT_FILE)

    if "relevance" not in results.columns:
        results["relevance"] = pd.NA

    query_ids = results["query_id"].drop_duplicates().tolist()

    total_queries = len(query_ids)

    for query_number, query_id in enumerate(
        query_ids,
        start=1,
    ):
        query_mask = results["query_id"] == query_id

        query_results = results.loc[query_mask].sort_values("rank").copy()

        existing_labels = query_results["relevance"].dropna()

        if len(existing_labels) == len(query_results):
            print(
                f"\n[{query_number}/{total_queries}] "
                f"Query {query_id} already labeled. Skipping."
            )
            continue

        print(f"\n[{query_number}/{total_queries}]")

        print(
            "Query:",
            query_results.iloc[0]["query"],
        )

        show_query_results(query_results)

        labels = get_relevance_labels(len(query_results))

        ordered_indices = query_results.index.tolist()

        results.loc[
            ordered_indices,
            "relevance",
        ] = labels

        results.to_csv(
            RESULT_FILE,
            index=False,
        )

        print(
            "Labels saved:",
            labels,
        )

    print("\nAll evaluation results labeled.")

    print(f"Saved to: {RESULT_FILE}")


if __name__ == "__main__":
    main()
