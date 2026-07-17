from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


from src.retriever.fashion_context_retriever import (
    FashionContextRetriever,
    minmax,
)

QUERY_FILE = PROJECT_ROOT / "evaluation" / "queries.csv"

OUTPUT_FILE = PROJECT_ROOT / "evaluation" / "ablation_results.csv"

TOP_K = 5
SEARCH_K = 150


def rank_config(results, config_name):
    ranked = results.copy()

    global_norm = minmax(ranked["global_score"])

    region_norm = minmax(ranked["region_score"])

    style_norm = minmax(ranked["style_score"])

    context_norm = minmax(ranked["context_score"])

    if config_name == "global_only":
        score = global_norm

    elif config_name == "global_region":
        score = 0.45 * global_norm + 0.55 * region_norm

    elif config_name == "region_coverage":
        score = (
            0.30 * global_norm
            + 0.40 * region_norm
            + 0.20 * ranked["coverage_score"]
            + 0.10 * ranked["all_clauses_matched"]
        )

    elif config_name == "full_model":
        components = [
            (0.25, global_norm),
        ]

        if ranked["region_score"].abs().sum() > 0:
            components.append((0.30, region_norm))

            components.append(
                (
                    0.15,
                    ranked["coverage_score"],
                )
            )

            if ranked["all_clauses_matched"].abs().sum() > 0:
                components.append(
                    (
                        0.10,
                        ranked["all_clauses_matched"],
                    )
                )

        if ranked["style_score"].abs().sum() > 0:
            components.append((0.15, style_norm))

        if ranked["context_score"].abs().sum() > 0:
            components.append((0.15, context_norm))

        total_weight = sum(weight for weight, _ in components)

        score = sum((weight / total_weight) * values for weight, values in components)

    else:
        raise ValueError(f"Unknown config: {config_name}")

    ranked["ablation_score"] = score

    return (
        ranked.sort_values(
            "ablation_score",
            ascending=False,
        )
        .head(TOP_K)
        .reset_index(drop=True)
    )


def main():
    queries = pd.read_csv(QUERY_FILE)

    retriever = FashionContextRetriever()

    configurations = [
        "global_only",
        "global_region",
        "region_coverage",
        "full_model",
    ]

    rows = []

    for query_index, row in queries.iterrows():
        query = str(row["query"]).strip()

        print(f"\n[{query_index + 1}/" f"{len(queries)}] {query}")

        candidates = retriever.search(
            query=query,
            top_k=SEARCH_K,
        )

        for config_name in configurations:
            ranked = rank_config(
                candidates,
                config_name,
            )

            top_images = ranked["image_id"].tolist()

            print(f"{config_name:18s} " f"{top_images}")

            for rank, (_, result) in enumerate(
                ranked.iterrows(),
                start=1,
            ):
                rows.append(
                    {
                        "query_id": query_index + 1,
                        "query": query,
                        "category": row.get(
                            "category",
                            "",
                        ),
                        "configuration": config_name,
                        "rank": rank,
                        "image_id": result["image_id"],
                        "image_path": result["image_path"],
                        "score": result["ablation_score"],
                    }
                )

    output = pd.DataFrame(rows)

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nAblation complete.")

    print(f"Rows: {len(output)}")

    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
