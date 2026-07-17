from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


from src.retriever.fashion_context_retriever import (
    FashionContextRetriever,
)

QUERY_FILE = PROJECT_ROOT / "evaluation" / "queries.csv"

OUTPUT_FILE = PROJECT_ROOT / "evaluation" / "retrieval_results.csv"

TOP_K = 5


def main():

    queries = pd.read_csv(QUERY_FILE)

    if "query" not in queries.columns:
        raise ValueError("queries.csv must contain " "a 'query' column.")

    retriever = FashionContextRetriever()

    all_results = []

    total_queries = len(queries)

    for query_number, row in queries.iterrows():

        query = str(row["query"]).strip()

        print(f"\n[{query_number + 1}/" f"{total_queries}]")

        print(f"Query: {query}")

        results = retriever.search(
            query=query,
            top_k=TOP_K,
        )

        for _, result in results.iterrows():

            all_results.append(
                {
                    "query_id": (query_number + 1),
                    "query": query,
                    "category": row.get(
                        "category",
                        "",
                    ),
                    "rank": int(result["rank"]),
                    "image_id": result["image_id"],
                    "image_path": result["image_path"],
                    "final_score": result["final_score"],
                    "global_score": result["global_score"],
                    "region_score": result["region_score"],
                    "coverage_score": result["coverage_score"],
                    "all_clauses_matched": result["all_clauses_matched"],
                    "style_score": result["style_score"],
                    "context_score": result["context_score"],
                    "relevance": "",
                }
            )

        print(
            "Top images:",
            results["image_id"].tolist(),
        )

    evaluation_results = pd.DataFrame(all_results)

    evaluation_results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nEvaluation retrieval complete.")

    print(f"Queries: {total_queries}")

    print(f"Judgments required: " f"{len(evaluation_results)}")

    print(f"Results: {OUTPUT_FILE}")

    print("\nNext: manually label the " "'relevance' column:")

    print("2 = highly relevant")

    print("1 = partially relevant")

    print("0 = irrelevant")


if __name__ == "__main__":
    main()
