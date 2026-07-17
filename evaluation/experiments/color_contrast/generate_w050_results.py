from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import src.retriever.fashion_context_retriever as fcr

QUERY_FILE = PROJECT_ROOT / "evaluation" / "queries.csv"

# Never written to -- only read, to reuse existing relevance labels.
BASELINE_LABELED_FILE = PROJECT_ROOT / "evaluation" / "retrieval_results.csv"

OUTPUT_FILE = Path(__file__).resolve().parent / "retrieval_results_w050.csv"

TOP_K = 5

# Evaluation-only override of the candidate weight under test. Set at
# runtime, not in the retriever source, so fashion_context_retriever.py
# stays unmodified for this comparison.
COLOR_CONTRAST_WEIGHT_UNDER_TEST = 0.50


def load_label_lookup():
    baseline = pd.read_csv(BASELINE_LABELED_FILE)

    baseline["relevance"] = pd.to_numeric(baseline["relevance"], errors="coerce")

    return {
        (int(row.query_id), row.image_id): row.relevance
        for row in baseline.itertuples()
        if not pd.isna(row.relevance)
    }


def main():
    queries = pd.read_csv(QUERY_FILE)

    if "query" not in queries.columns:
        raise ValueError("queries.csv must contain a 'query' column.")

    label_lookup = load_label_lookup()

    fcr.COLOR_CONTRAST_WEIGHT = COLOR_CONTRAST_WEIGHT_UNDER_TEST

    retriever = fcr.FashionContextRetriever()

    all_results = []
    new_pairs = []

    total_queries = len(queries)

    for query_number, row in queries.iterrows():
        query_id = query_number + 1
        query = str(row["query"]).strip()

        print(f"\n[{query_id}/{total_queries}]")
        print(f"Query: {query}")

        results = retriever.search(query=query, top_k=TOP_K)

        for _, result in results.iterrows():
            image_id = result["image_id"]
            key = (query_id, image_id)

            if key in label_lookup:
                relevance = label_lookup[key]
            else:
                relevance = ""
                new_pairs.append(key)

            all_results.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "category": row.get("category", ""),
                    "rank": int(result["rank"]),
                    "image_id": image_id,
                    "image_path": result["image_path"],
                    "final_score": result["final_score"],
                    "global_score": result["global_score"],
                    "region_score": result["region_score"],
                    "coverage_score": result["coverage_score"],
                    "all_clauses_matched": result["all_clauses_matched"],
                    "style_score": result["style_score"],
                    "context_score": result["context_score"],
                    "relevance": relevance,
                }
            )

        print("Top images:", results["image_id"].tolist())

    evaluation_results = pd.DataFrame(all_results)

    evaluation_results.to_csv(OUTPUT_FILE, index=False)

    print("\nWeight-0.50 evaluation retrieval complete.")
    print(f"Color contrast weight under test: {COLOR_CONTRAST_WEIGHT_UNDER_TEST}")
    print(f"Queries: {total_queries}")
    print(f"Rows: {len(evaluation_results)}")
    print(f"Reused labels: {len(evaluation_results) - len(new_pairs)}")
    print(f"New unlabeled pairs: {len(new_pairs)}")

    for pair in new_pairs:
        print(f"  {pair}")

    print(f"\nResults: {OUTPUT_FILE}")
    print(f"(baseline file untouched: {BASELINE_LABELED_FILE})")
    print("\nNext: python evaluation/experiments/color_contrast/label_new_results.py")


if __name__ == "__main__":
    main()
