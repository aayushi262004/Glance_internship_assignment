import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_FILE = PROJECT_ROOT / "evaluation" / "retrieval_results.csv"

TOP_K = 5

# A retrieved image counts as "relevant" for Precision@5 and mAP when its
# manually assigned relevance label is at least this value.
# (0 = irrelevant, 1 = partially relevant, 2 = highly relevant -- see the
# rubric printed by evaluation/label_results.py.) nDCG@5 uses the raw graded
# label directly and does not use this threshold.
RELEVANCE_MATCH_THRESHOLD = 1


def precision_at_k(relevance_labels):
    binary = [1 if label >= RELEVANCE_MATCH_THRESHOLD else 0 for label in relevance_labels]

    return sum(binary) / len(binary)


def average_precision(relevance_labels):
    """
    Average Precision over the retrieved top-K pool. Because only the top-K
    results per query were retrieved and labeled (no exhaustive corpus-wide
    relevance judgments exist), this is bounded to the K-item pool -- i.e.
    it is really AP@K, not corpus-complete AP. Reported as "mAP" per the
    request, but this scope limitation should be kept in mind.
    """

    binary = [1 if label >= RELEVANCE_MATCH_THRESHOLD else 0 for label in relevance_labels]

    num_relevant = sum(binary)

    if num_relevant == 0:
        return 0.0

    precision_sum = 0.0
    hits = 0

    for position, is_relevant in enumerate(binary, start=1):
        if is_relevant:
            hits += 1
            precision_sum += hits / position

    return precision_sum / num_relevant


def dcg(relevance_labels):
    return sum(
        (2**label - 1) / np.log2(position + 1)
        for position, label in enumerate(relevance_labels, start=1)
    )


def ndcg_at_k(relevance_labels):
    actual_dcg = dcg(relevance_labels)

    ideal_dcg = dcg(sorted(relevance_labels, reverse=True))

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def load_results(results_file):
    results = pd.read_csv(results_file)

    if "relevance" not in results.columns:
        raise ValueError(
            f"{results_file.name} has no 'relevance' column. "
            "Run evaluation/run_evaluation.py first."
        )

    results["relevance"] = pd.to_numeric(results["relevance"], errors="coerce")

    return results


def evaluate_query(group):
    """
    Compute metrics for one query's retrieved results. Unlabeled rows are
    never treated as irrelevant: a query with any missing label is marked
    incomplete/not-labeled and excluded from metric aggregation entirely,
    rather than silently scoring the missing rows as 0.
    """

    group = group.sort_values("rank")

    if len(group) != TOP_K:
        return {"status": f"unexpected result count ({len(group)} != {TOP_K})"}

    relevance = group["relevance"].tolist()

    labeled_mask = [not pd.isna(value) for value in relevance]

    num_labeled = sum(labeled_mask)

    if num_labeled == 0:
        return {"status": "not labeled", "num_labeled": 0, "num_expected": TOP_K}

    if num_labeled < TOP_K:
        missing_ranks = [
            int(rank) for rank, labeled in zip(group["rank"], labeled_mask) if not labeled
        ]

        return {
            "status": "incomplete",
            "num_labeled": num_labeled,
            "num_expected": TOP_K,
            "missing_ranks": missing_ranks,
        }

    relevance_labels = [int(value) for value in relevance]

    return {
        "status": "complete",
        "num_labeled": TOP_K,
        "num_expected": TOP_K,
        "precision_at_5": precision_at_k(relevance_labels),
        "average_precision": average_precision(relevance_labels),
        "ndcg_at_5": ndcg_at_k(relevance_labels),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results-file",
        type=Path,
        default=RESULTS_FILE,
        help="Path to a retrieval results CSV (default: evaluation/retrieval_results.csv).",
    )

    args = parser.parse_args()

    results_file = args.results_file

    # Derive a matching metrics output filename so a non-default input never
    # clobbers the default evaluation/metrics_results.csv.
    metrics_output_file = results_file.parent / results_file.name.replace(
        "retrieval_results", "metrics_results"
    )

    results = load_results(results_file)

    per_query_rows = []

    for query_id, group in results.groupby("query_id", sort=True):
        query_text = group["query"].iloc[0]
        category = group["category"].iloc[0]

        metrics = evaluate_query(group)

        per_query_rows.append(
            {
                "query_id": query_id,
                "query": query_text,
                "category": category,
                **metrics,
            }
        )

    per_query = pd.DataFrame(per_query_rows)

    per_query.to_csv(metrics_output_file, index=False)

    complete = per_query[per_query["status"] == "complete"]
    incomplete = per_query[per_query["status"] == "incomplete"]
    not_labeled = per_query[per_query["status"] == "not labeled"]

    print("=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)

    print(f"\nSource: {results_file}")
    print(f"Total queries: {len(per_query)}")
    print(f"  Fully labeled (used for metrics): {len(complete)}")
    print(f"  Partially labeled (excluded):     {len(incomplete)}")
    print(f"  Not labeled at all (excluded):    {len(not_labeled)}")

    if len(incomplete) or len(not_labeled):
        print("\nQueries excluded from metrics (need labeling):")

        excluded = pd.concat([incomplete, not_labeled]).sort_values("query_id")

        for _, row in excluded.iterrows():
            missing = row.get("missing_ranks")

            missing_description = "all ranks" if pd.isna(missing) or missing is None else missing

            print(f'  query_id={int(row["query_id"]):>2}  "{row["query"]}"  missing: {missing_description}')

    if len(complete) == 0:
        print("\nNo fully labeled queries available -- cannot compute metrics honestly.")
        print("Run: python evaluation/label_results.py")
        print(f"\nSaved (empty) per-query metrics to: {metrics_output_file}")
        return

    print(f"\n--- OVERALL (n={len(complete)} fully labeled queries) ---")
    print(f"Precision@5: {complete['precision_at_5'].mean():.4f}")
    print(f"mAP:         {complete['average_precision'].mean():.4f}")
    print(f"nDCG@5:      {complete['ndcg_at_5'].mean():.4f}")

    print("\n--- PER-QUERY (fully labeled only) ---")
    print(
        complete[
            ["query_id", "query", "category", "precision_at_5", "average_precision", "ndcg_at_5"]
        ].to_string(index=False)
    )

    print("\n--- PER-CATEGORY (fully labeled only) ---")

    category_summary = (
        complete.groupby("category")
        .agg(
            num_queries=("query_id", "count"),
            precision_at_5=("precision_at_5", "mean"),
            mAP=("average_precision", "mean"),
            ndcg_at_5=("ndcg_at_5", "mean"),
        )
        .reset_index()
    )

    print(category_summary.to_string(index=False))

    print(f"\nSaved per-query metrics to: {metrics_output_file}")


if __name__ == "__main__":
    main()
