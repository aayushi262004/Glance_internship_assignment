"""Reusable retrieval metrics + label-store scoring.

Given a retrieval_results DataFrame (columns: query, rank, image_id) and a
label store (query, image_id, relevance), compute P@5 / mAP@5 / nDCG@5 per
query and overall, and list any (query, image) pairs that are not yet labeled
(so they can be judged before the numbers are trusted).
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / "evaluation" / "harness" / "label_store.csv"
TOP_K = 5
REL_THRESHOLD = 1


def load_store():
    s = pd.read_csv(STORE)
    return {
        (str(r["query"]).strip(), str(r["image_id"]).strip()): int(r["relevance"])
        for _, r in s.iterrows()
    }


def precision_at_k(labels):
    b = [1 if x >= REL_THRESHOLD else 0 for x in labels]
    return sum(b) / len(b)


def average_precision(labels):
    b = [1 if x >= REL_THRESHOLD else 0 for x in labels]
    nrel = sum(b)
    if nrel == 0:
        return 0.0
    hits = 0
    s = 0.0
    for i, rel in enumerate(b, start=1):
        if rel:
            hits += 1
            s += hits / i
    return s / nrel


def dcg(labels):
    return sum((2 ** l - 1) / np.log2(i + 1) for i, l in enumerate(labels, start=1))


def ndcg_at_k(labels):
    ideal = dcg(sorted(labels, reverse=True))
    return dcg(labels) / ideal if ideal else 0.0


def evaluate(results_df, store=None, verbose=True):
    """results_df needs columns: query, rank, image_id, category (optional)."""
    if store is None:
        store = load_store()

    per_query = []
    unlabeled = []
    for query, g in results_df.groupby("query", sort=False):
        g = g.sort_values("rank").head(TOP_K)
        labels = []
        complete = True
        for _, r in g.iterrows():
            key = (str(query).strip(), str(r["image_id"]).strip())
            if key in store:
                labels.append(store[key])
            else:
                complete = False
                unlabeled.append({"query": query, "rank": int(r["rank"]),
                                  "image_id": r["image_id"],
                                  "image_path": r.get("image_path", "")})
                labels.append(None)
        cat = g["category"].iloc[0] if "category" in g.columns else ""
        if complete and len(labels) == TOP_K:
            per_query.append({
                "query": query, "category": cat,
                "precision_at_5": precision_at_k(labels),
                "mAP": average_precision(labels),
                "ndcg_at_5": ndcg_at_k(labels),
                "status": "complete",
            })
        else:
            per_query.append({"query": query, "category": cat, "status": "incomplete"})

    pq = pd.DataFrame(per_query)
    complete = pq[pq["status"] == "complete"]
    summary = {}
    if len(complete):
        summary = {
            "n_complete": len(complete),
            "precision_at_5": complete["precision_at_5"].mean(),
            "mAP": complete["mAP"].mean(),
            "ndcg_at_5": complete["ndcg_at_5"].mean(),
        }
    if verbose:
        if unlabeled:
            print(f"\n!!! {len(unlabeled)} UNLABELED (query, image) pairs — judge these before trusting metrics:")
            for u in unlabeled:
                print(f'    "{u["query"]}"  rank={u["rank"]}  {u["image_id"]}  {u["image_path"]}')
        if summary:
            print(f"\n=== OVERALL ({summary['n_complete']} fully-labeled queries) ===")
            print(f"Precision@5: {summary['precision_at_5']:.4f}")
            print(f"mAP@5:       {summary['mAP']:.4f}")
            print(f"nDCG@5:      {summary['ndcg_at_5']:.4f}")
            if len(complete) and "category" in complete.columns:
                print("\n--- per category ---")
                print(complete.groupby("category")[["precision_at_5", "mAP", "ndcg_at_5"]].mean().to_string())
    return pq, summary, unlabeled


if __name__ == "__main__":
    import sys
    df = pd.read_csv(sys.argv[1])
    evaluate(df)
