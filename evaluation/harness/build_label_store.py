"""Seed a reusable (query, image_id) -> relevance label store from the
existing human-labeled retrieval_results.csv.

The label store lets us evaluate ANY retriever configuration objectively:
for a (query, image) pair we have already judged, the label is reused; new
pairs surfaced by a new configuration are reported as UNLABELED so they can
be judged under the same rubric (2 = highly relevant, 1 = partial, 0 = none).
This mirrors the report's protocol (shared labels for identical pairs,
independent labeling for new ones).
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / "evaluation" / "harness" / "label_store.csv"

SOURCES = [
    ROOT / "evaluation" / "retrieval_results.csv",
    ROOT / "evaluation" / "experiments" / "global_baseline" / "retrieval_results_global_baseline.csv",
    ROOT / "evaluation" / "experiments" / "color_contrast" / "retrieval_results_w050.csv",
]


MANUAL = ROOT / "evaluation" / "harness" / "manual_labels.csv"


def main():
    rows = []
    for src in SOURCES:
        if not src.exists():
            continue
        df = pd.read_csv(src)
        if "relevance" not in df.columns:
            continue
        df = df.dropna(subset=["relevance"])
        for _, r in df.iterrows():
            rows.append(
                {
                    "query": str(r["query"]).strip(),
                    "image_id": str(r["image_id"]).strip(),
                    "relevance": int(float(r["relevance"])),
                }
            )

    # Manual judgments for (query, image) pairs surfaced only by improved
    # configurations (judged by inspection under the same 0/1/2 rubric). Kept in
    # a dedicated file so rebuilding the store never loses them.
    if MANUAL.exists():
        man = pd.read_csv(MANUAL)
        for _, r in man.iterrows():
            rows.append(
                {
                    "query": str(r["query"]).strip(),
                    "image_id": str(r["image_id"]).strip(),
                    "relevance": int(float(r["relevance"])),
                }
            )

    store = pd.DataFrame(rows)
    # Deduplicate; if a pair was labeled twice, keep the max (conservative:
    # a pair judged relevant anywhere stays relevant).
    store = (
        store.groupby(["query", "image_id"], as_index=False)["relevance"].max()
        .sort_values(["query", "image_id"])
    )
    STORE.parent.mkdir(parents=True, exist_ok=True)
    store.to_csv(STORE, index=False)
    print(f"Wrote {len(store)} labeled (query, image) pairs to {STORE}")
    print(store.groupby("query").size())


if __name__ == "__main__":
    main()
