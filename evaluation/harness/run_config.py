"""Run a retriever configuration over the eval queries and score it against
the reusable label store. Prints metrics and, crucially, lists any newly
surfaced (query, image) pairs that still need a human relevance judgment.

Usage:
  python evaluation/harness/run_config.py baseline
  python evaluation/harness/run_config.py v2 --clip-global
  python evaluation/harness/run_config.py v2 --clip-global --color-gate
  python evaluation/harness/run_config.py v2 --backbone marqo --clip-global --color-gate
"""
import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "evaluation" / "harness"))

import metrics as M

QUERIES = ROOT / "evaluation" / "queries.csv"


def get_retriever(args):
    if args.mode == "baseline":
        from src.retriever.fashion_context_retriever import FashionContextRetriever
        return FashionContextRetriever()
    from src.retriever.retriever_v2 import RetrieverV2
    return RetrieverV2(
        backbone=args.backbone,
        use_clip_global=args.clip_global,
        use_color_gate=args.color_gate,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["baseline", "v2"])
    ap.add_argument("--backbone", default="fashionclip", choices=["fashionclip", "marqo"])
    ap.add_argument("--clip-global", action="store_true")
    ap.add_argument("--color-gate", action="store_true")
    ap.add_argument("--tag", default=None, help="label for the saved results file")
    args = ap.parse_args()

    queries = pd.read_csv(QUERIES)
    retriever = get_retriever(args)

    rows = []
    for _, q in queries.iterrows():
        query = str(q["query"]).strip()
        res = retriever.search(query=query, top_k=5)
        for _, r in res.iterrows():
            rows.append({
                "query": query, "category": q.get("category", ""),
                "rank": int(r["rank"]), "image_id": r["image_id"],
                "image_path": r["image_path"], "final_score": r["final_score"],
            })
    df = pd.DataFrame(rows)

    tag = args.tag or (args.mode if args.mode == "baseline" else
                       f"v2_{args.backbone}" + ("_cg" if args.clip_global else "") + ("_col" if args.color_gate else ""))
    out = ROOT / "evaluation" / "harness" / f"results_{tag}.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved results -> {out}")

    print("\n" + "=" * 60)
    print(f"CONFIG: {tag}")
    print("=" * 60)
    M.evaluate(df)


if __name__ == "__main__":
    main()
