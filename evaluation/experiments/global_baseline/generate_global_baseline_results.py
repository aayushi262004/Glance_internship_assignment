"""
True whole-image global retrieval baseline ("vanilla CLIP").

Encoder:   CLIPEncoder (open_clip ViT-B-32, laion2b_s34b_b79k) -- the same
           encoder used to build the global CLIP index.
Index:     data/features/global_clip.faiss (whole-image embeddings only).
Query:     the full raw query string, encoded once with the CLIP text tower.
Retrieval: FAISS top-5 inner product (cosine on normalized embeddings).
Ranking:   raw FAISS similarity, descending. Nothing else.

Deliberately NOT used: query decomposition, garment-region retrieval or
scoring, clause coverage calibration, style/context components, score
fusion, color contrast. This is an independent retrieval path -- it never
sees candidates discovered by the final system.

Relevance labels are reused ONLY by exact (query_id, image_id) match from
the human-labeled evaluation/retrieval_results.csv; genuinely new pairs are
left blank for manual labeling. No labels are invented.
"""

from pathlib import Path
import sys

import faiss
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.indexer.encoders import CLIPEncoder

FEATURE_DIR = PROJECT_ROOT / "data" / "features"

QUERY_FILE = PROJECT_ROOT / "evaluation" / "queries.csv"

# Read-only source of existing human labels; never written to.
BASELINE_LABELED_FILE = PROJECT_ROOT / "evaluation" / "retrieval_results.csv"

OUTPUT_FILE = Path(__file__).resolve().parent / "retrieval_results_global_baseline.csv"

TOP_K = 5


def load_label_lookup():
    labeled = pd.read_csv(BASELINE_LABELED_FILE)

    labeled["relevance"] = pd.to_numeric(labeled["relevance"], errors="coerce")

    return {
        (int(row.query_id), row.image_id): row.relevance
        for row in labeled.itertuples()
        if not pd.isna(row.relevance)
    }


def main():
    queries = pd.read_csv(QUERY_FILE)

    label_lookup = load_label_lookup()

    index = faiss.read_index(str(FEATURE_DIR / "global_clip.faiss"))

    mapping = pd.read_csv(FEATURE_DIR / "image_mapping.csv")

    encoder = CLIPEncoder()

    all_results = []
    new_pairs = []

    for query_number, row in queries.iterrows():
        query_id = query_number + 1
        query = str(row["query"]).strip()

        print(f"\n[{query_id}/{len(queries)}] {query}")

        query_embedding = encoder.encode_texts([query])

        scores, indices = index.search(query_embedding, TOP_K)

        for rank, (score, position) in enumerate(
            zip(scores[0], indices[0]),
            start=1,
        ):
            image_row = mapping.iloc[int(position)]
            image_id = image_row["image_id"]

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
                    "rank": rank,
                    "image_id": image_id,
                    "image_path": image_row["image_path"],
                    "clip_score": float(score),
                    "relevance": relevance,
                }
            )

        top_images = [r["image_id"] for r in all_results[-TOP_K:]]
        print("Top images:", top_images)

    results = pd.DataFrame(all_results)

    results.to_csv(OUTPUT_FILE, index=False)

    print("\nGlobal baseline retrieval complete.")
    print(f"Rows: {len(results)}")
    print(f"Reused labels: {len(results) - len(new_pairs)}")
    print(f"New unlabeled pairs: {len(new_pairs)}")

    for pair in new_pairs:
        print(f"  {pair}")

    print(f"\nResults: {OUTPUT_FILE}")
    print("\nNext (manual labeling of blank rows):")
    print("  python evaluation/experiments/global_baseline/label_baseline_results.py")


if __name__ == "__main__":
    main()
