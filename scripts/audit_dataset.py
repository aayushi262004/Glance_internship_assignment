from pathlib import Path
import sys

import faiss
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT))


from src.indexer.encoders import CLIPEncoder


FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "dataset_audit"
)

TOP_K = 20


AUDIT_CONCEPTS = {
    "office": "a person inside a modern office",
    "urban_street": "a person walking on an urban city street",
    "park": "a person in a green park",
    "home": "a person inside a home living room",

    "formal": "a person wearing formal business clothing",
    "casual": "a person wearing casual everyday clothing",
    "outerwear": "a person wearing a coat or jacket",

    "yellow_raincoat": "a person wearing a bright yellow raincoat",
    "blue_shirt": "a person wearing a blue shirt",
    "red_tie": "a person wearing a red tie",
    "white_shirt": "a person wearing a white shirt",
}


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    index = faiss.read_index(
        str(
            FEATURE_DIR
            / "global_clip.faiss"
        )
    )

    mapping = pd.read_csv(
        FEATURE_DIR
        / "image_mapping.csv"
    )

    encoder = CLIPEncoder()

    audit_rows = []

    for concept, query in AUDIT_CONCEPTS.items():
        print(
            f"\nAuditing: {concept}"
        )

        query_embedding = (
            encoder.encode_texts([query])
        )

        scores, indices = index.search(
            query_embedding,
            TOP_K,
        )

        for rank, (
            score,
            index_position,
        ) in enumerate(
            zip(scores[0], indices[0]),
            start=1,
        ):
            row = mapping.iloc[index_position]

            audit_rows.append(
                {
                    "concept": concept,
                    "query": query,
                    "rank": rank,
                    "score": float(score),
                    "image_id": row["image_id"],
                    "image_path": row["image_path"],
                }
            )

    audit_df = pd.DataFrame(
        audit_rows
    )

    output_path = (
        OUTPUT_DIR
        / "semantic_audit.csv"
    )

    audit_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nAudit saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()