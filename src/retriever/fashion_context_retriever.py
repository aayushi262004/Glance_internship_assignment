from pathlib import Path
import re
import sys

import faiss
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.indexer.encoders import (
    CLIPEncoder,
    FashionCLIPEncoder,
)

FEATURE_DIR = PROJECT_ROOT / "data" / "features"

CANDIDATE_K = 100
CONTEXT_NEGATIVE_WEIGHT = 0.50

# R4: candidate-relative clause match calibration. A clause is "matched" for
# a candidate when its raw region score clears max(floor, this percentile of
# that same clause's score distribution across the current candidate pool).
# This replaces a single fixed cosine cutoff shared by every clause: rare
# garment categories (few annotated regions in the pool) naturally produce a
# low/zero pool percentile, so real-but-weak evidence can still count as a
# match, while common categories still need to beat roughly half the pool.
CLAUSE_MATCH_PERCENTILE = 50
CLAUSE_MATCH_ABSOLUTE_FLOOR = 0.05

# Region hits retrieved per fashion clause from the region-crop FAISS index.
# Clause-level region retrieval widens candidate recall for compositional
# queries (e.g. "red tie and white shirt") whose satisfying images may not
# rank inside the whole-query global candidates. Set to 0 to disable.
REGION_CANDIDATE_K = 50

# EXPERIMENTAL (disabled by default): contrastive color-prompt scoring.
# A "{color} {garment}" clause score is discounted by how well the SAME
# crop (already selected by the existing MaxSim logic) also matches every
# OTHER color for that garment, mirroring the CONTEXT_PROTOTYPES
# positive-minus-weighted-negative pattern. A labeled A/B at weight 0.50
# (evaluation/experiments/color_contrast/retrieval_results_w050.csv vs
# evaluation/retrieval_results.csv) showed
# a genuine trade-off: compositional nDCG@5 improved 0.912 -> 0.945 and a
# confirmed red-jacket false positive was removed, but overall P@5 dropped
# 0.933 -> 0.907 and yellow-attribute queries regressed (yellow-vs-beige
# is a boundary these embeddings cannot separate reliably). The submitted
# configuration therefore keeps this OFF (0.0 = disabled, exact validated
# post-R1/R4 baseline behavior); set to e.g. 0.50 to reproduce the A/B.
COLOR_CONTRAST_WEIGHT = 0.0


COLORS = {
    "red",
    "blue",
    "yellow",
    "white",
    "black",
    "green",
    "grey",
    "gray",
    "brown",
    "pink",
    "orange",
    "purple",
    "beige",
    "navy",
}


GARMENT_ALIASES = {
    "shirt": {0, 1},
    "blouse": {0},
    "top": {0, 1},
    "t-shirt": {1},
    "tshirt": {1},
    "sweatshirt": {1},
    "sweater": {2},
    "cardigan": {3},
    "jacket": {4},
    "blazer": {4},
    "vest": {5},
    "pants": {6},
    "trousers": {6},
    "shorts": {7},
    "skirt": {8},
    "coat": {9},
    "raincoat": {9, 4},
    "dress": {10},
    "jumpsuit": {11},
    "cape": {12},
    "glasses": {13},
    "hat": {14},
    "tie": {16},
    "glove": {17},
    "watch": {18},
    "belt": {19},
    "stockings": {21},
    "sock": {22},
    "shoe": {23},
    "shoes": {23},
    "bag": {24},
    "wallet": {24},
    "scarf": {25},
    "umbrella": {26},
}


STYLE_TERMS = {
    "professional",
    "business",
    "formal",
    "casual",
    "weekend",
    "smart",
    "relaxed",
    "elegant",
    "sporty",
    "streetwear",
}


CONTEXT_TERMS = {
    "office",
    "park",
    "bench",
    "street",
    "city",
    "home",
    "indoors",
    "inside",
    "outdoors",
    "urban",
    "walk",
    "walking",
    "sitting",
}


CONTEXT_PROTOTYPES = {
    "office": {
        "positive": [
            "inside a modern office",
            "professional office interior",
            "corporate workplace",
            "business office environment",
            "indoor office workspace",
        ],
        "negative": [
            "outdoor street",
            "urban sidewalk",
            "fashion runway",
            "park",
            "home interior",
            "closet or dressing room",
        ],
    },
    "park": {
        "positive": [
            "in a green park",
            "outdoor public park",
            "park with trees and grass",
            "sitting in a park",
            "park bench outdoors",
        ],
        "negative": [
            "office interior",
            "home interior",
            "fashion runway",
            "urban street",
            "closet or dressing room",
        ],
    },
    "city": {
        "positive": [
            "walking in a city",
            "urban street scene",
            "city sidewalk",
            "outdoor city walk",
            "walking through an urban area",
        ],
        "negative": [
            "office interior",
            "home interior",
            "green park",
            "fashion runway",
            "closet or dressing room",
        ],
    },
    "street": {
        "positive": [
            "urban street scene",
            "walking on a street",
            "city sidewalk outdoors",
            "street fashion outdoors",
        ],
        "negative": [
            "office interior",
            "home interior",
            "green park",
            "closet or dressing room",
        ],
    },
    "home": {
        "positive": [
            "inside a home",
            "home interior",
            "domestic living space",
            "casual indoor home setting",
        ],
        "negative": [
            "office interior",
            "urban street",
            "green park",
            "fashion runway",
        ],
    },
}


def parse_query(query):
    text = query.lower()

    normalized = re.sub(
        r"[^a-z0-9\- ]+",
        " ",
        text,
    )

    tokens = normalized.split()

    fashion_clauses = []

    for index in range(len(tokens) - 1):
        color = tokens[index]
        garment = tokens[index + 1]

        if color in COLORS and garment in GARMENT_ALIASES:
            fashion_clauses.append(
                {
                    "text": f"{color} {garment}",
                    "garment": garment,
                    "category_ids": GARMENT_ALIASES[garment],
                    "color": color,
                }
            )

    captured_garments = {clause["garment"] for clause in fashion_clauses}

    for token in tokens:
        if token not in GARMENT_ALIASES:
            continue

        if token in captured_garments:
            continue

        fashion_clauses.append(
            {
                "text": token,
                "garment": token,
                "category_ids": GARMENT_ALIASES[token],
                "color": None,
            }
        )

    style_tokens = [token for token in tokens if token in STYLE_TERMS]

    context_tokens = [token for token in tokens if token in CONTEXT_TERMS]

    return {
        "fashion_clauses": fashion_clauses,
        "style": " ".join(style_tokens),
        "context": " ".join(context_tokens),
    }


def get_context_prototypes(context_text):
    tokens = set(context_text.lower().split())

    for context_name in (
        "office",
        "park",
        "city",
        "street",
        "home",
    ):
        if context_name in tokens:
            return CONTEXT_PROTOTYPES[context_name]

    return None


def minmax(values):
    values = np.asarray(
        values,
        dtype=np.float32,
    )

    minimum = values.min()
    maximum = values.max()

    if maximum - minimum < 1e-8:
        return np.zeros_like(values)

    return (values - minimum) / (maximum - minimum)


class FashionContextRetriever:

    def __init__(self):
        print("Loading Fashion Context Retriever...")

        self.image_mapping = pd.read_csv(FEATURE_DIR / "image_mapping.csv")

        self.fashion_index = faiss.read_index(
            str(FEATURE_DIR / "global_fashionclip.faiss")
        )

        self.clip_index = faiss.read_index(str(FEATURE_DIR / "global_clip.faiss"))

        self.clip_embeddings = np.load(FEATURE_DIR / "global_clip.npy").astype(
            np.float32
        )

        self.fashion_embeddings = np.load(
            FEATURE_DIR / "global_fashionclip.npy"
        ).astype(np.float32)

        self.region_embeddings = np.load(FEATURE_DIR / "region_fashionclip.npy").astype(
            np.float32
        )

        self.region_mapping = pd.read_csv(FEATURE_DIR / "region_mapping.csv")

        if len(self.region_embeddings) != len(self.region_mapping):
            raise RuntimeError("Region embedding/mapping mismatch.")

        # Region-crop index used for clause-level candidate generation.
        # Row order matches region_mapping (checked above).
        self.region_index = faiss.read_index(
            str(FEATURE_DIR / "region_fashionclip.faiss")
        )

        # image_id -> row position in image_mapping (the coordinate system
        # of the global candidate pool and embedding matrices).
        self.image_positions = {
            image_id: position
            for position, image_id in enumerate(self.image_mapping["image_id"])
        }

        self.fashion_encoder = FashionCLIPEncoder()

        self.context_encoder = CLIPEncoder()

        self.region_groups = {
            image_id: group
            for image_id, group in self.region_mapping.groupby("image_id")
        }

        print("Retriever ready.")

    def search(self, query, top_k=5):

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        parsed = parse_query(query)

        fashion_clauses = parsed["fashion_clauses"]

        fashion_query_embedding = self.fashion_encoder.encode_texts([query])

        clip_query_embedding = self.context_encoder.encode_texts([query])

        if fashion_clauses:
            clause_texts = [clause["text"] for clause in fashion_clauses]

            clause_embeddings = self.fashion_encoder.encode_texts(clause_texts)

            # Contrastive color scoring (experimental, disabled when
            # COLOR_CONTRAST_WEIGHT is 0): for each clause with a captured
            # color, precompute embeddings for the same garment paired with
            # every other known color, once per query (not per candidate).
            clause_color_negative_embeddings = []

            for clause in fashion_clauses:
                color = clause.get("color")

                if color is None or COLOR_CONTRAST_WEIGHT <= 0:
                    clause_color_negative_embeddings.append(None)
                    continue

                other_colors = sorted(COLORS - {color})

                negative_texts = [
                    f"{other_color} {clause['garment']}" for other_color in other_colors
                ]

                clause_color_negative_embeddings.append(
                    self.fashion_encoder.encode_texts(negative_texts)
                )
        else:
            clause_texts = []
            clause_embeddings = None
            clause_color_negative_embeddings = []

        _, fashion_indices = self.fashion_index.search(
            fashion_query_embedding,
            CANDIDATE_K,
        )

        _, clip_indices = self.clip_index.search(
            clip_query_embedding,
            CANDIDATE_K,
        )

        # Clause-level region retrieval: search the region-crop index with
        # each parsed fashion clause (e.g. "red tie") so images containing a
        # matching garment region enter the candidate pool even when the
        # whole-query global searches miss them. This only improves candidate
        # recall for compositional queries; scoring and reranking below are
        # unchanged.
        region_candidate_positions = []

        if clause_embeddings is not None and REGION_CANDIDATE_K > 0:
            _, region_hits = self.region_index.search(
                clause_embeddings,
                REGION_CANDIDATE_K,
            )

            for clause_hits in region_hits:
                hit_image_ids = self.region_mapping["image_id"].iloc[clause_hits]

                for image_id in hit_image_ids:
                    position = self.image_positions.get(image_id)

                    if position is not None:
                        region_candidate_positions.append(position)

        candidate_indices = np.asarray(
            list(
                dict.fromkeys(
                    fashion_indices[0].tolist()
                    + clip_indices[0].tolist()
                    + region_candidate_positions
                )
            ),
            dtype=np.int64,
        )

        global_scores = (
            self.fashion_embeddings[candidate_indices] @ fashion_query_embedding[0]
        ).astype(np.float32)

        context_text = parsed["context"]

        if context_text:
            context_embedding = (self.context_encoder.encode_texts([context_text]))[0]
        else:
            context_embedding = None

        context_prototypes = get_context_prototypes(context_text)

        if context_prototypes is not None:
            positive_embeddings = self.context_encoder.encode_texts(
                context_prototypes["positive"]
            )

            negative_embeddings = self.context_encoder.encode_texts(
                context_prototypes["negative"]
            )
        else:
            positive_embeddings = None
            negative_embeddings = None

        style_text = parsed["style"]

        if style_text:
            style_embedding = (self.fashion_encoder.encode_texts([style_text]))[0]
        else:
            style_embedding = None

        result_rows = []

        for candidate_position, global_score in zip(
            candidate_indices,
            global_scores,
        ):

            image_row = self.image_mapping.iloc[candidate_position]

            image_id = image_row["image_id"]

            clause_scores = []

            if fashion_clauses and image_id in self.region_groups:

                image_regions = self.region_groups[image_id]

                for clause_index, clause in enumerate(fashion_clauses):

                    compatible_regions = image_regions[
                        image_regions["category_id"].isin(clause["category_ids"])
                    ]

                    if compatible_regions.empty:
                        clause_scores.append(0.0)
                        continue

                    region_indices = (
                        compatible_regions["region_index"].astype(int).to_numpy()
                    )

                    embeddings = self.region_embeddings[region_indices]

                    similarities = embeddings @ clause_embeddings[clause_index]

                    best_position = int(np.argmax(similarities))

                    positive_score = float(similarities[best_position])

                    # Contrastive color check: does the SAME crop chosen
                    # above (crop/category selection is unchanged) also
                    # match other colors for this garment about as well?
                    # If so, the match is likely driven by garment shape,
                    # not the queried color, and the score is discounted.
                    color_negative_embeddings = clause_color_negative_embeddings[
                        clause_index
                    ]

                    if color_negative_embeddings is not None:
                        best_crop_embedding = embeddings[best_position]

                        negative_score = float(
                            np.max(color_negative_embeddings @ best_crop_embedding)
                        )

                        clause_score = (
                            positive_score
                            - COLOR_CONTRAST_WEIGHT * negative_score
                        )
                    else:
                        clause_score = positive_score

                    clause_scores.append(clause_score)

            elif fashion_clauses:
                clause_scores = [0.0 for _ in fashion_clauses]

            if clause_scores:
                clause_array = np.asarray(
                    clause_scores,
                    dtype=np.float32,
                )

                mean_score = float(np.mean(clause_array))

                min_score = float(np.min(clause_array))

                if len(clause_array) > 1:
                    region_score = 0.60 * mean_score + 0.40 * min_score
                else:
                    region_score = mean_score

            else:
                region_score = 0.0

            # coverage_score / all_clauses_matched depend on the full
            # candidate pool (R4 calibration) and are filled in below, after
            # every candidate's raw clause_scores has been collected.
            coverage_score = 0.0
            all_matched = 0.0

            if context_embedding is not None:

                image_context = self.clip_embeddings[candidate_position]

                if positive_embeddings is not None:

                    positive_score = float(np.mean(positive_embeddings @ image_context))

                    negative_score = float(np.max(negative_embeddings @ image_context))

                    context_score = (
                        positive_score - CONTEXT_NEGATIVE_WEIGHT * negative_score
                    )

                else:
                    context_score = float(image_context @ context_embedding)

            else:
                context_score = 0.0

            if style_embedding is not None:
                style_score = float(
                    self.fashion_embeddings[candidate_position] @ style_embedding
                )
            else:
                style_score = 0.0

            result_rows.append(
                {
                    "image_id": image_id,
                    "image_path": image_row["image_path"],
                    "global_score": float(global_score),
                    "region_score": region_score,
                    "coverage_score": coverage_score,
                    "all_clauses_matched": all_matched,
                    "style_score": style_score,
                    "context_score": context_score,
                    "clause_scores": clause_scores,
                }
            )

        results = pd.DataFrame(result_rows)

        # R4: candidate-relative clause match calibration. Compare each
        # candidate's raw clause score against that clause's own score
        # distribution across this query's candidate pool, instead of one
        # fixed cosine cutoff shared by every clause regardless of how rare
        # or common its garment category is in the pool.
        if fashion_clauses:
            clause_matrix = np.asarray(
                results["clause_scores"].tolist(),
                dtype=np.float32,
            )

            relative_thresholds = np.percentile(
                clause_matrix,
                CLAUSE_MATCH_PERCENTILE,
                axis=0,
            )

            match_thresholds = np.maximum(
                relative_thresholds,
                CLAUSE_MATCH_ABSOLUTE_FLOOR,
            )

            matched_matrix = clause_matrix >= match_thresholds

            results["coverage_score"] = matched_matrix.mean(axis=1)

            results["all_clauses_matched"] = matched_matrix.all(axis=1).astype(
                np.float32
            )

        results["global_norm"] = minmax(results["global_score"])

        results["region_norm"] = minmax(results["region_score"])

        if style_embedding is not None:
            results["style_norm"] = minmax(results["style_score"])
        else:
            results["style_norm"] = 0.0

        if context_embedding is not None:
            results["context_norm"] = minmax(results["context_score"])
        else:
            results["context_norm"] = 0.0

        component_scores = {"global": results["global_norm"]}

        component_weights = {"global": 0.25}

        if fashion_clauses:
            component_scores["region"] = results["region_norm"]

            component_weights["region"] = 0.30

            component_scores["coverage"] = results["coverage_score"]

            component_weights["coverage"] = 0.15

            if len(fashion_clauses) > 1:
                component_scores["conjunction"] = results["all_clauses_matched"]

                component_weights["conjunction"] = 0.10

        if style_embedding is not None:
            component_scores["style"] = results["style_norm"]

            component_weights["style"] = 0.15

        if context_embedding is not None:
            component_scores["context"] = results["context_norm"]

            component_weights["context"] = 0.15

        total_weight = sum(component_weights.values())

        results["final_score"] = 0.0

        for name, score in component_scores.items():

            weight = component_weights[name] / total_weight

            results["final_score"] += weight * score

        results = results.sort_values(
            "final_score",
            ascending=False,
        ).reset_index(drop=True)

        results["rank"] = np.arange(len(results)) + 1

        results["query"] = query

        results["parsed_fashion"] = str(clause_texts)

        results["parsed_style"] = parsed["style"]

        results["parsed_context"] = parsed["context"]

        return results.head(top_k).copy()
