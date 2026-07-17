"""Improved multi-signal fashion retriever (configurable, backward compatible).

Extends the original FashionContextRetriever with three switchable upgrades so
each can be A/B'd independently against the shipped baseline:

  I1  use_clip_global   : add general-CLIP whole-query similarity as its own
                          fusion component. In the baseline the CLIP global
                          vector is only used for candidate recall and context
                          prototypes; its broad zero-shot semantics never
                          scored the ranking. Nearly free, helps scene/context.

  I2  use_color_gate    : verify garment *hue* with actual crop pixels
                          (data/features/region_colors.npy). CLIP crop
                          similarity is shape-dominated, so a red jacket can
                          rank for "blue jacket". A soft multiplicative gate
                          demotes hue-mismatched crops without the yellow/beige
                          regression the CLIP-embedding color-contrast had.

  backbone              : "fashionclip" (baseline) or "marqo"
                          (Marqo-FashionSigLIP, a stronger frozen fashion
                          encoder). Selects which precomputed feature files and
                          text encoder are used for the fashion + region signals.

With use_clip_global=False, use_color_gate=False, backbone="fashionclip" the
scoring is identical to the shipped system.
"""
from pathlib import Path
import json
import re
import sys

import faiss
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.indexer.encoders import CLIPEncoder, FashionCLIPEncoder
from src.retriever.fashion_context_retriever import (
    COLORS, GARMENT_ALIASES, STYLE_TERMS, CONTEXT_TERMS, CONTEXT_PROTOTYPES,
    parse_query, get_context_prototypes, minmax,
    CANDIDATE_K, CONTEXT_NEGATIVE_WEIGHT, REGION_CANDIDATE_K,
    CLAUSE_MATCH_PERCENTILE, CLAUSE_MATCH_ABSOLUTE_FLOOR,
)

FEATURE_DIR = PROJECT_ROOT / "data" / "features"

# --- I2 color-gate parameters ---
# gate = FLOOR + (1-FLOOR) * min(1, matched_frac / TARGET_FRAC)
# where matched_frac is the crop's pixel fraction in the queried color AND its
# perceptually adjacent buckets (see COLOR_COMPAT). A crop with >= TARGET_FRAC
# in the queried color family is unpenalized; a clear other-color crop is scaled
# toward FLOOR. Soft, so a partially-matching crop keeps most of its score.
COLOR_GATE_FLOOR = 0.45
COLOR_TARGET_FRAC = 0.22

# Perceptual color adjacency. Pixel-hue buckets are brittle at boundaries
# (mustard/golden "yellow" reads as orange; navy reads as blue; cream as
# white). When a query asks for a color we credit its neighbours with a partial
# weight, so a genuinely golden-yellow coat is not wrongly demoted for the
# "yellow" query, while a clearly red jacket still fails the "blue" query.
COLOR_COMPAT = {
    "red": {"red": 1.0, "orange": 0.4, "pink": 0.5, "brown": 0.3},
    "orange": {"orange": 1.0, "yellow": 0.5, "brown": 0.5, "red": 0.3},
    "yellow": {"yellow": 1.0, "orange": 0.7, "brown": 0.35},
    "green": {"green": 1.0, "yellow": 0.2},
    "blue": {"blue": 1.0, "purple": 0.3},
    "purple": {"purple": 1.0, "pink": 0.4, "blue": 0.3},
    "pink": {"pink": 1.0, "red": 0.5, "purple": 0.4},
    "brown": {"brown": 1.0, "orange": 0.5, "red": 0.2},
    "white": {"white": 1.0, "gray": 0.4},
    "gray": {"gray": 1.0, "white": 0.4, "black": 0.3},
    "grey": {"gray": 1.0, "white": 0.4, "black": 0.3},
    "black": {"black": 1.0, "gray": 0.4},
    "navy": {"blue": 1.0, "purple": 0.3},
    "beige": {"brown": 0.8, "white": 0.6, "yellow": 0.4, "orange": 0.4},
}


class MarqoEncoder:
    """Marqo-FashionSigLIP via open_clip. Same interface as the CLIP encoders
    (encode_texts / encode_images return L2-normalized float32)."""

    HF = "hf-hub:Marqo/marqo-fashionSigLIP"

    def __init__(self):
        import torch
        import open_clip
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Marqo-FashionSigLIP on {self.device}")
        self.model, self.preprocess = open_clip.create_model_from_pretrained(self.HF)
        self.tokenizer = open_clip.get_tokenizer(self.HF)
        self.model = self.model.to(self.device).eval()

    @staticmethod
    def _norm(e):
        n = np.linalg.norm(e, axis=1, keepdims=True)
        return e / np.clip(n, 1e-12, None)

    def encode_texts(self, texts):
        tok = self.tokenizer(texts).to(self.device)
        with self.torch.inference_mode():
            e = self.model.encode_text(tok)
        return self._norm(e.float().cpu().numpy()).astype(np.float32)

    def encode_images(self, images):
        from PIL import Image
        batch = []
        for im in images:
            if not isinstance(im, Image.Image):
                im = Image.open(im).convert("RGB")
            else:
                im = im.convert("RGB")
            batch.append(self.preprocess(im))
        x = self.torch.stack(batch).to(self.device)
        with self.torch.inference_mode():
            e = self.model.encode_image(x)
        return self._norm(e.float().cpu().numpy()).astype(np.float32)


class RetrieverV2:
    def __init__(self, backbone="fashionclip", use_clip_global=False,
                 use_color_gate=False, clip_global_weight=0.15):
        self.backbone = backbone
        self.use_clip_global = use_clip_global
        self.use_color_gate = use_color_gate
        self.clip_global_weight = clip_global_weight

        print(f"Loading RetrieverV2 (backbone={backbone}, clip_global={use_clip_global}, color_gate={use_color_gate})")
        self.image_mapping = pd.read_csv(FEATURE_DIR / "image_mapping.csv")
        self.region_mapping = pd.read_csv(FEATURE_DIR / "region_mapping.csv")

        suffix = "fashionclip" if backbone == "fashionclip" else "marqo"
        self.fashion_embeddings = np.load(FEATURE_DIR / f"global_{suffix}.npy").astype(np.float32)
        self.fashion_index = faiss.read_index(str(FEATURE_DIR / f"global_{suffix}.faiss"))
        self.region_embeddings = np.load(FEATURE_DIR / f"region_{suffix}.npy").astype(np.float32)
        self.region_index = faiss.read_index(str(FEATURE_DIR / f"region_{suffix}.faiss"))

        # General CLIP is always used for scene/context (and, if enabled, as an
        # extra global scoring component).
        self.clip_embeddings = np.load(FEATURE_DIR / "global_clip.npy").astype(np.float32)
        self.clip_index = faiss.read_index(str(FEATURE_DIR / "global_clip.faiss"))

        if len(self.region_embeddings) != len(self.region_mapping):
            raise RuntimeError("Region embedding/mapping mismatch.")

        self.image_positions = {img: pos for pos, img in enumerate(self.image_mapping["image_id"])}
        self.region_groups = {img: g for img, g in self.region_mapping.groupby("image_id")}

        if backbone == "fashionclip":
            self.fashion_encoder = FashionCLIPEncoder()
        else:
            self.fashion_encoder = MarqoEncoder()
        self.context_encoder = CLIPEncoder()

        # I2: per-region pixel color descriptor.
        self.region_colors = None
        self.color_cols = None
        self.color_aliases = {}
        if use_color_gate:
            self.region_colors = np.load(FEATURE_DIR / "region_colors.npy").astype(np.float32)
            meta = json.loads((FEATURE_DIR / "region_colors_meta.json").read_text())
            self.color_cols = {c: i for i, c in enumerate(meta["colors"])}
            self.color_aliases = meta.get("aliases", {})
        print("RetrieverV2 ready.")

    def _color_gate(self, region_index, queried_color):
        """Multiplicative [FLOOR,1] factor from the crop's pixel hue, using
        perceptual color adjacency so boundary hues (golden-yellow, navy,
        cream) are not wrongly penalized."""
        compat = COLOR_COMPAT.get(queried_color)
        if compat is None:
            return 1.0
        matched = 0.0
        for bucket, weight in compat.items():
            col = self.color_cols.get(bucket)
            if col is not None:
                matched += weight * float(self.region_colors[region_index, col])
        return COLOR_GATE_FLOOR + (1.0 - COLOR_GATE_FLOOR) * min(1.0, matched / COLOR_TARGET_FRAC)

    def similar(self, image_id, top_k=8):
        """Image-to-image retrieval: nearest looks to a given image in the
        fashion-backbone embedding space (same index used for text search)."""
        pos = self.image_positions.get(image_id)
        if pos is None:
            raise ValueError(f"Unknown image_id {image_id}")
        query_vec = self.fashion_embeddings[pos:pos + 1]
        scores, idx = self.fashion_index.search(query_vec, top_k + 1)
        out = []
        for score, i in zip(scores[0], idx[0]):
            row = self.image_mapping.iloc[i]
            if row["image_id"] == image_id:
                continue
            out.append({"image_id": row["image_id"], "image_path": row["image_path"],
                        "similarity": float(score)})
            if len(out) >= top_k:
                break
        return out

    def search(self, query, top_k=5, use_color_gate=None, weights=None):
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        gate_on = self.use_color_gate if use_color_gate is None else use_color_gate
        parsed = parse_query(query)
        fashion_clauses = parsed["fashion_clauses"]

        fashion_q = self.fashion_encoder.encode_texts([query])
        clip_q = self.context_encoder.encode_texts([query])

        if fashion_clauses:
            clause_texts = [c["text"] for c in fashion_clauses]
            clause_embeddings = self.fashion_encoder.encode_texts(clause_texts)
        else:
            clause_texts, clause_embeddings = [], None

        _, fashion_idx = self.fashion_index.search(fashion_q, CANDIDATE_K)
        _, clip_idx = self.clip_index.search(clip_q, CANDIDATE_K)

        region_cand = []
        if clause_embeddings is not None and REGION_CANDIDATE_K > 0:
            _, region_hits = self.region_index.search(clause_embeddings, REGION_CANDIDATE_K)
            for hits in region_hits:
                for img in self.region_mapping["image_id"].iloc[hits]:
                    pos = self.image_positions.get(img)
                    if pos is not None:
                        region_cand.append(pos)

        candidate_indices = np.asarray(
            list(dict.fromkeys(fashion_idx[0].tolist() + clip_idx[0].tolist() + region_cand)),
            dtype=np.int64,
        )

        global_scores = (self.fashion_embeddings[candidate_indices] @ fashion_q[0]).astype(np.float32)
        clip_global_scores = (self.clip_embeddings[candidate_indices] @ clip_q[0]).astype(np.float32)

        context_text = parsed["context"]
        context_embedding = self.context_encoder.encode_texts([context_text])[0] if context_text else None
        protos = get_context_prototypes(context_text)
        if protos is not None:
            pos_emb = self.context_encoder.encode_texts(protos["positive"])
            neg_emb = self.context_encoder.encode_texts(protos["negative"])
        else:
            pos_emb = neg_emb = None

        style_text = parsed["style"]
        style_embedding = self.fashion_encoder.encode_texts([style_text])[0] if style_text else None

        rows = []
        for ci, (cand, gscore, cgscore) in enumerate(zip(candidate_indices, global_scores, clip_global_scores)):
            image_row = self.image_mapping.iloc[cand]
            image_id = image_row["image_id"]

            clause_scores = []
            if fashion_clauses and image_id in self.region_groups:
                regions = self.region_groups[image_id]
                for k, clause in enumerate(fashion_clauses):
                    compat = regions[regions["category_id"].isin(clause["category_ids"])]
                    if compat.empty:
                        clause_scores.append(0.0)
                        continue
                    ridx = compat["region_index"].astype(int).to_numpy()
                    sims = self.region_embeddings[ridx] @ clause_embeddings[k]
                    best = int(np.argmax(sims))
                    score = float(sims[best])
                    if gate_on and clause.get("color"):
                        score *= self._color_gate(int(ridx[best]), clause["color"])
                    clause_scores.append(score)
            elif fashion_clauses:
                clause_scores = [0.0 for _ in fashion_clauses]

            if clause_scores:
                arr = np.asarray(clause_scores, dtype=np.float32)
                mean_s, min_s = float(arr.mean()), float(arr.min())
                region_score = 0.60 * mean_s + 0.40 * min_s if len(arr) > 1 else mean_s
            else:
                region_score = 0.0

            if context_embedding is not None:
                img_ctx = self.clip_embeddings[cand]
                if pos_emb is not None:
                    context_score = float(np.mean(pos_emb @ img_ctx)) - CONTEXT_NEGATIVE_WEIGHT * float(np.max(neg_emb @ img_ctx))
                else:
                    context_score = float(img_ctx @ context_embedding)
            else:
                context_score = 0.0

            style_score = float(self.fashion_embeddings[cand] @ style_embedding) if style_embedding is not None else 0.0

            rows.append({
                "image_id": image_id, "image_path": image_row["image_path"],
                "global_score": float(gscore), "clip_global_score": float(cgscore),
                "region_score": region_score, "coverage_score": 0.0,
                "all_clauses_matched": 0.0, "style_score": style_score,
                "context_score": context_score, "clause_scores": clause_scores,
            })

        results = pd.DataFrame(rows)

        clause_texts_list = [c["text"] for c in fashion_clauses]
        if fashion_clauses:
            cm = np.asarray(results["clause_scores"].tolist(), dtype=np.float32)
            thr = np.maximum(np.percentile(cm, CLAUSE_MATCH_PERCENTILE, axis=0), CLAUSE_MATCH_ABSOLUTE_FLOOR)
            matched = cm >= thr
            results["coverage_score"] = matched.mean(axis=1)
            results["all_clauses_matched"] = matched.all(axis=1).astype(np.float32)
            # Per-candidate list of which garment clauses were satisfied (for UI).
            results["matched_clauses"] = [
                [clause_texts_list[j] for j in range(len(clause_texts_list)) if row[j]]
                for row in matched
            ]
        else:
            results["matched_clauses"] = [[] for _ in range(len(results))]

        results["global_norm"] = minmax(results["global_score"])
        results["clip_global_norm"] = minmax(results["clip_global_score"])
        results["region_norm"] = minmax(results["region_score"])
        results["style_norm"] = minmax(results["style_score"]) if style_embedding is not None else 0.0
        results["context_norm"] = minmax(results["context_score"]) if context_embedding is not None else 0.0

        comp = {"global": results["global_norm"]}
        w = {"global": 0.25}
        if self.use_clip_global:
            comp["clip_global"] = results["clip_global_norm"]
            w["clip_global"] = self.clip_global_weight
        if fashion_clauses:
            comp["region"] = results["region_norm"]; w["region"] = 0.30
            comp["coverage"] = results["coverage_score"]; w["coverage"] = 0.15
            if len(fashion_clauses) > 1:
                comp["conjunction"] = results["all_clauses_matched"]; w["conjunction"] = 0.10
        if style_embedding is not None:
            comp["style"] = results["style_norm"]; w["style"] = 0.15
        if context_embedding is not None:
            comp["context"] = results["context_norm"]; w["context"] = 0.15

        # Optional per-request weight overrides (only for components active in
        # this query) so the UI can expose interactive fusion tuning.
        if weights:
            for name in list(w.keys()):
                if name in weights and weights[name] is not None:
                    w[name] = float(weights[name])

        total = sum(w.values()) or 1.0
        results["final_score"] = 0.0
        for name, s in comp.items():
            results["final_score"] += (w[name] / total) * s

        results = results.sort_values("final_score", ascending=False).reset_index(drop=True)
        results["rank"] = np.arange(len(results)) + 1
        results["query"] = query
        results["parsed_fashion"] = str(clause_texts)
        results["parsed_style"] = parsed["style"]
        results["parsed_context"] = parsed["context"]
        return results.head(top_k).copy()
