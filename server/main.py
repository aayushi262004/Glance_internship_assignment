"""FastAPI backend for the Fashion-Aware Context Retrieval demo.

Wraps the improved multi-signal retriever (Marqo-FashionSigLIP + region evidence
+ pixel-HSV color verification) and exposes it to the React frontend:

  POST /api/search    natural-language query -> ranked looks + per-signal
                      evidence + parsed interpretation. Supports live ablation
                      (backbone, color gate) and fusion-weight overrides.
  POST /api/similar   image-to-image retrieval (nearest looks to a given image)
  GET  /api/image/ID  serves a corpus image
  GET  /api/examples  curated example queries (incl. the 5 assignment prompts)
  GET  /api/config    available backbones / default weights / health
"""
import os

# faiss + torch share OpenMP; on macOS with duplicate libomp this segfaults
# unless duplicates are allowed and threading is constrained. Must be set
# before torch is imported anywhere.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pathlib import Path
from typing import Optional
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.retriever.retriever_v2 import RetrieverV2
from src.retriever.fashion_context_retriever import parse_query
from src.indexer.encoders import norm_path

FEATURE_DIR = PROJECT_ROOT / "data" / "features"
MARQO_AVAILABLE = (FEATURE_DIR / "global_marqo.faiss").exists()

app = FastAPI(title="Glance Fashion Retrieval API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazily-instantiated retrievers, one per backbone (each holds its own encoders).
_retrievers = {}


def get_retriever(backbone):
    if backbone not in ("marqo", "fashionclip"):
        raise HTTPException(400, f"unknown backbone {backbone}")
    if backbone == "marqo" and not MARQO_AVAILABLE:
        backbone = "fashionclip"
    if backbone not in _retrievers:
        # color gate decided per-request; instance flag off, index still loaded
        _retrievers[backbone] = RetrieverV2(backbone=backbone, use_color_gate=True)
    return _retrievers[backbone]


# image_id -> absolute path
_IMAGE_PATHS = None


def image_paths():
    global _IMAGE_PATHS
    if _IMAGE_PATHS is None:
        import pandas as pd
        m = pd.read_csv(FEATURE_DIR / "image_mapping.csv")
        _IMAGE_PATHS = {
            r["image_id"]: str(PROJECT_ROOT / norm_path(r["image_path"]))
            for _, r in m.iterrows()
        }
    return _IMAGE_PATHS


class SearchRequest(BaseModel):
    query: str
    top_k: int = 6
    backbone: str = "marqo"
    color_gate: bool = True
    weights: Optional[dict] = None


class SimilarRequest(BaseModel):
    image_id: str
    top_k: int = 6
    backbone: str = "marqo"


COMPONENT_KEYS = ["global", "region", "coverage", "conjunction", "style", "context"]


def result_payload(row):
    return {
        "image_id": row["image_id"],
        "image_url": f"/api/image/{row['image_id']}",
        "rank": int(row["rank"]),
        "final_score": round(float(row["final_score"]), 4),
        "components": {
            "global": round(float(row.get("global_norm", 0.0)), 3),
            "region": round(float(row.get("region_norm", 0.0)), 3),
            "coverage": round(float(row.get("coverage_score", 0.0)), 3),
            "style": round(float(row.get("style_norm", 0.0)), 3),
            "context": round(float(row.get("context_norm", 0.0)), 3),
        },
        "raw": {
            "global": round(float(row.get("global_score", 0.0)), 3),
            "region": round(float(row.get("region_score", 0.0)), 3),
            "style": round(float(row.get("style_score", 0.0)), 3),
            "context": round(float(row.get("context_score", 0.0)), 3),
        },
        "matched_clauses": list(row.get("matched_clauses", []) or []),
    }


@app.get("/api/config")
def config():
    return {
        "backbones": (["marqo", "fashionclip"] if MARQO_AVAILABLE else ["fashionclip"]),
        "default_backbone": "marqo" if MARQO_AVAILABLE else "fashionclip",
        "components": COMPONENT_KEYS,
        "default_weights": {"global": 0.25, "region": 0.30, "coverage": 0.15,
                            "conjunction": 0.10, "style": 0.15, "context": 0.15},
    }


@app.get("/api/examples")
def examples():
    return {
        "assignment": [
            {"label": "Bright yellow raincoat", "query": "A person in a bright yellow raincoat", "type": "Attribute"},
            {"label": "Business attire in a modern office", "query": "Professional business attire inside a modern office", "type": "Contextual"},
            {"label": "Blue shirt on a park bench", "query": "Someone wearing a blue shirt sitting on a park bench", "type": "Complex"},
            {"label": "Casual weekend city walk", "query": "Casual weekend outfit for a city walk", "type": "Style"},
            {"label": "Red tie & white shirt, formal", "query": "A red tie and a white shirt in a formal setting", "type": "Compositional"},
        ],
        "more": [
            {"label": "Black dress", "query": "A person wearing a black dress"},
            {"label": "Blue jacket", "query": "A person in a blue jacket"},
            {"label": "Formal blazer", "query": "A formal blazer outfit"},
            {"label": "Green park", "query": "Someone sitting in a green park"},
            {"label": "Blue shirt + black pants", "query": "A blue shirt with black pants"},
        ],
    }


@app.post("/api/search")
def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(400, "query is empty")
    retriever = get_retriever(req.backbone)
    df = retriever.search(req.query, top_k=req.top_k,
                          use_color_gate=req.color_gate, weights=req.weights)
    parsed = parse_query(req.query)
    return {
        "query": req.query,
        "backbone": req.backbone if (req.backbone == "fashionclip" or MARQO_AVAILABLE) else "fashionclip",
        "color_gate": req.color_gate,
        "interpretation": {
            "garments": [c["text"] for c in parsed["fashion_clauses"]],
            "style": parsed["style"],
            "context": parsed["context"],
        },
        "results": [result_payload(r) for _, r in df.iterrows()],
    }


@app.post("/api/similar")
def similar(req: SimilarRequest):
    retriever = get_retriever(req.backbone)
    try:
        hits = retriever.similar(req.image_id, top_k=req.top_k)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return {
        "image_id": req.image_id,
        "results": [
            {"image_id": h["image_id"], "image_url": f"/api/image/{h['image_id']}",
             "similarity": round(h["similarity"], 4)}
            for h in hits
        ],
    }


@app.get("/api/image/{image_id}")
def image(image_id: str):
    path = image_paths().get(image_id)
    if not path or not os.path.exists(path):
        raise HTTPException(404, "image not found")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/health")
def health():
    return {"ok": True, "marqo": MARQO_AVAILABLE}
