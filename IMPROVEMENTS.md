# Accuracy Improvements — Analysis & Results

This document records a focused pass over the submitted system: reproducing its
reported numbers, then measuring concrete ML improvements against them under an
identical human-judgment protocol. It supplements (does not replace) the
original report.

## TL;DR

| Configuration | Precision@5 | mAP@5 | nDCG@5 |
|---|---|---|---|
| Baseline (FashionCLIP + region fusion) — reproduced | 0.9333 | 0.9631 | 0.9627 |
| Baseline + pixel-HSV color gate | 0.9467 | 0.9719 | 0.9680 |
| Baseline + CLIP-global component *(rejected)* | 0.9200 | 0.9686 | 0.9507 |
| **Marqo-FashionSigLIP backbone** | **0.9600** | **0.9911** | 0.9641 |
| Marqo backbone + color gate *(shipped)* | 0.9600 | 0.9911 | 0.9618 |

The largest, most defensible win is **swapping the frozen fashion backbone from
FashionCLIP (ViT-B/32, 2023) to Marqo-FashionSigLIP** — a stronger fashion-adapted
encoder. It lifts Precision@5 from 0.933 → 0.960 and mAP from 0.963 → **0.991**,
and takes **three of the five mandatory assignment queries to a perfect 1.0**
(office, park-bench, red-tie-and-white-shirt). A second improvement,
**pixel-level HSV color verification**, directly fixes the color-binding failure
the original report named as its #1 limitation.

## 0. Reproducibility fixes (the project did not run as-is on macOS)

Three environment/version bugs had to be fixed before any measurement:

1. **faiss + torch OpenMP crash.** Importing `faiss` and running a torch encode
   in one process segfaults silently (duplicate libomp). Run with
   `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1`.
2. **Windows paths in the tracked CSVs** (`data\raw\img_0000.jpg`). Added
   `norm_path()` in `src/indexer/encoders.py`, used by every builder/retriever.
3. **transformers ≥4.5x** broke the FashionCLIP encoder's
   `vision_model(..., return_dict=True)` calls. Rewrote to `get_image_features()` /
   `get_text_features()` (identical embeddings, version-stable).

After these, the baseline reproduces the report's numbers **exactly**
(0.9333 / 0.9631 / 0.9627) with zero unlabeled results.

## 1. Objective evaluation harness

`evaluation/harness/` scores any retriever configuration against a reusable
label store:

- `label_store.csv` — `(query, image_id) → relevance` (0/1/2). Seeded from the
  original human judgments; newly surfaced (query, image) pairs were judged by
  visual inspection under the same rubric (the same protocol the original report
  used for its baseline vs. final comparison).
- `metrics.py` — Precision@5, mAP@5 (top-5-pool), nDCG@5; **flags any unlabeled
  pair so metrics are never computed on unseen results.**
- `run_config.py` — runs a config end-to-end and reports metrics.

```
python evaluation/harness/build_label_store.py
python evaluation/harness/run_config.py baseline
python evaluation/harness/run_config.py v2 --backbone marqo --color-gate
```

## 2. Improvements measured

All improvements live in `src/retriever/retriever_v2.py`, a configurable superset
of the original retriever (with all flags off it is byte-for-byte equivalent to
the baseline — verified).

### 2a. Marqo-FashionSigLIP backbone *(kept — biggest win)*

The original uses FashionCLIP for the fashion + region signals. Replacing it with
**Marqo-FashionSigLIP** (a frozen, fashion-adapted SigLIP encoder, loaded via
`open_clip`) — same architecture role, same zero-shot usage, richer 768-d
embeddings — is the single largest lever:

- Precision@5 0.9333 → **0.9600**; mAP 0.9631 → **0.9911**.
- 4 queries improve, 1 regresses (yellow-coat 1.0 → 0.6: Marqo ranks camel/beige
  coats as strong "coat" matches — a warm-tone confusion pixel hue can only
  partly separate, kept as an honest tradeoff).
- Mandatory queries: office 0.8→1.0, park-bench 0.8→1.0, red-tie 0.8→1.0.

Build: `python scripts/build_marqo_indexes.py`.

### 2b. Pixel-HSV color verification *(kept)*

The report's #1 limitation: CLIP-family crop similarity is shape-dominated, so a
**red** jacket scores as high as a blue one for "blue jacket". Their fix
(contrastive CLIP color prompts) improved one slice but hurt overall precision
and regressed yellow/beige, so it was disabled.

This pass verifies color from **actual crop pixels** instead of embeddings
(`scripts/build_region_colors.py` precomputes a per-region HSV color histogram;
the retriever applies a soft multiplicative gate to color clauses). Concretely it
demotes `img_0393` (a red jacket, human-labeled irrelevant, that the baseline
ranked #2 for "blue jacket").

The key subtlety: naive hue buckets misread golden/mustard yellow as "orange",
which would wrongly demote true yellow coats. A **perceptual color-adjacency**
map (`COLOR_COMPAT`) credits neighbouring hues, so the gate catches clear
mismatches (red≠blue) without the yellow regression that sank the embedding-based
attempt.

- On FashionCLIP: Precision@5 0.9333 → **0.9467**, mAP and nDCG both up, no query
  regresses. Fixes "blue jacket" (removes the red false positive).
- On Marqo: neutral on P@5/mAP (Marqo already resolves most color confusions).
  Kept in the shipped config as a robustness safeguard — a 15-query eval
  under-weights color-binding errors.

### 2c. CLIP-global scoring component *(measured, rejected)*

In the baseline the general-CLIP global vector is used only for candidate recall
and context prototypes; its whole-query similarity never scores the ranking.
Adding it as a fusion component improved mAP (better ordering) but **lowered
Precision@5 to 0.9200** (diluted fine-grained garment queries). Rejected —
documented here for the same reason the original report documented its
color-contrast experiment: measured model selection, not intuition.

## 3. Shipped configuration

The demo is a **React (Vite) frontend + FastAPI backend** (`web/` + `server/`;
the Streamlit UI was retired). The API loads
`RetrieverV2(backbone="marqo", use_color_gate=True)` (falls back to FashionCLIP
automatically if the Marqo indexes are absent) and the UI adds live backbone /
colour-gate ablation toggles, fusion-weight sliders, and image-to-image
"find similar looks". Reproduce:

```
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
python scripts/build_index.py && \
python scripts/build_fashion_index.py && \
python scripts/build_region_index.py && \
python scripts/build_region_colors.py && \
python scripts/build_marqo_indexes.py
./run.sh          # FastAPI backend + React frontend
```

## 4. Honest caveats

- **Evaluation scale.** 15 queries, 75 judgments — small; per-query deltas of one
  image (±0.067 P@5) are within labeling noise. Trends that move **both**
  Precision@5 and mAP together (Marqo) are the trustworthy signal.
- **Labeler variance.** Evaluating Marqo required judging 32 new images myself
  (the original labeler judged the baseline). Labels used a conservative reading
  of the original rubric; the mAP gain (0.963 → 0.991) is large and consistent
  with the qualitative quality of Marqo's results.
- **Not fixed.** Yellow-coat warm-tone confusion (Marqo); "sitting on a bench" is
  still inferred, not structurally verified; corpus sparsity (2 ties, thin office
  coverage) bounds any ranking change.

## 5. If continuing

- A light cross-encoder / VLM reranker over the top ~50 Marqo candidates (mAP is
  already 0.991, so headroom is mostly in the long tail and harder queries).
- A larger, multi-labeler evaluation set to make deltas statistically meaningful.
- Learned fusion weights and held-out calibration of the clause-match percentile
  (both currently hand-set), as the original report also proposes.
