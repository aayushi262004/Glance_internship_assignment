# Fashion-Aware Context Retrieval

> **Improvements addendum:** see [`IMPROVEMENTS.md`](IMPROVEMENTS.md) for a measured
> accuracy pass on top of this system — a Marqo-FashionSigLIP backbone
> (Precision@5 0.933 → 0.960, mAP 0.963 → 0.991) and pixel-HSV color verification
> that fixes the color-binding limitation. The app ships the improved retriever
> (`src/retriever/retriever_v2.py`); the objective eval harness is in
> `evaluation/harness/`.

Natural-language image search over a 1,000-image Fashionpedia corpus. Queries can combine
garment attributes, style, and scene context (e.g. *"a red tie and a white shirt in a formal
setting"*), and the system returns the top-k matching images. Built for the Glance ML
internship assignment (Multimodal Fashion & Context Retrieval).

## Approach

Two frozen encoders are used for what they are each good at: **FashionCLIP** for garment
semantics (what is worn) and **OpenCLIP ViT-B/32** for scene/context semantics (where it is
worn). Vanilla whole-query CLIP struggles with compositional fashion queries, so retrieval is
grounded at the garment-region level rather than relying on one sentence embedding.

At query time, a lightweight parser decomposes the query into **fashion clauses** ("red tie"
→ Fashionpedia category *tie*), a **style** phrase, and a **context** phrase. Candidates are
the union of three FAISS searches: global FashionCLIP, global CLIP, and — per clause — a
search of the **garment-region index**, so an image containing a matching garment crop is
found even when whole-image similarity misses it. Each candidate is then scored per
component: global similarity, per-clause region evidence (best crop of a compatible
category), **clause coverage** (a clause counts as matched relative to that clause's score
distribution across the current candidate pool — percentile plus an absolute floor — rather
than one fixed cutoff), an all-clauses-matched conjunction bonus, style similarity, and scene
context scored contrastively against positive/negative prototype sentences. The final score
is a weighted sum over only the components present in the query, with weights renormalized.

```
Query
  ↓
Query decomposition (fashion clauses / style / context)
  ↓
Global retrieval + garment-region retrieval (FAISS)
  ↓
Fashion / region / style / context scoring
  ↓
Clause coverage + dynamic score fusion
  ↓
Top-k results
```

**Offline indexing:** `scripts/prepare_annotated_dataset.py` samples 1,000 annotated
Fashionpedia validation images and writes an image manifest plus per-garment bounding boxes.
Three build scripts encode whole images (CLIP and FashionCLIP) and every garment crop
(FashionCLIP), storing L2-normalized embeddings and flat inner-product FAISS indexes under
`data/features/`.

## Setup

Python 3.11 on Windows:

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

(`requirements-lock.txt` pins the exact tested versions.)

Raw images, embeddings, and FAISS indexes are **not stored in Git** (only the metadata and
mapping CSVs are). To regenerate them, download the Fashionpedia val/test images
(`val_test2020`) and extract them under your `Downloads` folder, then run:

```
python scripts\prepare_annotated_dataset.py
python scripts\build_index.py
python scripts\build_fashion_index.py
python scripts\build_region_index.py
python scripts\build_region_colors.py     # pixel-HSV color descriptors (improvement)
python scripts\build_marqo_indexes.py     # Marqo-FashionSigLIP backbone (improvement)
```

**macOS note:** faiss and torch both bundle OpenMP; on macOS run every build /
eval / app command with `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1` to avoid a
silent segfault. (Not needed on the original Windows setup.)

Models (FashionCLIP, OpenCLIP weights) download automatically from Hugging Face on first
run. CPU is sufficient; CUDA is used automatically if available.

## Run

The demo is a **React (Vite) frontend + FastAPI backend**. The backend wraps the
retriever; the frontend is a Glance-themed search UI that proxies `/api` to it.

One command (starts both; open the printed Vite URL):

```
./run.sh
```

Or manually, in two terminals:

```
# terminal 1 — API (http://127.0.0.1:8000)
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
  ./venv/bin/python -m uvicorn server.main:app --port 8000

# terminal 2 — frontend (http://localhost:5173)
cd web && npm install && npm run dev
```

The UI exposes, beyond text search: parsed query interpretation, per-result
component score bars, live **backbone** (Marqo ↔ FashionCLIP) and **colour-gate**
ablation toggles, adjustable **fusion weights**, and **image-to-image "find
similar looks"**. Models download from Hugging Face on first run (CPU is fine).

Evaluation (separate from normal app usage):

```
python evaluation\run_evaluation.py    # retrieve top-5 for the 15 evaluation queries
python evaluation\label_results.py     # manually grade each result 0/1/2 (interactive)
python evaluation\compute_metrics.py   # Precision@5, mAP, nDCG@5 (overall / per query / per category)
```

## Evaluation

15 queries (including the five assignment evaluation prompts), 75 top-5 results, each
human-labeled with graded relevance (0 = irrelevant, 1 = partial, 2 = highly relevant).
Metrics: Precision@5 (relevance ≥ 1), mAP bounded to the retrieved top-5 pool, and nDCG@5 on
the graded labels.

| Slice                     | Precision@5 | mAP    | nDCG@5 |
|---------------------------|-------------|--------|--------|
| Overall (15 queries)      | 0.9333      | 0.9631 | 0.9632 |
| Compositional (4 queries) | 0.9000      | 0.9385 | 0.9117 |

These numbers come from the 15-query / 1,000-image assignment evaluation and should not be
read as broad statistical generalization.

Contrastive color-prompt scoring was evaluated but is disabled in the final system. It
improved compositional nDCG@5 from ~0.912 to ~0.945, but reduced overall Precision@5 from
~0.933 to ~0.907 and caused regressions on yellow/beige attribute queries.

## Scalability

The flat `IndexFlatIP` (exact search) is the right choice at 1,000 images. At ~1M images the
same retrieval logic would keep working with engineering substitutions: an approximate index
(IVF/HNSW, optionally with PQ compression), offline/distributed embedding generation, sharded
indexes, region reranking kept bounded to the global candidate pool (as it already is), and
compressed region-vector storage where needed. This is a proposed scaling design and was not
benchmarked at 1M images.

## Limitations

- Color-to-garment binding is imperfect: CLIP-family crop similarity is dominated by garment
  shape, so wrong-color matches can score competitively.
- Rare categories have sparse region annotations in the sampled corpus (e.g. 2 annotated
  ties), which bounds region-level evidence.
- The query parser uses a closed vocabulary; unknown garment/style/context words fall back to
  zero-shot whole-query embedding retrieval.
- Small evaluation set (15 queries; some categories contain a single query).
- mAP is bounded to the labeled top-5 pool, not corpus-wide relevance.
- Fusion weights are hand-set, not learned or cross-validated.
- The candidate-relative percentile calibration for clause coverage was selected from
  inspected score distributions, not a held-out validation set.

## Repository Structure

```
web/                        React (Vite) frontend — Glance-themed search UI
server/main.py              FastAPI backend wrapping the retriever
run.sh                      Launch backend + frontend together
src/indexer/encoders.py     CLIP + FashionCLIP (+ Marqo) encoder wrappers
src/retriever/              Query parsing, retrieval, scoring, fusion
                            (retriever_v2.py = improved, shipped)
scripts/                    Dataset prep, index building, Marqo + colour descriptors
evaluation/                 Queries, evaluation runner, labeling tool, metrics
evaluation/harness/         Objective config-vs-config eval + reusable label store
data/metadata/              Image manifest + Fashionpedia garment annotations (tracked)
data/raw/, data/features/   Images, embeddings, FAISS indexes (generated, not in Git)
```
