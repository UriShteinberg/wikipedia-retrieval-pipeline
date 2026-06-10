# Section B — Wikipedia Retrieval Pipeline

End-to-end retrieval over a corpus of ~27,000 Wikipedia-style pages. For each
query, `run(queries)` returns a ranked list of `page_id`s, scored by mean
NDCG@10 over the top 10 results.

**Method in one line:** BM25 retrieves candidate pages, a linear fusion of three
signals (page-dense cosine, BM25, chunk-dense max-pooled to page) ranks them, and
a cross-encoder reranks the top candidates — blended with the fusion score rather
than replacing it.

## Pipeline

| stage | file | what it does |
|-------|------|--------------|
| chunk | `chunk.py` | Paragraph-aware chunking: paragraphs are packed into ~150-word bins with one sentence of overlap; oversized paragraphs are windowed; the page title is prepended to every chunk. |
| embed | `embed.py` | `sentence-transformers/all-MiniLM-L6-v2`, L2-normalized 384-d embeddings (so inner product = cosine). |
| index | `index.py` | Offline build + load of all retrieval artifacts (chunk FAISS, page vectors, page-level BM25, page texts). |
| retrieve | `retrieve.py` | Query-time hybrid retrieval and reranking (timed path). |

### Retrieval detail (`retrieve.py`)

1. **Candidates** — top `CAND_M=50` pages by page-level BM25 (falls back to dense if no lexical hit).
2. **Fusion** — min-max normalized linear blend: `W_DENSE=0.3 · dense + W_BM25=0.5 · bm25 + W_CHUNK=0.2 · chunk`.
3. **Cross-encoder rerank** — the top `CE_TOPK=10` are scored by `cross-encoder/ms-marco-MiniLM-L-6-v2` and **blended** with the fusion score: `W_CE=0.85 · CE + 0.15 · fusion`. The blend (rather than letting the CE fully replace fusion) was tuned on the public set and avoids the CE's noise burying correctly-ranked pages.

## Artifacts (`artifacts/`, loaded by `run()` — not rebuilt at grading)

| file | content | format |
|------|---------|--------|
| `index.faiss` | chunk-level dense vectors (628,751 chunks) | FAISS `IndexFlatIP` (Git LFS) |
| `index_meta.json` | chunk-row → page_id map + build metadata | JSON |
| `page_vecs.npy` | one dense vector per page | float32 `(N, 384)` (Git LFS) |
| `page_meta.json` | page_ids aligned to `page_vecs.npy` | JSON |
| `bm25.npz` | page-level BM25 postings (precomputed weights, CSR-by-term) | npz (Git LFS) |
| `bm25_vocab.json` | term → term_id map | JSON |
| `page_texts.json` | per-page title+content (capped 400 words), cross-encoder input | JSON |

Large binaries are stored with **Git LFS**. After cloning, run `git lfs pull` to
fetch them.

## Setup

```bash
pip install -r requirements.txt
git lfs pull          # fetch the LFS-tracked artifacts
```

Dependencies (`requirements.txt`): `numpy`, `sentence-transformers`, `faiss-cpu`.
The MiniLM embedder and the cross-encoder are downloaded from the Hugging Face
hub on first use (network required for the first run).

## Run the evaluation

```bash
python scripts/eval_public.py
```

Prints mean NDCG@10 on the public query set. No index build is required — `run()`
loads the prebuilt artifacts from `artifacts/`.

Expected output:
```
public_queries=29
mean_ndcg@10=0.4597
query_phase_time≈19s
```

## Rebuilding the index (optional, offline)

The artifacts are committed, so this is **not** needed to run the evaluation. To
regenerate them from the corpus:

```bash
python scripts/build_index.py     # or: python main.py
```

This re-chunks, re-embeds, and rebuilds all artifacts. It is slow (CPU embedding
of ~628k chunks) and untimed — it is never run at grading. The committed
`chunk.py` reproduces the committed `index.faiss` exactly.

## Repository layout

```
main.py             run(queries) — graded entry point
chunk.py            paragraph-aware chunker
embed.py            MiniLM embeddings
index.py            artifact build + load
retrieve.py         query-time hybrid retrieval + CE rerank
utils.py            shared paths and helpers
eval.py             NDCG@10 utilities (read-only)
scripts/            eval_public.py, build_index.py (read-only)
artifacts/          prebuilt index + embeddings (Git LFS)
data/               corpus + public queries
```

## Video

[Presentation video](ADD_YOUR_LINK_HERE) — pipeline walkthrough and empirical results.
