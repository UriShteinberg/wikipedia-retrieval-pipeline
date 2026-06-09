# Section B — Retrieval Pipeline

Hybrid retrieval over the Wikipedia-style corpus. For each query we generate
page candidates with BM25, score them by a linear fusion of three signals
(page-dense cosine, BM25, chunk-dense max-pooled to page), and rerank the top
candidates with a cross-encoder.

## Pipeline
- **chunk.py** — overlapping title-prefixed word windows.
- **embed.py** — MiniLM (`all-MiniLM-L6-v2`), L2-normalized embeddings.
- **index.py** — builds/loads: chunk FAISS, page vectors, page-level BM25, page texts.
- **retrieve.py** — BM25 candidates → fused score (W_DENSE=0.3, W_BM25=0.5, W_CHUNK=0.2)
  → cross-encoder rerank of the top CE_TOPK=10.

## Artifacts (`artifacts/`, loaded by `run()`)
| file | content |
|------|---------|
| `index.faiss`, `index_meta.json` | chunk dense index + chunk→page map |
| `page_vecs.npy`, `page_meta.json` | page vectors + page_ids |
| `bm25.npz`, `bm25_vocab.json` | page BM25 postings + term→id map |
| `page_texts.json` | page text for the cross-encoder |

## Setup & build