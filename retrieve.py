"""Query-time hybrid retrieval (timed; includes query embedding).

Candidate pages from page-level BM25, scored by linear fusion of three signals —
page-dense cosine, BM25, and chunk-dense (max-pooled to page) — then the top
CE_TOPK are reranked by a cross-encoder. Artifacts load once and cache.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from embed import embed_queries
from index import load_chunk_index, load_hybrid, tokenize

# tuned config (selected by offline sweep on the public set)
CAND_M = 50                 # BM25 candidate pages fed to fusion
RETURN_K = 100              # pages returned per query (only first 10 are scored)
W_DENSE, W_BM25, W_CHUNK = 0.3, 0.5, 0.2   # linear fusion weights
CHUNK_TOPN = 256            # chunks fetched per query before max-pool to page
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CE_TOPK = 10                # candidates reranked by the cross-encoder

_H: Optional[Dict] = None
_CHUNK = None
_CE = None
_PID2ROW: Optional[Dict[int, int]] = None


def _hybrid(artifacts_dir: Optional[Path]) -> Dict:
    """Load and cache the page/BM25/text artifacts plus a page_id -> row map."""
    global _H, _PID2ROW
    if _H is None:
        _H = load_hybrid(artifacts_dir)
        _PID2ROW = {pid: i for i, pid in enumerate(_H["page_ids"])}
    return _H


def _chunk(artifacts_dir: Optional[Path]):
    """Load and cache the chunk FAISS index."""
    global _CHUNK
    if _CHUNK is None:
        _CHUNK = load_chunk_index(artifacts_dir)
    return _CHUNK


def _cross_encoder():
    """Load and cache the cross-encoder reranker."""
    global _CE
    if _CE is None:
        from sentence_transformers import CrossEncoder
        _CE = CrossEncoder(CE_MODEL)
    return _CE


def _minmax(x: np.ndarray) -> np.ndarray:
    """Scale a score vector to [0, 1]; flat input maps to zeros."""
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


def _bm25_scores(tokens: List[str], h: Dict) -> np.ndarray:
    """Page BM25 scores via gather + scatter-add over precomputed posting weights."""
    scores = np.zeros(len(h["page_ids"]), dtype=np.float32)
    vocab, ptr, pages, wts = h["vocab"], h["term_ptr"], h["post_pages"], h["post_weights"]
    for tok in tokens:
        tid = vocab.get(tok)
        if tid is None:
            continue
        a, b = int(ptr[tid]), int(ptr[tid + 1])
        if b > a:
            np.add.at(scores, pages[a:b], wts[a:b])
    return scores


def _chunk_page_scores(qvec: np.ndarray, artifacts_dir: Optional[Path]) -> Dict[int, float]:
    """Search the chunk index and max-pool chunk cosine to a per-page score."""
    index, chunk_page_ids = _chunk(artifacts_dir)
    sc, idx = index.search(qvec[None, :], min(CHUNK_TOPN, index.ntotal))
    best: Dict[int, float] = {}
    for s, row in zip(sc[0], idx[0]):
        if row < 0:                                   # FAISS pads with -1 when k > ntotal
            continue
        pid = chunk_page_ids[int(row)]
        if pid not in best or s > best[pid]:
            best[pid] = float(s)
    return best


def _rank_one(query: str, qvec: np.ndarray, h: Dict, artifacts_dir: Optional[Path]) -> List[int]:
    """Rank pages for one query: BM25 candidates -> fused score -> cross-encoder rerank."""
    page_ids = h["page_ids"]
    dense_all = h["page_vecs"] @ qvec
    bm25_all = _bm25_scores(tokenize(query), h)
    chunk_best = _chunk_page_scores(qvec, artifacts_dir)

    cand = np.argsort(-bm25_all)[:CAND_M]
    if int((bm25_all[cand] > 0).sum()) == 0:          # no lexical hit -> back off to dense
        cand = np.argsort(-dense_all)[:CAND_M]
    crows = [_PID2ROW[p] for p in chunk_best if p in _PID2ROW]
    if crows:                                         # let strong chunk hits enter the pool
        cand = np.union1d(cand, np.array(crows, dtype=cand.dtype))

    chunk_vec = np.array([chunk_best.get(page_ids[int(r)], 0.0) for r in cand], dtype=np.float32)
    fused = (W_DENSE * _minmax(dense_all[cand])
             + W_BM25 * _minmax(bm25_all[cand])
             + W_CHUNK * _minmax(chunk_vec))
    order = cand[np.argsort(-fused)]

    if h.get("page_texts") is not None and len(order):
        topk = order[:min(CE_TOPK, len(order))]
        pairs = [(query, h["page_texts"][int(r)]) for r in topk]
        ce = np.asarray(_cross_encoder().predict(pairs, show_progress_bar=False))
        order = np.concatenate([topk[np.argsort(-ce)], order[len(topk):]])

    out, seen = [], set()
    for row in order:
        pid = page_ids[int(row)]
        if pid not in seen:
            seen.add(pid)
            out.append(pid)
        if len(out) >= RETURN_K:
            break
    return out


def search_batch(queries: List[str], *, artifacts_dir: Optional[Path] = None) -> List[List[int]]:
    """Return one ranked list of page_id (best first) per query."""
    h = _hybrid(artifacts_dir)
    qvecs = embed_queries(queries)
    if qvecs.size == 0:
        return [[] for _ in queries]
    qvecs = np.ascontiguousarray(qvecs, dtype=np.float32)
    return [_rank_one(queries[i], qvecs[i], h, artifacts_dir) for i in range(len(queries))]