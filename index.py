"""Offline index build and load. The build is untimed; run() only loads from disk.

Writes three retrieval signals to artifacts/:
  index.faiss / index_meta.json   chunk-level dense FAISS (chunk channel)
  page_vecs.npy / page_meta.json  whole-page dense vectors (page channel)
  bm25.npz / bm25_vocab.json      page-level BM25 (precomputed posting weights)
  page_texts.json                 page text aligned to page_ids (cross-encoder)
FAISS is (de)serialized via Python bytes I/O so non-ASCII (e.g. Hebrew) paths work.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from chunk import Chunk, chunk_corpus
from embed import embed_texts
from utils import ARTIFACTS_DIR, ensure_artifacts_dir, entry_text, iter_entries

INDEX_FAISS_NAME = "index.faiss"
INDEX_META_NAME = "index_meta.json"
PAGE_VECS_NAME = "page_vecs.npy"
PAGE_META_NAME = "page_meta.json"
BM25_NAME = "bm25.npz"
BM25_VOCAB_NAME = "bm25_vocab.json"
PAGE_TEXTS_NAME = "page_texts.json"

EMBED_DIM = 384
PAGE_WORD_CAP = 400          # title+content cap; MiniLM truncates ~256 tokens anyway
BM25_K1, BM25_B = 1.5, 0.75
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokenizer shared by the BM25 build and query time."""
    return _TOKEN_RE.findall(text.lower())


def page_text(record: Dict) -> str:
    """Return a page's title+content, capped to PAGE_WORD_CAP words."""
    return " ".join(entry_text(record).split()[:PAGE_WORD_CAP])


def _write_faiss(index: faiss.Index, path: Path) -> None:
    """Serialize a FAISS index to bytes and write via Python (Unicode-safe)."""
    Path(path).write_bytes(faiss.serialize_index(index).tobytes())


def _read_faiss(path: Path) -> faiss.Index:
    """Read serialized FAISS bytes via Python and rebuild the index."""
    return faiss.deserialize_index(np.frombuffer(Path(path).read_bytes(), dtype=np.uint8).copy())


def build_bm25(page_token_lists: List[List[str]]) -> Dict:
    """Build a page-level BM25 index with posting weights precomputed, stored CSR-by-term."""
    n_pages = len(page_token_lists)
    doc_len = np.array([len(t) for t in page_token_lists], dtype=np.float64)
    avgdl = float(doc_len.mean()) if n_pages else 0.0

    vocab: Dict[str, int] = {}
    df: Dict[int, int] = {}
    page_tf: List[Dict[int, int]] = []
    for toks in page_token_lists:
        tf: Dict[int, int] = {}
        for w in toks:
            tid = vocab.setdefault(w, len(vocab))
            tf[tid] = tf.get(tid, 0) + 1
        page_tf.append(tf)
        for tid in tf:
            df[tid] = df.get(tid, 0) + 1

    idf = np.zeros(len(vocab), dtype=np.float64)
    for tid, d in df.items():
        idf[tid] = math.log((n_pages - d + 0.5) / (d + 0.5) + 1.0)  # +1 keeps idf >= 0

    per_term: List[List[Tuple[int, float]]] = [[] for _ in range(len(vocab))]
    for prow, tf in enumerate(page_tf):
        # BM25 length normalization: longer pages discount each term's contribution
        c = BM25_K1 * (1.0 - BM25_B + BM25_B * (doc_len[prow] / avgdl if avgdl else 0.0))
        for tid, f in tf.items():
            per_term[tid].append((prow, idf[tid] * (f * (BM25_K1 + 1.0)) / (f + c)))

    term_ptr = np.zeros(len(vocab) + 1, dtype=np.int64)
    pages, weights = [], []
    for tid, postings in enumerate(per_term):
        for prow, w in postings:
            pages.append(prow)
            weights.append(w)
        term_ptr[tid + 1] = len(pages)

    return {
        "vocab": vocab,
        "arrays": dict(term_ptr=term_ptr,
                       post_pages=np.asarray(pages, dtype=np.int32),
                       post_weights=np.asarray(weights, dtype=np.float32)),
    }


def build_index(*, entries_dir: Optional[Path] = None,
                artifacts_dir: Optional[Path] = None) -> Tuple[faiss.Index, List[int]]:
    """Build and persist all retrieval artifacts; returns (chunk_index, chunk_page_ids)."""
    out_dir = artifacts_dir or ensure_artifacts_dir()
    records = list(iter_entries(entries_dir))

    # chunk channel: dense FAISS over overlapping windows
    chunks: List[Chunk] = chunk_corpus(records)
    chunk_vectors = embed_texts([c.text for c in chunks])
    chunk_page_ids = [int(c.page_id) for c in chunks]
    dim = int(chunk_vectors.shape[1]) if chunk_vectors.size else EMBED_DIM
    chunk_index = faiss.IndexFlatIP(dim)              # vectors are L2-normalized -> IP = cosine
    if chunk_vectors.size:
        chunk_index.add(chunk_vectors)
    _write_faiss(chunk_index, out_dir / INDEX_FAISS_NAME)
    (out_dir / INDEX_META_NAME).write_text(json.dumps({
        "page_ids": chunk_page_ids,
        "chunk_ids": [int(c.chunk_id) for c in chunks],
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": dim, "num_vectors": len(chunk_page_ids), "num_pages": len(set(chunk_page_ids)),
    }, indent=2), encoding="utf-8")

    # page channel: one dense vector + raw text per page
    page_ids = [int(r["page_id"]) for r in records]
    page_texts = [page_text(r) for r in records]
    page_vecs = embed_texts(page_texts)
    if page_vecs.size == 0:
        page_vecs = np.zeros((0, dim), dtype=np.float32)
    np.save(out_dir / PAGE_VECS_NAME, page_vecs.astype(np.float32))
    (out_dir / PAGE_META_NAME).write_text(json.dumps({"page_ids": page_ids}, indent=2), encoding="utf-8")
    (out_dir / PAGE_TEXTS_NAME).write_text(json.dumps(page_texts), encoding="utf-8")

    # lexical channel: page-level BM25
    bm = build_bm25([tokenize(t) for t in page_texts])
    np.savez(out_dir / BM25_NAME, **bm["arrays"])
    (out_dir / BM25_VOCAB_NAME).write_text(json.dumps(bm["vocab"]), encoding="utf-8")

    return chunk_index, chunk_page_ids


def load_chunk_index(artifacts_dir: Optional[Path] = None) -> Tuple[faiss.Index, List[int]]:
    """Load the chunk FAISS index and its chunk-row -> page_id map."""
    root = artifacts_dir or ARTIFACTS_DIR
    meta = json.loads((root / INDEX_META_NAME).read_text(encoding="utf-8"))
    return _read_faiss(root / INDEX_FAISS_NAME), [int(x) for x in meta["page_ids"]]


def load_hybrid(artifacts_dir: Optional[Path] = None) -> Dict:
    """Load page vectors, page-level BM25, and page texts used by the hybrid retriever."""
    root = artifacts_dir or ARTIFACTS_DIR
    page_ids = json.loads((root / PAGE_META_NAME).read_text(encoding="utf-8"))["page_ids"]
    bm = np.load(root / BM25_NAME)
    pt = root / PAGE_TEXTS_NAME
    return {
        "page_vecs": np.ascontiguousarray(np.load(root / PAGE_VECS_NAME), dtype=np.float32),
        "page_ids": [int(x) for x in page_ids],
        "term_ptr": bm["term_ptr"], "post_pages": bm["post_pages"], "post_weights": bm["post_weights"],
        "vocab": json.loads((root / BM25_VOCAB_NAME).read_text(encoding="utf-8")),
        "page_texts": json.loads(pt.read_text(encoding="utf-8")) if pt.exists() else None,
    }