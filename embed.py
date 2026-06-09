"""Embedding utilities (sentence-transformers/all-MiniLM-L6-v2 only)."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from utils import EMBEDDING_MODEL_NAME

_model: SentenceTransformer | None = None

EMBED_DIM = 384       # MiniLM-L6-v2 output dimensionality
MAX_SEQ_LENGTH = 256  # model token cap


def get_model() -> SentenceTransformer:
    """Lazily load and cache the encoder (uses GPU automatically if present)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        _model.max_seq_length = MAX_SEQ_LENGTH
    return _model


def embed_texts(texts: Sequence[str], *, batch_size: int = 64) -> np.ndarray:
    """Return L2-normalized float32 embeddings, shape (n, EMBED_DIM)."""
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    model = get_model()
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.ascontiguousarray(vectors, dtype=np.float32)


def embed_queries(queries: List[str], *, batch_size: int = 64) -> np.ndarray:
    return embed_texts(queries, batch_size=batch_size)