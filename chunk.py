"""Preprocessing and chunking.

Adaptive word-window chunking: short pages stay one chunk; long pages split into
overlapping windows so content past MiniLM's token limit is still represented.
The title is prepended to every chunk so each unit carries the entity name.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

WORDS_PER_CHUNK = 180        # stays under MiniLM's ~256-token cap once the title is added
WORD_OVERLAP = 40            # keeps facts that straddle a window boundary in both windows


@dataclass
class Chunk:
    """One retrieval unit: its page, its index within the page, and its text."""
    page_id: int
    chunk_id: int
    text: str


def _window_words(text: str, size: int, overlap: int) -> List[str]:
    """Split text into overlapping word windows (one window if short enough)."""
    words = text.split()
    if not words:
        return []
    if len(words) <= size:
        return [" ".join(words)]
    step = max(1, size - overlap)
    windows: List[str] = []
    for start in range(0, len(words), step):
        windows.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return windows


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    """Split one corpus entry into one or more title-prefixed chunks."""
    page_id = int(record["page_id"])
    title = str(record.get("title", "")).strip()
    content = str(record.get("content", "")).strip()
    windows = _window_words(content, WORDS_PER_CHUNK, WORD_OVERLAP) or ([title] if title else [""])
    return [Chunk(page_id=page_id, chunk_id=cid,
                  text=f"{title}\n\n{w}".strip() if title else w)
            for cid, w in enumerate(windows)]


def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    """Chunk every record into a flat list of retrieval units."""
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(chunk_entry(record))
    return chunks