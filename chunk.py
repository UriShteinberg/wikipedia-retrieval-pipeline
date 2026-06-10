"""Preprocessing and chunking.

Paragraph-aware chunking: paragraphs are packed greedily into bins of a target
word count, with one sentence of overlap carried between bins. Oversized
paragraphs are split into fixed word windows. The title is prepended to every
chunk so each unit carries the entity name. This reproduces the artifacts under
artifacts/ (628,751 chunks over the corpus).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

TARGET_WORDS = 150           # target words per chunk before flushing a bin
HARD_MAX = 200               # paragraphs longer than this are windowed alone
OVERLAP_SENTS = 1            # sentences carried from one bin to the next

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    """One retrieval unit: its page, its index within the page, and its text."""
    page_id: int
    chunk_id: int
    text: str


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences on terminal punctuation."""
    return [p for p in _SENT_SPLIT.split(text.strip()) if p]


def _chunk_text(title: str, content: str) -> List[str]:
    """Pack paragraphs into ~TARGET_WORDS bins with sentence overlap; prefix title."""
    title = (title or "").strip()
    content = (content or "").strip()
    if not content:
        return [title] if title else [""]

    paras = [p.strip() for p in content.split("\n\n") if p.strip()] or [content]
    bins: List[str] = []
    cur: List[str] = []
    cur_n = 0
    carry: List[str] = []

    def flush() -> None:
        nonlocal cur, cur_n, carry
        if cur:
            bins.append(" ".join(cur))
            sents = _split_sentences(" ".join(cur))
            carry = sents[-OVERLAP_SENTS:] if OVERLAP_SENTS else []
            cur = list(carry)
            cur_n = sum(len(s.split()) for s in carry)

    for para in paras:
        pw = para.split()
        if len(pw) > HARD_MAX:                      # oversized: window it alone
            flush()
            step = max(1, TARGET_WORDS - 30)
            for s in range(0, len(pw), step):
                bins.append(" ".join(pw[s:s + TARGET_WORDS]))
                if s + TARGET_WORDS >= len(pw):
                    break
            cur, cur_n, carry = [], 0, []
            continue
        if cur and cur_n + len(pw) > TARGET_WORDS:
            flush()
        cur.append(para)
        cur_n += len(pw)

    if cur and cur != carry:
        bins.append(" ".join(cur))

    pref = (title + "\n\n") if title else ""
    return [(pref + b).strip() for b in bins] if bins else [title]


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    """Split one corpus entry into one or more title-prefixed chunks."""
    page_id = int(record["page_id"])
    title = str(record.get("title", ""))
    content = str(record.get("content", ""))
    return [Chunk(page_id=page_id, chunk_id=cid, text=text)
            for cid, text in enumerate(_chunk_text(title, content))]


def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    """Chunk every record into a flat list of retrieval units."""
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(chunk_entry(record))
    return chunks