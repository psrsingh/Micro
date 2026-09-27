"""Chunking that never crosses a structural boundary and keeps exact char offsets."""

from __future__ import annotations

import re
from dataclasses import dataclass

_WORD = re.compile(r"\S+")


@dataclass(frozen=True)
class ChunkSpan:
    start: int
    end: int
    n_tokens: int


def chunk_span(text: str, start: int, end: int, *, target: int, max_tokens: int,
               overlap: int) -> list[ChunkSpan]:
    """Split text[start:end] into overlapping windows of ~`target` tokens.

    A token here is a whitespace-delimited word (Phase 0). A section that fits in
    `max_tokens` becomes exactly one chunk. Spans index into `text`, so
    `text[c.start:c.end]` is the chunk text verbatim.
    """
    if not 0 <= overlap < target <= max_tokens:
        raise ValueError("need 0 <= overlap < target <= max_tokens")
    words = [(m.start() + start, m.end() + start) for m in _WORD.finditer(text[start:end])]
    if not words:
        return []
    if len(words) <= max_tokens:
        return [ChunkSpan(words[0][0], words[-1][1], len(words))]

    spans: list[ChunkSpan] = []
    step = target - overlap
    i = 0
    while i < len(words):
        j = min(i + target, len(words))
        # Fold a short tail into this window instead of emitting a tiny last chunk.
        if len(words) - j < overlap + 1 and len(words) - i <= max_tokens:
            j = len(words)
        spans.append(ChunkSpan(words[i][0], words[j - 1][1], j - i))
        if j == len(words):
            break
        i += step
    return spans
