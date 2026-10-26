"""Small deterministic text helpers (sentence splitting, tokens, overlap)."""
from __future__ import annotations

import re

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are", "was",
    "were", "be", "it", "its", "as", "at", "by", "with", "that", "this", "which", "who",
    "what", "how", "why", "when", "where", "does", "do", "did", "has", "have", "had",
    "from", "about", "into", "than", "then", "they", "you", "your",
}


def split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def tokens(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


def content_tokens(text: str) -> set[str]:
    return {t for t in tokens(text) if t not in _STOP and len(t) > 1}


def overlap_score(a: str, b: str) -> float:
    """Jaccard-style content-word overlap in [0, 1]."""
    ta, tb = content_tokens(a), content_tokens(b)
    if not ta or not tb:
        return 0.0
    inter = ta & tb
    return len(inter) / len(ta | tb)
