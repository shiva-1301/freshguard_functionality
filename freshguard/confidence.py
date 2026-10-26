"""Confidence score computation (FR13).

Combines three signals into a 0-100% score:
  * evidence support ratio  (fraction of statements supported/review)
  * mean evidence relevance (how on-topic the retrieved docs are)
  * freshness alignment     (did we retrieve when the query looked stale, and vice-versa)
"""
from __future__ import annotations

from freshguard.verify import VerifiedStatement


def support_ratio(statements: list[VerifiedStatement]) -> float:
    if not statements:
        return 0.0
    good = sum(1 for s in statements if s.label in ("supported", "review"))
    return good / len(statements)


def freshness_alignment(freshness: float, retrieved: bool) -> float:
    """1.0 when the action matched the freshness signal: retrieved a stale-looking query,
    or answered a fresh-looking query directly. Degrades smoothly otherwise."""
    return (freshness if retrieved else 1.0 - freshness)


def compute_confidence(statements: list[VerifiedStatement], mean_relevance: float,
                       freshness: float, retrieved: bool) -> float:
    sr = support_ratio(statements)
    fa = freshness_alignment(freshness, retrieved)
    raw = 0.5 * sr + 0.3 * max(0.0, min(1.0, mean_relevance)) + 0.2 * fa
    # penalise an empty answer
    if not statements:
        raw *= 0.3
    return round(100.0 * max(0.0, min(1.0, raw)), 1)
