"""Explicit temporal features shared by the freshness trainer and runtime (FR3).

Kept in a module of its own because the trained sklearn Pipeline pickles a reference to
`temporal_features`; joblib must be able to import it by the same path at load time.
"""
from __future__ import annotations

import re
import numpy as np

_TEMPORAL_WORDS = [
    "latest", "current", "currently", "now", "today", "recent", "recently", "this year",
    "this month", "nowadays", "up to date", "updated", "newest", "new", "right now",
    "at the moment", "as of", "present", "ongoing", "live",
]
_EVENT_WORDS = [
    "price", "rate", "won", "winner", "score", "election", "champion", "stock", "news",
    "release", "released", "launch", "announced", "version", "who is the", "standings",
]
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_STATIC_WORDS = [
    "define", "definition", "formula", "theorem", "what is the", "how does", "why does",
    "capital of", "boiling point", "symbol for", "calculate", "explain how", "law of",
]


def _row(q: str) -> list[float]:
    s = q.lower()
    temporal = sum(1 for w in _TEMPORAL_WORDS if w in s)
    event = sum(1 for w in _EVENT_WORDS if w in s)
    static = sum(1 for w in _STATIC_WORDS if w in s)
    years = _YEAR_RE.findall(s)
    has_recent_year = 1.0 if any(int(q[m.start():m.end()]) >= 2020
                                 for m in _YEAR_RE.finditer(s)) else 0.0
    return [
        float(temporal),
        float(event),
        float(static),
        float(len(years)),
        has_recent_year,
        float(len(s.split())),
    ]


def temporal_features(queries) -> np.ndarray:
    """Vectorised transform: list[str] -> (n, 6) float array."""
    if isinstance(queries, str):
        queries = [queries]
    return np.asarray([_row(q) for q in queries], dtype=float)
