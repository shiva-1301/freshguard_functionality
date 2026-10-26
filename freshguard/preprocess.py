"""Query cleaning, normalisation and classification (FR2)."""
from __future__ import annotations

import re
from dataclasses import dataclass

from freshguard.features import _YEAR_RE, _TEMPORAL_WORDS

_WS = re.compile(r"\s+")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

_CATEGORY_KEYWORDS = {
    "sports": ["world cup", "olympics", "match", "won", "champion", "score", "league", "tournament"],
    "finance": ["price", "rate", "stock", "inflation", "market", "gdp", "interest", "bitcoin", "economy"],
    "technology": ["smartphone", "chipset", "gpu", "ai", "model", "laptop", "software", "release", "version"],
    "science": ["photosynthesis", "theorem", "gravity", "atom", "speed of light", "boiling", "formula", "cell"],
    "geography": ["capital", "country", "city", "continent", "river", "mountain"],
    "politics": ["president", "election", "prime minister", "government", "policy", "minister"],
}


@dataclass
class PreprocessResult:
    raw: str
    clean: str
    time_sensitive: bool
    category: str


def clean_text(q: str) -> str:
    q = _CTRL.sub(" ", q)
    q = _WS.sub(" ", q).strip()
    return q


def classify_category(q: str) -> str:
    s = q.lower()
    best, best_hits = "general", 0
    for cat, kws in _CATEGORY_KEYWORDS.items():
        hits = sum(1 for k in kws if k in s)
        if hits > best_hits:
            best, best_hits = cat, hits
    return best


def is_time_sensitive(q: str) -> bool:
    s = q.lower()
    if any(w in s for w in _TEMPORAL_WORDS):
        return True
    for m in _YEAR_RE.finditer(s):
        if int(s[m.start():m.end()]) >= 2020:
            return True
    return False


def preprocess(query: str) -> PreprocessResult:
    clean = clean_text(query)
    return PreprocessResult(
        raw=query,
        clean=clean,
        time_sensitive=is_time_sensitive(clean),
        category=classify_category(clean),
    )
