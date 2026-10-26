"""Knowledge-freshness estimation (FR3) and numeric freshness score (FR4).

Loads the learned LogisticRegression pipeline and returns P(knowledge is outdated) for a
query. If the model file is missing it is trained on first use. Also exposes a model
*self-confidence* estimate used by the retrieval decision (FR5).
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import joblib

from freshguard.config import FRESHNESS_MODEL_PATH
from freshguard.data.make_freshness import train_and_save


@lru_cache(maxsize=1)
def _load_model(path_str: str):
    path = Path(path_str)
    if not path.exists():
        train_and_save(path)
    return joblib.load(path)


def freshness_score(query: str) -> float:
    """Return a learned probability in [0, 1] that the LLM's knowledge is stale."""
    model = _load_model(str(FRESHNESS_MODEL_PATH))
    prob = float(model.predict_proba([query])[0][1])
    return round(prob, 4)


def freshness_label(score: float, threshold: float) -> str:
    return "stale" if score >= threshold else "fresh"


def model_self_confidence(query: str, fresh: float) -> float:
    """Heuristic estimate of how confident a parametric LLM would be answering *without*
    retrieval. Static-looking queries (low freshness) -> high confidence; time-sensitive
    queries -> low confidence. Monotonic, deterministic, in [0, 1]."""
    # logistic shaping of (1 - freshness), centred at 0.5
    conf = 1.0 / (1.0 + math.exp(-6.0 * (0.5 - fresh)))
    return round(conf, 4)
