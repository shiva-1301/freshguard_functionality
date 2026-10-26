"""Adaptive retrieval decision (FR5) and skip logic (FR6)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Decision:
    retrieve: bool
    reason: str


def decide(freshness: float, model_conf: float,
           freshness_threshold: float, confidence_threshold: float) -> Decision:
    """Retrieve when knowledge looks stale OR the model is not confident enough to answer
    directly; otherwise skip retrieval to save latency and cost (FR6)."""
    if freshness >= freshness_threshold:
        return Decision(True, f"freshness {freshness:.2f} >= threshold {freshness_threshold:.2f}")
    if model_conf < confidence_threshold:
        return Decision(True, f"model confidence {model_conf:.2f} < threshold {confidence_threshold:.2f}")
    return Decision(False, f"knowledge estimated fresh (freshness {freshness:.2f}, "
                           f"confidence {model_conf:.2f}); answered without retrieval")
