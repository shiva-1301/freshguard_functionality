"""Central configuration (FR19). Values may be overridden at runtime by the admin
config stored in SQLite; see storage.get_config / set_config."""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "freshguard" / "data"
ARTIFACT_DIR = ROOT / "artifacts"
DB_PATH = ARTIFACT_DIR / "freshguard.db"
FRESHNESS_MODEL_PATH = ARTIFACT_DIR / "freshness_estimator.joblib"
FRONTEND_DIR = ROOT / "frontend"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# A fixed reference "now" keeps scoring deterministic for tests; the API overrides
# nothing by default, so behaviour is reproducible. Year chosen to match the corpus.
REFERENCE_YEAR = 2025


@dataclass
class Settings:
    # Decision thresholds (FR5/FR6) --------------------------------------------
    freshness_threshold: float = 0.55   # retrieve if freshness_score >= this
    confidence_threshold: float = 0.60  # or if model self-confidence < this
    # Retrieval / ranking (FR7/FR8) --------------------------------------------
    top_k: int = 4
    min_relevance: float = 0.05
    recency_half_life_years: float = 2.0
    web_search_enabled: bool = False    # offline by default
    # Verification (FR10/FR11) -------------------------------------------------
    support_hi: float = 0.32            # >= supported
    support_lo: float = 0.15            # < unsupported (between => weak/review)
    # Provider selection (FR9/FR19) --------------------------------------------
    llm_provider: str = "grounded"      # grounded | openai | anthropic | grok
    llm_model: str = "offline-grounded-v1"

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULTS = Settings()


def env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)
