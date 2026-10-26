"""Pydantic models for API I/O (FR1, FR15, FR16)."""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


class Source(BaseModel):
    doc_id: str
    title: str
    url: str
    date: str
    relevance: float
    recency: float
    score: float


class Statement(BaseModel):
    text: str
    label: Literal["supported", "unsupported", "contradicted", "review"]
    support: float
    evidence_doc_id: Optional[str] = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    confidence: float                      # 0-100 (FR13)
    freshness_score: float                 # 0-1  (FR4)
    freshness_label: str                   # "fresh" | "stale" (FR16)
    retrieved: bool                        # FR5/FR6
    skipped_reason: Optional[str] = None
    statements: list[Statement] = []       # FR10/FR11/FR14
    sources: list[Source] = []             # FR15
    warnings: list[str] = []               # FR12/FR22
    query_category: str = "general"
    model_confidence: float = 0.0
    history_id: Optional[int] = None


class FeedbackRequest(BaseModel):
    history_id: int
    rating: Optional[int] = Field(None, ge=1, le=5)
    report_incorrect: bool = False
    comment: Optional[str] = Field(None, max_length=1000)


class ConfigUpdate(BaseModel):
    freshness_threshold: Optional[float] = None
    confidence_threshold: Optional[float] = None
    top_k: Optional[int] = None
    web_search_enabled: Optional[bool] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None


class KBDocument(BaseModel):
    doc_id: Optional[str] = None
    title: str
    url: str = ""
    date: str = "2025-01-01"
    text: str
