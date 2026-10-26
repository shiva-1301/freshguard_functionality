"""Evidence retrieval and ranking (FR7, FR8) built on LangChain + FAISS, plus an optional
web-search adapter and knowledge-base management (FR20).

The vector store is a LangChain FAISS index over TF-IDF embeddings (see embeddings.py).
Final ranking combines semantic relevance with document recency so that, for equally
relevant documents, fresher ones rank higher (FR8).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document

from freshguard.config import REFERENCE_YEAR, DEFAULTS
from freshguard.embeddings import TfidfEmbeddings


@dataclass
class RetrievedDoc:
    doc_id: str
    title: str
    url: str
    date: str
    text: str
    relevance: float   # cosine similarity in [0, 1]
    recency: float     # recency weight in [0, 1]
    score: float       # combined ranking score

    def as_source(self) -> dict:
        d = asdict(self)
        d.pop("text")
        return d


def _parse_year(d: str) -> int:
    try:
        return int(d[:4])
    except Exception:
        return REFERENCE_YEAR


def _recency_weight(doc_date: str, half_life_years: float, now_year: int) -> float:
    age = max(0, now_year - _parse_year(doc_date))
    return float(0.5 ** (age / max(0.1, half_life_years)))


class VectorStore:
    """Thin wrapper around a LangChain FAISS index with recency-aware ranking and
    add/update/remove operations (FR20). Rebuilds the index on mutation; the corpus is
    small, so this stays simple and deterministic."""

    def __init__(self, docs: list[dict], half_life_years: float | None = None):
        self._docs: dict[str, dict] = {d["doc_id"]: dict(d) for d in docs}
        self.half_life = half_life_years if half_life_years is not None else DEFAULTS.recency_half_life_years
        self._build()

    # -- index lifecycle -------------------------------------------------------
    def _build(self) -> None:
        texts = [d["text"] for d in self._docs.values()] or ["placeholder"]
        self._emb = TfidfEmbeddings([f"{d['title']}. {d['text']}" for d in self._docs.values()] or texts)
        lc_docs = [
            Document(
                page_content=f"{d['title']}. {d['text']}",
                metadata={"doc_id": d["doc_id"], "title": d["title"], "url": d["url"], "date": d["date"]},
            )
            for d in self._docs.values()
        ]
        if lc_docs:
            self._store = FAISS.from_documents(
                lc_docs, self._emb, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
            )
        else:
            self._store = None

    # -- retrieval (FR7/FR8) ---------------------------------------------------
    def search(self, query: str, top_k: int, min_relevance: float,
               now_year: int = REFERENCE_YEAR) -> list[RetrievedDoc]:
        if self._store is None or not self._docs:
            return []
        k = min(max(top_k * 2, top_k), len(self._docs))
        hits = self._store.similarity_search_with_score_by_vector(
            self._emb.embed_query(query), k=k
        )
        out: list[RetrievedDoc] = []
        for doc, score in hits:
            # with unit vectors + MAX_INNER_PRODUCT, score ~ cosine similarity
            relevance = max(0.0, min(1.0, float(score)))
            if relevance < min_relevance:
                continue
            meta = doc.metadata
            rec = _recency_weight(meta["date"], self.half_life, now_year)
            combined = relevance * (0.6 + 0.4 * rec)
            out.append(RetrievedDoc(
                doc_id=meta["doc_id"], title=meta["title"], url=meta["url"], date=meta["date"],
                text=self._docs[meta["doc_id"]]["text"],
                relevance=round(relevance, 4), recency=round(rec, 4), score=round(combined, 6),
            ))
        # rank by combined score, then recency, then doc_id for deterministic ties (FR8)
        out.sort(key=lambda r: (-r.score, -r.recency, r.doc_id))
        return out[:top_k]

    # -- knowledge-base management (FR20) --------------------------------------
    def upsert(self, doc: dict) -> str:
        doc = dict(doc)
        if not doc.get("doc_id"):
            n = 1
            while f"u{n:03d}" in self._docs:
                n += 1
            doc["doc_id"] = f"u{n:03d}"
        doc.setdefault("url", "")
        doc.setdefault("date", date.today().isoformat())
        self._docs[doc["doc_id"]] = doc
        self._build()
        return doc["doc_id"]

    def remove(self, doc_id: str) -> bool:
        if doc_id in self._docs:
            del self._docs[doc_id]
            self._build()
            return True
        return False

    def all_docs(self) -> list[dict]:
        return [dict(d) for d in self._docs.values()]


def web_search_adapter(query: str) -> list[dict]:
    """Optional live web-search source (FR7). Offline by default: returns nothing unless a
    real backend is wired in. Kept as a seam so production can plug in a search API without
    touching the pipeline."""
    return []
