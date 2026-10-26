"""End-to-end orchestration (FR1 input handling through FR16 display), with evidence
highlighting (FR14), citations (FR15) and graceful error handling (FR22)."""
from __future__ import annotations

from freshguard import freshness as fresh_mod
from freshguard.config import REFERENCE_YEAR
from freshguard.confidence import compute_confidence
from freshguard.data.corpus import get_seed_docs
from freshguard.decision import decide
from freshguard.llm import make_llm
from freshguard.preprocess import preprocess
from freshguard.retrieval import VectorStore, web_search_adapter
from freshguard.storage import Storage
from freshguard.verify import verify_statements, corrective_action


class FreshGuard:
    """Holds the vector store, LLM and storage; exposes `answer(query)`."""

    def __init__(self, storage: Storage | None = None, extra_docs: list[dict] | None = None):
        self.storage = storage or Storage()
        docs = get_seed_docs() + (extra_docs or []) + self.storage.kb_all()
        # de-dup by doc_id (DB/extra override seed)
        merged: dict[str, dict] = {}
        for d in docs:
            merged[d["doc_id"]] = d
        self._all_docs = list(merged.values())
        settings = self.storage.get_settings()
        self.store = VectorStore(self._all_docs, half_life_years=settings.recency_half_life_years)
        self.llm = make_llm(settings.llm_provider, settings.llm_model, self._all_docs)

    # -- KB management reflects into the live index + DB (FR20) ----------------
    def kb_add(self, doc: dict) -> str:
        doc_id = self.store.upsert(doc)
        stored = next(d for d in self.store.all_docs() if d["doc_id"] == doc_id)
        self.storage.kb_upsert(stored)
        self._refresh_llm()
        return doc_id

    def kb_remove(self, doc_id: str) -> bool:
        ok = self.store.remove(doc_id)
        if ok:
            self.storage.kb_delete(doc_id)
            self._refresh_llm()
        return ok

    def kb_list(self) -> list[dict]:
        return self.store.all_docs()

    def _refresh_llm(self) -> None:
        s = self.storage.get_settings()
        self.llm = make_llm(s.llm_provider, s.llm_model, self.store.all_docs())

    # -- main flow -------------------------------------------------------------
    def answer(self, query: str, persist: bool = True, now_year: int = REFERENCE_YEAR) -> dict:
        try:
            return self._answer(query, persist, now_year)
        except Exception as exc:  # FR22 graceful fallback
            resp = _fallback_response(query, f"internal error: {type(exc).__name__}")
            if persist:
                try:
                    resp["history_id"] = self.storage.add_history(resp)
                except Exception:
                    pass
            return resp

    def _answer(self, query: str, persist: bool, now_year: int) -> dict:
        settings = self.storage.get_settings()
        pre = preprocess(query)
        if not pre.clean:
            resp = _fallback_response(query, "empty query after preprocessing")
            if persist:
                resp["history_id"] = self.storage.add_history(resp)
            return resp

        # FR3/FR4 freshness
        fscore = fresh_mod.freshness_score(pre.clean)
        model_conf = fresh_mod.model_self_confidence(pre.clean, fscore)
        flabel = fresh_mod.freshness_label(fscore, settings.freshness_threshold)

        # FR5/FR6 decision
        dec = decide(fscore, model_conf, settings.freshness_threshold, settings.confidence_threshold)

        # FR7/FR8 retrieval
        evidence: list[dict] = []
        sources: list[dict] = []
        if dec.retrieve:
            docs = self.store.search(pre.clean, settings.top_k, settings.min_relevance, now_year)
            if settings.web_search_enabled:
                for w in web_search_adapter(pre.clean):
                    docs.append(w)
            evidence = [{"doc_id": d.doc_id, "title": d.title, "url": d.url,
                         "date": d.date, "text": d.text} for d in docs]
            sources = [d.as_source() for d in docs]

        # FR9 generation
        raw_answer = self.llm.generate(pre.clean, evidence)

        # FR10/FR11 verification + FR12 correction
        if dec.retrieve and evidence:
            statements = verify_statements(raw_answer, evidence, settings.support_hi, settings.support_lo)
            corrected, warnings = corrective_action(statements)
            final_answer = corrected or raw_answer
        else:
            # no retrieval: statements are from parametric memory, not independently verified
            statements = []
            warnings = []
            final_answer = raw_answer

        if not final_answer:
            final_answer = ("I don't have enough reliable, up-to-date information to answer this "
                            "confidently.")
            warnings.append("No supported answer could be produced.")

        mean_rel = (sum(s["relevance"] for s in sources) / len(sources)) if sources else model_conf
        confidence = compute_confidence(
            statements if statements else [], mean_rel, fscore, dec.retrieve
        )
        # when retrieval was skipped, confidence is model-driven
        if not dec.retrieve:
            confidence = round(100.0 * (0.7 * model_conf + 0.3 * (1.0 - fscore)), 1)

        resp = {
            "query": query,
            "answer": final_answer,
            "confidence": confidence,
            "freshness_score": fscore,
            "freshness_label": flabel,
            "retrieved": dec.retrieve,
            "skipped_reason": None if dec.retrieve else dec.reason,
            "statements": [
                {"text": s.text, "label": s.label, "support": s.support,
                 "evidence_doc_id": s.evidence_doc_id} for s in statements
            ],
            "sources": sources,          # FR15 citations
            "warnings": warnings,        # FR12/FR22
            "query_category": pre.category,
            "model_confidence": model_conf,
            "history_id": None,
        }
        if persist:
            resp["history_id"] = self.storage.add_history(resp)  # FR17
        return resp


def _fallback_response(query: str, reason: str) -> dict:
    return {
        "query": query,
        "answer": "The system could not complete this request reliably. Please try rephrasing; "
                  "this is a low-confidence fallback response.",
        "confidence": 0.0,
        "freshness_score": 0.0,
        "freshness_label": "unknown",
        "retrieved": False,
        "skipped_reason": reason,
        "statements": [],
        "sources": [],
        "warnings": [f"Graceful fallback triggered ({reason})."],
        "query_category": "general",
        "model_confidence": 0.0,
        "history_id": None,
    }
