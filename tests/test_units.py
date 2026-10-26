"""Unit tests for individual components (FR2-FR13)."""
import pytest

from freshguard.preprocess import preprocess
from freshguard.freshness import freshness_score, model_self_confidence, freshness_label
from freshguard.decision import decide
from freshguard.retrieval import VectorStore
from freshguard.data.corpus import get_seed_docs
from freshguard.llm import GroundedExtractiveLLM, GrokLLM, make_llm
from freshguard.verify import verify_statements, corrective_action
from freshguard.confidence import compute_confidence


# -- FR2 -----------------------------------------------------------------------
def test_preprocess_cleans_and_classifies():
    r = preprocess("  Who   won the 2022 World\tCup?  ")
    assert r.clean == "Who won the 2022 World Cup?"
    assert r.time_sensitive is True
    assert r.category == "sports"


def test_preprocess_static_query():
    r = preprocess("what is the speed of light")
    assert r.time_sensitive is False
    assert r.category == "science"


# -- FR3 / FR4 -----------------------------------------------------------------
@pytest.mark.parametrize("q", [
    "what is the latest smartphone chipset in 2025",
    "who won the world cup this year",
    "current interest rate right now",
])
def test_freshness_high_for_timesensitive(q):
    s = freshness_score(q)
    assert 0.0 <= s <= 1.0
    assert s > 0.5, f"expected stale for {q!r}, got {s}"


@pytest.mark.parametrize("q", [
    "what is the speed of light",
    "define photosynthesis",
    "what is the capital of Australia",
])
def test_freshness_low_for_static(q):
    s = freshness_score(q)
    assert s < 0.5, f"expected fresh for {q!r}, got {s}"


def test_model_confidence_monotonic():
    assert model_self_confidence("x", 0.1) > model_self_confidence("x", 0.9)
    assert freshness_label(0.8, 0.55) == "stale"
    assert freshness_label(0.2, 0.55) == "fresh"


# -- FR5 / FR6 -----------------------------------------------------------------
def test_decision_retrieve_when_stale():
    d = decide(0.9, 0.2, 0.55, 0.6)
    assert d.retrieve is True


def test_decision_skip_when_fresh_and_confident():
    d = decide(0.1, 0.95, 0.55, 0.6)
    assert d.retrieve is False
    assert "fresh" in d.reason


# -- FR7 / FR8 / FR20 ----------------------------------------------------------
def test_vectorstore_retrieves_relevant_doc():
    vs = VectorStore(get_seed_docs())
    hits = vs.search("who won the 2022 world cup", top_k=3, min_relevance=0.0)
    assert hits
    assert hits[0].doc_id == "d001"
    assert 0.0 <= hits[0].relevance <= 1.0


def test_vectorstore_recency_affects_ranking():
    vs = VectorStore(get_seed_docs())
    hits = vs.search("latest chipset 2025", top_k=4, min_relevance=0.0)
    assert any(h.doc_id == "d003" for h in hits)


def test_vectorstore_add_and_remove():
    vs = VectorStore(get_seed_docs())
    n0 = len(vs.all_docs())
    did = vs.upsert({"title": "Mars colony 2025", "url": "", "date": "2025-07-01",
                     "text": "A new Mars research outpost was announced in 2025."})
    assert len(vs.all_docs()) == n0 + 1
    hits = vs.search("mars research outpost 2025", top_k=3, min_relevance=0.0)
    assert any(h.doc_id == did for h in hits)
    assert vs.remove(did) is True
    assert len(vs.all_docs()) == n0


# -- FR9 -----------------------------------------------------------------------
def test_grounded_llm_uses_evidence():
    llm = GroundedExtractiveLLM(get_seed_docs())
    ev = [{"doc_id": "d001", "text": "Argentina won the 2022 FIFA World Cup, defeating France."}]
    out = llm.generate("who won the 2022 world cup", ev)
    assert "argentina" in out.lower()


# -- FR10 / FR11 / FR12 --------------------------------------------------------
def test_verify_supported_and_unsupported():
    ev = [{"doc_id": "d001", "text": "Argentina won the 2022 FIFA World Cup in Qatar."}]
    stmts = verify_statements(
        "Argentina won the 2022 FIFA World Cup. The match was played on the moon.",
        ev, support_hi=0.32, support_lo=0.15)
    labels = {s.text[:10]: s.label for s in stmts}
    assert stmts[0].label == "supported"
    assert stmts[1].label in ("unsupported", "review")


def test_verify_detects_contradiction():
    ev = [{"doc_id": "d001", "text": "Argentina won the 2022 FIFA World Cup, defeating France."}]
    stmts = verify_statements("France won the 2022 FIFA World Cup.", ev, 0.32, 0.15)
    assert stmts[0].label == "contradicted"


def test_corrective_action_drops_unsupported():
    ev = [{"doc_id": "d001", "text": "Argentina won the 2022 FIFA World Cup."}]
    stmts = verify_statements(
        "Argentina won the 2022 FIFA World Cup. Aliens attended the final.", ev, 0.32, 0.15)
    corrected, warnings = corrective_action(stmts)
    assert "argentina" in corrected.lower()
    assert "aliens" not in corrected.lower()
    assert warnings


# -- FR13 ----------------------------------------------------------------------
def test_confidence_in_range_and_rewards_support():
    ev = [{"doc_id": "d001", "text": "Argentina won the 2022 FIFA World Cup."}]
    good = verify_statements("Argentina won the 2022 FIFA World Cup.", ev, 0.32, 0.15)
    bad = verify_statements("The moon is made of cheese entirely.", ev, 0.32, 0.15)
    c_good = compute_confidence(good, 0.8, 0.9, True)
    c_bad = compute_confidence(bad, 0.1, 0.9, True)
    assert 0 <= c_bad <= 100 and 0 <= c_good <= 100
    assert c_good > c_bad


# -- FR9 (pluggable providers, incl. Grok) -------------------------------------
def test_make_llm_selects_grok_and_falls_back_without_key(monkeypatch):
    docs = [{"doc_id": "d001", "date": "2019-01-01",
             "text": "Paris is the capital of France."}]
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    llm = make_llm("grok", "grok-2-latest", docs)
    assert isinstance(llm, GrokLLM) and llm.name == "grok"
    assert llm._available() is False
    # with no key it must fall back to the grounded generator (no network call)
    out = llm.generate("what is the capital of France", [])
    assert "paris" in out.lower()
    # the key presence is detected from either env var
    monkeypatch.setenv("GROK_API_KEY", "x")
    assert llm._available() is True
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    monkeypatch.setenv("XAI_API_KEY", "x")
    assert llm._available() is True
