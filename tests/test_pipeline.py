"""End-to-end pipeline tests (FR1, FR5/FR6, FR14-FR17, FR22) + determinism."""
import copy


def test_fresh_query_skips_retrieval(fg):
    r = fg.answer("what is the speed of light")
    assert r["retrieved"] is False
    assert r["freshness_label"] == "fresh"
    assert r["skipped_reason"]
    assert "299" in r["answer"]            # answered from parametric memory
    assert r["history_id"] is not None     # FR17 persisted


def test_stale_query_retrieves_and_cites(fg):
    r = fg.answer("who won the 2022 world cup")
    assert r["retrieved"] is True
    assert r["freshness_label"] == "stale"
    assert r["sources"], "expected cited sources (FR15)"
    assert r["sources"][0]["doc_id"] == "d001"
    assert all({"title", "url", "date", "relevance"} <= set(s) for s in r["sources"])
    assert "argentina" in r["answer"].lower()
    assert r["statements"], "expected per-statement verification (FR10/FR14)"
    assert 0 <= r["confidence"] <= 100


def test_hallucination_detected_and_corrected(fg):
    # force the LLM to emit an unsupported + contradictory answer
    class FakeLLM:
        name = "fake"
        def generate(self, query, evidence):
            return ("Argentina won the 2022 FIFA World Cup. "
                    "The final was held on Mars in front of robots.")
    fg.llm = FakeLLM()
    r = fg.answer("who won the 2022 world cup")
    labels = [s["label"] for s in r["statements"]]
    assert "supported" in labels
    assert any(l in ("unsupported", "contradicted") for l in labels)
    assert r["warnings"], "corrective action should warn (FR12)"
    assert "mars" not in r["answer"].lower()   # unsupported statement removed


def test_empty_query_graceful(fg):
    r = fg.answer("    ")
    assert r["confidence"] == 0.0
    assert r["warnings"]


def test_error_handling_fallback(fg):
    class BoomLLM:
        name = "boom"
        def generate(self, *a, **k):
            raise RuntimeError("boom")
    fg.llm = BoomLLM()
    r = fg.answer("who won the 2022 world cup")   # retrieval path hits the LLM
    assert r["confidence"] == 0.0
    assert any("fallback" in w.lower() for w in r["warnings"])


def test_history_and_feedback(fg):
    r = fg.answer("what is the capital of Australia")
    hid = r["history_id"]
    assert fg.storage.history_exists(hid)
    fg.storage.add_feedback(hid, rating=5, report_incorrect=False, comment="great")
    summ = fg.storage.feedback_summary()
    assert summ["total"] == 1 and summ["avg_rating"] == 5.0
    items = fg.storage.get_history(limit=10)
    assert items and items[0]["query"] == "what is the capital of Australia"


def test_config_changes_behaviour(fg):
    # a static query is answered directly by default ...
    assert fg.answer("what is the speed of light")["retrieved"] is False
    # ... but raising the confidence threshold forces retrieval for the same query (FR5/FR19)
    fg.storage.set_config({"confidence_threshold": 0.999})
    assert fg.answer("what is the speed of light")["retrieved"] is True
    # and a stale query always retrieves regardless
    assert fg.answer("who won the 2024 world cup")["retrieved"] is True


def test_kb_management_round_trip(fg):
    did = fg.kb_add({"title": "New 2025 Fact", "url": "http://x", "date": "2025-08-01",
                     "text": "A brand new 2025 discovery about quantum batteries was reported."})
    r = fg.answer("what is the latest 2025 discovery about quantum batteries")
    assert any(s["doc_id"] == did for s in r["sources"])
    assert fg.kb_remove(did) is True


def test_determinism_same_output(fg):
    r1 = fg.answer("who won the 2022 world cup", persist=False)
    r2 = fg.answer("who won the 2022 world cup", persist=False)
    for key in ("answer", "confidence", "freshness_score", "retrieved"):
        assert r1[key] == r2[key], f"non-deterministic field: {key}"
    assert [s["doc_id"] for s in r1["sources"]] == [s["doc_id"] for s in r2["sources"]]
