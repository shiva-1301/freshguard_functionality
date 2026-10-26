"""API tests via FastAPI TestClient (FR1, FR16-FR21)."""
import pytest
from fastapi.testclient import TestClient

from freshguard.api.main import create_app
from freshguard.pipeline import FreshGuard
from freshguard.storage import Storage


@pytest.fixture()
def client(tmp_path):
    fg = FreshGuard(storage=Storage(tmp_path / "api.db"))
    return TestClient(create_app(fg))


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_query_endpoint(client):
    r = client.post("/api/query", json={"query": "who won the 2022 world cup"})
    assert r.status_code == 200
    d = r.json()
    assert d["retrieved"] is True
    assert d["sources"]
    assert "argentina" in d["answer"].lower()
    assert d["history_id"] is not None


def test_query_validation(client):
    r = client.post("/api/query", json={"query": ""})
    assert r.status_code == 422   # min_length violated


def test_feedback_flow(client):
    q = client.post("/api/query", json={"query": "what is the speed of light"}).json()
    r = client.post("/api/feedback", json={"history_id": q["history_id"], "rating": 4})
    assert r.status_code == 200
    assert r.json()["summary"]["total"] == 1
    bad = client.post("/api/feedback", json={"history_id": 999999, "rating": 4})
    assert bad.status_code == 404


def test_history_endpoint(client):
    client.post("/api/query", json={"query": "define photosynthesis"})
    r = client.get("/api/history")
    assert r.status_code == 200
    assert r.json()["items"]


def test_config_endpoint(client):
    r = client.post("/api/config", json={"top_k": 6})
    assert r.status_code == 200
    assert r.json()["top_k"] == 6
    assert client.get("/api/config").json()["top_k"] == 6


def test_kb_endpoints(client):
    add = client.post("/api/kb", json={"title": "Doc", "text": "Some new 2025 fact about fusion power.",
                                       "date": "2025-05-05", "url": "http://x"})
    assert add.status_code == 200
    did = add.json()["doc_id"]
    assert any(d["doc_id"] == did for d in client.get("/api/kb").json()["items"])
    rm = client.delete(f"/api/kb/{did}")
    assert rm.status_code == 200


def test_eval_endpoint(client):
    r = client.get("/api/eval")
    assert r.status_code == 200
    d = r.json()
    assert "freshness" in d and "hallucination" in d
    assert 0.0 <= d["freshness"]["freshness_accuracy"] <= 1.0
