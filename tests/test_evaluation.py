"""Evaluation module tests (FR21) — also assert the system meets quality bars."""
from freshguard import evaluation


def test_freshness_accuracy(fg):
    r = evaluation.evaluate_freshness(fg)
    assert r["freshness_accuracy"] >= 0.9, r


def test_factuality(fg):
    r = evaluation.evaluate_factuality(fg)
    assert r["factuality"] >= 0.75, r


def test_hallucination_detection(fg):
    s = fg.storage.get_settings()
    r = evaluation.evaluate_hallucination(s.support_hi, s.support_lo)
    assert r["recall"] >= 0.9, r      # catch (almost) all hallucinations
    assert r["f1"] >= 0.8, r


def test_calibration_reported(fg):
    r = evaluation.evaluate_calibration(fg)
    assert 0.0 <= r["ece"] <= 1.0


def test_run_all_shape(fg):
    r = evaluation.run_all(fg)
    assert set(r) == {"freshness", "factuality", "hallucination", "calibration"}
