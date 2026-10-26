# FreshGuard

**Freshness-aware adaptive RAG with statement-level hallucination detection.**

FreshGuard decides *when* an LLM's own knowledge is likely stale, retrieves fresh
evidence only when needed, generates a grounded answer, then verifies every statement
against the evidence — returning a calibrated confidence score, a freshness indicator and
cited sources. Built on **FastAPI + LangChain + FAISS + scikit-learn**, fully offline and
deterministic by default.

See [`docs/plan.md`](docs/plan.md) for the design and the full
[functional-requirements](docs/FreshGuard_Functional_Requirements.pdf) → component mapping
(FR1–FR22).

## Quick start

```bash
# from the project root, using a Python 3.11+ environment
pip install -r requirements.txt

python run.py --train      # train the learned freshness estimator (one-off)
python run.py              # serve the API + web UI at http://127.0.0.1:8000
```

Open <http://127.0.0.1:8000>, ask a question (try the example chips), and switch to the
**Admin** tab to change thresholds, manage the knowledge base, and run the evaluation.

## Testing

```bash
python -m pytest -q        # 41 unit / end-to-end / API / evaluation tests
python eval/run_eval.py    # writes eval/metrics.json
```

The suite is deterministic and was verified to pass across 12 consecutive runs.

## Results (controlled benchmark)

| Metric | Value |
|---|---|
| Retrieval-decision (freshness) accuracy | 1.00 |
| Factuality (gold fact present) | 1.00 |
| Hallucination precision / recall / F1 | 1.00 / 1.00 / 1.00 |
| Expected calibration error (ECE) | 0.10 |

## Layout

```
freshguard/        core package (preprocess, freshness, decision, retrieval, llm,
                   verify, confidence, pipeline, storage, evaluation, api, data)
frontend/          single-page web UI (query, confidence, verification, citations, admin)
tests/             pytest suite
eval/              evaluation runner + metrics.json
paper/             IEEE two-column paper (freshguard.tex, freshguard.pdf, figures/)
docs/              plan.md + functional requirements
```

## Design notes

- **Learned freshness estimator** (not a fixed time-decay rule): LogisticRegression over
  TF-IDF word+char n-grams plus explicit temporal features.
- **Adaptive retrieval**: retrieve iff `freshness ≥ τ` or `model_confidence < τ_c`.
- **Retrieval**: LangChain FAISS index with offline TF-IDF embeddings; relevance × recency
  ranking. A web-search source and a transformer encoder are pluggable seams.
- **Pluggable LLM**: offline grounded generator by default; OpenAI, Anthropic and **xAI Grok**
  adapters used only if an API key is present (`XAI_API_KEY`/`GROK_API_KEY` for Grok). Grok uses
  its OpenAI-compatible endpoint; without a key the system falls back to the grounded generator.
- **Verification**: per-statement support labelling (supported / unsupported /
  contradicted), with entity-swap and numeric-mismatch contradiction detection.

> Research prototype — not for production decisions. Verify important answers independently.
