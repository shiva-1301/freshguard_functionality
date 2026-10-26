# FreshGuard — Project Plan

**FreshGuard** is an *adaptive, freshness-aware Retrieval-Augmented Generation (RAG)* system.
It decides **when** an LLM's own knowledge is likely stale, retrieves fresh evidence **only when
needed**, generates a grounded answer, then **verifies every statement** against the evidence to
catch hallucinations — returning a calibrated confidence score, a freshness indicator and cited
sources.

This plan maps the 22 functional requirements in
[`FreshGuard_Functional_Requirements.pdf`](FreshGuard_Functional_Requirements.pdf) to a concrete,
testable architecture.

---

## 1. Problem & approach

Static RAG retrieves for *every* query, wasting latency/cost on questions the model already knows,
and vanilla LLMs answer confidently even when their knowledge is outdated. FreshGuard adds two
learned gates around generation:

1. **Freshness gate (before generation)** — a learned estimator predicts the probability that the
   model's parametric knowledge is outdated for this query, and a decision policy retrieves only
   when that probability (or low model confidence) crosses a threshold.
2. **Verification gate (after generation)** — the answer is split into atomic statements and each is
   checked against the retrieved evidence (supported / unsupported / contradicted); unsupported
   content is corrected or flagged, and a confidence score is computed from all three signals.

The whole pipeline is **deterministic and offline-capable**: retrieval uses a local TF-IDF vector
store, the LLM is a *pluggable provider* whose default is a grounded extractive generator (a real
OpenAI/Anthropic adapter can be dropped in), and web search is an optional adapter. This makes the
system fully runnable and reproducibly testable without any network or API key.

---

## 2. Architecture

```
          ┌──────────────────────── FastAPI backend ────────────────────────┐
  Web UI  │  /query  /feedback  /history  /admin/config  /admin/kb  /eval    │
 (static) │                                                                  │
   │      │   Pipeline.run(query):                                           │
   │ POST  │     preprocess ─▶ freshness ─▶ decision ─▶ [retrieve ─▶ rank]   │
   └──────▶│       ─▶ generate ─▶ verify ─▶ correct ─▶ confidence ─▶ cite    │
          │                              │                                    │
          │   LogisticRegression    TF-IDF vector store    SQLite (history,   │
          │   freshness estimator    + optional web        feedback, config)  │
          └──────────────────────────────────────────────────────────────────┘
```

**Package layout**

```
freshguard/
  config.py        thresholds, paths, provider selection          (FR19)
  schemas.py       pydantic request/response models
  preprocess.py    clean, normalize, classify query               (FR1 input, FR2)
  freshness.py     learned freshness estimator + score            (FR3, FR4)
  decision.py      retrieve-or-skip policy                        (FR5, FR6)
  retrieval.py     TF-IDF vector store + web adapter + ranking    (FR7, FR8)
  llm.py           pluggable LLM (offline grounded default)       (FR9)
  verify.py        statement split, support labels, correction    (FR10, FR11, FR12)
  confidence.py    confidence score from 3 signals                (FR13)
  pipeline.py      orchestrates the full flow + error handling    (FR14, FR16, FR22)
  storage.py       SQLite history / feedback / config / KB        (FR17, FR18, FR20)
  evaluation.py    factuality, freshness acc, halluc. P/R/F1, ECE (FR21)
  data/
    corpus.py          seed documents for the vector store
    make_freshness.py  synthesise labelled data + train estimator
  api/main.py      FastAPI app, all endpoints, serves frontend
frontend/          index.html + app.js + style.css (query box, answer,
                   confidence meter, freshness badge, highlighted evidence,
                   citations, history, feedback)                  (FR14, FR15, FR16, FR18)
tests/             pytest suite (unit + end-to-end, deterministic)
eval/run_eval.py   runs the evaluation module, writes metrics JSON
paper/             IEEE two-column LaTeX paper + refs.bib
```

---

## 3. Requirement → component mapping

| FR | Requirement | Component |
|----|-------------|-----------|
| FR1  | NL query input via web UI | `frontend/`, `api /query` |
| FR2  | Clean / normalize / classify query | `preprocess.py` |
| FR3  | Estimate knowledge-staleness likelihood | `freshness.py` |
| FR4  | Numeric freshness score (learned, not fixed decay) | `freshness.py` (LogisticRegression) |
| FR5  | Adaptive retrieval decision | `decision.py` |
| FR6  | Skip retrieval when fresh | `decision.py` / `pipeline.py` |
| FR7  | Evidence retrieval (vector DB / web) | `retrieval.py` |
| FR8  | Rank + filter by relevance & recency | `retrieval.py` |
| FR9  | Grounded answer generation | `llm.py` |
| FR10 | Statement-level verification | `verify.py` |
| FR11 | Hallucination detection (supported/unsupported/contradicted) | `verify.py` |
| FR12 | Corrective action / warn | `verify.py`, `pipeline.py` |
| FR13 | Confidence score (model + evidence + freshness) | `confidence.py` |
| FR14 | Evidence highlighting per claim | `pipeline.py` output + `frontend` |
| FR15 | Source citation (link, title, date) | `retrieval.py` metadata + `frontend` |
| FR16 | Final answer + confidence + freshness + links | `pipeline.py`, `frontend` |
| FR17 | Query history storage | `storage.py` |
| FR18 | User feedback (rate / report) | `storage.py`, `api /feedback` |
| FR19 | Admin config (thresholds, sources, LLM) | `config.py`, `storage.py`, `api /admin/config` |
| FR20 | Knowledge-base management | `storage.py`, `retrieval.py`, `api /admin/kb` |
| FR21 | Performance evaluation module | `evaluation.py`, `eval/run_eval.py` |
| FR22 | Error handling + graceful fallback | `pipeline.py` (try/except, low-confidence warning) |

---

## 4. Key design decisions

- **Learned freshness estimator (FR4).** `LogisticRegression` over TF-IDF word+char n-grams of the
  query plus hand-crafted features (temporal keywords, explicit years, "latest/current/now" cues,
  category). Trained on a synthesised, labelled dataset (`data/make_freshness.py`). Outputs a
  calibrated probability — a *learned* estimator, explicitly not a fixed time-decay rule.
- **Retrieval decision (FR5/FR6).** Retrieve if `freshness_score ≥ τ_fresh` **or**
  `model_confidence < τ_conf`; otherwise answer directly (skip) to save latency/cost. Thresholds are
  admin-configurable.
- **Vector store (FR7/FR8).** TF-IDF + cosine over a seeded corpus; each doc carries a `date`; final
  ranking score = `relevance × recency_decay`, filtered by a minimum relevance floor.
- **Pluggable LLM (FR9).** `llm.py` exposes `generate(query, evidence) -> text`. Default
  `GroundedExtractiveLLM` composes the answer from top evidence sentences (deterministic, offline).
  `OpenAILLM` / `AnthropicLLM` adapters use env keys when present. Selected via config (FR19).
- **Verification (FR10/FR11).** Split answer into sentences; for each, compute max cosine similarity
  to evidence sentences → `supported` (≥ hi), `unsupported` (< lo); `contradicted` when a negation
  mismatch is detected against an otherwise-similar evidence sentence.
- **Correction (FR12).** Unsupported statements are dropped from the grounded answer and the answer
  is recomposed from supported evidence; if nothing is supported, return a low-confidence warning.
- **Confidence (FR13).** `0.5·support_ratio + 0.3·mean_evidence_relevance + 0.2·freshness_alignment`,
  scaled to 0–100 %.
- **Determinism for testing.** Fixed random seeds, sorted tie-breaks, no wall-clock in scoring
  (corpus dates are fixed, "now" is injectable) → identical outputs across ≥10 runs.

---

## 5. Build phases

- **P0 — Scaffold:** package, `config`, `schemas`, `requirements.txt`, seed corpus.
- **P1 — Freshness:** synthesise data, train + persist estimator, `freshness.py` (FR3/FR4).
- **P2 — Retrieval:** TF-IDF store, ranking, KB mgmt, web adapter stub (FR7/FR8/FR20).
- **P3 — Generate + verify:** `llm.py`, `verify.py`, `confidence.py` (FR9–FR13).
- **P4 — Pipeline + storage:** orchestration, highlighting, citations, SQLite, error handling
  (FR14–FR18, FR22).
- **P5 — API + frontend:** FastAPI endpoints + static web UI (FR1, FR16, FR19).
- **P6 — Evaluation:** metrics module + runner (FR21).
- **P7 — Tests:** unit + end-to-end pytest; run ≥10× to prove stability.
- **P8 — Paper:** IEEE two-column LaTeX, compiled and cross-checked.

## 6. Acceptance criteria

- `pytest` green; end-to-end `/query` returns answer + confidence + freshness + cited evidence.
- Fresh (static) queries skip retrieval; stale/time-sensitive queries retrieve (FR5/FR6 proven in
  tests).
- Injected hallucination is detected and corrected/flagged (FR10–FR12 proven in tests).
- Evaluation module reports factuality, freshness accuracy, hallucination F1 and calibration (ECE).
- Full test suite passes **10+ consecutive times** with identical results (determinism).
- IEEE paper compiles to PDF, two-column, with abstract, method, results and references.

> Research prototype — not production advice; answers may be imperfect and should be verified.
