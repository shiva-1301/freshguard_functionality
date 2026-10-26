# FreshGuard — Setup Guide

Freshness-aware adaptive RAG with statement-level hallucination detection.
Fully **offline and deterministic** by default — no API key or internet needed to run it.

---

## 1. Prerequisites

- **Python 3.11 or newer** — check with `python --version`
- **pip** (ships with Python)
- ~500 MB free disk for the virtual environment and dependencies
- No GPU required; runs on CPU.

> Windows, macOS and Linux are all supported. Commands below show **Windows (PowerShell)** first
> and the **macOS / Linux (bash)** equivalent underneath.

---

## 2. Get the code

Unzip `freshguard.zip` anywhere, then open a terminal in the extracted
`freshguard_functionality` folder (the one containing `run.py`).

```powershell
cd path\to\freshguard_functionality
```

---

## 3. Create a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
If PowerShell blocks activation, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**macOS / Linux (bash):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` at the start of your prompt.

---

## 4. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs FastAPI, Uvicorn, scikit-learn, FAISS, LangChain-core/community and the
test tools. First install takes a couple of minutes.

---

## 5. Train the freshness estimator (one-off)

FreshGuard learns *when* to retrieve from a small labelled query set. Build the model once:

```bash
python run.py --train
```

This writes the trained estimator into `artifacts/`. You only need to repeat it if you
change the training data.

---

## 6. Run the app

```bash
python run.py
```

Then open **http://127.0.0.1:8000** in your browser.

- **Ask** tab — type a question (try the example chips) and get a grounded answer with a
  confidence score, freshness badge, per-statement verification and cited sources.
- **History** tab — recent queries.
- **Admin** tab — change thresholds, add knowledge-base documents, pick the LLM provider,
  and run the evaluation.

Stop the server with **Ctrl+C**.

---

## 7. Run the tests (optional but recommended)

```bash
python -m pytest -q        # 42 unit / end-to-end / API / evaluation tests
python eval/run_eval.py     # writes eval/metrics.json (benchmark numbers)
```

The suite is deterministic and verified to pass across repeated runs.

---

## 8. Using a real LLM provider (optional)

By default FreshGuard uses its **offline grounded generator** — no key needed. To route
generation through a hosted model instead, set an environment variable **before** starting
the server, then pick the provider in the **Admin → LLM provider** dropdown.

| Provider | Dropdown value | Environment variable |
|---|---|---|
| Offline (default) | `grounded` | — none — |
| OpenAI | `openai` | `OPENAI_API_KEY` |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` |
| xAI **Grok** | `grok` | `XAI_API_KEY` (or `GROK_API_KEY`) |

**Windows (PowerShell):**
```powershell
$env:XAI_API_KEY = "your-key-here"
python run.py
```

**macOS / Linux (bash):**
```bash
export XAI_API_KEY="your-key-here"
python run.py
```

Grok uses xAI's OpenAI-compatible endpoint (`https://api.x.ai/v1`). If no key is present,
the system automatically falls back to the offline grounded generator, so nothing breaks.

> Never commit your API key. Keep it in your shell/session only.

---

## 9. Project layout

```
freshguard_functionality/
├─ run.py                 # launcher: --train, or serve API + UI
├─ requirements.txt       # Python dependencies
├─ freshguard/            # core package
│  ├─ preprocess.py       #   query cleaning + classification
│  ├─ freshness.py        #   learned freshness / staleness estimator
│  ├─ decision.py         #   retrieve-or-skip gate
│  ├─ retrieval.py        #   FAISS index, relevance × recency ranking
│  ├─ llm.py              #   pluggable LLM: grounded / openai / anthropic / grok
│  ├─ verify.py           #   per-statement support labelling
│  ├─ confidence.py       #   0–100 calibrated confidence
│  ├─ pipeline.py         #   ties the stages together
│  ├─ api/                #   FastAPI app
│  └─ data/               #   seed corpus + estimator training
├─ frontend/              # single-page web UI (served at /)
├─ tests/                 # pytest suite
├─ eval/                  # evaluation runner + metrics.json
├─ docs/                  # project report (.docx / .pdf) + plan
└─ paper/                 # IEEE paper (.docx / .pdf) + LaTeX source
```

---

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `python` not found | Use `py` on Windows, or install Python 3.11+ and re-open the terminal. |
| Activation blocked in PowerShell | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then re-activate. |
| `faiss` install fails | Ensure `pip` is upgraded; the package is `faiss-cpu` (already in requirements). |
| Port 8000 busy | Edit the port in `run.py` (`uvicorn.run(..., port=8000)`). |
| Answers seem offline-only | That's expected unless you set an API key (see §8). |

---

> Research prototype — not for production decisions. Verify important answers independently.
