"""Generate all figures for the FreshGuard IEEE paper from the real system."""
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

from freshguard.pipeline import FreshGuard          # noqa: E402
from freshguard.storage import Storage              # noqa: E402
from freshguard import evaluation                    # noqa: E402
from freshguard.freshness import freshness_score     # noqa: E402

NAVY = "#1f3a5f"; TEAL = "#2dd4bf"; ORANGE = "#e8792b"; GREY = "#64748b"
LBLUE = "#e8eef7"; LTEAL = "#dcf5f0"; LOR = "#fbe6d4"; GOOD = "#16a34a"; BAD = "#dc2626"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def _box(ax, x, y, w, h, text, fc, ec, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=1.4, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color="#10202f")


def _arrow(ax, p1, p2, color=GREY, style="-|>", lw=1.5, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=13,
                                 lw=lw, color=color, ls=ls, shrinkA=1, shrinkB=1))


# ---------------------------------------------------------------- architecture
def architecture():
    fig, ax = plt.subplots(figsize=(7.2, 4.4)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    _box(ax, 33, 88, 34, 9, "Web UI (React-style SPA)\nquery · answer · confidence · citations", LBLUE, NAVY, 8.5)
    _arrow(ax, (50, 88), (50, 82)); _arrow(ax, (50, 76), (50, 82), color=GREY)
    _box(ax, 30, 74, 40, 8, "FastAPI service  /query /feedback /kb /config /eval", "#fff", GREY, 8.5)
    # pipeline stages
    stages = ["Preprocess\n& classify", "Freshness\nestimator", "Retrieve-or-\nskip policy",
              "FAISS retrieval\n+ recency rank", "Grounded\ngeneration", "Statement\nverification",
              "Confidence\n& citations"]
    cols = [(2 + i * 14, 56) for i in range(7)]
    fcs = [LTEAL, LOR, LBLUE, LTEAL, LOR, LBLUE, LTEAL]
    ecs = [TEAL, ORANGE, NAVY, TEAL, ORANGE, NAVY, TEAL]
    for (x, y), s, fc, ec in zip(cols, stages, fcs, ecs):
        _box(ax, x, y, 12.5, 14, s, fc, ec, 7.2)
    for i in range(6):
        _arrow(ax, (cols[i][0] + 12.5, 63), (cols[i + 1][0], 63))
    _arrow(ax, (50, 74), (8.3, 70.2), color=GREY)
    # supports
    _box(ax, 6, 30, 24, 10, "Learned freshness\nestimator (LogReg\n+ temporal feats)", "#fff", ORANGE, 7.8)
    _box(ax, 38, 30, 24, 10, "FAISS vector store\n(LangChain, TF-IDF\nembeddings, dated)", "#fff", TEAL, 7.8)
    _box(ax, 70, 30, 24, 10, "SQLite\nhistory · feedback\nconfig · KB", "#fff", NAVY, 7.8)
    _arrow(ax, (16, 56), (17, 40), color=ORANGE, ls="--")
    _arrow(ax, (44, 56), (49, 40), color=TEAL, ls="--")
    _arrow(ax, (86, 56), (84, 40), color=NAVY, ls="--")
    _box(ax, 20, 10, 60, 10, "Pluggable LLM provider  —  offline grounded (default) | OpenAI | Anthropic",
         LBLUE, GREY, 8.2)
    _arrow(ax, (58, 56), (55, 20), color=GREY, ls=":")
    ax.text(50, 3, "Privacy/offline: no external calls in the default configuration; fully deterministic.",
            ha="center", fontsize=7.5, color=GREY, style="italic")
    fig.savefig(OUT / "fig_architecture.png", dpi=220, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- decision flow
def pipeline_flow():
    fig, ax = plt.subplots(figsize=(7.2, 2.5)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    _box(ax, 1, 55, 16, 26, "Query +\nfreshness\nscore s", LOR, ORANGE, 8)
    _box(ax, 24, 55, 18, 26, "s ≥ τ  or\nconf < τc ?", LBLUE, NAVY, 8, bold=True)
    _arrow(ax, (17, 68), (24, 68))
    _box(ax, 50, 74, 22, 22, "Retrieve evidence\n(FAISS) → generate\n→ verify claims", LTEAL, TEAL, 7.8)
    _box(ax, 50, 30, 22, 22, "Answer directly\nfrom parametric\nknowledge (skip)", "#fff", GREY, 7.8)
    _arrow(ax, (42, 72), (50, 83)); ax.text(44, 80, "stale", fontsize=7.5, color=TEAL)
    _arrow(ax, (42, 64), (50, 41)); ax.text(44, 52, "fresh", fontsize=7.5, color=GREY)
    _box(ax, 80, 52, 18, 26, "Confidence +\ncited answer", LBLUE, NAVY, 8, bold=True)
    _arrow(ax, (72, 85), (85, 78)); _arrow(ax, (72, 41), (85, 55))
    fig.savefig(OUT / "fig_pipeline.png", dpi=220, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- freshness separation
def freshness_separation():
    static = ["what is the speed of light", "define photosynthesis", "capital of Australia",
              "state the pythagorean theorem", "boiling point of water",
              "how many chambers in the human heart"]
    stale = ["who won the 2022 world cup", "federal reserve interest rate in 2025",
             "latest flagship smartphone chipset 2025", "which city hosted the 2024 olympics",
             "ev battery price per kwh in 2025", "llm context windows in 2025"]
    ss = [freshness_score(q) for q in static]
    st = [freshness_score(q) for q in stale]
    fig, ax = plt.subplots(figsize=(5.0, 2.7))
    ax.scatter([0] * len(ss), ss, c=GOOD, s=55, label="static (no retrieval)", zorder=3, edgecolor="w")
    ax.scatter([1] * len(st), st, c=ORANGE, s=55, label="time-sensitive (retrieve)", zorder=3, edgecolor="w")
    ax.axhline(0.55, color=NAVY, ls="--", lw=1.3)
    ax.text(1.42, 0.57, "threshold τ = 0.55", fontsize=8, color=NAVY, ha="right")
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.05, 1.05); ax.set_xticks([0, 1])
    ax.set_xticklabels(["static", "time-sensitive"]); ax.set_ylabel("freshness score")
    ax.legend(fontsize=7.5, loc="center right"); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "fig_freshness.png", dpi=220); plt.close(fig)


# ---------------------------------------------------------------- metrics bars
def metrics_bar(results):
    labels = ["Freshness\naccuracy", "Factuality", "Halluc.\nprecision", "Halluc.\nrecall", "Halluc.\nF1"]
    vals = [results["freshness"]["freshness_accuracy"], results["factuality"]["factuality"],
            results["hallucination"]["precision"], results["hallucination"]["recall"],
            results["hallucination"]["f1"]]
    fig, ax = plt.subplots(figsize=(5.0, 2.7))
    bars = ax.bar(labels, vals, color=[TEAL, NAVY, ORANGE, ORANGE, ORANGE], edgecolor="#223", width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.15); ax.set_ylabel("score"); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "fig_metrics.png", dpi=220); plt.close(fig)


# ---------------------------------------------------------------- latency saving
def latency(fg):
    static_qs = ["what is the speed of light", "define photosynthesis", "capital of Australia",
                 "state the pythagorean theorem", "boiling point of water"]
    stale_qs = ["who won the 2022 world cup", "federal reserve interest rate in 2025",
                "latest smartphone chipset 2025", "which city hosted the 2024 olympics",
                "ev battery price per kwh in 2025"]

    def mean_ms(qs):
        ts = []
        for q in qs:
            t0 = time.perf_counter(); fg.answer(q, persist=False); ts.append((time.perf_counter() - t0) * 1000)
        return sum(ts) / len(ts)

    skip_ms = mean_ms(static_qs); retr_ms = mean_ms(stale_qs)
    # adaptive = each query takes its own path; always-retrieve = force retrieval everywhere
    fg.storage.set_config({"confidence_threshold": 1.0, "freshness_threshold": 0.0})
    allr_static = mean_ms(static_qs)
    fg.storage.set_config({"confidence_threshold": 0.6, "freshness_threshold": 0.55})
    fig, ax = plt.subplots(figsize=(5.0, 2.7))
    ax.bar(["fresh query\n(adaptive skip)", "fresh query\n(always retrieve)", "stale query\n(retrieve)"],
           [skip_ms, allr_static, retr_ms], color=[GOOD, GREY, ORANGE], width=0.6, edgecolor="#223")
    for i, v in enumerate([skip_ms, allr_static, retr_ms]):
        ax.text(i, v + max(skip_ms, retr_ms, allr_static) * 0.02, f"{v:.1f} ms", ha="center", fontsize=8)
    ax.set_ylabel("mean latency per query (ms)"); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "fig_latency.png", dpi=220); plt.close(fig)
    return skip_ms, allr_static, retr_ms


def main():
    fg = FreshGuard(storage=Storage(ROOT / "artifacts" / "fig.db"))
    results = evaluation.run_all(fg)
    architecture(); pipeline_flow(); freshness_separation(); metrics_bar(results)
    lat = latency(fg)
    print("figures written to", OUT)
    print("latency (skip, always-retrieve, retrieve) ms:", [round(x, 2) for x in lat])


if __name__ == "__main__":
    main()
