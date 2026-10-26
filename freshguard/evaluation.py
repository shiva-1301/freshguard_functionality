"""Performance evaluation module (FR21).

Reports:
  * freshness_accuracy  - did the retrieval decision match the gold need-retrieval label
  * factuality          - fraction of answers that contain the gold key fact
  * hallucination       - precision/recall/F1 of flagging unsupported+contradicted claims
  * calibration (ECE)   - expected calibration error of the confidence score vs correctness
"""
from __future__ import annotations

from freshguard.textutil import content_tokens
from freshguard.verify import verify_statements

# -- labelled evaluation sets --------------------------------------------------
# query, should_retrieve (gold), gold answer keywords
EVAL_QUERIES = [
    ("what is the speed of light", False, ["299", "792", "458"]),
    ("define photosynthesis", False, ["light", "oxygen"]),
    ("what is the capital of Australia", False, ["canberra"]),
    ("state the pythagorean theorem", False, ["hypotenuse", "squares"]),
    ("what is the boiling point of water", False, ["100", "celsius"]),
    ("how many chambers does the human heart have", False, ["four", "chambers"]),
    ("who won the 2022 world cup", True, ["argentina"]),
    ("what is the current federal reserve interest rate in 2025", True, ["4", "percent"]),
    ("latest flagship smartphone chipset 2025", True, ["3", "nanometre", "nanometer", "nm"]),
    ("which city hosted the 2024 summer olympics", True, ["paris"]),
    ("electric vehicle battery price per kwh in 2025", True, ["100", "kilowatt"]),
    ("how large are llm context windows in 2025", True, ["tokens"]),
]

# answer sentence, evidence text, is_hallucination (gold)
HALLUCINATION_SET = [
    ("Argentina won the 2022 FIFA World Cup.",
     "Argentina won the 2022 FIFA World Cup, defeating France in the final.", False),
    ("France won the 2022 FIFA World Cup.",
     "Argentina won the 2022 FIFA World Cup, defeating France in the final.", True),
    ("The speed of light is 299,792,458 metres per second.",
     "The speed of light in vacuum is exactly 299,792,458 metres per second.", False),
    ("The speed of light is about 150,000 kilometres per second.",
     "The speed of light in vacuum is exactly 299,792,458 metres per second.", True),
    ("Paris hosted the 2024 Summer Olympics.",
     "The 2024 Summer Olympic Games were hosted by Paris, France.", False),
    ("Los Angeles hosted the 2024 Summer Olympics.",
     "The 2024 Summer Olympic Games were hosted by Paris, France.", True),
    ("Water boils at 100 degrees Celsius at standard pressure.",
     "At standard atmospheric pressure water boils at 100 degrees Celsius.", False),
    ("Water boils at 50 degrees Celsius at standard pressure.",
     "At standard atmospheric pressure water boils at 100 degrees Celsius.", True),
]


def _contains(answer: str, keywords: list[str]) -> bool:
    toks = content_tokens(answer) | set(answer.lower().replace(",", " ").split())
    low = answer.lower()
    return any(k.lower() in low for k in keywords)


def evaluate_freshness(fg) -> dict:
    correct = 0
    for q, gold_retrieve, _ in EVAL_QUERIES:
        r = fg.answer(q, persist=False)
        if r["retrieved"] == gold_retrieve:
            correct += 1
    return {"freshness_accuracy": round(correct / len(EVAL_QUERIES), 4),
            "n": len(EVAL_QUERIES)}


def evaluate_factuality(fg) -> dict:
    hits = 0
    for q, _, keywords in EVAL_QUERIES:
        r = fg.answer(q, persist=False)
        if _contains(r["answer"], keywords):
            hits += 1
    return {"factuality": round(hits / len(EVAL_QUERIES), 4), "n": len(EVAL_QUERIES)}


def evaluate_hallucination(support_hi: float = 0.32, support_lo: float = 0.15) -> dict:
    tp = fp = tn = fn = 0
    for sent, ev_text, is_hallu in HALLUCINATION_SET:
        evidence = [{"doc_id": "e", "text": ev_text}]
        vs = verify_statements(sent, evidence, support_hi, support_lo)[0]
        flagged = vs.label in ("unsupported", "contradicted")
        if is_hallu and flagged:
            tp += 1
        elif is_hallu and not flagged:
            fn += 1
        elif not is_hallu and flagged:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (tp + tn) / len(HALLUCINATION_SET)
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "accuracy": round(acc, 4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def evaluate_calibration(fg, n_bins: int = 5) -> dict:
    confs, corrects = [], []
    for q, _, keywords in EVAL_QUERIES:
        r = fg.answer(q, persist=False)
        confs.append(r["confidence"] / 100.0)
        corrects.append(1.0 if _contains(r["answer"], keywords) else 0.0)
    ece = 0.0
    n = len(confs)
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, c in enumerate(confs) if (c > lo or (b == 0 and c == 0)) and c <= hi]
        if not idx:
            continue
        avg_conf = sum(confs[i] for i in idx) / len(idx)
        avg_acc = sum(corrects[i] for i in idx) / len(idx)
        ece += (len(idx) / n) * abs(avg_conf - avg_acc)
    return {"ece": round(ece, 4), "mean_confidence": round(sum(confs) / n, 4),
            "mean_accuracy": round(sum(corrects) / n, 4)}


def run_all(fg) -> dict:
    s = fg.storage.get_settings()
    return {
        "freshness": evaluate_freshness(fg),
        "factuality": evaluate_factuality(fg),
        "hallucination": evaluate_hallucination(s.support_hi, s.support_lo),
        "calibration": evaluate_calibration(fg),
    }
