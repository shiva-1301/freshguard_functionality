"""Statement-level verification (FR10), hallucination labelling (FR11) and corrective
action (FR12).

Each answer sentence is checked against the retrieved evidence and labelled:
  * supported    - its content is covered by the evidence and no conflict is found
  * unsupported  - its content is largely absent from the evidence (incl. a named entity
                   the evidence never mentions)
  * contradicted - it conflicts with the evidence: a negation flip, a numeric mismatch on
                   an otherwise-matching claim, or an entity swap (asserting that the party
                   the evidence names as the *loser* is the subject of the claim)
  * review       - partial / weak support

The design is deliberately bag-of-words + light linguistic cues (no large model) so the
whole system stays offline and deterministic; the rules target the hallucination patterns a
grounded RAG answer realistically exhibits.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from freshguard.textutil import split_sentences, overlap_score, content_tokens, tokens

_NEG = {"not", "no", "never", "cannot", "none", "false", "incorrect", "didnt", "isnt", "wasnt"}
_DEFEAT_RE = re.compile(r"\b(?:defeat(?:ing|ed)?|beat(?:ing|en)?|downed|edged|over|against)\s+"
                        r"([A-Z][a-zA-Z]+)")
_PROPER_RE = re.compile(r"\b([A-Z][a-zA-Z]+)")


@dataclass
class VerifiedStatement:
    text: str
    label: str
    support: float
    evidence_doc_id: str | None


def _stem(t: str) -> str:
    return t[:-1] if len(t) > 3 and t.endswith("s") else t


def _stems(text: str) -> set[str]:
    return {_stem(t) for t in content_tokens(text)}


def _numbers(s: str) -> set[str]:
    return {t for t in tokens(s) if any(c.isdigit() for c in t)}


def _proper_nouns(sentence: str) -> list[str]:
    """Capitalised tokens that are not the first word of the sentence (so sentence-initial
    capitalisation is not mistaken for a proper noun)."""
    out = []
    for m in _PROPER_RE.finditer(sentence):
        if m.start() == 0:
            continue
        out.append(m.group(1))
    return out


def _coverage(stmt: str, ev: str) -> float:
    s, e = _stems(stmt), _stems(ev)
    if not s:
        return 0.0
    return len(s & e) / len(s)


def _first_proper(sentence: str) -> str | None:
    m = _PROPER_RE.search(sentence)
    return m.group(1) if m else None


def _contradiction(stmt: str, ev: str) -> bool:
    ov = overlap_score(stmt, ev)
    if ov < 0.3:
        return False
    # (a) negation flip
    neg_s = bool(set(tokens(stmt)) & _NEG) or "n't" in stmt.lower()
    neg_e = bool(set(tokens(ev)) & _NEG) or "n't" in ev.lower()
    if neg_s != neg_e:
        return True
    # (b) numeric mismatch on an otherwise-matching claim
    sn, en = _numbers(stmt), _numbers(ev)
    if sn and en and not (sn & en):
        return True
    # (c) entity swap: evidence names a loser after "defeating/beating/over"; the statement
    #     makes that loser its subject while omitting the evidence's winner
    m = _DEFEAT_RE.search(ev)
    if m:
        loser = _stem(m.group(1).lower())
        winner = _first_proper(ev)
        winner_stem = _stem(winner.lower()) if winner else None
        s_stems = _stems(stmt)
        if loser in s_stems and (winner_stem is None or winner_stem not in s_stems):
            return True
    return False


def verify_statements(answer: str, evidence: list[dict],
                      support_hi: float, support_lo: float) -> list[VerifiedStatement]:
    ev_items: list[tuple[str, str]] = []  # (doc_id, sentence)
    for d in evidence:
        for sent in split_sentences(d.get("text", "")):
            ev_items.append((d.get("doc_id", ""), sent))
    all_ev_stems: set[str] = set()
    for _, sent in ev_items:
        all_ev_stems |= _stems(sent)

    results: list[VerifiedStatement] = []
    for stmt in split_sentences(answer):
        best_cov, best_doc = 0.0, None
        contradicted = False
        for doc_id, sent in ev_items:
            cov = _coverage(stmt, sent)
            if cov > best_cov:
                best_cov, best_doc = cov, doc_id
            if _contradiction(stmt, sent):
                contradicted = True
                best_doc = best_doc or doc_id

        # a named entity the evidence never mentions => fabricated detail
        missing_entity = any(_stem(p.lower()) not in all_ev_stems for p in _proper_nouns(stmt))

        if contradicted:
            label = "contradicted"
        elif missing_entity and best_cov < 0.95:
            label = "unsupported"
        elif best_cov >= support_hi:
            label = "supported"
        elif best_cov < support_lo:
            label = "unsupported"
        else:
            label = "review"
        results.append(VerifiedStatement(
            stmt, label, round(best_cov, 4),
            best_doc if label in ("supported", "contradicted", "review") else None))
    return results


def corrective_action(statements: list[VerifiedStatement]) -> tuple[str, list[str]]:
    """FR12: keep supported/review statements, drop unsupported, flag contradicted."""
    warnings: list[str] = []
    kept: list[str] = []
    for s in statements:
        if s.label in ("supported", "review"):
            kept.append(s.text)
        elif s.label == "unsupported":
            warnings.append(f"Removed an unsupported statement: \"{_short(s.text)}\"")
        elif s.label == "contradicted":
            warnings.append(f"Flagged a statement that contradicts the evidence: \"{_short(s.text)}\"")
    corrected = " ".join(kept).strip()
    if not corrected:
        warnings.append("No statement in the generated answer was supported by the evidence; "
                        "returning a low-confidence response.")
    return corrected, warnings


def _short(t: str, n: int = 80) -> str:
    return t if len(t) <= n else t[: n - 1] + "…"
