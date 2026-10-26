"""Pluggable answer generation (FR9).

`generate(query, evidence)` returns an answer string. Two situations:
  * retrieval happened  -> the answer is grounded in the retrieved evidence;
  * retrieval was skipped -> the LLM answers from its own parametric memory.

The default `GroundedExtractiveLLM` is fully offline and deterministic. Its parametric
memory holds only *timeless* facts (modelling a knowledge cutoff), so time-sensitive
queries cannot be answered well without retrieval — exactly the case FreshGuard's freshness
gate is meant to catch. Real providers (OpenAI/Anthropic) can be selected via config and are
used only when an API key is present; otherwise the system falls back to the grounded LLM.
"""
from __future__ import annotations

import os
from typing import Protocol

from freshguard.textutil import split_sentences, overlap_score


class LLM(Protocol):
    name: str
    def generate(self, query: str, evidence: list[dict]) -> str: ...


def _compose(query: str, docs: list[dict], max_sentences: int = 2) -> str:
    """Pick the sentences across `docs` most relevant to the query and join them."""
    scored: list[tuple[float, str, str]] = []
    for d in docs:
        for sent in split_sentences(d.get("text", "")):
            scored.append((overlap_score(query, sent), d.get("doc_id", ""), sent))
    # highest overlap first; deterministic tie-break on doc_id then sentence text
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    picked = [s for sc, _, s in scored[:max_sentences] if sc > 0.0]
    return " ".join(picked)


class GroundedExtractiveLLM:
    name = "offline-grounded-v1"

    def __init__(self, parametric_docs: list[dict] | None = None):
        # parametric memory = only timeless documents (knowledge cutoff model)
        self.parametric = [d for d in (parametric_docs or []) if int(str(d.get("date", "2000"))[:4]) < 2020]

    def generate(self, query: str, evidence: list[dict]) -> str:
        if evidence:
            ans = _compose(query, evidence, max_sentences=2)
            return ans
        # no retrieval: answer from parametric memory only
        ans = _compose(query, self.parametric, max_sentences=2)
        return ans


class _ApiLLM:
    """Base for hosted providers; falls back to grounded if unavailable."""
    key_env = ""
    name = "api"

    def __init__(self, model: str, fallback: GroundedExtractiveLLM):
        self.model = model
        self.fallback = fallback

    def _available(self) -> bool:
        return bool(os.environ.get(self.key_env))

    def generate(self, query: str, evidence: list[dict]) -> str:  # pragma: no cover - needs key
        if not self._available():
            return self.fallback.generate(query, evidence)
        return self._call(query, evidence)

    def _call(self, query: str, evidence: list[dict]) -> str:  # pragma: no cover
        raise NotImplementedError


class OpenAILLM(_ApiLLM):
    key_env = "OPENAI_API_KEY"
    name = "openai"

    def _call(self, query, evidence):  # pragma: no cover - needs key/network
        from openai import OpenAI
        client = OpenAI()
        ctx = "\n".join(f"- {d['text']}" for d in evidence)
        prompt = (f"Answer the question using only the evidence when provided.\n"
                  f"Evidence:\n{ctx}\n\nQuestion: {query}\nAnswer:")
        r = client.chat.completions.create(model=self.model,
                                           messages=[{"role": "user", "content": prompt}],
                                           temperature=0)
        return r.choices[0].message.content.strip()


class AnthropicLLM(_ApiLLM):
    key_env = "ANTHROPIC_API_KEY"
    name = "anthropic"

    def _call(self, query, evidence):  # pragma: no cover - needs key/network
        import anthropic
        client = anthropic.Anthropic()
        ctx = "\n".join(f"- {d['text']}" for d in evidence)
        prompt = (f"Answer using only the evidence when provided.\n"
                  f"Evidence:\n{ctx}\n\nQuestion: {query}")
        r = client.messages.create(model=self.model, max_tokens=512, temperature=0,
                                   messages=[{"role": "user", "content": prompt}])
        return r.content[0].text.strip()


class GrokLLM(_ApiLLM):
    """xAI Grok via its OpenAI-compatible endpoint (https://api.x.ai/v1).

    Reads the key from XAI_API_KEY, falling back to GROK_API_KEY. Requires the
    `openai` client package; if the key or client is missing the grounded LLM is used.
    """
    key_env = "XAI_API_KEY"
    name = "grok"
    base_url = "https://api.x.ai/v1"

    def _available(self) -> bool:
        return bool(os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY"))

    def _call(self, query, evidence):  # pragma: no cover - needs key/network
        from openai import OpenAI
        key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
        client = OpenAI(api_key=key, base_url=self.base_url)
        ctx = "\n".join(f"- {d['text']}" for d in evidence)
        prompt = (f"Answer the question using only the evidence when provided.\n"
                  f"Evidence:\n{ctx}\n\nQuestion: {query}\nAnswer:")
        r = client.chat.completions.create(model=self.model,
                                           messages=[{"role": "user", "content": prompt}],
                                           temperature=0)
        return r.choices[0].message.content.strip()


def make_llm(provider: str, model: str, parametric_docs: list[dict]) -> LLM:
    grounded = GroundedExtractiveLLM(parametric_docs)
    if provider == "openai":
        return OpenAILLM(model, grounded)
    if provider == "anthropic":
        return AnthropicLLM(model, grounded)
    if provider == "grok":
        return GrokLLM(model, grounded)
    return grounded
