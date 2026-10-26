"""A deterministic, offline LangChain Embeddings implementation backed by scikit-learn
TF-IDF. Produces fixed-dimension unit-normalised dense vectors so they can be indexed by
FAISS and compared with inner-product = cosine similarity.

Using TF-IDF (rather than a downloaded transformer) keeps the whole system offline and
reproducible, which is what makes the end-to-end tests stable across many runs.
"""
from __future__ import annotations

import numpy as np
from langchain_core.embeddings import Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbeddings(Embeddings):
    def __init__(self, corpus: list[str]):
        # fit on the corpus; char n-grams add robustness to spelling/word overlap
        self._vec = TfidfVectorizer(
            ngram_range=(1, 2), analyzer="word", min_df=1, sublinear_tf=True, stop_words="english"
        )
        self._vec.fit(corpus if corpus else ["placeholder document"])
        self.dim = len(self._vec.vocabulary_)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        mat = self._vec.transform(texts).toarray().astype("float32")
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = mat / norms
        return mat.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]
