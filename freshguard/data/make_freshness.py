"""Synthesise a labelled dataset and train the learned freshness estimator (FR3/FR4).

The target label is 1 when a query's correct answer is likely to depend on information
that changes over time (so a static LLM's parametric knowledge may be stale and retrieval
is warranted), and 0 when the answer is stable/timeless.

The estimator is a scikit-learn Pipeline (TF-IDF word+char n-grams -> LogisticRegression)
combined with a few explicit temporal features. It is a *learned* probability, deliberately
not a fixed time-decay rule (FR4). Training is seeded for reproducibility.
"""
from __future__ import annotations

import random
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from freshguard.config import FRESHNESS_MODEL_PATH
from freshguard.features import temporal_features  # shared with freshness.py

SEED = 20251024

# ---- templates ---------------------------------------------------------------
STALE_TEMPLATES = [
    "what is the latest {thing}",
    "who won the {event} this year",
    "current {metric} of {entity}",
    "what is today's {metric}",
    "most recent {thing} in 2025",
    "who is the current {role}",
    "what is the newest {thing}",
    "{entity} stock price right now",
    "latest news about {entity}",
    "what happened to {entity} recently",
    "updated {metric} for {entity}",
    "who holds the {role} position now",
    "this year's {event} winner",
    "what is the price of {thing} today",
    "recent developments in {thing}",
    # explicit-recent-year phrasings WITHOUT a "latest/current" cue -----------
    "who won the {event} in {year}",
    "which team won the {year} {event}",
    "which city hosted the {year} {event}",
    "where was the {year} {event} held",
    "what was the {metric} of {entity} in {year}",
    "what is the {metric} in {year}",
    "how large are {thing} in {year}",
    "how big is the {thing} market in {year}",
    "how fast is the {year} {thing}",
    "{event} {year} winner",
    "{entity} {metric} in {year}",
    "what happened at the {event} in {year}",
    "specifications of the {year} {thing}",
    "{thing} price per unit in {year}",
    "how much does the {year} {thing} cost",
]
FRESH_TEMPLATES = [
    "what is the {concept}",
    "define {concept}",
    "explain how {process} works",
    "what is the capital of {place}",
    "how do you calculate {concept}",
    "what is the formula for {concept}",
    "why does {process} happen",
    "what are the parts of the {thing_static}",
    "state the {concept}",
    "what is the boiling point of water",
    "who wrote the theory of {concept}",
    "what is the chemical symbol for {element}",
    "how many sides does a {shape} have",
    "what is {a} plus {b}",
]

THINGS = ["smartphone", "gpu", "ai model", "electric car", "chipset", "laptop", "vaccine"]
EVENTS = ["world cup", "olympics", "election", "super bowl", "grand prix", "champions league"]
METRICS = ["interest rate", "inflation rate", "exchange rate", "stock price", "market share", "gdp growth"]
ENTITIES = ["Tesla", "Apple", "the Federal Reserve", "Bitcoin", "the economy", "the government", "Nvidia"]
ROLES = ["president", "prime minister", "ceo", "champion", "world number one", "chairman"]
CONCEPTS = ["Pythagorean theorem", "speed of light", "photosynthesis", "gravity", "Ohm's law",
            "second law of thermodynamics", "area of a circle", "Newton's first law"]
PROCESSES = ["photosynthesis", "digestion", "evaporation", "mitosis", "combustion", "osmosis"]
PLACES = ["Australia", "France", "Japan", "Canada", "Brazil", "Egypt", "Norway"]
STATIC_THINGS = ["human heart", "atom", "plant cell", "solar system", "digestive system"]
ELEMENTS = ["gold", "oxygen", "hydrogen", "sodium", "carbon", "iron"]
SHAPES = ["triangle", "hexagon", "pentagon", "square", "octagon"]


YEARS = [2021, 2022, 2023, 2024, 2025, 2026]


def _fill(t: str, rng: random.Random) -> str:
    return t.format(
        thing=rng.choice(THINGS), event=rng.choice(EVENTS), metric=rng.choice(METRICS),
        entity=rng.choice(ENTITIES), role=rng.choice(ROLES), concept=rng.choice(CONCEPTS),
        process=rng.choice(PROCESSES), place=rng.choice(PLACES), thing_static=rng.choice(STATIC_THINGS),
        element=rng.choice(ELEMENTS), shape=rng.choice(SHAPES), a=rng.randint(2, 99), b=rng.randint(2, 99),
        year=rng.choice(YEARS),
    )


def build_dataset(n_per_class: int = 400) -> tuple[list[str], list[int]]:
    rng = random.Random(SEED)
    X, y = [], []
    for _ in range(n_per_class):
        X.append(_fill(rng.choice(STALE_TEMPLATES), rng)); y.append(1)
        X.append(_fill(rng.choice(FRESH_TEMPLATES), rng)); y.append(0)
    # de-duplicate while keeping labels aligned
    seen, Xd, yd = set(), [], []
    for q, lab in zip(X, y):
        if q not in seen:
            seen.add(q); Xd.append(q); yd.append(lab)
    return Xd, yd


def build_pipeline() -> Pipeline:
    text_feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)),
    ])
    full = FeatureUnion([
        ("text", text_feats),
        ("temporal", FunctionTransformer(temporal_features, validate=False)),
    ])
    return Pipeline([
        ("features", full),
        ("clf", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced", random_state=SEED)),
    ])


def train_and_save(path: Path = FRESHNESS_MODEL_PATH) -> dict:
    X, y = build_dataset()
    pipe = build_pipeline()
    pipe.fit(X, y)
    acc = pipe.score(X, y)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)
    return {"train_size": len(X), "train_accuracy": round(float(acc), 4), "model_path": str(path)}


if __name__ == "__main__":
    info = train_and_save()
    print("Freshness estimator trained:", info)
