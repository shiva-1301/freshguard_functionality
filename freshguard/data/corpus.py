"""Seed corpus for the vector store (FR7). Each document has a date so the ranker can
reward recency (FR8). Dates are fixed so retrieval is deterministic for tests.

This is a small, self-contained knowledge base mixing time-sensitive facts (which a
stale LLM would get wrong) with stable background documents.
"""
from __future__ import annotations

SEED_DOCS: list[dict] = [
    {
        "doc_id": "d001",
        "title": "FIFA World Cup 2022 final result",
        "url": "https://example.org/wc2022",
        "date": "2022-12-18",
        "text": "Argentina won the 2022 FIFA World Cup, defeating France in the final. "
                "The match finished 3-3 after extra time and Argentina won 4-2 on penalties. "
                "Lionel Messi was named the tournament's best player.",
    },
    {
        "doc_id": "d002",
        "title": "US Federal Reserve policy rate, mid 2025",
        "url": "https://example.org/fed-rate-2025",
        "date": "2025-06-15",
        "text": "As of mid 2025 the US Federal Reserve held its benchmark federal funds rate "
                "in a target range around 4.25 to 4.50 percent after a series of adjustments. "
                "The central bank signalled a data-dependent stance for the rest of the year.",
    },
    {
        "doc_id": "d003",
        "title": "Latest flagship smartphone chipset 2025",
        "url": "https://example.org/chipset-2025",
        "date": "2025-09-01",
        "text": "The newest flagship mobile chipset announced in late 2025 uses a 3 nanometre "
                "process and emphasises on-device AI acceleration with a dedicated neural engine. "
                "It improves energy efficiency over the previous generation by about 20 percent.",
    },
    {
        "doc_id": "d004",
        "title": "Speed of light (physical constant)",
        "url": "https://example.org/speed-of-light",
        "date": "2010-01-01",
        "text": "The speed of light in vacuum is exactly 299,792,458 metres per second. "
                "It is a fundamental physical constant denoted c and is the same for all observers.",
    },
    {
        "doc_id": "d005",
        "title": "Photosynthesis overview",
        "url": "https://example.org/photosynthesis",
        "date": "2009-05-01",
        "text": "Photosynthesis is the process by which green plants convert light energy, water "
                "and carbon dioxide into glucose and oxygen. It takes place mainly in the "
                "chloroplasts using the pigment chlorophyll.",
    },
    {
        "doc_id": "d006",
        "title": "Capital of Australia",
        "url": "https://example.org/canberra",
        "date": "2008-01-01",
        "text": "The capital city of Australia is Canberra, located in the Australian Capital "
                "Territory. Canberra was purpose-built as the capital; it is not Sydney or "
                "Melbourne, which are larger cities.",
    },
    {
        "doc_id": "d007",
        "title": "Summer Olympics 2024 host city",
        "url": "https://example.org/olympics-2024",
        "date": "2024-08-11",
        "text": "The 2024 Summer Olympic Games were hosted by Paris, France. The games featured "
                "new events and a widely discussed opening ceremony held along the river Seine.",
    },
    {
        "doc_id": "d008",
        "title": "Pythagorean theorem",
        "url": "https://example.org/pythagoras",
        "date": "2005-01-01",
        "text": "The Pythagorean theorem states that in a right-angled triangle the square of the "
                "hypotenuse equals the sum of the squares of the other two sides, a squared plus "
                "b squared equals c squared.",
    },
    {
        "doc_id": "d009",
        "title": "Large language model context windows, 2025",
        "url": "https://example.org/llm-context-2025",
        "date": "2025-03-20",
        "text": "By 2025 several leading large language models offered context windows of hundreds "
                "of thousands of tokens, with some research systems reaching one million tokens, "
                "enabling long-document reasoning and retrieval-augmented generation.",
    },
    {
        "doc_id": "d010",
        "title": "Water boiling point",
        "url": "https://example.org/boiling-point",
        "date": "2007-01-01",
        "text": "At standard atmospheric pressure water boils at 100 degrees Celsius, which is "
                "212 degrees Fahrenheit. The boiling point decreases at higher altitude where "
                "atmospheric pressure is lower.",
    },
    {
        "doc_id": "d011",
        "title": "Electric vehicle battery prices, 2025",
        "url": "https://example.org/ev-battery-2025",
        "date": "2025-02-10",
        "text": "Average lithium-ion battery pack prices for electric vehicles continued to fall in "
                "2025, approaching roughly 100 US dollars per kilowatt-hour, driven by cheaper "
                "lithium iron phosphate cells and manufacturing scale.",
    },
    {
        "doc_id": "d012",
        "title": "Human heart basics",
        "url": "https://example.org/heart",
        "date": "2006-01-01",
        "text": "The human heart is a muscular organ with four chambers: two atria and two "
                "ventricles. It pumps blood through the circulatory system, supplying oxygen and "
                "nutrients to the body.",
    },
]


def get_seed_docs() -> list[dict]:
    return [dict(d) for d in SEED_DOCS]
