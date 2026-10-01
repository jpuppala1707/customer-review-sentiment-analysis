"""
Rule-based aspect extraction for customer reviews.

A review like "Great blender, but delivery took three weeks" is neither fully
positive nor fully negative - it is positive about QUALITY and negative about
DELIVERY. Splitting reviews into sentences and tagging each sentence with the
business aspect it talks about lets us score sentiment per aspect, which is
what an operations or product team can actually act on.
"""

from __future__ import annotations

import re

import pandas as pd

# Keyword dictionary built by reading a sample of reviews and grouping terms
# by the team that owns the problem (product, logistics, pricing, support...).
ASPECT_KEYWORDS: dict[str, list[str]] = {
    "Product Quality": [
        "quality", "made", "materials", "sturdy", "durable", "flimsy", "broke", "broken",
        "crack", "defective", "stopped working", "falling apart", "overheats", "noise",
        "solid", "premium", "works", "photos", "as described",
    ],
    "Delivery & Shipping": [
        "delivery", "delivered", "shipping", "arrived", "tracking", "late", "delayed",
        "transit", "reshipped", "driver", "estimated",
    ],
    "Price & Value": [
        "price", "priced", "value", "expensive", "overpriced", "cheaper", "bargain",
        "worth", "money", "sale", "rip off", "penny",
    ],
    "Customer Service": [
        "customer service", "support", "refund", "return", "warranty", "on hold",
        "responded", "emails", "replaced", "contacted",
    ],
    "Packaging": ["packaging", "packed", "box", "plastic", "crushed", "torn"],
    "Assembly & Ease of Use": [
        "assembly", "assemble", "instructions", "setup", "screws", "panels", "drilled",
        "easy to use", "hard to use", "app", "disconnecting", "screwdriver",
    ],
}

_PATTERNS = {
    aspect: re.compile(r"\b(" + "|".join(re.escape(k) for k in kws) + r")\b", re.IGNORECASE)
    for aspect, kws in ASPECT_KEYWORDS.items()
}

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(str(text)) if s.strip()]


def tag_aspects(sentence: str) -> list[str]:
    """Return every aspect whose keywords appear in the sentence."""
    return [aspect for aspect, pat in _PATTERNS.items() if pat.search(sentence)]


def explode_aspects(df: pd.DataFrame, text_col: str = "review_text",
                    id_cols: tuple[str, ...] = ("review_id",)) -> pd.DataFrame:
    """One row per (review, sentence, aspect). Sentences with no aspect are dropped."""
    rows = []
    for rec in df[[*id_cols, text_col]].itertuples(index=False):
        rec = rec._asdict()
        for sent in split_sentences(rec[text_col]):
            for aspect in tag_aspects(sent):
                rows.append({**{c: rec[c] for c in id_cols}, "sentence": sent, "aspect": aspect})
    return pd.DataFrame(rows)
