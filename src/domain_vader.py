"""
VADER sentiment, adapted to e-commerce review language.

Off-the-shelf VADER was trained on social-media text. It has no opinion on
words such as "overpriced", "flimsy" or "disconnecting", and it reads
"Customer service never responded to my emails" as neutral because none of
the words are negative on their own. For a retailer, those are exactly the
complaints that matter.

Two light-touch fixes:
1. Single-word lexicon updates (VADER's built-in mechanism).
2. Multi-word phrases rewritten to a single token before scoring, so that
   "stopped working" or "rip off" can carry a sentiment of its own.
Scores use VADER's -4..+4 word-valence scale.
"""

from __future__ import annotations

import re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

DOMAIN_WORDS = {
    # product quality
    "flimsy": -2.0, "defective": -2.3, "overheats": -2.0, "crack": -1.5, "loose": -1.0,
    "sturdy": 1.8, "durable": 1.8, "premium": 1.5, "solid": 1.3,
    # delivery
    "delayed": -1.8, "late": -1.3, "slow": -1.5, "reshipped": -1.2, "early": 1.0,
    "accurate": 1.2,
    # customer service ("support" is positive in stock VADER - here it's just a department)
    "support": 0.0, "refused": -1.8,
    # usability
    "instantly": 1.2,
    # price
    "overpriced": -2.3, "expensive": -1.3, "bargain": 1.8,
    # assembly / usability
    "missing": -1.5, "disconnecting": -1.8, "straightforward": 1.5,
    # packaging
    "crushed": -1.8, "torn": -1.5, "rattling": -1.0,
}

DOMAIN_PHRASES = {
    r"stopped working": "stopped_working",
    r"falling apart": "falling_apart",
    r"never responded": "never_responded",
    r"never updated": "never_updated",
    r"on hold for": "on_hold_for",
    r"call dropped": "call_dropped",
    r"different answers": "different_answers",
    r"rip off": "rip_off",
    r"days later": "days_later",
    r"nothing like the photos": "nothing_like_photos",
    r"(?:don't|do not) line up": "dont_line_up",
    r"drilled wrong": "drilled_wrong",
    r"lost in transit": "lost_in_transit",
    r"past the estimated date": "past_estimated_date",
}
PHRASE_SCORES = {
    "stopped_working": -2.5, "falling_apart": -2.5, "never_responded": -2.3,
    "never_updated": -1.8, "on_hold_for": -1.8, "call_dropped": -1.5,
    "different_answers": -1.5, "rip_off": -2.5, "days_later": -1.2,
    "nothing_like_photos": -2.0, "dont_line_up": -2.0, "drilled_wrong": -2.0,
    "lost_in_transit": -2.3, "past_estimated_date": -1.8,
}
_PHRASE_RE = [(re.compile(p, re.IGNORECASE), tok) for p, tok in DOMAIN_PHRASES.items()]


class DomainVader:
    """Drop-in wrapper: DomainVader().compound(text) -> float in [-1, 1]."""

    def __init__(self, use_domain: bool = True):
        self.analyzer = SentimentIntensityAnalyzer()
        self.use_domain = use_domain
        if use_domain:
            self.analyzer.lexicon.update(DOMAIN_WORDS)
            self.analyzer.lexicon.update(PHRASE_SCORES)

    def _rewrite(self, text: str) -> str:
        for pat, tok in _PHRASE_RE:
            text = pat.sub(tok, text)
        return text

    def compound(self, text: str) -> float:
        text = str(text)
        if self.use_domain:
            text = self._rewrite(text)
        return self.analyzer.polarity_scores(text)["compound"]
