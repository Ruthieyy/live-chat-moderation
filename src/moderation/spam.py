"""High-precision spam rules for livestream chat.

Jigsaw labels toxicity, not spam. A learned spam head would need a different
corpus and would still lose to an obvious URL or "free followers" pattern on
the latency budget we care about. These rules run on the raw comment, before
the normalizer squashes repeated characters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_URL_RE = re.compile(r"(https?://|www\.)\S+", re.IGNORECASE)
_CONTACT_RE = re.compile(
    r"([a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,})|(\+?\d[\d\-\s().]{7,}\d)",
    re.IGNORECASE,
)
_SOLICIT_RE = re.compile(
    r"\b("
    r"free followers|click here|subscribe to my|check (?:out )?my (?:channel|page)|"
    r"buy followers|earn money fast|work from home|crypto giveaway|"
    r"onlyfans|promo code|limited offer|act now|giveaway|dm me"
    r")\b",
    re.IGNORECASE,
)
_REPEAT_RE = re.compile(r"(.)\1{5,}")
_WORD_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)


@dataclass(frozen=True)
class SpamHit:
    name: str
    score: float


def _repeated_token_score(text: str) -> float:
    words = [word.lower() for word in _WORD_RE.findall(text)]
    if len(words) < 6:
        return 0.0
    most = max(words.count(word) for word in set(words))
    ratio = most / len(words)
    if most >= 6 and ratio >= 0.6:
        return 0.72
    return 0.0


def spam_hits(text: str) -> list[SpamHit]:
    hits: list[SpamHit] = []
    urls = _URL_RE.findall(text)
    if len(urls) >= 2:
        hits.append(SpamHit("spam_url", 0.9))
    elif len(urls) == 1:
        hits.append(SpamHit("spam_url", 0.78))
    if _CONTACT_RE.search(text):
        hits.append(SpamHit("spam_contact", 0.82))
    if _SOLICIT_RE.search(text):
        hits.append(SpamHit("spam_solicitation", 0.88))
    if _REPEAT_RE.search(text):
        hits.append(SpamHit("spam_repetition", 0.6))
    repeated = _repeated_token_score(text)
    if repeated:
        hits.append(SpamHit("spam_repeated_token", repeated))
    letters = [ch for ch in text if ch.isalpha()]
    if len(letters) >= 20 and sum(ch.isupper() for ch in letters) / len(letters) >= 0.9:
        hits.append(SpamHit("spam_shouting", 0.45))
    return hits


def combine_spam(hits: list[SpamHit]) -> float:
    """Noisy-OR so two moderate signals outrank either one alone."""
    keep = 1.0
    for hit in hits:
        keep *= 1.0 - hit.score
    return 1.0 - keep


def score_spam(text: str) -> tuple[float, list[str]]:
    hits = spam_hits(text)
    return combine_spam(hits), [hit.name for hit in hits]
