"""Map model scores and spam hits to allow / review / block.

Block is reserved for high-precision toxicity, hate, and threats. Spam and
borderline toxicity go to a review queue: hiding a real viewer is worse than
queueing a human for a second look during a live.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from moderation.labels import TOXIC_FAMILY

Action = Literal["allow", "review", "block"]


@dataclass(frozen=True)
class Thresholds:
    toxic_review: float
    toxic_block: float
    hate_review: float
    hate_block: float
    threat_block: float
    spam_review: float

    @classmethod
    def from_dict(cls, payload: dict[str, float]) -> Thresholds:
        return cls(
            toxic_review=float(payload["toxic_review"]),
            toxic_block=float(payload["toxic_block"]),
            hate_review=float(payload["hate_review"]),
            hate_block=float(payload["hate_block"]),
            threat_block=float(payload["threat_block"]),
            spam_review=float(payload["spam_review"]),
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "toxic_review": self.toxic_review,
            "toxic_block": self.toxic_block,
            "hate_review": self.hate_review,
            "hate_block": self.hate_block,
            "threat_block": self.threat_block,
            "spam_review": self.spam_review,
        }


@dataclass(frozen=True)
class Decision:
    action: Action
    reasons: list[str]
    scores: dict[str, float]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def toxic_score(attributes: dict[str, float]) -> float:
    return max(float(attributes[name]) for name in TOXIC_FAMILY)


def product_scores(attributes: dict[str, float], spam_score: float) -> dict[str, float]:
    return {
        "toxic": toxic_score(attributes),
        "hate": float(attributes["identity_hate"]),
        "spam": float(spam_score),
    }


def decide(
    attributes: dict[str, float],
    spam_score: float,
    spam_reasons: list[str],
    thresholds: Thresholds,
) -> Decision:
    scores = product_scores(attributes, spam_score)
    hate = scores["hate"]
    threat = float(attributes["threat"])
    toxic = scores["toxic"]

    block_reasons: list[str] = []
    if hate >= thresholds.hate_block:
        block_reasons.append("hate")
    if threat >= thresholds.threat_block:
        block_reasons.append("threat")
    if toxic >= thresholds.toxic_block:
        block_reasons.append("toxic")

    reasons: list[str] = []
    action: Action = "allow"
    if block_reasons:
        action = "block"
        reasons.extend(block_reasons)
    else:
        if hate >= thresholds.hate_review:
            reasons.append("hate")
        if toxic >= thresholds.toxic_review:
            reasons.append("toxic")
        if reasons:
            action = "review"

    if spam_score >= thresholds.spam_review:
        if action == "allow":
            action = "review"
        reasons.append("spam")
        reasons.extend(spam_reasons)

    return Decision(action=action, reasons=_dedupe(reasons), scores=scores)
