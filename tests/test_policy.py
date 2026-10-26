from __future__ import annotations

from moderation.labels import MODEL_LABELS
from moderation.policy import Thresholds, decide


def _thresholds() -> Thresholds:
    return Thresholds(
        toxic_review=0.40,
        toxic_block=0.80,
        hate_review=0.30,
        hate_block=0.70,
        threat_block=0.60,
        spam_review=0.70,
    )


def _attributes(**overrides: float) -> dict[str, float]:
    base = {label: 0.0 for label in MODEL_LABELS}
    base.update(overrides)
    return base


def test_clean_comment_is_allowed() -> None:
    decision = decide(_attributes(), 0.0, [], _thresholds())
    assert decision.action == "allow"
    assert decision.reasons == []


def test_high_toxicity_blocks() -> None:
    decision = decide(_attributes(toxic=0.91, insult=0.2), 0.0, [], _thresholds())
    assert decision.action == "block"
    assert "toxic" in decision.reasons


def test_borderline_toxicity_is_reviewed() -> None:
    decision = decide(_attributes(insult=0.55), 0.0, [], _thresholds())
    assert decision.action == "review"
    assert decision.reasons == ["toxic"]


def test_threat_blocks_below_the_toxic_threshold() -> None:
    decision = decide(_attributes(threat=0.66), 0.0, [], _thresholds())
    assert decision.action == "block"
    assert "threat" in decision.reasons


def test_hate_blocks() -> None:
    decision = decide(_attributes(identity_hate=0.77), 0.0, [], _thresholds())
    assert decision.action == "block"
    assert "hate" in decision.reasons


def test_spam_alone_goes_to_review() -> None:
    decision = decide(_attributes(), 0.9, ["spam_url"], _thresholds())
    assert decision.action == "review"
    assert decision.reasons == ["spam", "spam_url"]
    assert decision.scores["spam"] == 0.9


def test_spam_does_not_downgrade_a_block() -> None:
    decision = decide(_attributes(toxic=0.95), 0.9, ["spam_url"], _thresholds())
    assert decision.action == "block"
    assert decision.reasons[0] == "toxic"
    assert "spam" in decision.reasons
