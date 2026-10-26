from __future__ import annotations

from moderation.spam import score_spam


def test_ordinary_chat_is_not_spam() -> None:
    score, reasons = score_spam("Welcome to the stream, glad you could make it tonight.")
    assert score < 0.2
    assert reasons == []


def test_url_and_solicitation_score_high() -> None:
    score, reasons = score_spam("Get free followers now http://spam.example/deal and click here")
    assert score >= 0.7
    assert "spam_url" in reasons
    assert "spam_solicitation" in reasons


def test_contact_details_are_flagged() -> None:
    score, reasons = score_spam("Contact me at spam-bot@example.com or call 555-010-0199")
    assert "spam_contact" in reasons
    assert score >= 0.7


def test_repeated_token_flood_is_flagged() -> None:
    score, reasons = score_spam("lol lol lol lol lol lol lol lol lol lol")
    assert "spam_repeated_token" in reasons
    assert score >= 0.7


def test_two_signals_outrank_one() -> None:
    url_only, _ = score_spam("see https://example.com/a for the patch notes")
    both, _ = score_spam("free followers https://example.com/a")
    assert both > url_only
