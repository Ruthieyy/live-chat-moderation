from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from moderation.api import create_app


@pytest.fixture(scope="session")
def client() -> TestClient:
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_health_and_ready(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["model_version"]


def test_model_card_lists_thresholds(client: TestClient) -> None:
    body = client.get("/v1/model").json()
    assert body["labels"][0] == "toxic"
    assert "toxic_block" in body["thresholds"]
    assert body["dataset"]["revision"]


def test_benign_chat_is_allowed(client: TestClient) -> None:
    response = client.post(
        "/v1/moderate",
        json={
            "text": "Thanks for the thoughtful edit, I agree with your summary.",
            "room_id": "live-1",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["action"] == "allow"
    assert 0.0 <= body["scores"]["toxic"] <= 1.0
    assert body["model_version"]
    assert response.headers["content-type"].startswith("application/json")


def test_insult_scores_higher_than_a_benign_comment(client: TestClient) -> None:
    benign = client.post(
        "/v1/moderate",
        json={"text": "Thanks for the thoughtful edit, I agree with your summary."},
    ).json()
    rude = client.post(
        "/v1/moderate",
        json={"text": "You are a complete idiot and nobody wants you here."},
    ).json()
    assert rude["scores"]["toxic"] > benign["scores"]["toxic"] + 0.25
    assert rude["action"] in {"review", "block"}


def test_spam_rules_fire_on_the_sync_path(client: TestClient) -> None:
    body = client.post(
        "/v1/moderate",
        json={"text": "Get free followers now http://spam.example/deal and click here"},
    ).json()
    assert body["scores"]["spam"] >= 0.7
    assert "spam" in body["reasons"]
    assert body["action"] in {"review", "block"}


def test_scores_are_deterministic(client: TestClient) -> None:
    payload = {"text": "Good luck in the next match, you have played really well."}
    first = client.post("/v1/moderate", json=payload).json()
    second = client.post("/v1/moderate", json=payload).json()
    assert first["scores"] == second["scores"]
    assert first["action"] == second["action"]


def test_empty_and_huge_payloads_are_rejected(client: TestClient) -> None:
    assert client.post("/v1/moderate", json={"text": "   "}).status_code == 422
    assert client.post("/v1/moderate", json={"text": "a" * 2001}).status_code == 422
    assert client.post("/v1/moderate", json={"text": "a" * 9000}).status_code == 422


def test_batch_limit(client: TestClient) -> None:
    comments = [{"text": "nice"} for _ in range(17)]
    assert client.post("/v1/moderate/batch", json={"comments": comments}).status_code == 422
    ok = client.post(
        "/v1/moderate/batch",
        json={"comments": [{"text": "nice"}, {"text": "gg"}]},
    )
    assert ok.status_code == 200
    assert len(ok.json()) == 2


def test_metrics_expose_the_latency_histogram(client: TestClient) -> None:
    client.post("/v1/moderate", json={"text": "gg"})
    body = client.get("/metrics").text
    assert "moderation_request_latency_seconds_bucket" in body
    assert 'endpoint="sync"' in body
    assert "moderation_decisions_total" in body


def test_drift_report_starts_cold(client: TestClient) -> None:
    body = client.get("/v1/drift").json()
    assert "psi" in body
    assert body["samples"] >= 1
