from __future__ import annotations

import fakeredis
from fastapi.testclient import TestClient

from moderation.api import create_app
from moderation.queue import JobQueue
from moderation.worker import handle_batch


def test_async_job_is_scored_by_the_worker() -> None:
    fake = fakeredis.FakeRedis(decode_responses=True)
    queue = JobQueue(fake, result_ttl_seconds=60)
    queue.ensure_group()
    app = create_app(queue=queue)
    with TestClient(app) as client:
        accepted = client.post(
            "/v1/moderate/async",
            json={"text": "Welcome to the stream, glad you could make it tonight.", "room_id": "live-9"},
        )
        assert accepted.status_code == 202
        job_id = accepted.json()["job_id"]
        queued = client.get(f"/v1/jobs/{job_id}")
        assert queued.status_code == 200
        assert queued.json()["status"] == "queued"
        messages = queue.read_new("test-worker", count=8, block_ms=200)
        assert len(messages) == 1
        handled = handle_batch(app.state.engine, queue, messages, app.state.drift)
        assert handled == 1
        done = client.get(f"/v1/jobs/{job_id}").json()
        assert done["status"] == "done"
        assert done["result"]["action"] in {"allow", "review", "block"}
        assert done["result"]["room_id"] == "live-9"
        assert done["queue_ms"] is not None


def test_unknown_job_is_404() -> None:
    fake = fakeredis.FakeRedis(decode_responses=True)
    queue = JobQueue(fake, result_ttl_seconds=60)
    queue.ensure_group()
    app = create_app(queue=queue)
    with TestClient(app) as client:
        response = client.get("/v1/jobs/does-not-exist")
        assert response.status_code == 404


def test_async_without_redis_is_unavailable() -> None:
    from moderation.settings import Settings

    settings = Settings(redis_enabled=False, model_dir=Settings().model_dir)
    app = create_app(settings=settings, queue=None)
    with TestClient(app) as client:
        response = client.post("/v1/moderate", json={"text": "hello from chat"})
        assert response.status_code == 200
        rejected = client.post("/v1/moderate/async", json={"text": "hello from chat"})
        assert rejected.status_code == 503
