"""Redis Streams job queue.

Streams give the async path a consumer group, acknowledgements, and a way to
reclaim a message whose worker died mid-batch. Results live on a separate key
with a TTL so the API can poll without scanning the stream.
"""

from __future__ import annotations

import json
from typing import Any

import redis

STREAM = "moderation:jobs"
GROUP = "moderation-workers"
RESULT_PREFIX = "moderation:result:"


class JobQueue:
    def __init__(self, client: redis.Redis, *, result_ttl_seconds: int) -> None:
        self.client = client
        self.result_ttl_seconds = result_ttl_seconds

    def close(self) -> None:
        self.client.close()

    def ensure_group(self) -> None:
        try:
            self.client.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
        except redis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    def enqueue(self, payload: dict[str, Any]) -> str:
        job_id = str(payload["job_id"])
        self.client.xadd(STREAM, {"data": json.dumps(payload)})
        self.store_result(
            job_id,
            {"job_id": job_id, "status": "queued", "result": None, "error": None},
        )
        return job_id

    def store_result(self, job_id: str, body: dict[str, Any]) -> None:
        self.client.set(
            RESULT_PREFIX + job_id,
            json.dumps(body),
            ex=self.result_ttl_seconds,
        )

    def get_result(self, job_id: str) -> dict[str, Any] | None:
        raw = self.client.get(RESULT_PREFIX + job_id)
        if raw is None:
            return None
        return json.loads(raw)

    def read_new(self, consumer: str, *, count: int, block_ms: int) -> list[tuple[str, dict[str, Any]]]:
        response = self.client.xreadgroup(
            GROUP,
            consumer,
            {STREAM: ">"},
            count=count,
            block=block_ms,
        )
        return _parse(response)

    def ack(self, message_ids: list[str]) -> None:
        if message_ids:
            self.client.xack(STREAM, GROUP, *message_ids)

    def depth(self) -> int:
        return int(self.client.xlen(STREAM))

    def claim_stale(self, consumer: str, min_idle_ms: int, count: int) -> list[tuple[str, dict[str, Any]]]:
        claimed = self.client.xautoclaim(
            STREAM,
            GROUP,
            consumer,
            min_idle_time=min_idle_ms,
            start_id="0-0",
            count=count,
        )
        messages = claimed[1] if isinstance(claimed, (list, tuple)) and len(claimed) >= 2 else []
        return _parse([(STREAM, messages)])


def connect_queue(url: str, *, result_ttl_seconds: int, socket_timeout: float | None) -> JobQueue:
    client = redis.Redis.from_url(
        url,
        decode_responses=True,
        socket_connect_timeout=0.5,
        socket_timeout=socket_timeout,
        health_check_interval=30,
    )
    client.ping()
    queue = JobQueue(client, result_ttl_seconds=result_ttl_seconds)
    queue.ensure_group()
    return queue


def _parse(response: Any) -> list[tuple[str, dict[str, Any]]]:
    if not response:
        return []
    parsed: list[tuple[str, dict[str, Any]]] = []
    for _stream, messages in response:
        for message_id, fields in messages:
            raw = fields.get("data")
            if not raw:
                continue
            payload = json.loads(raw)
            parsed.append((str(message_id), payload))
    return parsed
