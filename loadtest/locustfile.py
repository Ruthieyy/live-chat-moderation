"""Load test for the sync moderation path, with a little async traffic mixed in."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

from locust import HttpUser, between, task

_COMMENTS = [
    json.loads(line)["text"]
    for line in Path(__file__).with_name("comments.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]


class ChatUser(HttpUser):
    wait_time = between(0.01, 0.05)

    @task(10)
    def sync_moderate(self) -> None:
        self.client.post(
            "/v1/moderate",
            json={"text": random.choice(_COMMENTS), "room_id": "live-1"},
            name="/v1/moderate",
        )

    @task(2)
    def batch_moderate(self) -> None:
        comments = [{"text": random.choice(_COMMENTS), "room_id": "live-1"} for _ in range(4)]
        self.client.post(
            "/v1/moderate/batch",
            json={"comments": comments},
            name="/v1/moderate/batch",
        )

    @task(1)
    def async_moderate(self) -> None:
        response = self.client.post(
            "/v1/moderate/async",
            json={"text": random.choice(_COMMENTS), "room_id": "live-1"},
            name="/v1/moderate/async",
        )
        if response.status_code != 202:
            return
        job_id = response.json()["job_id"]
        deadline = time.perf_counter() + 2.0
        while time.perf_counter() < deadline:
            status = self.client.get(f"/v1/jobs/{job_id}", name="/v1/jobs/[id]")
            if status.status_code == 200 and status.json().get("status") == "done":
                return
            time.sleep(0.02)
