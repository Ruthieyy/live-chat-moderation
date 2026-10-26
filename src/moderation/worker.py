"""Async micro-batch worker.

The sync API scores one comment at a time so a viewer sees a decision inside
the latency budget. This process drains the Redis stream, waits a few
milliseconds to coalesce a batch, and runs one ONNX call for the group.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import threading
import time
from typing import Any

import redis
from prometheus_client import start_http_server

from moderation.inference import InferenceEngine
from moderation.logsetup import configure_logging
from moderation.metrics import (
    BATCH_SIZE,
    ERRORS,
    QUEUE_DEPTH,
    QUEUE_WAIT,
    REQUESTS,
    observe_decision,
    observe_psi,
)
from moderation.queue import JobQueue, connect_queue
from moderation.schemas import Attributes, ModerationResponse, Scores
from moderation.settings import load_settings

logger = logging.getLogger("moderation.worker")


def collect_batch(
    queue: JobQueue,
    consumer: str,
    *,
    batch_size: int,
    block_ms: int,
    coalesce_ms: int,
) -> list[tuple[str, dict[str, Any]]]:
    first = queue.read_new(consumer, count=batch_size, block_ms=block_ms)
    if not first or len(first) >= batch_size or coalesce_ms <= 0:
        return first
    rest = queue.read_new(consumer, count=batch_size - len(first), block_ms=coalesce_ms)
    return first + rest


def handle_batch(
    engine: InferenceEngine,
    queue: JobQueue,
    messages: list[tuple[str, dict[str, Any]]],
    drift: Any,
) -> int:
    if not messages:
        return 0
    now = time.time()
    texts = [str(payload["text"]) for _, payload in messages]
    try:
        predictions, inference_s = engine.predict(texts)
    except Exception:
        logger.exception("worker inference failed")
        ERRORS.labels(endpoint="worker", type="inference").inc()
        REQUESTS.labels(endpoint="worker", status="500").inc()
        acked: list[str] = []
        for message_id, payload in messages:
            try:
                queue.store_result(
                    str(payload["job_id"]),
                    {
                        "job_id": payload["job_id"],
                        "status": "error",
                        "result": None,
                        "error": "inference failed",
                        "queue_ms": _queue_ms(payload, now),
                    },
                )
                acked.append(message_id)
            except redis.RedisError:
                logger.exception("failed to store error result")
        if acked:
            queue.ack(acked)
        return 0

    per_example = inference_s / len(predictions)
    acked = []
    for (message_id, payload), prediction in zip(messages, predictions, strict=True):
        decision = prediction.decision
        response = ModerationResponse(
            comment_id=payload.get("comment_id"),
            room_id=payload.get("room_id"),
            action=decision.action,
            scores=Scores(**{key: round(value, 6) for key, value in decision.scores.items()}),
            attributes=Attributes(**{key: round(value, 6) for key, value in prediction.attributes.items()}),
            reasons=decision.reasons,
            model_version=engine.model_version,
            inference_ms=round(per_example * 1000, 3),
            truncated=prediction.truncated,
        )
        body = {
            "job_id": payload["job_id"],
            "status": "done",
            "result": response.model_dump(),
            "error": None,
            "queue_ms": _queue_ms(payload, now),
        }
        try:
            queue.store_result(str(payload["job_id"]), body)
        except redis.RedisError:
            logger.exception("failed to store result for %s", payload.get("job_id"))
            continue
        acked.append(message_id)
        observe_decision(
            endpoint="worker",
            action=response.action,
            scores=response.scores.model_dump(),
            reasons=response.reasons,
            inference_s=per_example,
        )
        if drift is not None:
            observe_psi(drift.observe(response.scores.model_dump()))
        queue_ms = body["queue_ms"]
        if queue_ms is not None:
            QUEUE_WAIT.observe(queue_ms / 1000)
    if acked:
        queue.ack(acked)
    BATCH_SIZE.observe(len(messages))
    REQUESTS.labels(endpoint="worker", status="200").inc()
    return len(acked)


def _queue_ms(payload: dict[str, Any], now: float) -> float | None:
    enqueued = payload.get("enqueued_at")
    if enqueued is None:
        return None
    return round((now - float(enqueued)) * 1000, 3)


def main() -> None:
    settings = load_settings()
    configure_logging(settings.log_level)
    stop = threading.Event()

    def _handle_signal(signum: int, _frame: Any) -> None:
        logger.info("signal %s, shutting down", signum)
        stop.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    start_http_server(settings.worker_metrics_port)
    engine = InferenceEngine.from_model_dir(
        settings.model_dir,
        intra_op_threads=settings.ort_intra_op_threads,
        max_concurrent=settings.inference_slots,
    )
    engine.warmup()
    from moderation.drift import DriftMonitor

    drift = DriftMonitor.from_model_dir(
        settings.model_dir,
        window=settings.drift_window,
        min_samples=settings.drift_min_samples,
    )
    queue = connect_queue(
        settings.redis_url,
        result_ttl_seconds=settings.result_ttl_seconds,
        socket_timeout=None,
    )
    consumer = settings.worker_consumer or f"{socket.gethostname()}-{os.getpid()}"
    logger.info("worker online consumer=%s batch=%s", consumer, settings.worker_batch_size)
    idle_loops = 0
    while not stop.is_set():
        if idle_loops % 10 == 0:
            try:
                stale = queue.claim_stale(consumer, settings.worker_claim_idle_ms, settings.worker_batch_size)
                if stale:
                    handle_batch(engine, queue, stale, drift)
                    idle_loops = 0
                    continue
            except redis.RedisError:
                logger.exception("claim failed")
        try:
            QUEUE_DEPTH.set(queue.depth())
            batch = collect_batch(
                queue,
                consumer,
                batch_size=settings.worker_batch_size,
                block_ms=settings.worker_block_ms,
                coalesce_ms=settings.worker_batch_wait_ms,
            )
        except redis.RedisError:
            logger.exception("read failed")
            time.sleep(0.5)
            continue
        if not batch:
            idle_loops += 1
            continue
        idle_loops = 0
        handle_batch(engine, queue, batch, drift)
    queue.close()
    logger.info("worker stopped")


if __name__ == "__main__":
    main()
