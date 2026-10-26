"""Prometheus metrics. Buckets are dense below 100ms because that is the SLO."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

REQUEST_LATENCY_BUCKETS = (
    0.001,
    0.002,
    0.005,
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.15,
    0.25,
    0.5,
    1.0,
    2.5,
)

INFERENCE_LATENCY_BUCKETS = (
    0.0002,
    0.0005,
    0.001,
    0.002,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
)

REQUEST_LATENCY = Histogram(
    "moderation_request_latency_seconds",
    "Handler latency, including tokenization, inference, and policy.",
    ["endpoint"],
    buckets=REQUEST_LATENCY_BUCKETS,
)
INFERENCE_LATENCY = Histogram(
    "moderation_inference_latency_seconds",
    "Per-comment ONNX Runtime session time.",
    ["endpoint"],
    buckets=INFERENCE_LATENCY_BUCKETS,
)
QUEUE_WAIT = Histogram(
    "moderation_queue_wait_seconds",
    "Time from enqueue until the worker starts the batch.",
    buckets=REQUEST_LATENCY_BUCKETS,
)
REQUESTS = Counter(
    "moderation_requests_total",
    "HTTP requests and worker batches.",
    ["endpoint", "status"],
)
COMMENTS = Counter(
    "moderation_comments_total",
    "Comments scored.",
    ["endpoint", "action"],
)
ERRORS = Counter(
    "moderation_errors_total",
    "Failed requests and batches.",
    ["endpoint", "type"],
)
DECISIONS = Counter(
    "moderation_decisions_total",
    "Moderation actions.",
    ["action"],
)
LABEL_FLAGS = Counter(
    "moderation_label_flags_total",
    "Decision reasons at the product-label level.",
    ["label"],
)
LABEL_SCORE = Histogram(
    "moderation_label_score",
    "Product score distribution.",
    ["label"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
QUEUE_DEPTH = Gauge(
    "moderation_queue_depth",
    "Redis stream length (approximate backlog).",
)
PSI = Gauge(
    "moderation_psi",
    "Population stability index versus the training reference.",
    ["label"],
)
BATCH_SIZE = Histogram(
    "moderation_worker_batch_size",
    "Micro-batch size on the async path.",
    buckets=(1, 2, 4, 8, 16, 32, 64),
)

_FLAG_LABELS = {"toxic", "hate", "threat", "spam"}


def observe_decision(
    *,
    endpoint: str,
    action: str,
    scores: dict[str, float],
    reasons: list[str],
    inference_s: float,
) -> None:
    COMMENTS.labels(endpoint=endpoint, action=action).inc()
    DECISIONS.labels(action=action).inc()
    INFERENCE_LATENCY.labels(endpoint=endpoint).observe(inference_s)
    for label, score in scores.items():
        LABEL_SCORE.labels(label=label).observe(score)
    for reason in reasons:
        if reason in _FLAG_LABELS:
            LABEL_FLAGS.labels(label=reason).inc()


def observe_psi(psi: dict[str, float]) -> None:
    for label, value in psi.items():
        PSI.labels(label=label).set(value)
