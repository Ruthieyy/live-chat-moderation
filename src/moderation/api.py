"""FastAPI application: a sync low-latency path and an async enqueue path."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import redis
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from moderation import __version__
from moderation.inference import InferenceEngine, Prediction
from moderation.logsetup import configure_logging
from moderation.metrics import (
    ERRORS,
    QUEUE_DEPTH,
    REQUEST_LATENCY,
    REQUESTS,
    observe_decision,
    observe_psi,
)
from moderation.queue import JobQueue, connect_queue
from moderation.schemas import (
    AsyncAccepted,
    Attributes,
    BatchRequest,
    JobResponse,
    ModerateRequest,
    ModerationResponse,
    Scores,
)
from moderation.settings import Settings, load_settings, resolve_port

logger = logging.getLogger("moderation.api")


def create_app(
    settings: Settings | None = None,
    *,
    engine: InferenceEngine | None = None,
    queue: JobQueue | None = None,
) -> FastAPI:
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if app.state.engine is None:
            app.state.engine = InferenceEngine.from_model_dir(
                settings.model_dir,
                intra_op_threads=settings.ort_intra_op_threads,
                max_concurrent=settings.inference_slots,
            )
            app.state.engine.warmup()
            logger.info(
                "model loaded version=%s slots=%s",
                app.state.engine.model_version,
                settings.inference_slots,
            )
        from moderation.drift import DriftMonitor

        app.state.drift = DriftMonitor.from_model_dir(
            settings.model_dir,
            window=settings.drift_window,
            min_samples=settings.drift_min_samples,
        )
        if queue is None and settings.redis_enabled:
            try:
                app.state.queue = connect_queue(
                    settings.redis_url,
                    result_ttl_seconds=settings.result_ttl_seconds,
                    socket_timeout=2.0,
                )
                logger.info("redis connected")
            except redis.RedisError:
                logger.warning("redis unavailable; async moderation disabled")
                app.state.queue = None
        yield
        active = app.state.queue
        if active is not None:
            active.close()

    app = FastAPI(
        title="Live Chat Moderation",
        version=__version__,
        summary="Real-time toxicity, hate, and spam scoring for livestream chat.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.engine = engine
    app.state.queue = queue
    app.state.drift = None

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        endpoint = _route_name(request.url.path)
        ERRORS.labels(endpoint=endpoint, type="validation").inc()
        REQUESTS.labels(endpoint=endpoint, status="422").inc()
        return JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors())})

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, HTTPException):
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        endpoint = _route_name(request.url.path)
        ERRORS.labels(endpoint=endpoint, type="unhandled").inc()
        REQUESTS.labels(endpoint=endpoint, status="500").inc()
        logger.exception("unhandled error")
        return JSONResponse(status_code=500, content={"detail": "internal error"})

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict[str, Any]:
        loaded = app.state.engine is not None
        if not loaded:
            raise HTTPException(status_code=503, detail="model not loaded")
        return {
            "status": "ready",
            "model_version": app.state.engine.model_version,
            "redis": app.state.queue is not None,
        }

    @app.get("/v1/model")
    def model_info() -> dict[str, Any]:
        config_path = settings.model_dir / "config.json"
        import json

        config = json.loads(config_path.read_text(encoding="utf-8"))
        return {
            "model_version": config["model_version"],
            "onnx_file": config["onnx_file"],
            "labels": config["labels"],
            "thresholds": config["thresholds"],
            "dataset": config.get("dataset"),
            "metrics": config.get("metrics"),
            "max_len": config.get("max_len"),
            "max_tokens": config.get("max_tokens"),
        }

    @app.get("/v1/drift")
    def drift() -> dict[str, Any]:
        monitor = app.state.drift
        if monitor is None:
            return {"ready": False, "detail": "no reference distribution"}
        return monitor.report()

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/moderate", response_model=ModerationResponse)
    def moderate(body: ModerateRequest) -> ModerationResponse:
        started = time.perf_counter()
        _check_length(body.text, settings, "sync")
        try:
            prediction, inference_s = _score_one(app, body.text)
        except Exception:
            ERRORS.labels(endpoint="sync", type="inference").inc()
            REQUESTS.labels(endpoint="sync", status="500").inc()
            logger.exception("inference failed")
            raise HTTPException(status_code=500, detail="inference failed") from None
        response = _response(app, prediction, body, inference_s)
        _record(app, "sync", response, inference_s, time.perf_counter() - started, len(body.text))
        REQUESTS.labels(endpoint="sync", status="200").inc()
        return response

    @app.post("/v1/moderate/batch", response_model=list[ModerationResponse])
    def moderate_batch(body: BatchRequest) -> list[ModerationResponse]:
        started = time.perf_counter()
        if len(body.comments) > settings.sync_batch_limit:
            ERRORS.labels(endpoint="batch", type="validation").inc()
            REQUESTS.labels(endpoint="batch", status="422").inc()
            raise HTTPException(
                status_code=422,
                detail=f"batch limit is {settings.sync_batch_limit}",
            )
        for comment in body.comments:
            _check_length(comment.text, settings, "batch")
        engine: InferenceEngine = app.state.engine
        try:
            predictions, inference_s = engine.predict([comment.text for comment in body.comments])
        except Exception:
            ERRORS.labels(endpoint="batch", type="inference").inc()
            REQUESTS.labels(endpoint="batch", status="500").inc()
            logger.exception("batch inference failed")
            raise HTTPException(status_code=500, detail="inference failed") from None
        per_example = inference_s / len(predictions)
        responses = [
            _response(app, prediction, comment, per_example)
            for prediction, comment in zip(predictions, body.comments, strict=True)
        ]
        elapsed = time.perf_counter() - started
        REQUEST_LATENCY.labels(endpoint="batch").observe(elapsed)
        for response in responses:
            _record_comment(app, "batch", response, per_example)
        REQUESTS.labels(endpoint="batch", status="200").inc()
        return responses

    @app.post("/v1/moderate/async", response_model=AsyncAccepted, status_code=202)
    def moderate_async(body: ModerateRequest) -> AsyncAccepted:
        started = time.perf_counter()
        _check_length(body.text, settings, "async")
        job_queue: JobQueue | None = app.state.queue
        if job_queue is None:
            ERRORS.labels(endpoint="async", type="redis").inc()
            REQUESTS.labels(endpoint="async", status="503").inc()
            raise HTTPException(status_code=503, detail="queue unavailable")
        job_id = uuid.uuid4().hex
        payload = {
            "job_id": job_id,
            "text": body.text,
            "comment_id": body.comment_id,
            "room_id": body.room_id,
            "enqueued_at": time.time(),
        }
        try:
            job_queue.enqueue(payload)
            depth = job_queue.depth()
        except redis.RedisError:
            ERRORS.labels(endpoint="async", type="redis").inc()
            REQUESTS.labels(endpoint="async", status="503").inc()
            logger.exception("enqueue failed")
            raise HTTPException(status_code=503, detail="queue unavailable") from None
        QUEUE_DEPTH.set(depth)
        REQUEST_LATENCY.labels(endpoint="async").observe(time.perf_counter() - started)
        REQUESTS.labels(endpoint="async", status="202").inc()
        return AsyncAccepted(job_id=job_id, status="queued")

    @app.get("/v1/jobs/{job_id}", response_model=JobResponse)
    def job_status(job_id: str) -> JobResponse:
        if not _valid_job_id(job_id):
            raise HTTPException(status_code=422, detail="invalid job id")
        job_queue: JobQueue | None = app.state.queue
        if job_queue is None:
            raise HTTPException(status_code=503, detail="queue unavailable")
        try:
            body = job_queue.get_result(job_id)
        except redis.RedisError:
            raise HTTPException(status_code=503, detail="queue unavailable") from None
        if body is None:
            raise HTTPException(status_code=404, detail="job not found")
        return JobResponse(**body)

    return app


def _valid_job_id(job_id: str) -> bool:
    if not 8 <= len(job_id) <= 64:
        return False
    return all(char.isalnum() or char in {"-", "_"} for char in job_id)


def _check_length(text: str, settings: Settings, endpoint: str) -> None:
    if len(text) > settings.max_text_chars:
        ERRORS.labels(endpoint=endpoint, type="validation").inc()
        REQUESTS.labels(endpoint=endpoint, status="422").inc()
        raise HTTPException(
            status_code=422,
            detail=f"text exceeds {settings.max_text_chars} characters",
        )


def _score_one(app: FastAPI, text: str) -> tuple[Prediction, float]:
    engine: InferenceEngine = app.state.engine
    predictions, elapsed = engine.predict([text])
    return predictions[0], elapsed


def _response(
    app: FastAPI,
    prediction: Prediction,
    request: ModerateRequest,
    inference_s: float,
) -> ModerationResponse:
    decision = prediction.decision
    return ModerationResponse(
        comment_id=request.comment_id,
        room_id=request.room_id,
        action=decision.action,
        scores=Scores(**{key: round(value, 6) for key, value in decision.scores.items()}),
        attributes=Attributes(**{key: round(value, 6) for key, value in prediction.attributes.items()}),
        reasons=decision.reasons,
        model_version=app.state.engine.model_version,
        inference_ms=round(inference_s * 1000, 3),
        truncated=prediction.truncated,
    )


def _record(
    app: FastAPI,
    endpoint: str,
    response: ModerationResponse,
    inference_s: float,
    elapsed: float,
    text_len: int,
) -> None:
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)
    _record_comment(app, endpoint, response, inference_s)
    # Debug, not info: a log line per comment moves p99 more than the model does.
    logger.debug(
        "moderated endpoint=%s action=%s inference_ms=%.3f total_ms=%.3f text_len=%s reasons=%s",
        endpoint,
        response.action,
        response.inference_ms,
        elapsed * 1000,
        text_len,
        ",".join(response.reasons) or "-",
    )


def _record_comment(app: FastAPI, endpoint: str, response: ModerationResponse, inference_s: float) -> None:
    scores = response.scores.model_dump()
    observe_decision(
        endpoint=endpoint,
        action=response.action,
        scores=scores,
        reasons=response.reasons,
        inference_s=inference_s,
    )
    monitor = app.state.drift
    if monitor is not None:
        observe_psi(monitor.observe(scores))


def _route_name(path: str) -> str:
    if path.startswith("/v1/moderate/batch"):
        return "batch"
    if path.startswith("/v1/moderate/async"):
        return "async"
    if path.startswith("/v1/moderate"):
        return "sync"
    return "other"


def main() -> None:
    settings = load_settings()
    configure_logging(settings.log_level)
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=resolve_port(settings),
        access_log=settings.access_log,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
