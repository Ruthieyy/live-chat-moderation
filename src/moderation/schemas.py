"""Request and response bodies."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from moderation.labels import SCHEMA_MAX_TEXT_CHARS

Action = Literal["allow", "review", "block"]


class ModerateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=SCHEMA_MAX_TEXT_CHARS)
    comment_id: str | None = Field(default=None, max_length=64)
    room_id: str | None = Field(default=None, max_length=64)

    @field_validator("text")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("text is empty")
        return stripped


class BatchRequest(BaseModel):
    comments: list[ModerateRequest] = Field(min_length=1, max_length=64)


class Scores(BaseModel):
    toxic: float
    hate: float
    spam: float


class Attributes(BaseModel):
    toxic: float
    severe_toxic: float
    obscene: float
    threat: float
    insult: float
    identity_hate: float


class ModerationResponse(BaseModel):
    comment_id: str | None = None
    room_id: str | None = None
    action: Action
    scores: Scores
    attributes: Attributes
    reasons: list[str]
    model_version: str
    inference_ms: float
    truncated: bool


class AsyncAccepted(BaseModel):
    job_id: str
    status: Literal["queued"]


class JobResponse(BaseModel):
    job_id: str
    status: Literal["queued", "done", "error"]
    result: ModerationResponse | None = None
    error: str | None = None
    queue_ms: float | None = None
