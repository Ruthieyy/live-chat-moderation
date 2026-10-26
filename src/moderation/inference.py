"""ONNX Runtime scorer. One session, shared across FastAPI worker threads."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from moderation.labels import MODEL_LABELS
from moderation.policy import Decision, Thresholds, decide
from moderation.spam import score_spam
from moderation.tokenize import Vocabulary

WARMUP_TEXTS = (
    "hello from the livestream",
    "thanks for joining the chat",
    "this is a warmup comment",
    "great stream today",
)


@dataclass(frozen=True)
class Prediction:
    attributes: dict[str, float]
    decision: Decision
    truncated: bool


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


class InferenceEngine:
    def __init__(
        self,
        session: ort.InferenceSession,
        vocabulary: Vocabulary,
        thresholds: Thresholds,
        model_version: str,
        label_names: tuple[str, ...],
        max_concurrent: int,
    ) -> None:
        self.session = session
        self.vocabulary = vocabulary
        self.thresholds = thresholds
        self.model_version = model_version
        self.label_names = label_names
        self._slots = threading.Semaphore(max_concurrent)
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name

    @classmethod
    def from_model_dir(cls, model_dir: Path, *, intra_op_threads: int, max_concurrent: int) -> InferenceEngine:
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        vocabulary = Vocabulary.load(model_dir / "vocab.json")
        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            str(model_dir / config["onnx_file"]),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        labels = tuple(config.get("labels", MODEL_LABELS))
        return cls(
            session=session,
            vocabulary=vocabulary,
            thresholds=Thresholds.from_dict(config["thresholds"]),
            model_version=str(config["model_version"]),
            label_names=labels,
            max_concurrent=max_concurrent,
        )

    def warmup(self, rounds: int = 8) -> None:
        texts = list(WARMUP_TEXTS)
        for _ in range(rounds):
            self.predict(texts)

    def predict(self, texts: list[str]) -> tuple[list[Prediction], float]:
        encoded, truncated = self.vocabulary.encode_batch(texts)
        ids = np.asarray(encoded, dtype=np.int64)
        started = time.perf_counter()
        with self._slots:
            logits = self.session.run([self.output_name], {self.input_name: ids})[0]
        elapsed = time.perf_counter() - started
        probs = _sigmoid(np.asarray(logits, dtype=np.float64))
        predictions: list[Prediction] = []
        for row, text, was_truncated in zip(probs, texts, truncated, strict=True):
            attributes = {name: float(row[index]) for index, name in enumerate(self.label_names)}
            spam_score, spam_reasons = score_spam(text)
            decision = decide(attributes, spam_score, spam_reasons, self.thresholds)
            predictions.append(Prediction(attributes=attributes, decision=decision, truncated=was_truncated))
        return predictions, elapsed
