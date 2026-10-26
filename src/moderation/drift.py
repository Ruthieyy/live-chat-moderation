"""Population Stability Index against the training reference distribution.

PSI is the standard batch drift check for a score: under 0.1 is noise, 0.1 to
0.25 is worth a look, above 0.25 means the live score mix has moved. The
monitor keeps a fixed window of recent product scores in-process. That is the
right scope for one replica; a fleet would aggregate the same histograms.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from moderation.labels import PRODUCT_LABELS

DEFAULT_EDGES: tuple[float, ...] = tuple(round(i / 10, 1) for i in range(11))


def histogram(scores: Iterable[float], edges: np.ndarray) -> np.ndarray:
    values = np.asarray(list(scores), dtype=np.float64)
    if values.size == 0:
        return np.zeros(len(edges) - 1, dtype=np.float64)
    clipped = np.clip(values, edges[0], np.nextafter(edges[-1], edges[0]))
    indexes = np.digitize(clipped, edges[1:-1], right=False)
    return np.bincount(indexes, minlength=len(edges) - 1).astype(np.float64)


def population_stability_index(
    expected_counts: np.ndarray,
    actual_counts: np.ndarray,
    *,
    eps: float = 1e-4,
) -> float:
    expected = np.asarray(expected_counts, dtype=np.float64) + eps
    actual = np.asarray(actual_counts, dtype=np.float64) + eps
    expected /= expected.sum()
    actual /= actual.sum()
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def interpret_psi(value: float) -> str:
    if value < 0.10:
        return "stable"
    if value < 0.25:
        return "moderate"
    return "significant"


class DriftMonitor:
    def __init__(
        self,
        reference: dict,
        *,
        window: int = 2000,
        min_samples: int = 200,
        update_every: int = 25,
    ) -> None:
        counts = reference["counts"]
        unknown = set(counts) - set(PRODUCT_LABELS)
        if unknown:
            raise ValueError(f"unexpected drift labels: {sorted(unknown)}")
        self.reference = reference
        self.edges = np.asarray(reference["edges"], dtype=np.float64)
        self.window = window
        self.min_samples = min_samples
        self.update_every = update_every
        self._samples: dict[str, deque[float]] = {label: deque(maxlen=window) for label in counts}
        self._seen = 0
        self._lock = threading.Lock()
        self.psi = {label: 0.0 for label in counts}
        self.ready = False

    @classmethod
    def from_model_dir(
        cls,
        model_dir: Path,
        *,
        window: int,
        min_samples: int,
    ) -> DriftMonitor | None:
        path = model_dir / "reference.json"
        if not path.exists():
            return None
        reference = json.loads(path.read_text(encoding="utf-8"))
        return cls(reference, window=window, min_samples=min_samples)

    def observe(self, scores: dict[str, float]) -> dict[str, float]:
        with self._lock:
            self._seen += 1
            for label, series in self._samples.items():
                if label in scores:
                    series.append(float(scores[label]))
            filled = len(next(iter(self._samples.values()))) if self._samples else 0
            if filled >= self.min_samples and self._seen % self.update_every == 0:
                self.psi = self._compute_unlocked()
                self.ready = True
            return dict(self.psi)

    def _compute_unlocked(self) -> dict[str, float]:
        report: dict[str, float] = {}
        for label, series in self._samples.items():
            actual = histogram(series, self.edges)
            expected = np.asarray(self.reference["counts"][label], dtype=np.float64)
            report[label] = population_stability_index(expected, actual)
        return report

    def report(self) -> dict:
        with self._lock:
            filled = len(next(iter(self._samples.values()))) if self._samples else 0
            psi = dict(self.psi)
            ready = self.ready
        return {
            "ready": ready,
            "samples": filled,
            "window": self.window,
            "min_samples": self.min_samples,
            "psi": psi,
            "status": {label: interpret_psi(value) for label, value in psi.items()},
            "reference_size": self.reference.get("sample_size"),
        }
