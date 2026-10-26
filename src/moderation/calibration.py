"""Threshold selection on a validation slice.

Block thresholds maximize recall subject to a precision floor, and they are
only eligible when enough comments would actually be blocked — a perfect
precision on three examples is not an operating point. Review thresholds
maximize precision subject to a recall floor, then get clamped under the
block threshold so the two bands stay ordered.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    precision: float
    recall: float
    f1: float
    predicted_positives: int
    support: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "threshold": self.threshold,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "predicted_positives": self.predicted_positives,
            "support": self.support,
        }


def confusion(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[int, int, int]:
    y = np.asarray(y_true).astype(bool)
    pred = np.asarray(scores) >= threshold
    tp = int(np.sum(pred & y))
    fp = int(np.sum(pred & ~y))
    fn = int(np.sum(~pred & y))
    return tp, fp, fn


def operating_point(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> OperatingPoint:
    y = np.asarray(y_true).astype(bool)
    tp, fp, fn = confusion(y, scores, threshold)
    predicted = tp + fp
    support = int(y.sum())
    precision = tp / predicted if predicted else 0.0
    recall = tp / support if support else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return OperatingPoint(float(threshold), precision, recall, f1, predicted, support)


def _candidates(scores: np.ndarray) -> np.ndarray:
    grid = np.linspace(0.05, 0.95, 181)
    if len(scores) == 0:
        return grid
    quantiles = np.quantile(scores, np.linspace(0.0, 1.0, 101))
    return np.unique(np.concatenate([grid, quantiles]))


def threshold_for_precision(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    min_precision: float,
    min_predicted: int = 20,
    default: float = 0.8,
) -> OperatingPoint:
    """Lowest threshold that still meets the precision floor (max recall)."""
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    support = int(y.sum())
    best: OperatingPoint | None = None
    best_f1: OperatingPoint | None = None
    for threshold in _candidates(s):
        point = operating_point(y, s, float(threshold))
        if point.predicted_positives < min_predicted:
            continue
        if best_f1 is None or point.f1 > best_f1.f1:
            best_f1 = point
        if point.precision + 1e-12 >= min_precision and (best is None or point.recall > best.recall + 1e-12):
            best = point
    if best is not None:
        return best
    if best_f1 is not None:
        return best_f1
    return operating_point(y, s, default) if support else OperatingPoint(default, 0.0, 0.0, 0.0, 0, 0)


def threshold_for_recall(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    min_recall: float,
    min_predicted: int = 20,
    default: float = 0.3,
) -> OperatingPoint:
    """Highest threshold that still meets the recall floor (max precision)."""
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    best: OperatingPoint | None = None
    for threshold in _candidates(s):
        point = operating_point(y, s, float(threshold))
        if point.predicted_positives < min_predicted:
            continue
        if point.recall + 1e-12 >= min_recall and (best is None or point.precision > best.precision + 1e-12):
            best = point
    if best is not None:
        return best
    return threshold_for_precision(y, s, min_precision=0.0, min_predicted=min_predicted, default=default)


def clamp_review_below_block(review: float, block: float) -> float:
    if review < block:
        return float(review)
    return float(max(0.05, min(review, block) * 0.75))
