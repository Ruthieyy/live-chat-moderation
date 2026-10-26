from __future__ import annotations

import numpy as np

from moderation.calibration import (
    clamp_review_below_block,
    operating_point,
    threshold_for_precision,
    threshold_for_recall,
)


def test_precision_floor_picks_a_high_recall_threshold() -> None:
    y_true = np.array([0] * 50 + [1] * 50)
    scores = np.array([0.1] * 50 + [0.9] * 50)
    point = threshold_for_precision(y_true, scores, min_precision=0.9, min_predicted=10)
    assert point.precision >= 0.9
    assert point.recall >= 0.9
    assert point.predicted_positives >= 10


def test_recall_floor_stays_above_the_requested_recall() -> None:
    y_true = np.array([0] * 80 + [1] * 20)
    scores = np.concatenate([np.linspace(0.0, 0.4, 80), np.linspace(0.6, 1.0, 20)])
    point = threshold_for_recall(y_true, scores, min_recall=0.5, min_predicted=5)
    assert point.recall >= 0.5


def test_operating_point_counts() -> None:
    point = operating_point(np.array([0, 0, 1, 1]), np.array([0.1, 0.8, 0.7, 0.9]), 0.75)
    assert point.predicted_positives == 2
    assert point.support == 2
    assert point.precision == 0.5
    assert point.recall == 0.5


def test_review_threshold_is_clamped_under_block() -> None:
    assert clamp_review_below_block(0.2, 0.8) == 0.2
    assert clamp_review_below_block(0.9, 0.8) < 0.8
