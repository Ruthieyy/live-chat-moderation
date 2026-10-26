from __future__ import annotations

import numpy as np

from moderation.drift import (
    DEFAULT_EDGES,
    DriftMonitor,
    histogram,
    interpret_psi,
    population_stability_index,
)
from moderation.labels import PRODUCT_LABELS


def test_identical_distributions_have_zero_psi() -> None:
    counts = np.array([10, 20, 30, 40, 10, 5, 5, 5, 5, 5], dtype=np.float64)
    psi = population_stability_index(counts, counts.copy())
    assert abs(psi) < 1e-9


def test_shifted_distribution_is_significant() -> None:
    expected = np.zeros(10)
    actual = np.zeros(10)
    expected[0] = 1000
    actual[-1] = 1000
    psi = population_stability_index(expected, actual)
    assert psi > 0.25
    assert interpret_psi(psi) == "significant"
    assert interpret_psi(0.0) == "stable"
    assert interpret_psi(0.15) == "moderate"


def test_histogram_places_endpoints_in_the_end_bins() -> None:
    edges = np.asarray(DEFAULT_EDGES, dtype=np.float64)
    counts = histogram([0.0, 0.09, 0.95, 1.0], edges)
    assert counts[0] == 2
    assert counts[-1] == 2
    assert counts.sum() == 4


def test_monitor_flags_a_shift_once_the_window_fills() -> None:
    reference = {
        "edges": list(DEFAULT_EDGES),
        "sample_size": 1000,
        "counts": {label: [1000] + [0] * 9 for label in PRODUCT_LABELS},
    }
    monitor = DriftMonitor(reference, window=100, min_samples=20, update_every=1)
    for _ in range(25):
        monitor.observe({"toxic": 0.95, "hate": 0.95, "spam": 0.95})
    report = monitor.report()
    assert report["ready"] is True
    assert report["psi"]["toxic"] > 0.25
    assert report["status"]["hate"] == "significant"
