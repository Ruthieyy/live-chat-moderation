from moderation.metrics import REQUEST_LATENCY_BUCKETS


def test_latency_buckets_include_the_100ms_slo() -> None:
    assert 0.1 in REQUEST_LATENCY_BUCKETS
    assert list(REQUEST_LATENCY_BUCKETS) == sorted(REQUEST_LATENCY_BUCKETS)
