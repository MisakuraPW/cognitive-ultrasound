"""Acceleration must obey numerical gates, resource limits and amortization."""

import pytest

from cognitive_ultrasound.belief_filter.preflight import choose
from cognitive_ultrasound.hardware import thread_candidates


def measurement(cold, warm, passed=True, headroom=True):
    return dict(cold_s=cold, median_s=warm, passed=passed, headroom_ok=headroom)


def test_short_run_avoids_compilation_long_run_uses_graph():
    results = [dict(threads=4, modes=dict(eager=measurement(1, 1), graph=measurement(20, 0.1)))]
    assert choose(results, 5)["execution"] == "eager"
    assert choose(results, 1000)["execution"] == "graph"


def test_fast_wrong_or_memory_unsafe_candidate_never_selected():
    results = [
        dict(
            threads=1,
            modes=dict(eager=measurement(1, 1), graph=measurement(1, 0.001, passed=False)),
        ),
        dict(threads=8, modes=dict(graph=measurement(1, 0.01, headroom=False))),
    ]
    assert choose(results, 1000)["execution"] == "eager"
    with pytest.raises(RuntimeError, match="No configuration"):
        choose(results[1:], 1000)


def test_threads_respect_container_quota():
    assert thread_candidates(1) == [1]
    assert thread_candidates(2) == [1, 2]
    assert thread_candidates(16) == [1, 4, 8]


def test_marginal_graph_timing_gain_keeps_reference():
    results = [dict(threads=4, modes=dict(eager=measurement(1, 1), graph=measurement(1, 0.95)))]
    assert choose(results, 1000)["execution"] == "eager"
