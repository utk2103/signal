"""Diagnostics distinguish proved-safe plans from best-effort fallback."""

from types import SimpleNamespace

from laya_mlx.trex.pilot import Decision, Stats


def test_summary_includes_planning_latency_and_best_effort_count():
    stats = Stats()
    for seq, (elapsed, robust) in enumerate([(1.0, True), (3.0, False), (5.0, True)]):
        stats.record(Decision(seq, 0, 0, 0.0, plan_ms=elapsed, robust=robust))
    game = SimpleNamespace(deaths=0, score=10, crashed=False)
    brain = SimpleNamespace(model="test", detail="test")
    report = stats.summary(game, brain)
    assert report["planner_ms_p50"] == 3.0
    assert report["planner_ms_p95"] == 5.0
    assert report["best_effort_decisions"] == 1


def test_empty_summary_has_no_planning_latency():
    report = Stats().summary(
        SimpleNamespace(deaths=0, score=0, crashed=False),
        SimpleNamespace(model="test", detail="test"),
    )
    assert report["planner_ms_p50"] is None
    assert report["planner_ms_p95"] is None
    assert report["best_effort_decisions"] == 0
