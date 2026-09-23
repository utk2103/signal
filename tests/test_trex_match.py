from types import SimpleNamespace

from laya_mlx.trex.course import phase, shape
from laya_mlx.trex.engine import Spec
from laya_mlx.trex.match import Match


def test_round_counts_distance_across_deaths_and_keeps_match_wins():
    match = Match(["Laya", "Jev"], seconds=1, rounds=2)
    match.advance(1, [30, 20])
    match.advance(2, [0, 25])  # Laya restarts; past distance is retained.
    assert match.advance(60, [10, 30])
    assert match.last_result["points"] == [40, 30]
    assert match.wins == [1, 0]
    assert match.number == 2 and match.points == [0, 0]
    assert match.advance(120, [20, 40])
    assert match.finished and match.wins == [1, 1]
    assert match.state(120)["remaining"] == 0


def test_tied_round_does_not_award_a_win():
    match = Match(["Laya", "Jev"], seconds=1)
    match.advance(60, [20, 20])
    assert match.last_result["winner"] is None
    assert match.wins == [0, 0]


def test_course_phases_preserve_original_bird_speed_threshold():
    assert phase(SimpleNamespace(speed=8.4, obstacle_index=10))["name"] == "Sprint"
    assert phase(SimpleNamespace(speed=8.5, obstacle_index=10))["name"] == "Bird attack"
    assert phase(SimpleNamespace(speed=9.2, obstacle_index=10))["name"] == "Final challenge"
    game = SimpleNamespace(speed=6, obstacle_index=0)
    spec = shape(Spec("cactusLarge", size=3, gap=0), game)
    assert spec.size == 1 and spec.gap == 0.85


def test_equal_distance_uses_live_saves_and_resets_assist_baseline():
    match = Match(["Laya", "Jev"], seconds=1)
    match.advance(60, [20, 20], [5, 2])
    assert match.last_result["winner"] == "Jev"
    assert match.last_result["reason"] == "fewer live saves"
    match.advance(120, [20, 20], [6, 4])
    assert match.last_result["assists"] == [1, 2]
    assert match.last_result["winner"] == "Laya"
