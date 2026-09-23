"""Answers must still be relevant when they land, not just when requested."""

import time
from types import SimpleNamespace

from laya_mlx.trex.pilot import Decision, Pilot


def idle_pilot(monkeypatch):
    monkeypatch.setattr(Pilot, "work", lambda self: None)
    brain = SimpleNamespace(name="test", inflight=2, usd_per_token=0, close=lambda: None)
    pilot = Pilot(brain)
    pilot.game.press_jump()
    return pilot


def test_older_answer_cannot_override_newer_applied_answer(monkeypatch):
    pilot = idle_pilot(monkeypatch)
    for seq, action in [(2, "run"), (1, "duck")]:
        pilot.results.put(
            Decision(
                seq, pilot.game.run_index, 0, time.perf_counter(), executed=action, proposed=action
            )
        )
        pilot.collect(1)
    assert pilot.held == "run"
    assert pilot.stats.discarded == 1


def test_failed_pending_answer_is_not_treated_as_applied(monkeypatch):
    pilot = idle_pilot(monkeypatch)
    pilot.results.put(
        Decision(
            1,
            pilot.game.run_index,
            0,
            time.perf_counter(),
            executed="jump",
            error="network timeout",
        )
    )
    pilot.collect(1)
    dependent = Decision(
        2,
        pilot.game.run_index,
        0,
        time.perf_counter(),
        pending_seq=1,
        pending="jump",
        executed="duck",
    )
    assert not pilot.premise_holds(dependent)


def imminent_cactus(pilot):
    from laya_mlx.trex.engine import Obstacle, Spec

    for _ in range(300):
        pilot.game.step()
    obstacle = Obstacle(Spec("cactusSmall"), pilot.game.speed)
    obstacle.x = pilot.game.trex.x + 100
    pilot.game.obstacles = [obstacle]


def test_late_answer_rechecked_against_live_physics(monkeypatch):
    pilot = idle_pilot(monkeypatch)
    imminent_cactus(pilot)
    pilot.results.put(
        Decision(
            1, pilot.game.run_index, 0, time.perf_counter(), executed="run", expected_first=(0, 2)
        )
    )
    pilot.collect(51)
    assert pilot.game.trex.jumping
    assert pilot.stats.arrival_saves == 1


def test_emergency_protects_stalled_model_but_unassisted_does_not(monkeypatch):
    for guarded in (True, False):
        pilot = idle_pilot(monkeypatch)
        pilot.guarded = guarded
        imminent_cactus(pilot)
        pilot.emergency(51)
        assert pilot.game.trex.jumping == guarded
        assert pilot.stats.emergency_saves == int(guarded)


def test_timing_adapts_to_slowdown_and_stays_bounded(monkeypatch):
    pilot = idle_pilot(monkeypatch)
    for _ in range(80):
        pilot.observe_timing(1, 1)
    pilot.observe_timing(30, 31)
    assert pilot.latency_frames >= 20
    pilot.observe_timing(1000, 1031)
    assert pilot.stagger <= 12
    assert pilot.longest_wait <= 24
    pilot.reset_run()
    assert len(pilot.observed) == 0


def test_crash_trace_keeps_only_two_seconds_and_is_serializable(monkeypatch):
    import json

    pilot = idle_pilot(monkeypatch)
    for frame in range(200):
        pilot.capture_tick(frame)
    crash = pilot.on_crash(199)
    assert len(crash["trace"]) == 120
    assert crash["trace"][0]["frame"] == 80
    json.dumps(crash)


def test_emergency_jump_actually_clears_obstacle(monkeypatch):
    pilot = idle_pilot(monkeypatch)
    imminent_cactus(pilot)
    pilot.emergency(51)
    for frame in range(52, 94):
        pilot.emergency(frame)
        pilot.enforce()
        pilot.game.step()
        assert not pilot.game.crashed
    assert pilot.stats.emergency_saves == 1


def test_changed_arrival_action_invalidates_dependent_plan(monkeypatch):
    pilot = idle_pilot(monkeypatch)
    imminent_cactus(pilot)
    pilot.results.put(Decision(1, pilot.game.run_index, 0, time.perf_counter(), executed="run"))
    pilot.collect(51)
    dependent = Decision(
        2,
        pilot.game.run_index,
        50,
        time.perf_counter(),
        pending_seq=1,
        pending="run",
        executed="duck",
    )
    assert not pilot.premise_holds(dependent)


def test_unassisted_arrival_does_not_change_model_action(monkeypatch):
    pilot = idle_pilot(monkeypatch)
    pilot.guarded = False
    imminent_cactus(pilot)
    pilot.results.put(Decision(1, pilot.game.run_index, 0, time.perf_counter(), executed="run"))
    pilot.collect(51)
    assert not pilot.game.trex.jumping
    assert pilot.stats.arrival_saves == 0


def test_stalled_model_emergency_survives_three_seeded_courses(monkeypatch):
    for seed in (0, 1, 2):
        pilot = idle_pilot(monkeypatch)
        from laya_mlx.trex.engine import Game

        pilot.game = Game(seed)
        pilot.game.press_jump()
        for frame in range(6000):
            pilot.emergency(frame)
            pilot.enforce()
            pilot.game.step()
            assert not pilot.game.crashed, (seed, frame)
        assert pilot.game.score > 1200
        assert pilot.stats.emergency_saves > 0


def test_spectator_shows_emergency_action_not_previous_model_choice(monkeypatch):
    pilot = idle_pilot(monkeypatch)
    monkeypatch.setattr("laya_mlx.trex.pilot.protect", lambda snap, action: "jump")
    pilot.emergency(10)
    assert pilot.spectator(10)["last_action"] == "jump"


def test_salient_notice_persists_but_every_answer_is_logged(monkeypatch):
    pilot = idle_pilot(monkeypatch)
    pilot.note(10, "Arrival shield saved it")
    pilot.note(11, "Answer → run")
    assert pilot.spectator(11)["event"] == "Arrival shield saved it"
    assert pilot.events[-1]["event"] == "Answer → run"
    pilot.note(70, "Answer → duck")
    assert pilot.spectator(70)["event"] == "Answer → duck"


def test_successful_stale_run_answer_still_counts_billed_tokens(monkeypatch):
    pilot = idle_pilot(monkeypatch)
    pilot.brain.usd_per_token = 0.001
    pilot.results.put(
        Decision(1, pilot.game.run_index - 1, 0, time.perf_counter(), input_tokens=20)
    )
    pilot.collect(5)
    assert pilot.stats.tokens == 20
    assert pilot.stats.cost == 0.02
    assert pilot.stats.discarded == 1


def test_routine_discard_does_not_replace_shield_notice(monkeypatch):
    pilot = idle_pilot(monkeypatch)
    pilot.note(10, "Emergency shield → jump")
    pilot.note(11, "Late answer discarded")
    assert pilot.spectator(11)["event"] == "Emergency shield → jump"


def test_repeated_discard_notices_do_not_flash_every_frame(monkeypatch):
    pilot = idle_pilot(monkeypatch)
    pilot.note(100, "Late answer discarded")
    pilot.note(101, "Late answer discarded")
    assert pilot.spectator(101)["event_frame"] == 100
    assert pilot.events[-1]["frame"] == 101


def test_discard_notice_cooldown_and_expiry(monkeypatch):
    pilot = idle_pilot(monkeypatch)
    pilot.note(100, "Late answer discarded")
    assert pilot.spectator(159)["event"] == "Late answer discarded"
    assert pilot.spectator(160)["event"] == ""
    pilot.note(161, "Answer → run")
    pilot.note(399, "Late answer discarded")
    assert pilot.spectator(399)["event"] == "Answer → run"
    pilot.note(400, "Late answer discarded")
    assert pilot.spectator(400)["event"] == "Late answer discarded"
    assert len(pilot.events) == 4


def test_airborne_jump_releasing_duck_changes_pending_answer_premise(monkeypatch):
    pilot = idle_pilot(monkeypatch)
    pilot.held = "duck"
    assert pilot.game.trex.jumping
    assert pilot.changes_keys("jump")


def test_collect_release_from_airborne_jump_invalidates_old_epoch(monkeypatch):
    pilot = idle_pilot(monkeypatch)
    pilot.held = "duck"
    pilot.enforce()
    assert pilot.game.trex.speed_drop
    epoch = pilot.epoch
    pilot.results.put(
        Decision(1, pilot.game.run_index, 0, time.perf_counter(), executed="jump", epoch=epoch)
    )
    pilot.collect(1)
    assert pilot.epoch == epoch + 1
    pilot.enforce()
    assert not pilot.game.trex.speed_drop
    pilot.results.put(
        Decision(2, pilot.game.run_index, 0, time.perf_counter(), executed="duck", epoch=epoch)
    )
    pilot.collect(2)
    assert pilot.held == "run"
    assert pilot.stats.discarded == 1
