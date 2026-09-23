"""Round, recording and replay behavior tested without models or network calls."""

from types import SimpleNamespace

import pytest

from laya_mlx.trex.cli import parse
from laya_mlx.trex.pilot import Arena, Pilot
from laya_mlx.trex.replay import Capture, ReplayBuffer


def make_arena(monkeypatch, *, seconds=1, rounds=2, guarded=True):
    monkeypatch.setattr(Pilot, "work", lambda self: None)
    pilots = [
        Pilot(
            SimpleNamespace(
                name=name,
                inflight=1,
                usd_per_token=0,
                model="test",
                detail="test",
                close=lambda: None,
            ),
            guarded=guarded,
        )
        for name in ("Laya", "Jev")
    ]
    return Arena(pilots, round_seconds=seconds, rounds=rounds)


def test_arena_rounds_restart_together_and_finish(monkeypatch):
    arena = make_arena(monkeypatch)
    arena.start()
    for _ in range(60):
        arena.tick()
    assert arena.match.number == 2
    assert [p.game.run_index for p in arena.pilots] == [1, 1]
    assert all(not p.game.crashed for p in arena.pilots)
    for _ in range(60):
        arena.tick()
    assert arena.finished
    frame = arena.frame
    arena.tick()
    assert arena.frame == frame
    assert len(arena.report()["match"]["rounds"]) == 2


def test_replay_emits_once_and_is_bounded_to_two_seconds():
    buffer = ReplayBuffer(1)
    for frame in range(1, 182):
        clips = buffer.capture(
            frame,
            [{"crashed": frame == 181}],
            [{"deaths": int(frame >= 181), "score": 20, "exec": "run"}],
            [0],
        )
    assert len(clips) == 1
    assert clips[0]["frames"][0]["frame"] == 61
    assert len(clips[0]["frames"]) == 121
    assert buffer.capture(181, [{}], [{"deaths": 1, "score": 20}], [0]) == []
    assert buffer.capture(182, [{}], [{"deaths": 1, "score": 0}], [1]) == []
    assert len(buffer.history[0]) == 1


def test_crash_record_has_trace_and_replay_and_renders(monkeypatch, tmp_path):
    pygame = pytest.importorskip("pygame")
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from laya_mlx.trex.app import Painter

    arena = make_arena(monkeypatch, seconds=60, rounds=1, guarded=False)
    capture = Capture(arena, parse(["--course", "random"]))
    arena.start()
    frames = []
    for _ in range(1200):
        arena.tick()
        frame = capture.frame()
        frames.append(frame)
        if frame["replays"]:
            break
    assert frame["replays"] and frame["crashes"]
    assert frame["crashes"][0]["trace"]
    assert len(frame["crashes"][0]["trace"]) <= 120
    assert frame["match"]["round"] == 1
    pygame.init()
    pygame.display.set_mode((1, 1))
    try:
        for layout, scale in (("window", 1), ("video", None)):
            painter = Painter(layout, capture.meta(), scale)
            surface = pygame.Surface(painter.size, pygame.SRCALPHA, 32)
            for previous in frames:
                painter.observe(previous)
            painter.draw(surface, frame)
            pygame.image.save(surface, tmp_path / f"{layout}.png")
            assert painter.replay_clips
    finally:
        pygame.quit()


@pytest.mark.parametrize(
    "args",
    [
        ["--round-seconds", "nan"],
        ["--round-seconds", "-1"],
        ["--rounds", "2", "--round-seconds", "0"],
        ["--players", "laya,laya"],
        ["--duration", "inf"],
    ],
)
def test_show_options_reject_invalid_values(args):
    with pytest.raises(SystemExit):
        parse(args)


def test_round_boundary_preserves_hud_high_score(monkeypatch):
    arena = make_arena(monkeypatch)
    arena.start()
    for _ in range(59):
        arena.tick()
    distance = arena.pilots[0].game.distance
    assert distance > 0
    arena.tick()
    assert arena.pilots[0].game.high_score >= distance
