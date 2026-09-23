"""States reconstructed from the first recorded two-round match, before harmful inputs."""

import json
from pathlib import Path

import pytest

from laya_mlx.trex.planner import ObstacleView, Snapshot
from laya_mlx.trex.safety import protect

CASES = json.loads((Path(__file__).parent / "fixtures/trex_live_arrivals.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["player"])
def test_live_arrival_rejects_premature_jump_or_drop(case):
    raw = case["snapshot"]
    obstacles = tuple(
        ObstacleView(**dict(o, boxes=tuple(map(tuple, o["boxes"])))) for o in raw["obstacles"]
    )
    snap = Snapshot(**dict(raw, trex=tuple(raw["trex"]), obstacles=obstacles))
    assert protect(snap, case["action"]) != case["action"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["player"])
def test_live_arrival_and_emergency_survive_until_obstacles_pass(case, monkeypatch):
    from types import SimpleNamespace

    from laya_mlx.trex.engine import KINDS, Box, Obstacle, Spec
    from laya_mlx.trex.pilot import Pilot
    from laya_mlx.trex.planner import snapshot

    monkeypatch.setattr(Pilot, "work", lambda self: None)
    pilot = Pilot(SimpleNamespace(name="replay", inflight=1))
    game = pilot.game
    game.press_jump()
    for _ in range(300):
        game.step()
    raw = case["snapshot"]
    game.speed = raw["speed"]
    t = game.trex
    (t.y, t.velocity, t.jumping, t.ducking, t.speed_drop, t.reached_min_height, status) = raw[
        "trex"
    ]
    t.status = {"jump": "jumping", "run": "running", "duck": "ducking"}[status]
    t.x = raw["trex_x"]
    pilot.held = raw["held"]
    game.obstacles = []
    for source in raw["obstacles"]:
        kind = (
            "pterodactyl"
            if source["bird"]
            else ("cactusLarge" if "large" in source["label"] else "cactusSmall")
        )
        size = source["width"] // KINDS[kind].width
        obstacle = Obstacle(Spec(kind, size=size, y=source["y"]), game.speed)
        obstacle.width = source["width"]
        obstacle.boxes = [Box(*box) for box in source["boxes"]]
        obstacle.speed_offset = source["offset"]
        obstacle.id = source["id"]
        obstacle.x = source["x"]
        obstacle.following_created = True
        game.obstacles.append(obstacle)
    ids = {o.id for o in game.obstacles}
    pilot.apply(protect(snapshot(game, pilot.held), case["action"]))
    for frame in range(120):
        pilot.emergency(frame)
        pilot.enforce()
        game.step()
        assert not game.crashed, (case["player"], frame)
        if not any(o.id in ids for o in game.obstacles):
            break
    assert not any(o.id in ids for o in game.obstacles)
