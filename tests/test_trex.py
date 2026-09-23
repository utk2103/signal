"""The T-Rex planner must predict the engine exactly; the shield's soundness rests on it."""

import copy
import random

import pytest

from laya_mlx.trex.engine import Game
from laya_mlx.trex.planner import (
    Planner,
    World,
    end_of_frame,
    enforce,
    impulse_jump,
    snapshot,
    trex_frame,
)


class Keys:
    """The pilot's key handling, without a model."""

    def __init__(self, game):
        self.game, self.held = game, "run"

    def apply(self, action):
        game = self.game
        if action == "jump":
            if not game.trex.jumping:
                if game.trex.ducking:
                    game.release_duck()
                game.press_jump()
            self.held = "run"
        else:
            self.held = action

    def enforce(self):
        game, trex = self.game, self.game.trex
        if self.held == "duck":
            if trex.jumping:
                if not trex.speed_drop:
                    game.press_duck()
            elif not trex.ducking:
                game.press_duck()
        elif trex.speed_drop or not trex.jumping and trex.ducking:
            game.release_duck()


def started(seed, frames, invincible=True):
    game = Game(seed)
    game.invincible = invincible
    game.press_jump()
    for _ in range(frames):
        game.step()
    return game


@pytest.mark.parametrize("seed", range(8))
def test_dino_model_matches_engine_under_random_keys(seed):
    rng = random.Random(seed)
    game = started(seed, 40)
    keys = Keys(game)
    state = snapshot(game, keys.held).trex
    for frame in range(3000):
        action = rng.choice(["jump", "duck", "run", None, None])
        if action:
            keys.apply(action)
            if action == "jump":
                state = impulse_jump(state, game.speed)
        keys.enforce()
        state = end_of_frame(trex_frame(enforce(state, keys.held)))
        game.step()
        assert snapshot(game, keys.held).trex == state, f"frame {frame} after {action}"


@pytest.mark.parametrize("seed", range(8))
def test_world_predicts_engine_collisions(seed):
    rng = random.Random(seed)
    game = started(seed, 200)
    keys = Keys(game)
    checked = 0
    for _ in range(2500):
        if rng.random() < 0.08:
            keys.apply(rng.choice(["jump", "duck", "run"]))
        keys.enforce()
        game.step()
        if not game.obstacles or rng.random() > 0.05:
            continue
        # From here, replay one random key script in a real copy and in the planner's world.
        script = {k: rng.choice(["jump", "duck", "run"]) for k in rng.sample(range(30), 4)}
        real = copy.deepcopy(game)
        real.invincible = False
        real_keys = Keys(real)
        real_keys.held = keys.held
        snap = snapshot(game, keys.held)
        world, state, held = World(snap, 30), snap.trex, keys.held
        for k in range(1, 31):
            if action := script.get(k - 1):
                real_keys.apply(action)
                if action == "jump":
                    state, held = impulse_jump(state, world.speed[k]), "run"
                else:
                    held = action
            real_keys.enforce()
            real.step()
            state = trex_frame(enforce(state, held))
            predicted = world.hit(k, state[0], state[3])
            assert predicted == real.crashed, (
                f"frame +{k}: planner {predicted}, engine {real.crashed}"
            )
            if predicted:
                break
            state = end_of_frame(state)
        checked += 1
    assert checked > 20


def choose(planner, game, keys, rng, timing, pending=None):
    snap = snapshot(game, keys.held)
    if pending:
        snap = type(snap)(**{**vars(snap), "pending": pending})
    plan = planner.plan(snap, timing, timing)
    return rng.choice(plan.safe_actions) if plan.robust else plan.best


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("timing", [(0, 1), (0, 2), (1, 3)])
def test_any_safe_action_survives_with_jittery_answers(seed, timing):
    """A player that picks at random among the actions labelled safe must never crash.

    Timing follows the pilot. Each answer is planned from the newest frame at the moment
    the previous answer finishes (so that one is still pending), and it is applied
    `timing = (lo, hi)` frames later, before the following step.
    """
    rng = random.Random(seed)
    game = started(seed, 0, invincible=False)
    keys, planner = Keys(game), Planner()
    frame, inflight = 0, None
    while frame < 2500:
        if inflight is None:
            inflight = (frame + rng.randint(*timing), choose(planner, game, keys, rng, timing))
        if inflight[0] == frame:
            landing = inflight[1]
            following = (
                frame + rng.randint(*timing),
                choose(planner, game, keys, rng, timing, pending=landing),
            )
            keys.apply(landing)
            inflight = following
            if following[0] == frame:  # Two answers on one frame; the next needs a fresh view.
                keys.apply(following[1])
                inflight = None
        keys.enforce()
        game.step()
        frame += 1
        assert not game.crashed, f"crashed at frame {frame}, score {game.score}"


@pytest.mark.parametrize("timing", [(0, 1), (1, 3), (18, 22)])
def test_following_the_recommended_move_survives(timing):
    """The recommendation must always be a safe move; a fast player then never crashes."""
    rng = random.Random(1)
    game = started(1, 0, invincible=False)
    keys, planner = Keys(game), Planner()
    frame, inflight = 0, None

    def best(pending=None):
        snap = snapshot(game, keys.held)
        if pending:
            snap = type(snap)(**{**vars(snap), "pending": pending})
        plan = planner.plan(snap, timing, timing)
        assert plan.safe[plan.best] or not plan.safe_actions
        return plan.best

    while frame < 4000 and not game.crashed:
        if inflight is None:
            inflight = (frame + rng.randint(*timing), best())
        if inflight[0] == frame:
            landing = inflight[1]
            inflight = (frame + rng.randint(*timing), best(pending=landing))
            keys.apply(landing)
            if inflight[0] == frame:
                keys.apply(inflight[1])
                inflight = None
        keys.enforce()
        game.step()
        frame += 1
    if timing[1] <= 3:
        assert not game.crashed, f"crashed at frame {frame}, score {game.score}"
    else:
        # A third of a second of latency cannot clear every obstacle at speed.
        assert game.score > 150


def test_recording_round_trips_and_renders(tmp_path, monkeypatch):
    """A recorded frame must load back unchanged and draw in both layouts without a display."""
    pygame = pytest.importorskip("pygame")
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from laya_mlx.trex.app import Painter
    from laya_mlx.trex.replay import Recorder, capture_game, load

    game = started(2, 300)
    panel = {
        "p": {"jump": 0.7, "duck": 0.1, "run": 0.2}, "exec": "jump", "prop": "duck", "veto": True,
        "score": game.score, "top": game.score, "deaths": 0, "ms": 25.0, "model_ms": 14.0, "n": 10,
        "rate": 50.0, "saves": 1, "agree": 0.8, "cost": 0.0, "err": None, "new": [24.0, 26.0],
    }  # fmt: skip
    players = [
        {"name": "Laya", "detail": "local", "guarded": True, "paid": False},
        {"name": "Jev", "detail": "api", "guarded": True, "paid": True},
    ]
    meta = {"type": "metadata", "seed": 2, "mode": "real time", "course": "Jev", "players": players}
    frame = {
        "type": "frame", "t": 5.0, "f": 300, "paused": False, "g": [capture_game(game)] * 2,
        "s": [panel, panel], "c": {"designed": 9, "fallbacks": 0, "errors": 0},
    }  # fmt: skip
    recorder = Recorder(tmp_path / "run.jsonl", meta)
    recorder.write(frame)
    recorder.write(frame)  # The same game frame is stored once.
    recorder.close()
    loaded_meta, frames = load(tmp_path / "run.jsonl")
    assert loaded_meta == meta and frames == [frame]
    with pytest.raises(FileExistsError):
        Recorder(tmp_path / "run.jsonl", meta)

    pygame.init()
    pygame.display.set_mode((1, 1))
    try:
        for layout, scale in (("video", None), ("window", 1.0)):
            painter = Painter(layout, loaded_meta, scale)
            surface = pygame.Surface(painter.size, pygame.SRCALPHA, 32)
            painter.observe(frames[0])
            painter.draw(surface, frames[0], badge="RECORDED RUN")
            assert (
                surface.get_at((painter.rows[0]["game"][0] + 5, painter.rows[0]["game"][1] + 5))[:3]
                == (255,) * 3
            )
    finally:
        pygame.quit()


def slow_player(seed, inflight, latency=19, jitter=7, frames=3000):
    """A player with a third of a second of latency, mirroring the pilot's bookkeeping:
    questions every `period` frames, and answers discarded once the keys have changed."""
    rng = random.Random(seed)
    game = started(seed, 0, invincible=False)
    keys, planner = Keys(game), Planner()
    stagger = max(1, round((latency + jitter / 2) / inflight))
    landing = (latency, latency + jitter)
    frame, epoch, asked, flying, waits = 0, 0, None, [], [2 * stagger]
    while frame < frames and not game.crashed:
        for answer in sorted((a for a in flying if a[0] <= frame), key=lambda a: a[0]):
            flying.remove(answer)
            _, action, asked_in = answer
            if asked_in != epoch:
                continue  # The keys changed after this was asked; its premise is gone.
            epoch += (not game.trex.jumping) if action == "jump" else action != keys.held
            keys.apply(action)
        keys.enforce()
        game.step()
        frame += 1
        if len(flying) < inflight and (asked is None or frame - asked >= stagger):
            if asked is not None:
                waits = (waits + [frame - asked])[-60:]
            asked = frame
            period = max(stagger, max(waits)) if inflight > 1 else None
            plan = planner.plan(snapshot(game, keys.held), landing, landing, period)
            flying.append((frame + rng.randint(*landing), plan.best, epoch))
    return game


@pytest.mark.parametrize("seed", range(3))
def test_requests_in_flight_let_a_slow_player_keep_up(seed):
    """One question per round trip leaves a slow player one chance per obstacle. Several in
    flight give it a turn every few frames, and then the same latency is survivable."""
    alone = slow_player(seed, inflight=1)
    together = slow_player(seed, inflight=5)
    assert alone.crashed and alone.score < 200
    assert not together.crashed, f"crashed at score {together.score}"


@pytest.mark.parametrize("seed", range(3))
def test_a_quick_player_with_two_questions_in_flight_survives(seed):
    game = slow_player(seed, inflight=2, latency=1, jitter=2)
    assert not game.crashed, f"crashed at score {game.score}"
