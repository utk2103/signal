"""A decision model designs the obstacle course that every player faces.

For each obstacle the designer answers two questions: which obstacle comes next (a
Choice over the options the game's rules allow at that moment) and how much room
follows it (a Score over the original game's gap range). The choice is sampled from the
model's probabilities with a stream seeded per obstacle, so a course is reproducible.

The designer works ahead of play. A private, collision-free copy of the game runs the
course forward, so every question sees the exact speed, score and history at the moment
that obstacle will appear. Courses are shared: all players read the same stored design.
If a player reaches an obstacle before it is designed, the original random rule fills
that slot, is stored for everyone, and counts as a fallback.
"""

import random
import threading
import time
from dataclasses import replace

from .course import StagedCourse, phase, shape
from .engine import KINDS, Game, RandomCourse, Spec

MENU = (
    (Spec("cactusSmall", 1), "One small cactus. The easiest obstacle."),
    (Spec("cactusSmall", 2), "Two small cacti side by side."),
    (Spec("cactusSmall", 3), "Three small cacti: a wide jump."),
    (Spec("cactusLarge", 1), "One tall cactus."),
    (Spec("cactusLarge", 2), "Two tall cacti: a long, high jump."),
    (Spec("cactusLarge", 3), "Three tall cacti: the widest jump, very hard at speed."),
    (Spec("pterodactyl", 1, 100), "A bird skimming the ground: the dino must jump it."),
    (Spec("pterodactyl", 1, 75), "A bird at head height: the dino must duck or jump."),
    (Spec("pterodactyl", 1, 50), "A bird flying high: the dino can run under it."),
)
SPACE = (
    "The minimum gap: a tight, demanding sequence",
    "A short gap",
    "A comfortable gap",
    "The maximum gap: a breather",
)
# Keep options clear of the game's speed thresholds, since the rule check happens at spawn.
SPEED_MARGIN = 0.05


def allowed(game):
    """Menu entries the original rules permit for the obstacle `game` is about to spawn."""
    options = []
    for spec, description in MENU:
        kind = KINDS[spec.kind]
        if game.too_many(kind.name):
            continue
        if game.speed < kind.min_speed + (SPEED_MARGIN if kind.min_speed else 0):
            continue
        if spec.size > 1 and game.speed < kind.multiple_speed + SPEED_MARGIN:
            continue
        options.append((spec, description))
    return options


def design_question(game, index, history, staged=False):
    state = {
        "game": "Chrome dino runner. The dinosaur runs right; it jumps cacti and low birds, "
        "ducks head-height birds, and runs under high birds.",
        "obstacle_number": index + 1,
        "speed": round(game.speed, 2),
        "top_speed": 13,
        "score": game.score,
        "recent_obstacles": [spec.label for spec in history[-5:]],
    }
    if staged:
        stage = phase(game)
        state["phase"] = stage["name"]
        state["direction"] = stage["hint"]
        state["minimum_gap_fraction"] = stage["gap_floor"]
    options = {spec.label: description for spec, description in allowed(game)}
    questions = {
        "next": {
            "type": "choice",
            "instructions": "Design the next obstacle of this dino runner course. Keep the course "
            "varied and fair, and raise the difficulty as the speed and score increase.",
            "criteria": options,
        },
        "space": {
            "type": "score",
            "instructions": "How much running room should follow this obstacle before the next "
            "one? Less room is harder.",
            "criteria": list(SPACE),
        },
    }
    return state, questions


class CourseDirector:
    def __init__(self, backend, seed, lookahead=8, staged=False):
        self.backend = backend
        self.seed = seed
        self.lookahead = lookahead
        self.staged = staged
        self.fallback = StagedCourse(seed) if staged else RandomCourse(seed)
        self.specs = {}
        self.wanted = {}
        self.ghosts = {}
        self.cond = threading.Condition()
        self.stopped = False
        self.designed = 0
        self.fallbacks = 0
        self.errors = 0
        self.last_error = None
        self.latencies = []
        self.tokens = 0
        self.log = []
        self.thread = threading.Thread(target=self.work, name="course-director", daemon=True)
        self.thread.start()

    @property
    def name(self):
        return self.backend.name

    # -- Called by players (main thread) -------------------------------------------------
    def demand(self, run, upto):
        with self.cond:
            if upto > self.wanted.get(run, -1):
                self.wanted[run] = upto
                self.cond.notify_all()

    def ready(self, run, upto):
        with self.cond:
            return all((run, i) in self.specs for i in range(upto + 1))

    def spec(self, game, index):
        key = (game.run_index, index)
        with self.cond:
            if key not in self.specs:
                if game.invincible:
                    return None  # The ghost designs it below, outside the lock.
                self.specs[key] = self.fallback.spec(game, index, source="fallback")
                self.fallbacks += 1
            return self.specs[key]

    def close(self):
        with self.cond:
            self.stopped = True
            self.cond.notify_all()
        self.thread.join(timeout=1.0)

    # -- Designer thread ------------------------------------------------------------------
    def work(self):
        try:
            while True:
                with self.cond:
                    while not self.stopped and not (job := self.next_job()):
                        self.cond.wait(0.25)
                    if self.stopped:
                        return
                run, ghost = job
                ghost.step()
        finally:
            # The worker owns the backend so an active request finishes before closure.
            self.backend.close()

    def next_job(self):
        for run in sorted(self.wanted):
            ghost = self.ghosts.get(run)
            if ghost is None:
                ghost = self.ghosts[run] = self.ghost(run)
            if ghost.obstacle_index <= self.wanted[run]:
                return run, ghost
        return None

    def ghost(self, run):
        game = Game(self.seed, course=GhostCourse(self))
        game.invincible = True
        if run == 0:
            game.press_jump()
        else:
            # Enter run `run` through the same restart path a real player takes.
            game.activated = game.crashed = True
            game.run_index = run - 1
            game.restart()
        return game

    def design(self, game, index):
        history = [self.specs[(game.run_index, i)] for i in range(index)]
        state, questions = design_question(game, index, history, self.staged)
        rng = random.Random(f"design-{self.seed}-{game.run_index}-{index}")
        try:
            answers, elapsed, tokens = self.backend.ask(state, questions)
            probabilities = answers["next"]["probabilities"]
            options = [spec for spec, _ in allowed(game)]
            weights = [max(0.0, float(probabilities.get(spec.label, 0.0))) for spec in options]
            if self.staged and phase(game)["name"] == "Bird attack":
                weights = [
                    w * (4 if s.kind == "pterodactyl" else 1) for s, w in zip(options, weights)
                ]
            pick = rng.choices(options, weights=weights)[0] if sum(weights) > 0 else options[0]
            level = min(1.0, max(0.0, float(answers["space"]["score"]) / (len(SPACE) - 1)))
            spec = replace(pick, gap=level, faster=rng.random() > 0.5, source=self.backend.name)
            if self.staged:
                spec = shape(spec, game)
            self.log.append(
                {
                    "run": game.run_index,
                    "index": index,
                    "speed": round(game.speed, 3),
                    "score": game.score,
                    "probabilities": {s.label: w for s, w in zip(options, weights)},
                    "pick": spec.label,
                    "space": round(level, 3),
                    "ms": round(elapsed, 1),
                }
            )
            self.latencies.append(elapsed)
            self.tokens += tokens
            self.designed += 1
        except Exception as error:  # A failed call must not stall the course.
            self.errors += 1
            self.last_error = str(error)[:160]
            spec = self.fallback.spec(game, index, source="fallback")
            time.sleep(0.2)
        return spec


class GhostCourse:
    """Course source for the designer's collision-free look-ahead game."""

    def __init__(self, director):
        self.director = director

    def spec(self, game, index):
        director = self.director
        existing = director.spec(game, index)
        if existing is not None:
            return existing
        designed = director.design(game, index)
        with director.cond:
            # A player may have needed this slot first; theirs stands.
            return director.specs.setdefault((game.run_index, index), designed)


def summary(director):
    lat = sorted(director.latencies)
    return {
        "designer": director.name,
        "designed": director.designed,
        "fallbacks": director.fallbacks,
        "errors": director.errors,
        "p50_ms": lat[len(lat) // 2] if lat else None,
        "tokens": director.tokens,
    }
