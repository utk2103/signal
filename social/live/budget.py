"""Jev spend accounting, every-Nth sampling, and one post through one backend.

Three small independent pieces, deliberately not one object: how much has been spent,
which posts Jev is shown, and how an answer becomes `scores` rows.
"""

from datetime import datetime

from laya_mlx.trex.backends import JEV_USD_PER_TOKEN

from ..rubric import RUBRIC_VERSION, post_questions, post_state
from ..score import answer_to_row

# A rubric call's floor cost, used as the reservation before any real call has returned.
# Deliberately low: it only has to make a cap smaller than one call refuse the first call.
FIRST_CALL_TOKENS = 500


class Budget:
    """A hard per-session cap on Jev spend. Reports exhaustion; never raises.

    `cap_usd <= 0` disables Jev entirely.
    """

    def __init__(self, cap_usd: float):
        self.cap_usd = cap_usd
        self.tokens = 0
        self.calls = 0
        # A cap can only be honoured by refusing a call BEFORE making it, and a call's cost
        # is unknown until it returns -- so allow() reserves room for one more call as
        # expensive as the priciest so far, seeded so the first call is capped too.
        self.worst_tokens = FIRST_CALL_TOKENS

    def allow(self) -> bool:
        return self.cap_usd > 0 and (
            (self.tokens + self.worst_tokens) * JEV_USD_PER_TOKEN <= self.cap_usd
        )

    def charge(self, tokens: int) -> None:
        self.tokens += tokens
        self.calls += 1
        self.worst_tokens = max(self.worst_tokens, tokens)

    @property
    def spent_usd(self) -> float:
        return self.tokens * JEV_USD_PER_TOKEN

    @property
    def exhausted(self) -> bool:
        return not self.allow()


class Sampler:
    """Every Nth post goes to Jev, counting posts from 1: with `every=5`, posts 1, 6, 11.

    `every=0` disables Jev; `every=1` takes every post.
    """

    def __init__(self, every: int):
        self.every = every
        self.seen = 0
        self.taken = 0

    def take(self) -> bool:
        self.seen += 1
        hit = self.every > 0 and (self.seen - 1) % self.every == 0
        if hit:
            self.taken += 1
        return hit

    @property
    def skipped(self) -> int:
        return self.seen - self.taken


def score_one(backend, post, engine: str) -> tuple[list[tuple], float, int]:
    """Score one post with one backend. Returns (`scores` rows, elapsed ms, input tokens)."""
    answers, elapsed_ms, tokens = backend.ask(post_state(post), post_questions())
    scored_at = datetime.now().astimezone().isoformat()
    rows = [
        (
            post.url, question, RUBRIC_VERSION, engine, *answer_to_row(answer),
            float(answer["confidence"]), scored_at,
        )
        for question, answer in answers.items()
    ]
    return rows, elapsed_ms, tokens
