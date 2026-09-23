"""Jev budget, sampling, and row shape. No network, no model, no browser.

Sampling convention asserted here: posts are counted from 1, and `every=N` takes
posts 1, N+1, 2N+1, ... so the first post of a session always reaches Jev.
"""

from laya_mlx.trex.backends import JEV_USD_PER_TOKEN
from social.live.budget import Budget, Sampler, score_one
from social.models import Post
from social.rubric import RUBRIC_VERSION, post_questions

CALL_TOKENS = 1200


class FakeBackend:
    """Shaped like LayaBackend/JevBackend: .ask(state, questions) -> (answers, ms, tokens)."""

    def __init__(self, tokens=CALL_TOKENS):
        self.tokens = tokens
        self.calls = 0

    def ask(self, state, questions):
        self.calls += 1
        answers = {}
        for name, question in questions.items():
            if question["type"] == "choice":
                first = next(iter(question["criteria"]))
                answers[name] = {
                    "type": "choice",
                    "choice": first,
                    "probabilities": {first: 0.71},
                    "confidence": 0.8,
                }
            elif question["type"] == "score":
                answers[name] = {"type": "score", "score": 2.5, "confidence": 0.6}
            else:
                answers[name] = {"type": "noul", "noul": 0.9, "confidence": 0.55}
        return answers, 311.0, self.tokens


def a_post(url="https://www.linkedin.com/feed/update/urn:li:activity:1/"):
    return Post(
        url=url,
        platform="linkedin",
        corpus="live",
        author="Someone",
        text="We took a client from $40k to $310k MRR in 11 months. Here is the playbook.",
        posted_at="2026-09-22T10:00:00+00:00",
        likes=12,
        comments=3,
        reposts=1,
    )


def test_usd_maths_uses_the_existing_constant():
    budget = Budget(cap_usd=1.00)
    budget.charge(1_000_000)
    assert budget.spent_usd == 1_000_000 * JEV_USD_PER_TOKEN == 0.042


def test_budget_refuses_before_the_call_that_would_exceed_the_cap():
    # Room for exactly three calls of CALL_TOKENS, plus a little change.
    cap = 3.5 * CALL_TOKENS * JEV_USD_PER_TOKEN
    budget = Budget(cap_usd=cap)
    backend = FakeBackend()
    for _ in range(10):
        if not budget.allow():
            break
        _, _, tokens = score_one(backend, a_post(), "jev")
        budget.charge(tokens)

    assert backend.calls == 3, "the fourth call would have crossed the cap"
    assert budget.exhausted
    assert budget.spent_usd <= cap
    assert not budget.allow(), "exhaustion is permanent for the session"


def test_cap_of_zero_disables_jev():
    budget = Budget(cap_usd=0)
    assert not budget.allow()
    assert budget.exhausted
    assert budget.spent_usd == 0.0


def test_sampler_takes_every_nth_post_counting_from_one():
    sampler = Sampler(every=5)
    taken = [i for i in range(1, 17) if sampler.take()]
    assert taken == [1, 6, 11, 16]
    assert (sampler.seen, sampler.taken, sampler.skipped) == (16, 4, 12)

    assert [Sampler(every=0).take() for _ in range(5)] == [False] * 5

    every_one = Sampler(every=1)
    assert all(every_one.take() for _ in range(5))
    assert every_one.skipped == 0


def test_score_one_emits_rows_matching_the_live_scores_schema():
    post = a_post()
    rows, elapsed_ms, tokens = score_one(FakeBackend(), post, "jev")

    assert (elapsed_ms, tokens) == (311.0, CALL_TOKENS)
    assert len(rows) == len(post_questions())
    questions = {row[1] for row in rows}
    assert questions == set(post_questions())

    for url, question, version, model, value, label, confidence, scored_at in rows:
        assert (url, version, model) == (post.url, RUBRIC_VERSION, "jev")
        assert isinstance(value, float) and isinstance(confidence, float)
        assert label is None or isinstance(label, str)
        assert scored_at.startswith("20")

    by_question = {row[1]: row for row in rows}
    assert by_question["hook_type"][4:6] == (0.71, "curiosity_gap")
    assert by_question["specificity"][4:6] == (2.5, None)
    assert by_question["takes_a_position"][4:6] == (0.9, None)


def test_score_one_rows_insert_into_the_real_scores_table(tmp_path):
    from social.models import connect, insert_posts

    conn = connect(tmp_path / "live.sqlite")
    post = a_post()
    insert_posts(conn, [post])
    for engine in ("laya", "jev"):
        rows, _, _ = score_one(FakeBackend(), post, engine)
        with conn:
            conn.executemany("INSERT INTO scores VALUES (?,?,?,?,?,?,?,?)", rows)

    # Both engines survive side by side; that is what the `model` column is for.
    models = conn.execute("SELECT DISTINCT model FROM scores ORDER BY model").fetchall()
    assert [r[0] for r in models] == ["jev", "laya"]
    assert conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0] == 2 * len(post_questions())


def test_the_feed_tab_is_found_however_you_opened_it():
    """The bug this exists for: you open LinkedIn in a new tab and the session watches
    about:blank forever. Tabs are matched by URL because under CDP every tab claims to be
    visible and focused, so the browser cannot say which one you are reading."""
    from social.live.browser import feed_tab

    class Tab:
        def __init__(self, url):
            self.url = url

    blank, feed, other = Tab("about:blank"), Tab("https://www.linkedin.com/feed/"), Tab("x.com/home")

    assert feed_tab([blank, feed], "linkedin") is feed
    assert feed_tab([feed, blank], "linkedin") is feed, "tab order must not matter"
    assert feed_tab([blank], "linkedin") is None, "no feed tab yet is not an error"
    assert feed_tab([blank, other], "x") is other
    assert feed_tab([blank, feed], "x") is None, "a LinkedIn tab is not an X feed"

    newest = Tab("https://www.linkedin.com/feed/?new")
    assert feed_tab([feed, newest], "linkedin") is newest, "the tab you just opened wins"
