"""End-to-end integration: a fake feed page through the real extract -> score -> overlay chain.

The pieces each have their own tests. This one exists because they are wired together across
three seams -- extraction's data-laya-i tag, overlay's card lookup, and the payload shape --
and a mismatch at any of them is invisible until a real session fails.

No network, no LinkedIn account, no model: the backend is a stub.
"""

import json

import pytest

from social.live import overlay
from social.live.session import Session
from social.models import connect
from social.rubric import CHOICE_QUESTIONS, post_questions

playwright_api = pytest.importorskip("playwright.sync_api")

URN1 = "urn:li:activity:7123456789012345678"
URN2 = "urn:li:activity:7123456789012345679"

FEED = """
<!doctype html><meta charset=utf-8><main>
  <div data-urn="{u1}">
    <div class="update-components-actor__title"><span aria-hidden="true">Priya Raman</span></div>
    <div class="update-components-actor__description">Founder · 4,200 followers</div>
    <div class="update-components-text">We cut onboarding from 14 days to 3. Activation
      went 31% to 58% in Q2.</div>
    <button aria-label="1,204 reactions"></button>
    <button aria-label="96 comments"></button>
    <button aria-label="41 reposts"></button>
  </div>
  <div data-urn="{u2}" data-is-sponsored="true">
    <div class="update-components-actor__title"><span aria-hidden="true">AdCo</span></div>
    <div class="update-components-actor__description">Promoted</div>
    <div class="update-components-text">Buy our thing today.</div>
    <button aria-label="3 reactions"></button>
  </div>
</main>
""".format(u1=URN1, u2=URN2)


class FakeBackend:
    """Answers the 9-question rubric in Laya's own answer shape."""

    def __init__(self, tokens=120):
        self.tokens = tokens
        self.calls = 0

    def ask(self, state, questions):
        self.calls += 1
        answers = {}
        for name, q in questions.items():
            if q["type"] == "choice":
                label = next(iter(q["criteria"]))
                answers[name] = {"type": "choice", "choice": label, "confidence": 0.8,
                                 "probabilities": {label: 0.7}}
            elif q["type"] == "score":
                answers[name] = {"type": "score", "score": 1.5, "confidence": 0.8}
            else:
                answers[name] = {"type": "noul", "noul": 0.6, "confidence": 0.8}
        return answers, 5.0, self.tokens

    def close(self):
        """LayaBackend owns a child process; Session.close() must reach it."""


class FakeBrowser:
    def __init__(self, page):
        self.page = page

    def active_page(self, platform):
        return self.page

    def settle(self, **_):
        return 0.0


@pytest.fixture
def feed(tmp_path):
    page_file = tmp_path / "feed.html"
    page_file.write_text(FEED)
    with playwright_api.sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        page.goto(page_file.as_uri())
        yield page
        browser.close()


def make_session(page, tmp_path, monkeypatch, *, jev_every=1, budget=1.0):
    laya, jev = FakeBackend(), FakeBackend()
    monkeypatch.setattr("social.live.session.backends.create",
                        lambda kind, **kw: laya if kind == "laya" else jev)
    conn = connect(tmp_path / "corpus.sqlite")
    # A nested --session path whose directory does not exist yet: the normal invocation.
    session = Session(FakeBrowser(page), conn, "t", jev_every=jev_every,
                      jev_budget_usd=budget, record=str(tmp_path / "artifacts/live/s.jsonl"))
    overlay.install(page)
    return session, laya, jev, conn


def test_scores_a_real_card_and_skips_the_ad(feed, tmp_path, monkeypatch):
    session, laya, jev, conn = make_session(feed, tmp_path, monkeypatch)
    post = session.tick()

    assert post is not None, "the non-sponsored card should have been scored"
    assert post.author == "Priya Raman"
    assert (post.likes, post.comments, post.reposts) == (1204, 96, 41)
    assert post.author_followers == 4200
    assert session.stats["sponsored"] == 1, "the ad must be counted, not silently dropped"

    rows = conn.execute("SELECT model, COUNT(*) FROM scores GROUP BY model").fetchall()
    written = {r[0]: r[1] for r in rows}
    n = len(post_questions())
    assert written == {"laya": n, "jev": n}, "both engines write, neither overwrites the other"
    session.close()


def test_dedup_and_overlay_mark_find_the_card(feed, tmp_path, monkeypatch):
    session, laya, _, _ = make_session(feed, tmp_path, monkeypatch)
    post = session.tick()
    before = laya.calls

    # The card is still on the page; a second pass must not rescore it.
    assert session.tick() is None
    assert laya.calls == before

    # mark() must actually find the card extract.py tagged -- the seam this test exists for.
    found = feed.evaluate(
        "(u) => !!document.querySelector('[data-laya-i=' + JSON.stringify(u) + ']')", post.url
    )
    assert found, "extract.py tagged the card with data-laya-i and overlay must locate it"
    railed = feed.evaluate(
        "(u) => { const c = document.querySelector('[data-laya-i=' + JSON.stringify(u) + ']');"
        "  return !!c && !!c.querySelector('[data-laya-rail]'); }", post.url
    )
    assert railed, "overlay.mark did not paint a rail on the scored card"
    session.close()


def test_the_panel_survives_you_navigating_to_your_feed(feed, tmp_path, monkeypatch):
    # The session starts on a blank tab and you open your own feed -- the one navigation the
    # tool will never do for you. That wipes every page.evaluate injection, panel included.
    session, _, _, _ = make_session(feed, tmp_path, monkeypatch)
    session.tick()
    feed.reload()
    assert feed.evaluate("() => !!window.__laya") is False

    session.tick()
    assert feed.evaluate("() => !!window.__laya"), "tick must put the panel back"
    session.close()


def test_click_to_save_uses_the_same_identity(feed, tmp_path, monkeypatch):
    session, _, _, conn = make_session(feed, tmp_path, monkeypatch)
    post = session.tick()

    feed.locator(f'[data-laya-i="{post.url}"]').click()
    session._save_clicked()

    saved = conn.execute("SELECT url, reason FROM swipe").fetchall()
    assert [tuple(r) for r in saved] == [(post.url, "manual")], (
        "a card click must star the same url the corpus keys on"
    )
    session.close()


def test_payload_matches_the_overlay_contract(feed, tmp_path, monkeypatch):
    session, _, _, _ = make_session(feed, tmp_path, monkeypatch)
    post = session.tick()
    payload = session._payload(post, {"laya": [], "jev": None})

    assert set(payload) >= {"post", "rubric", "latency", "counters", "budget", "series"}
    assert payload["post"]["urn"] == post.url
    overlay.update(feed, payload)  # raises if the panel rejects the shape
    session.close()


def test_jev_stops_at_budget_but_laya_continues(feed, tmp_path, monkeypatch):
    # A cap far below one call's cost: Jev must never be invoked, Laya must still score.
    session, laya, jev, conn = make_session(feed, tmp_path, monkeypatch, budget=1e-9)
    post = session.tick()

    assert post is not None and laya.calls == 1
    assert jev.calls == 0, "Jev must be refused before the call, not after"
    models = {r[0] for r in conn.execute("SELECT DISTINCT model FROM scores")}
    assert models == {"laya"}
    session.close()


def test_session_log_is_written(feed, tmp_path, monkeypatch):
    session, _, _, _ = make_session(feed, tmp_path, monkeypatch)
    session.tick()
    session.close()

    log = tmp_path / "artifacts/live/s.jsonl"
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert lines and lines[0]["url"].startswith("https://")


def test_choice_questions_land_in_the_label_column(feed, tmp_path, monkeypatch):
    session, _, _, conn = make_session(feed, tmp_path, monkeypatch)
    session.tick()
    labelled = {r[0] for r in conn.execute("SELECT question FROM scores WHERE label IS NOT NULL")}
    assert labelled == set(CHOICE_QUESTIONS)
    session.close()
