"""Overlay behaviour against a local file:// page. No network, no LinkedIn, no account.

The fixture stands in for a feed: two cards carrying `data-urn`, and a full-width strip along
the bottom of the viewport that counts clicks. The strip sits exactly where the panel is
painted, so it is the probe for the one thing that would make the page feel broken -- a stray
click eaten by the dashboard instead of reaching the page underneath.
"""

import pytest

from social.live import overlay

pytest.importorskip("playwright.sync_api")

FIXTURE = """<!doctype html>
<html><body style="margin:0">
<div id="feed">
  <div class="card" data-urn="urn:li:activity:1" style="height:80px">post one</div>
  <div class="card" data-urn="urn:li:activity:2" style="height:80px">post two</div>
</div>
<div id="target" style="position:fixed;left:0;right:0;bottom:0;height:40vh"></div>
<script>
window.hits = 0;
document.getElementById('target').addEventListener('click', () => { window.hits += 1; });
</script>
</body></html>
"""

FRAME_A = {
    "post": {
        "urn": "urn:li:activity:1",
        "author": "Jane Roe",
        "permalink": "https://www.linkedin.com/feed/update/urn:li:activity:1/",
        "state": "scored",
    },
    "rubric": {
        "specificity": {"laya": {"value": 2.7}, "jev": {"value": 2.1}},
        "actionability": {"laya": {"value": 1.4}, "jev": None},
        "hook_type": {"laya": {"label": "contrarian"}, "jev": {"label": "curiosity_gap"}},
        "takes_a_position": {"laya": {"value": 1.0, "label": "true"}, "jev": None},
    },
    "latency": {"laya": 14.0, "jev": 310.0},
    "counters": {"seen": 214, "scored": 198, "dropped": 16, "sponsored": 3},
    "budget": {"spent": 0.41, "cap": 2.0, "exhausted": False},
    "agreement": 0.62,
    "band": None,
    "paused": False,
    "note": "no pattern mined yet",
    "series": {"laya": [2.0, 2.4, 2.7], "jev": [None, 2.1, 2.1], "agreement": [0.5, 0.6, 0.62]},
}

FRAME_B = {
    "post": {"urn": "urn:li:activity:2", "author": "Rob Poe", "permalink": "", "state": "pending"},
    "rubric": {
        "specificity": {"laya": {"value": 0.8}, "jev": None},
        "hook_type": {"laya": {"label": "story_open"}, "jev": None},
    },
    "latency": {"laya": 9.0, "jev": None},
    "counters": {"seen": 215, "scored": 198, "dropped": 17, "sponsored": 3},
    "budget": {"spent": 2.0, "cap": 2.0, "exhausted": True},
    "agreement": 0.61,
    "band": {"predicted": 180.0, "low": 120.0, "high": 260.0, "actual": 210.0},
    "paused": False,
    "note": "",
    "series": {"laya": [2.0, 2.4, 2.7, 0.8], "lat_laya": [14.0, 13.0, 11.0, 9.0]},
}

NODE_COUNT = """() => {
  const walker = document.createTreeWalker(
    document.getElementById('__laya-overlay').shadowRoot, NodeFilter.SHOW_ALL);
  let n = 0;
  while (walker.nextNode()) n += 1;
  return n;
}"""

SHADOW_TEXT = """() =>
  document.getElementById('__laya-overlay').shadowRoot.textContent"""


@pytest.fixture
def page(tmp_path):
    from playwright.sync_api import sync_playwright

    path = tmp_path / "feed.html"
    path.write_text(FIXTURE)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1200, "height": 800})
        page.goto(path.as_uri())
        yield page
        browser.close()


def test_install_is_idempotent(page):
    overlay.install(page)
    overlay.install(page)
    assert page.eval_on_selector_all("#__laya-overlay", "nodes => nodes.length") == 1
    assert page.evaluate("() => document.body.style.paddingBottom") == "42vh"


def test_update_mutates_in_place(page):
    overlay.install(page)
    overlay.update(page, FRAME_A)
    before = page.evaluate(NODE_COUNT)
    text_a = page.evaluate(SHADOW_TEXT)

    overlay.update(page, FRAME_B)
    after = page.evaluate(NODE_COUNT)
    text_b = page.evaluate(SHADOW_TEXT)

    assert after == before, "update() rebuilt DOM; it must only write values in place"
    assert text_a != text_b
    assert "2.7" in text_a and "contrarian" in text_a and "214" in text_a
    assert "0.8" in text_b and "story_open" in text_b and "215" in text_b
    assert "budget exhausted" in text_b

    # Chart paths are attributes on nodes that already exist, so plotting adds no nodes either.
    assert page.evaluate(
        "() => [...document.getElementById('__laya-overlay').shadowRoot"
        ".querySelectorAll('.chart path')].some(p => p.getAttribute('d'))"
    )


def test_clicks_reach_the_page_but_buttons_do_not_fall_through(page):
    overlay.install(page)
    overlay.update(page, FRAME_A)

    # A point inside the panel's footprint but not on a control: must reach the page.
    page.mouse.click(40, 780)
    assert page.evaluate("() => window.hits") == 1

    box = page.locator("#__laya-overlay button", has_text="SAVE").bounding_box()
    page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    assert page.evaluate("() => window.hits") == 1, "the panel button leaked its click to the page"
    assert [c["source"] for c in overlay.poll_clicks(page)] == ["save"]


def test_poll_clicks_drains(page):
    overlay.install(page)
    overlay.update(page, FRAME_A)
    page.locator('[data-urn="urn:li:activity:2"]').click()

    first = overlay.poll_clicks(page)
    assert [(c["source"], c["urn"]) for c in first] == [("card", "urn:li:activity:2")]
    assert isinstance(first[0]["ts"], (int, float))
    assert overlay.poll_clicks(page) == []


def test_every_injected_node_carries_the_overlay_attribute(page):
    overlay.install(page)
    overlay.update(page, FRAME_A)
    overlay.mark(page, "urn:li:activity:1", "both")

    unmarked = page.evaluate(
        """() => {
          const host = document.getElementById('__laya-overlay');
          const ours = [host, ...host.shadowRoot.querySelectorAll('*'),
                        ...document.querySelectorAll('[data-laya-rail]'),
                        ...document.head.querySelectorAll('style')];
          return ours.filter(n => !n.hasAttribute('data-laya-overlay'))
                     .map(n => n.tagName + (n.className.baseVal ?? n.className));
        }"""
    )
    assert unmarked == []
    assert page.evaluate(
        """() => document.querySelector('[data-urn="urn:li:activity:1"]')
                        .firstElementChild.getAttribute('data-laya-rail')"""
    ) == "both"

    overlay.teardown(page)
    assert page.eval_on_selector_all("#__laya-overlay", "n => n.length") == 0
    assert page.eval_on_selector_all("[data-laya-rail]", "n => n.length") == 0
