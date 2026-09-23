"""Extraction against fixture HTML that mimics LinkedIn's feed. Never touches the network.

The fixtures are served over file:// and run the REAL EXTRACT_JS, so a selector change is
caught here rather than during a live scroll session. Navigating to a fixture is the only
navigation in this file; the package under test never navigates.
"""

import pytest

from social.live.extract import (
    ExtractionFailed,
    extract_posts,
    extract_raw,
    parse_count,
)

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None

needs_browser = pytest.mark.skipif(sync_playwright is None, reason="playwright not installed")

URN_A = "urn:li:activity:7123456789012345678"
URN_SPONSORED = "urn:li:activity:7222222222222222222"
URN_OVERLAY = "urn:li:activity:7333333333333333333"
URL_A = f"https://www.linkedin.com/feed/update/{URN_A}/"

CARD_A = f"""
<div class="feed-shared-update-v2" data-urn="{URN_A}">
  <div class="update-components-actor__title"><span aria-hidden="true">Ada Lovelace</span></div>
  <div class="update-components-actor__description">Computing pioneer &middot; 3,120 followers</div>
  <div class="update-components-text">Shipping beats planning.
     Three months of shipping taught me this.</div>
  <button class="feed-shared-inline-show-more-text__see-more-less-toggle">see more</button>
  <a href="{URL_A}">Copy link to post</a>
  <div class="social-details-social-counts">
    <button aria-label="1,204 reactions">1,204</button>
    <button aria-label="3,405 comments">3,405 comments</button>
    <button aria-label="87 reposts">87 reposts</button>
  </div>
</div>
"""

CARD_SPONSORED = f"""
<div class="feed-shared-update-v2" data-urn="{URN_SPONSORED}">
  <div class="update-components-actor__title"><span aria-hidden="true">Acme Cloud</span></div>
  <div class="update-components-actor__description">Promoted</div>
  <div class="update-components-text">Migrate to Acme Cloud today.</div>
  <div class="social-details-social-counts">
    <button aria-label="12 reactions">12</button>
  </div>
</div>
"""

CARD_IN_OVERLAY = f"""
<div data-laya-overlay="panel">
  <div class="feed-shared-update-v2" data-urn="{URN_OVERLAY}">
    <div class="update-components-actor__title"><span aria-hidden="true">Laya</span></div>
    <div class="update-components-text">specificity 2.7 / actionability 1.4</div>
  </div>
</div>
"""

# Feed container present, nothing inside it that extraction recognises: the DOM-changed signal.
CARD_UNRECOGNISABLE = '<div class="some-new-linkedin-class">Post body</div>'

# Cards LinkedIn still marks with a urn, but whose body no longer matches the text selector.
CARD_NO_BODY = f'<div class="feed-shared-update-v2" data-urn="{URN_A}">Ada Lovelace</div>'


def feed(*cards):
    return (
        '<html><body><main role="main"><div class="scaffold-finite-scroll__content">'
        + "".join(cards)
        + "</div></main></body></html>"
    )


@pytest.fixture(scope="module")
def page():
    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch()
        except Exception as exc:
            pytest.skip(f"chromium is not installed: {exc}")
        yield browser.new_page()
        browser.close()


def load(page, tmp_path, html):
    path = tmp_path / "feed.html"
    path.write_text(html, encoding="utf-8")
    page.goto(path.as_uri())  # fixture navigation, not page navigation
    return page


def test_parse_count_abbreviations():
    assert parse_count("1,204") == 1204
    assert parse_count("1.2K") == 1200
    assert parse_count("3.4K") == 3400
    assert parse_count("12") == 12
    assert parse_count("") == 0
    assert parse_count(None) == 0
    assert parse_count("3,405 comments") == 3405
    assert parse_count("1,204 reactions") == 1204
    assert parse_count("2.5M") == 2_500_000
    assert parse_count("Be the first to comment") == 0


@needs_browser
def test_extracts_fields_and_counts(page, tmp_path):
    load(page, tmp_path, feed(CARD_A))
    (post,) = extract_posts(page, corpus="acme")

    assert post.url == URL_A
    assert post.author == "Ada Lovelace"
    assert post.text.startswith("Shipping beats planning.")
    assert post.likes == 1204
    assert post.comments == 3405
    assert post.reposts == 87
    assert post.author_followers == 3120
    assert post.posted_at.startswith("2023-")

    (raw,) = extract_raw(page)
    assert raw["urn"] == URN_A
    assert raw["truncated_by_platform"] is True
    # The tag is the permalink, so one string identifies a post in the DB, the dedup set
    # and the DOM. browser.py's observer ignores exactly this attribute name.
    assert page.get_attribute(f'[data-urn="{URN_A}"]', "data-laya-i") == post.url


@needs_browser
def test_synthesises_permalink_when_card_has_no_link(page, tmp_path):
    load(page, tmp_path, feed(CARD_SPONSORED))
    (raw,) = extract_raw(page)
    assert raw["url"] == f"https://www.linkedin.com/feed/update/{URN_SPONSORED}/"


@needs_browser
def test_skips_our_own_overlay_nodes(page, tmp_path):
    load(page, tmp_path, feed(CARD_A, CARD_IN_OVERLAY))
    assert [r["urn"] for r in extract_raw(page)] == [URN_A]


@needs_browser
def test_sponsored_posts_are_flagged_not_dropped(page, tmp_path):
    load(page, tmp_path, feed(CARD_A, CARD_SPONSORED))

    flags = {r["urn"]: r["sponsored"] for r in extract_raw(page)}
    assert flags == {URN_A: False, URN_SPONSORED: True}
    # Flagged for session.py to count, and kept out of the corpus.
    assert [p.url for p in extract_posts(page, corpus="acme")] == [URL_A]


@needs_browser
def test_feed_container_with_no_parseable_posts_raises(page, tmp_path):
    load(page, tmp_path, feed(CARD_UNRECOGNISABLE))
    with pytest.raises(ExtractionFailed):
        extract_posts(page, corpus="acme")


@needs_browser
def test_cards_that_yield_no_body_raise(page, tmp_path):
    load(page, tmp_path, feed(CARD_NO_BODY))
    with pytest.raises(ExtractionFailed):
        extract_posts(page, corpus="acme")


@needs_browser
def test_no_feed_container_is_not_an_error(page, tmp_path):
    load(page, tmp_path, "<html><body><p>Signed out</p></body></html>")
    assert extract_posts(page, corpus="acme") == []
