"""LinkedIn feed extraction: the DOM in, `Post` records out.

Every CSS selector in this package lives in `SELECTORS` below and nowhere else, so a
LinkedIn DOM change is a one-dict edit. The selectors are a *guess* -- they were written
without saved feed HTML -- and are expected to be wrong until verified against a real feed.

Techniques borrowed from jev-browser's page-script.mjs (MIT, Ying-Kai Liao; see NOTICE):
the shadow-DOM `walk`, the rect+computed-style `visible()` check, `clean()` text
normalisation, and tagging nodes with a `data-*` attribute so they can be found again.
Its ENUMERATE targets interactive elements, which is the wrong target here, and is not used.

Nothing in this module navigates, scrolls, or touches the page. It reads and it tags.
"""

import re
from datetime import datetime, timezone

from ..models import Post

# Selectors only. No logic, no attribute names, no regexes -- those live in EXTRACT_JS.
# Each value is a comma-separated fallback chain, most stable hook first.
SELECTORS = {
    # The scrolling feed column. Its presence is what turns "zero posts" into an alarm.
    "feed_container": "[role=main], main, .scaffold-finite-scroll__content, .core-rail",
    # One post card. data-urn/data-id carry the activity URN on current LinkedIn.
    "post": (
        "[data-urn^='urn:li:activity'], [data-id^='urn:li:activity'], "
        "[data-chameleon-result-urn], .feed-shared-update-v2, [role=article]"
    ),
    # Our own injected dashboard and per-post markers. Never extracted from.
    "overlay": "[data-laya-overlay]",
    # Anchor whose href carries the post URN; doubles as the permalink source.
    "permalink": "a[href*='/feed/update/urn:li:']",
    # Author display name inside the card's actor block.
    "author": (
        ".update-components-actor__title span[aria-hidden=true], "
        ".update-components-actor__title, .update-components-actor__name"
    ),
    # Actor subtitle. Carries "Promoted"/"Sponsored" on ads, and a follower count on the
    # minority of cards that expose one.
    "actor_subtitle": (
        ".update-components-actor__description, .update-components-actor__sub-description"
    ),
    # Explicit ad markup, where LinkedIn emits it.
    "sponsored": "[data-ad-banner], [data-is-sponsored='true']",
    # The post body.
    "text": (
        ".update-components-text, .feed-shared-update-v2__description, "
        ".update-components-update-v2__commentary"
    ),
    # "…see more" toggle. Its presence means LinkedIn truncated the body in the DOM.
    "see_more": (
        ".feed-shared-inline-show-more-text__see-more-less-toggle, "
        "[aria-label*='see more' i]"
    ),
    "reactions": (
        "button[aria-label*='reaction' i], .social-details-social-counts__reactions-count, "
        "[aria-label*=' like' i]"
    ),
    "comments": "button[aria-label*='comment' i], .social-details-social-counts__comments",
    "reposts": "button[aria-label*='repost' i], .social-details-social-counts__item--right",
}

# Self-contained: Playwright serialises this into the page. Takes SELECTORS as its argument
# so no selector is ever written here. Returns raw strings; Python parses them.
EXTRACT_JS = """
(SEL) => {
  const clean = (s) => (s || "").replace(/\\s+/g, " ").trim();

  const walk = (root, out) => {
    for (const el of root.querySelectorAll("*")) {
      out.push(el);
      if (el.shadowRoot) walk(el.shadowRoot, out);
    }
    return out;
  };

  const visible = (el) => {
    const r = el.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) return false;
    const st = getComputedStyle(el);
    return st.visibility !== "hidden" && st.display !== "none" && +st.opacity > 0.05;
  };

  const text = (card, sel) => {
    const n = card.querySelector(sel);
    return n ? clean(n.innerText) : "";
  };

  // aria-label first: LinkedIn spells the number out there ("1,204 reactions") even when
  // the visible text is an abbreviation or an icon.
  const count = (card, sel) => {
    for (const n of card.querySelectorAll(sel)) {
      const v = clean(n.getAttribute("aria-label") || n.innerText);
      if (/\\d/.test(v)) return v;
    }
    return "";
  };

  const urnOf = (card) => {
    for (const a of card.attributes) if (/^urn:li:/.test(a.value)) return a.value;
    const link = card.querySelector(SEL.permalink);
    const m = link && link.getAttribute("href").match(/urn:li:[A-Za-z]+:[0-9]+/);
    return m ? m[0] : "";
  };

  const all = walk(document, []);
  const container = all.some((el) => el.matches(SEL.feed_container));

  let cards = all.filter(
    (el) => el.matches(SEL.post) && !el.closest(SEL.overlay) && visible(el)
  );
  // A card can match several hooks at different nesting depths; keep the outermost.
  cards = cards.filter((c) => !cards.some((o) => o !== c && o.contains(c)));

  const posts = [];
  for (const card of cards) {
    const urn = urnOf(card);
    if (!urn) continue;
    const link = card.querySelector(SEL.permalink);
    // One identity for a post everywhere: DB primary key, dedup set, DOM tag, overlay mark.
    // A card's own link is preferred; LinkedIn's canonical form is built from the urn when
    // there is none, so url is always a real openable link and never a bare urn.
    const url = link && link.href.startsWith("https://")
      ? link.href
      : "https://www.linkedin.com/feed/update/" + urn + "/";
    const subtitle = text(card, SEL.actor_subtitle);
    posts.push({
      urn: urn,
      url: url,
      author: text(card, SEL.author),
      subtitle: subtitle,
      text: text(card, SEL.text),
      truncated_by_platform: !!card.querySelector(SEL.see_more),
      sponsored: !!card.querySelector(SEL.sponsored) || /promoted|sponsored/i.test(subtitle),
      reactions_raw: count(card, SEL.reactions),
      comments_raw: count(card, SEL.comments),
      reposts_raw: count(card, SEL.reposts),
    });
    // browser.py's MutationObserver ignores exactly this attribute name; renaming it means
    // our own tagging retriggers the observer and the page never settles.
    card.setAttribute("data-laya-i", url);
  }
  return { container: container, posts: posts };
}
"""


class ExtractionFailed(RuntimeError):
    """The feed exists but nothing parsed out of it -- LinkedIn's DOM changed.

    Raised loudly on purpose: the failure mode this prevents is an hour of silent
    harvesting that returns [] after every scroll.
    """


_COUNT = re.compile(r"(\d[\d,.]*)\s*([KkMmBb])?")
_MULT = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}


def parse_count(raw) -> int:
    """LinkedIn renders counts as '1,204', '1.2K', '3,405 comments'. Missing count is 0."""
    match = _COUNT.search(raw or "")
    if not match:
        return 0
    number = float(match.group(1).replace(",", "").rstrip("."))
    return int(number * _MULT.get((match.group(2) or "").lower(), 1))


def extract_raw(page) -> list[dict]:
    """Every feed post currently in the DOM, as raw dicts. Sponsored ones are included and
    flagged, so a caller can count the ads it is skipping."""
    result = page.evaluate(EXTRACT_JS, SELECTORS)
    if result["container"] and not result["posts"]:
        raise ExtractionFailed(
            f"feed container matched {SELECTORS['feed_container']!r} but no post matched "
            f"{SELECTORS['post']!r}. LinkedIn's DOM has changed; update SELECTORS."
        )
    return result["posts"]


def post_from_raw(raw, corpus, platform="linkedin") -> Post:
    """Map one raw extraction dict onto `Post`. `Post` decides what is acceptable."""
    return Post(
        url=raw["url"],
        platform=platform,
        corpus=corpus,
        author=raw["author"],
        text=raw["text"],
        posted_at=_posted_at(raw["urn"]),
        likes=parse_count(raw["reactions_raw"]),
        comments=parse_count(raw["comments_raw"]),
        reposts=parse_count(raw["reposts_raw"]),
        author_followers=_followers(raw["subtitle"]),
    )


def extract_posts(page, corpus: str, platform: str = "linkedin") -> list[Post]:
    """Feed posts currently in the DOM, as corpus records. Ads are dropped."""
    raws = extract_raw(page)
    posts = [
        post_from_raw(raw, corpus, platform)
        for raw in raws
        # No body text is not scoreable by a text rubric, and no URN is not dedupable.
        if not raw["sponsored"] and raw["text"] and raw["urn"]
    ]
    if raws and not posts and not all(r["sponsored"] for r in raws):
        raise ExtractionFailed(
            f"{len(raws)} feed cards matched but none yielded an author and a body. "
            "LinkedIn's DOM has changed; update SELECTORS."
        )
    return posts


def _posted_at(urn):
    """Feed cards only show a relative age ('2h'), but a LinkedIn activity id carries its
    creation time: the top 41 bits of the id are the unix timestamp in milliseconds."""
    digits = re.search(r"\d{16,}", urn or "")
    if not digits:
        return datetime.now(timezone.utc).isoformat()
    value = int(digits.group())
    millis = value >> (value.bit_length() - 41)
    return datetime.fromtimestamp(millis / 1000, timezone.utc).isoformat()


_FOLLOWERS = re.compile(r"([\d][\d,.]*\s*[KkMmBb]?)\s*followers", re.IGNORECASE)


def _followers(subtitle):
    """None, never a guess: most feed cards do not expose a follower count at all."""
    match = _FOLLOWERS.search(subtitle or "")
    return parse_count(match.group(1)) if match else None
