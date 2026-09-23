"""Persistent browser context and DOM-settle detection.

Ported from jev-browser (MIT, Ying-Kai Liao); see NOTICE. Two mechanisms carry over: a
persistent profile so you log in once, and settle-by-mutation-timestamp so we never read a
post whose engagement counts are still rendering.

THE NO-NAVIGATION RULE: `login()` is the only function in this package permitted to navigate,
and only to a platform login page at the user's explicit command. Nothing else may call goto,
scroll, or click on page content -- the tool is a passive observer of a session you drive, and
that is what keeps it from looking like automation to LinkedIn. tests/test_no_navigation.py
enforces this mechanically.
"""

import time
from pathlib import Path

LOGIN_URLS = {
    "linkedin": "https://www.linkedin.com/login",
    "x": "https://x.com/login",
}

FEED_HOSTS = {
    "linkedin": ("linkedin.com",),
    "x": ("x.com", "twitter.com"),
}


def feed_tab(pages, platform):
    """The newest open tab on the platform -- the one you just opened -- or None."""
    for page in reversed(list(pages)):
        try:
            url = page.url
        except Exception:
            continue  # a tab mid-close has no url and is not the one you are reading
        if any(host in url for host in FEED_HOSTS.get(platform, ())):
            return page
    return None

# Installed before any page script runs, so it survives navigation the user performs.
# The attributeName guard is load-bearing: tagging posts with their URN is itself a DOM
# mutation, so without it the observer retriggers on its own writes and the page never settles.
INIT_SCRIPT = """
window.__layaMut = performance.now();
const mo = new MutationObserver(recs => {
  if (recs.some(r => r.attributeName !== 'data-laya-i')) window.__layaMut = performance.now();
});
const go = () => mo.observe(document, {
  subtree: true, childList: true, attributes: true, characterData: true,
});
document ? go() : addEventListener('DOMContentLoaded', go);
"""

IDLE_JS = ("() => document.readyState === 'loading' ? 0 "
           ": performance.now() - (window.__layaMut ?? 0)")


class Browser:
    """A persistent Chrome profile with DOM-quiet detection. Never drives the page."""

    def __init__(self, profile, headed=True, viewport=None):
        from playwright.sync_api import sync_playwright

        self.profile = Path(profile).expanduser()
        self.profile.mkdir(parents=True, exist_ok=True)
        self._pw = sync_playwright().start()
        self.context = self._pw.chromium.launch_persistent_context(
            str(self.profile), headless=not headed, viewport=viewport,
        )
        self.context.add_init_script(INIT_SCRIPT)
        self._inflight = {}
        self.page = self.context.pages[0] if self.context.pages else self.context.new_page()
        # Every tab, not just the first: you open your feed in whichever one you like.
        self.context.on("page", self._watch)
        for page in self.context.pages:
            self._watch(page)

    def _watch(self, page):
        page.on("request", self._track)
        page.on("requestfinished", self._untrack)
        page.on("requestfailed", self._untrack)

    def active_page(self, platform):
        """The tab showing your feed, so opening it in a new tab needs nothing from you.

        Chosen by URL, not by focus: under CDP every tab reports `visibilityState: visible`
        and `hasFocus()` true, so the browser will not tell us which one you are reading.
        Until a feed tab exists we stay on the tab we have, and the panel goes there.
        """
        match = feed_tab(self.context.pages, platform)
        if match is not None:
            self.page = match
        if self.page.is_closed() and self.context.pages:
            self.page = self.context.pages[-1]
        return self.page

    def _track(self, request):
        if request.resource_type in ("fetch", "xhr", "document"):
            self._inflight[request] = time.monotonic()

    def _untrack(self, request):
        self._inflight.pop(request, None)

    def settle(self, quiet_ms=400, max_ms=8000):
        """Block until the DOM stops changing and network is idle. Returns ms waited.

        Stale requests are dropped after 5s: LinkedIn holds long-poll connections open, and
        without the cutoff a single hung request means settle never returns.
        """
        t0 = time.monotonic()
        while (time.monotonic() - t0) * 1000 < max_ms:
            now = time.monotonic()
            net = sum(1 for started in self._inflight.values() if now - started < 5.0)
            try:
                idle = self.page.evaluate(IDLE_JS)
            except Exception:
                idle = 0
            if net == 0 and idle >= quiet_ms:
                break
            time.sleep(0.1)
        return (time.monotonic() - t0) * 1000

    def login(self, platform):
        """The single permitted navigation in this package. Opens the login page and waits."""
        if platform not in LOGIN_URLS:
            raise ValueError(f"platform must be one of {tuple(LOGIN_URLS)}, got {platform!r}")
        self.page.goto(LOGIN_URLS[platform])  # noqa: laya-allow-navigation
        print(f"Sign in to {platform}, then press Enter here to save the session.")
        input()

    def close(self):
        try:
            self.context.close()
        finally:
            self._pw.stop()
