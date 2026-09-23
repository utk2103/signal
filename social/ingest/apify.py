"""Apify actor adapters for LinkedIn and X.

Both return plain `Post` records, so nothing downstream knows Apify exists. Feature-gated
on APIFY_TOKEN, read at call time: importing this module must never break a subcommand
that does not scrape.
"""

import os

import httpx

from . import post_from_row

API_BASE = "https://api.apify.com/v2"

# Actor ids are configuration, not code: pass `actor=`, or set APIFY_LINKEDIN_ACTOR /
# APIFY_X_ACTOR. These defaults are only the starting point.
DEFAULT_LINKEDIN_ACTOR = "apimaestro~linkedin-profile-posts"
DEFAULT_X_ACTOR = "apidojo~tweet-scraper"

PAGE_SIZE = 1000
DEFAULT_TIMEOUT = 900.0  # run-sync blocks for the whole scrape, and scrapes are slow


class MissingApifyToken(RuntimeError):
    """APIFY_TOKEN is unset. cli.py catches this, prints it, and exits non-zero."""


def fetch_linkedin(profiles, corpus, *, actor=None, max_posts=50, timeout=DEFAULT_TIMEOUT):
    """Scrape recent posts for each LinkedIn profile URL or handle. Returns list[Post]."""
    actor = actor or os.environ.get("APIFY_LINKEDIN_ACTOR", DEFAULT_LINKEDIN_ACTOR)
    items = run_actor(
        actor,
        {"urls": list(profiles), "maxPosts": max_posts},
        timeout=timeout,
    )
    return [post_from_row(item, corpus, platform="linkedin") for item in items]


def fetch_x(handles, corpus, *, actor=None, max_posts=50, timeout=DEFAULT_TIMEOUT):
    """Scrape recent posts for each X handle. Returns list[Post]."""
    actor = actor or os.environ.get("APIFY_X_ACTOR", DEFAULT_X_ACTOR)
    items = run_actor(
        actor,
        {"twitterHandles": list(handles), "maxItems": max_posts},
        timeout=timeout,
    )
    return [post_from_row(item, corpus, platform="x") for item in items]


def run_actor(actor, actor_input, *, timeout=DEFAULT_TIMEOUT):
    """Run an actor to completion and return its dataset items."""
    token = _token()
    with httpx.Client(timeout=timeout) as client:
        response = client.post(
            f"{API_BASE}/acts/{actor}/run-sync-get-dataset-items",
            params={"token": token, "limit": PAGE_SIZE},
            json=actor_input,
        )
        _check(response, actor)
        items = list(response.json())
        total = int(response.headers.get("X-Apify-Pagination-Total", len(items)))
        # Re-POSTing the sync endpoint at a higher offset would run and bill the actor a
        # second time, so the remaining pages come out of the run we already paid for.
        while len(items) < total:
            page = client.get(
                f"{API_BASE}/acts/{actor}/runs/last/dataset/items",
                params={
                    "token": token,
                    "status": "SUCCEEDED",
                    "offset": len(items),
                    "limit": PAGE_SIZE,
                },
            )
            _check(page, actor)
            batch = list(page.json())
            if not batch:
                break
            items.extend(batch)
    return items


def _token():
    token = os.environ.get("APIFY_TOKEN", "").strip()
    if not token:
        raise MissingApifyToken(
            "APIFY_TOKEN is not set, so Apify ingest is unavailable. Export APIFY_TOKEN, "
            "or import a local export with: laya-social ingest --from-json <file>"
        )
    return token


def _check(response, actor):
    if response.is_success:
        return
    raise httpx.HTTPStatusError(
        f"Apify actor {actor!r} returned HTTP {response.status_code}: {response.text[:500]}",
        request=response.request,
        response=response,
    )
