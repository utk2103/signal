"""Ingest maps external shapes onto Post; Post decides what is valid.

The Apify tests never touch the network: httpx.Client is replaced with a fake serving
fixtures shaped like real actor output.
"""

import json

import httpx
import pytest

from social.ingest import apify
from social.ingest.jsonfile import read_json_posts

LINKEDIN_ITEMS = [
    {
        "urn": "urn:li:activity:7100",
        "post_url": "https://www.linkedin.com/posts/jane-doe_activity-7100",
        "text": "We cut onboarding from 11 days to 2. Here is the whole runbook.",
        "posted_at": {"date": "2026-08-14 09:12:00", "timestamp": 1786698720},
        "author": {
            "full_name": "Jane Doe",
            "username": "jane-doe",
            "followers": 12400,
        },
        "stats": {"total_reactions": 812, "comments": 96, "reposts": 41},
    },
    {
        "post_url": "https://www.linkedin.com/posts/sam-lee_activity-7101",
        "text": "Unpopular opinion: your ICP doc is a wish list.",
        "posted_at": {"date": "2026-08-15 14:00:00"},
        "author": {"full_name": "Sam Lee", "username": "sam-lee"},
        "stats": {"total_reactions": 44, "comments": 7, "reposts": 0},
    },
]

X_ITEMS = [
    {
        "url": "https://x.com/devops_dan/status/1899",
        "text": "Shipped the migration at 3am. Nothing broke. Suspicious.",
        "createdAt": "2026-08-14T09:12:00.000Z",
        "likeCount": 530,
        "replyCount": 22,
        "retweetCount": 61,
        "author": {"userName": "devops_dan", "followers": 8300},
    },
    {
        "url": "https://x.com/devops_dan/status/1900",
        "text": "Follow-up thread on the rollback plan.",
        "createdAt": 1786698720,
        "likeCount": 12,
        "replyCount": 1,
        "retweetCount": 0,
        "author": {"userName": "devops_dan", "followers": 8300},
    },
]


class FakeResponse:
    def __init__(self, payload, headers=None, status_code=200, text=""):
        self._payload = payload
        self.headers = headers or {}
        self.status_code = status_code
        self.text = text
        self.request = httpx.Request("POST", apify.API_BASE)

    @property
    def is_success(self):
        return 200 <= self.status_code < 300

    def json(self):
        return self._payload


class FakeClient:
    def __init__(self, responses, calls, **kwargs):
        self.responses = responses
        self.calls = calls
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, params=None, json=None):
        self.calls.append(("POST", url, params, json))
        return self.responses.pop(0)

    def get(self, url, params=None):
        self.calls.append(("GET", url, params, None))
        return self.responses.pop(0)


@pytest.fixture
def fake_apify(monkeypatch):
    """Serve canned responses to apify.run_actor and record every request made."""
    monkeypatch.setenv("APIFY_TOKEN", "test-token")
    calls = []

    def install(*responses):
        queue = list(responses)
        monkeypatch.setattr(apify.httpx, "Client", lambda **kw: FakeClient(queue, calls, **kw))
        return calls

    return install


def test_linkedin_actor_item_maps_onto_post(fake_apify):
    fake_apify(FakeResponse(LINKEDIN_ITEMS))
    posts = apify.fetch_linkedin(["https://www.linkedin.com/in/jane-doe"], "acme")
    assert [p.url for p in posts] == [item["post_url"] for item in LINKEDIN_ITEMS]
    first = posts[0]
    assert first.platform == "linkedin"
    assert first.corpus == "acme"
    assert first.author == "Jane Doe"
    assert first.posted_at == "2026-08-14 09:12:00"
    assert (first.likes, first.comments, first.reposts) == (812, 96, 41)
    assert first.author_followers == 12400
    assert first.engagement == 949


def test_x_actor_item_maps_onto_post(fake_apify):
    fake_apify(FakeResponse(X_ITEMS))
    posts = apify.fetch_x(["devops_dan"], "acme")
    assert [p.platform for p in posts] == ["x", "x"]
    assert posts[0].author == "devops_dan"
    assert (posts[0].likes, posts[0].comments, posts[0].reposts) == (530, 22, 61)
    assert posts[0].author_followers == 8300
    assert posts[1].posted_at.startswith("2026-08-14T09:12:00")  # epoch seconds normalized


def test_missing_follower_count_stays_none(fake_apify):
    fake_apify(FakeResponse([LINKEDIN_ITEMS[1]]))
    (post,) = apify.fetch_linkedin(["https://www.linkedin.com/in/sam-lee"], "acme")
    assert post.author_followers is None


def test_pagination_reads_the_rest_from_the_run_we_paid_for(fake_apify):
    calls = fake_apify(
        FakeResponse(X_ITEMS[:1], headers={"X-Apify-Pagination-Total": "2"}),
        FakeResponse(X_ITEMS[1:]),
    )
    posts = apify.fetch_x(["devops_dan"], "acme")
    assert len(posts) == 2
    assert calls[0][0] == "POST" and calls[0][1].endswith("/run-sync-get-dataset-items")
    assert calls[1][0] == "GET" and calls[1][1].endswith("/runs/last/dataset/items")
    assert calls[1][2]["offset"] == 1


def test_actor_id_is_configuration(fake_apify, monkeypatch):
    calls = fake_apify(FakeResponse([]), FakeResponse([]))
    monkeypatch.setenv("APIFY_X_ACTOR", "agency~custom-x-scraper")
    apify.fetch_x(["devops_dan"], "acme")
    apify.fetch_linkedin(["jane-doe"], "acme", actor="agency~custom-li-scraper")
    assert "agency~custom-x-scraper" in calls[0][1]
    assert "agency~custom-li-scraper" in calls[1][1]


def test_missing_token_names_the_variable(monkeypatch):
    monkeypatch.delenv("APIFY_TOKEN", raising=False)
    with pytest.raises(apify.MissingApifyToken) as excinfo:
        apify.fetch_linkedin(["jane-doe"], "acme")
    assert "APIFY_TOKEN" in str(excinfo.value)


def test_http_error_surfaces_status_and_body(fake_apify):
    fake_apify(FakeResponse(None, status_code=402, text="monthly usage hard limit exceeded"))
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        apify.fetch_x(["devops_dan"], "acme")
    assert "402" in str(excinfo.value)
    assert "hard limit exceeded" in str(excinfo.value)


def test_jsonl_export_maps_alias_fields(tmp_path):
    rows = [
        {
            "postUrl": "https://www.linkedin.com/posts/jane-doe_activity-1",
            "content": "Alias spellings everywhere.",
            "authorName": "Jane Doe",
            "postedAt": "2026-08-14T09:12:00Z",
            "numLikes": "1,204",
            "numComments": 33,
            "numShares": 9,
            "followersCount": 12400,
            "platform": "LinkedIn",
        },
        {
            "link": "https://x.com/devops_dan/status/1899",
            "postText": "Short one.",
            "username": "devops_dan",
            "date": 1786698720,
            "likes": 5,
            "network": "x",
        },
    ]
    path = tmp_path / "export.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")

    posts, errors = read_json_posts(path, "acme")
    assert errors == []
    assert [p.platform for p in posts] == ["linkedin", "x"]
    assert posts[0].likes == 1204
    assert posts[0].author_followers == 12400
    assert (posts[1].comments, posts[1].reposts) == (0, 0)
    assert posts[1].author_followers is None


def test_bad_rows_are_skipped_with_index_and_reason(tmp_path):
    good = {
        "url": "https://x.com/devops_dan/status/1899",
        "text": "Fine.",
        "author": "devops_dan",
        "posted_at": "2026-08-14T09:12:00Z",
        "likes": 5,
    }
    path = tmp_path / "export.json"
    path.write_text(
        json.dumps(
            [
                good,
                {**good, "url": "not-a-url"},
                {**good, "url": "https://x.com/a/2", "posted_at": "last tuesday"},
                "just a string",
                {**good, "url": "https://x.com/a/3", "likes": "many"},
                {**good, "url": "https://x.com/a/4"},
            ]
        )
    )

    posts, errors = read_json_posts(path, "acme", platform="x")
    assert len(posts) == 2
    assert len(errors) == 4
    assert errors[0].startswith("row 1:") and "not-a-url" in errors[0]
    assert errors[1].startswith("row 2:") and "ISO 8601" in errors[1]
    assert errors[2].startswith("row 3:") and "JSON object" in errors[2]
    assert errors[3].startswith("row 4:") and "likes" in errors[3]


def test_malformed_jsonl_line_does_not_stop_the_import(tmp_path):
    good = json.dumps(
        {
            "url": "https://x.com/devops_dan/status/1899",
            "text": "Fine.",
            "author": "devops_dan",
            "posted_at": "2026-08-14T09:12:00Z",
            "platform": "x",
        }
    )
    path = tmp_path / "export.jsonl"
    path.write_text(f"{good}\n{{oops\n{good.replace('/1899', '/1900')}\n")

    posts, errors = read_json_posts(path, "acme")
    assert len(posts) == 2
    assert errors[0].startswith("row 1:") and "invalid JSON" in errors[0]
