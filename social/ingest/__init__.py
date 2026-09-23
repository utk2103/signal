"""Ingest: the trust boundary. External shapes in, `Post` records out.

Exports vary wildly between Apify actors and hand-rolled JSON dumps, so field names are
matched tolerantly here. Nothing in this package validates: it reshapes, and `Post`
decides what is acceptable. Both importers share this mapping so a LinkedIn export and a
LinkedIn actor land in the corpus identically.

`apify` is not imported here on purpose: it needs httpx (the `social` extra), and
`--from-json` must keep working without it.
"""

from datetime import datetime, timezone

from ..models import Post

URL_KEYS = ("url", "postUrl", "post_url", "link", "permalink", "twitterUrl")
TEXT_KEYS = ("text", "full_text", "content", "postText", "post_text", "body")
AUTHOR_KEYS = ("author", "authorName", "author_name", "authorFullName", "username", "handle")
NAME_KEYS = ("name", "fullName", "full_name", "userName", "username", "screen_name", "handle")
POSTED_AT_KEYS = (
    "posted_at", "postedAt", "createdAt", "created_at", "publishedAt", "published_at",
    "date", "timestamp", "time",
)
# Explicit count names come first: some exports put a list of comment objects under the
# bare "comments" key, and a truncated list is not a count.
LIKES_KEYS = (
    "likeCount", "likesCount", "numLikes", "reactionsCount", "total_reactions",
    "favoriteCount", "likes", "reactions",
)
COMMENTS_KEYS = ("commentCount", "commentsCount", "numComments", "replyCount", "comments")
REPOSTS_KEYS = (
    "repostCount", "repostsCount", "retweetCount", "shareCount", "numShares", "reposts", "shares",
)
FOLLOWERS_KEYS = (
    "author_followers", "authorFollowers", "followersCount", "followerCount", "followers_count",
    "numFollowers", "follower_count", "followers",
)
PLATFORM_KEYS = ("platform", "network", "source")
# Actors commonly nest the engagement counts one level down.
STATS_KEYS = ("stats", "statistics", "engagement", "metrics")


def pick(row, *keys, default=None):
    """First key present with a non-empty value. Exports disagree on spelling."""
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return default


def to_int(value, field, default=None):
    """Coerce a count to int. Counts arrive as ints, floats, or strings like '1,234'."""
    if value is None or value == "":
        return default
    if isinstance(value, bool) or isinstance(value, (list, dict)):
        raise ValueError(f"{field} must be a number, got {value!r}")
    if isinstance(value, str):
        value = value.replace(",", "").strip()
    try:
        return int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a number, got {value!r}") from exc


def to_iso(value):
    """Normalize a timestamp to a string `Post` can parse. Epoch seconds become ISO;
    everything else passes through untouched for `Post` to accept or reject."""
    if isinstance(value, dict):
        value = pick(value, "date", "iso", "timestamp")
    if value is None:
        return ""
    if isinstance(value, bool):
        return repr(value)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, timezone.utc).isoformat()
    return str(value)


def post_from_row(row, corpus, platform=None):
    """Map one external row onto `Post`. Raises ValueError for rows `Post` rejects."""
    nested_author = row.get("author") if isinstance(row.get("author"), dict) else {}
    nested_stats = next((row[k] for k in STATS_KEYS if isinstance(row.get(k), dict)), {})

    def field(*keys, nested=nested_stats):
        value = pick(row, *keys)
        return value if value is not None else pick(nested, *keys)

    author = pick(row, *AUTHOR_KEYS, default="")
    if isinstance(author, dict):
        author = pick(author, *NAME_KEYS, default="")
    return Post(
        url=str(pick(row, *URL_KEYS, default="")),
        platform=str(platform or pick(row, *PLATFORM_KEYS, default="")).strip().lower(),
        corpus=corpus,
        author=str(author),
        text=str(pick(row, *TEXT_KEYS, default="")),
        posted_at=to_iso(pick(row, *POSTED_AT_KEYS)),
        likes=to_int(field(*LIKES_KEYS), "likes", default=0),
        comments=to_int(field(*COMMENTS_KEYS), "comments", default=0),
        reposts=to_int(field(*REPOSTS_KEYS), "reposts", default=0),
        # None, never a guess: follower tiers are what mitigate the raw-engagement confound.
        author_followers=to_int(field(*FOLLOWERS_KEYS, nested=nested_author), "author_followers"),
    )
