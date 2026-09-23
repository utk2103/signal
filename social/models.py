"""Post records and the SQLite corpus that stores them.

The corpus is the durable local archive: `posts.text` holds the full body, so every
downstream stage reads from here and nothing needs a second store.
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PLATFORMS = ("linkedin", "x")

# Mirrors email.clean_email_body's budget. The tokenizer truncates to the model's
# max_len anyway; this just avoids tokenizing text that can never fit.
MAX_POST_CHARS = 3000

SCHEMA = """
CREATE TABLE IF NOT EXISTS posts (
  url              TEXT PRIMARY KEY,
  platform         TEXT    NOT NULL,
  corpus           TEXT    NOT NULL,
  author           TEXT    NOT NULL,
  author_followers INTEGER,
  text             TEXT    NOT NULL,
  posted_at        TEXT    NOT NULL,
  likes            INTEGER NOT NULL,
  comments         INTEGER NOT NULL,
  reposts          INTEGER NOT NULL,
  scraped_at       TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS posts_corpus ON posts(corpus);

CREATE TABLE IF NOT EXISTS scores (
  url            TEXT NOT NULL REFERENCES posts(url),
  question       TEXT NOT NULL,
  rubric_version TEXT NOT NULL,
  model          TEXT NOT NULL,
  value          REAL NOT NULL,
  label          TEXT,
  confidence     REAL NOT NULL,
  scored_at      TEXT NOT NULL,
  PRIMARY KEY (url, question, rubric_version, model)
);

CREATE TABLE IF NOT EXISTS swipe (
  url      TEXT PRIMARY KEY REFERENCES posts(url),
  added_at TEXT NOT NULL,
  reason   TEXT NOT NULL,
  note     TEXT
);
"""


@dataclass(frozen=True)
class Post:
    """One scraped post. Constructed at the ingest trust boundary, so it validates here."""

    url: str
    platform: str
    corpus: str
    author: str
    text: str
    posted_at: str
    likes: int
    comments: int
    reposts: int
    author_followers: int | None = None

    def __post_init__(self):
        if not self.url.startswith(("http://", "https://")):
            raise ValueError(f"Post url must be an http(s) URL: {self.url!r}")
        if self.platform not in PLATFORMS:
            raise ValueError(f"platform must be one of {PLATFORMS}, got {self.platform!r}")
        for field in ("corpus", "author", "text"):
            if not getattr(self, field).strip():
                raise ValueError(f"Post {field} must be a nonempty string")
        try:
            datetime.fromisoformat(self.posted_at)
        except ValueError as exc:
            raise ValueError(f"posted_at must be ISO 8601: {self.posted_at!r}") from exc
        for field in ("likes", "comments", "reposts"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"Post {field} must be a non-negative int, got {value!r}")
        followers = self.author_followers
        if followers is not None and (
            not isinstance(followers, int) or isinstance(followers, bool) or followers < 0
        ):
            raise ValueError(f"author_followers must be a non-negative int or None, got {followers!r}")

    @property
    def engagement(self) -> int:
        return self.likes + self.comments + self.reposts

    @property
    def is_truncated(self) -> bool:
        """True when scoring sees less than the full body. Full text is still stored."""
        return len(self.text) > MAX_POST_CHARS


def connect(path) -> sqlite3.Connection:
    """Open the corpus, creating the schema if needed."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    existing = {r[1] for r in conn.execute("PRAGMA table_info(scores)")}
    if existing and "model" not in existing:
        raise ValueError(
            f"{path} predates the `model` column in `scores` (two engines would overwrite "
            "each other). Delete it and re-ingest; there is no migration."
        )
    conn.executescript(SCHEMA)
    return conn


def insert_posts(conn, posts, scraped_at=None) -> int:
    """Upsert posts, refreshing engagement counts on re-scrape. Returns rows written."""
    scraped_at = scraped_at or datetime.now().astimezone().isoformat()
    rows = [
        (
            p.url, p.platform, p.corpus, p.author, p.author_followers,
            p.text, p.posted_at, p.likes, p.comments, p.reposts, scraped_at,
        )
        for p in posts
    ]
    with conn:
        conn.executemany(
            """INSERT INTO posts VALUES (?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(url) DO UPDATE SET
                 likes=excluded.likes, comments=excluded.comments,
                 reposts=excluded.reposts, author_followers=excluded.author_followers,
                 scraped_at=excluded.scraped_at""",
            rows,
        )
    return len(rows)


def load_posts(conn, corpus) -> list[Post]:
    rows = conn.execute("SELECT * FROM posts WHERE corpus = ? ORDER BY posted_at", (corpus,))
    return [
        Post(
            url=r["url"], platform=r["platform"], corpus=r["corpus"], author=r["author"],
            text=r["text"], posted_at=r["posted_at"], likes=r["likes"], comments=r["comments"],
            reposts=r["reposts"], author_followers=r["author_followers"],
        )
        for r in rows
    ]
