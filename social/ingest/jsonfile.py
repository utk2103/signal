"""Import posts from a local JSON array or JSONL export.

This is a first-class ingest path, not a fallback: the whole pipeline runs, and is fully
testable, with no Apify account. A 5,000-row export with 3 bad rows imports 4,997 and
reports the three.
"""

import json
from pathlib import Path

from . import post_from_row


def read_json_posts(path, corpus, platform=None):
    """Parse an export into Posts.

    Returns `(posts, errors)`: imported = len(posts), skipped = len(errors). Each error
    reads 'row N: reason', N counting records (blank JSONL lines are not records).
    `platform` overrides the row's own platform field; without either, `Post` rejects
    the row.
    """
    posts, errors = [], []
    for index, row in enumerate(_rows(path)):
        if isinstance(row, json.JSONDecodeError):
            errors.append(f"row {index}: invalid JSON ({row.msg})")
            continue
        if not isinstance(row, dict):
            errors.append(f"row {index}: expected a JSON object, got {type(row).__name__}")
            continue
        try:
            posts.append(post_from_row(row, corpus, platform))
        except ValueError as exc:
            errors.append(f"row {index}: {exc}")
    return posts, errors


def _rows(path):
    """Yield one record per post: a JSON array or JSONL, decided by the first character.
    A malformed JSONL line yields its JSONDecodeError so the caller can skip just it."""
    text = Path(path).expanduser().read_text()
    if text.lstrip().startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{path}: top-level JSON must be an array of post objects")
        yield from data
        return
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            yield exc
