"""Score a corpus against the rubric and write the `scores` table.

`cache_prompts=True` is the point: every post asks the same ten questions, so the rubric
prefix is tokenized and built once by PrefixCache and reused for the whole corpus.
"""

import sys
import time
from datetime import datetime

import laya_mlx as laya

from .models import load_posts
from .rubric import RUBRIC_VERSION, post_questions, post_state

DEFAULT_MODEL = "aac6fef/laya-mlx"
# Which engine produced a row. The live probe writes "jev" for the same rubric.
ENGINE = "laya"


def answer_to_row(answer: dict) -> tuple[float, str | None]:
    """Map one laya answer onto the (value, label) pair the `scores` table stores."""
    kind = answer["type"]
    if kind == "choice":
        label = answer["choice"]
        return float(answer["probabilities"][label]), label
    if kind == "score":
        return float(answer["score"]), None
    return float(answer["noul"]), None


def score_corpus(
    conn,
    corpus: str,
    *,
    model: str = DEFAULT_MODEL,
    batch_size: int = 32,
    dtype: str = "float16",
) -> dict:
    """Score every post in `corpus` not yet scored at the current RUBRIC_VERSION.

    Returns a run summary: posts scored, posts skipped, posts truncated, wall seconds,
    and posts/sec. Re-running after a completed run scores nothing and costs one query.
    """
    questions = post_questions()
    posts = load_posts(conn, corpus)
    done = {
        r["url"]
        for r in conn.execute(
            "SELECT DISTINCT url FROM scores WHERE rubric_version = ? AND model = ?",
            (RUBRIC_VERSION, ENGINE),
        )
    }
    # A post's ten rows commit in one transaction, so any row present means fully scored.
    pending = [p for p in posts if p.url not in done]
    summary = {
        "corpus": corpus,
        "total": len(posts),
        "skipped": len(posts) - len(pending),
        "scored": 0,
        "truncated": sum(p.is_truncated for p in pending),
        "seconds": 0.0,
        "posts_per_sec": 0.0,
    }
    if not pending:
        print(f"[score] {corpus}: nothing to do, {summary['skipped']} posts already at "
              f"rubric {RUBRIC_VERSION}", file=sys.stderr)
        return summary

    agent = laya.load(model, cache_prompts=True, batch_size=batch_size, dtype=dtype)

    # Two different chunkings, easy to conflate:
    #   - laya's own batch_size batches the QUESTIONS inside one predict() call, because
    #     predict() takes a single state. With batch_size >= 10 each post is one forward
    #     pass over its ten questions.
    #   - `chunk` below batches POSTS, and exists only to bound the transaction: one
    #     commit per chunk, so an interrupt loses at most this chunk instead of the run.
    # A 32-post chunk is therefore 320 decisions across 32 forward passes, committed once.
    start = time.perf_counter()
    for offset in range(0, len(pending), batch_size):
        chunk = pending[offset : offset + batch_size]
        rows = []
        for post in chunk:
            try:
                answers = agent.predict(post_state(post), questions)["answers"]
            except FloatingPointError as exc:
                raise FloatingPointError(
                    f"{exc} (post {post.url}); pass dtype='float32' to score_corpus"
                ) from exc
            scored_at = datetime.now().astimezone().isoformat()
            for question, answer in answers.items():
                value, label = answer_to_row(answer)
                rows.append(
                    (post.url, question, RUBRIC_VERSION, ENGINE, value, label,
                     float(answer["confidence"]), scored_at)
                )
        with conn:
            conn.executemany("INSERT OR REPLACE INTO scores VALUES (?,?,?,?,?,?,?,?)", rows)
        summary["scored"] += len(chunk)
        elapsed = time.perf_counter() - start
        print(
            f"[score] {summary['scored']}/{len(pending)} posts, {elapsed:.1f}s, "
            f"{summary['scored'] / elapsed:.2f} posts/sec",
            file=sys.stderr,
        )

    summary["seconds"] = round(time.perf_counter() - start, 2)
    summary["posts_per_sec"] = round(summary["scored"] / summary["seconds"], 2)
    print(
        f"[score] done: {summary['scored']} scored, {summary['skipped']} skipped, "
        f"{summary['truncated']} truncated at MAX_POST_CHARS, "
        f"{summary['seconds']}s, {summary['posts_per_sec']} posts/sec",
        file=sys.stderr,
    )
    return summary
