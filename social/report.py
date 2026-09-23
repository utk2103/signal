"""PATTERN.md and swipe.md writers, plus the auto half of the swipe table.

Markdown only: the SQLite corpus is already the machine-readable store, so a second
machine format would be redundant.
"""

from datetime import datetime
from pathlib import Path

from .analyze import QUESTIONS, load_scores
from .models import load_posts
from .rubric import CHOICE_QUESTIONS

AUTO_REASON = "auto:top-decile"


def score_line(scores: dict) -> str:
    """One readable line of all ten rubric answers for a post."""
    parts = []
    for q in QUESTIONS:
        if q not in scores:
            continue
        value, label = scores[q]
        parts.append(f"{q}={label} ({value:.2f})" if q in CHOICE_QUESTIONS else f"{q} {value:.2f}")
    return " · ".join(parts)


def refresh_auto_swipe(conn, result) -> int:
    """Replace this corpus's auto:top-decile rows from the current top decile per bucket.
    Manual rows are never touched."""
    urls = sorted({url for b in result.buckets for url in b.top_decile_urls})
    added_at = datetime.now().astimezone().isoformat()
    with conn:
        conn.execute(
            """DELETE FROM swipe WHERE reason = ?
                 AND url IN (SELECT url FROM posts WHERE corpus = ?)""",
            (AUTO_REASON, result.corpus),
        )
        conn.executemany(
            """INSERT INTO swipe (url, added_at, reason, note) VALUES (?, ?, ?, NULL)
               ON CONFLICT(url) DO NOTHING""",
            [(u, added_at, AUTO_REASON) for u in urls],
        )
    return len(urls)


def add_manual_swipe(conn, url, note=None) -> None:
    """Star a post by hand. Manual rows survive refresh_auto_swipe, and their notes are the
    seed corpus for the next rubric revision."""
    if conn.execute("SELECT 1 FROM posts WHERE url = ?", (url,)).fetchone() is None:
        raise ValueError(f"No post in the corpus with url {url!r}; ingest it first")
    with conn:
        conn.execute(
            """INSERT INTO swipe (url, added_at, reason, note) VALUES (?, ?, 'manual', ?)
               ON CONFLICT(url) DO UPDATE SET reason='manual', note=excluded.note""",
            (url, datetime.now().astimezone().isoformat(), note),
        )


def write_pattern(conn, result, path) -> Path:
    posts = {p.url: p for p in load_posts(conn, result.corpus)}
    scored = load_scores(conn, result.corpus, result.rubric_version)
    out = [f"# Pattern report — {result.corpus}", ""]

    if result.confound_unmitigated:
        out += [
            "> **Confound unmitigated.** Follower bucketing was skipped "
            f"(`author_followers` present for {result.follower_coverage:.0%} of scored posts), "
            "so one regression ran across all account sizes. Raw engagement counts are "
            "confounded by audience size: these coefficients may be measuring "
            "*big account* rather than *good post*.",
            "",
        ]

    out += ["## Corpus", ""]
    out.append(f"- Posts: {result.n_posts} ({result.n_scored} scored at rubric "
               f"{result.rubric_version})")
    if result.date_range:
        out.append(f"- Date range: {result.date_range[0]} → {result.date_range[1]}")
    out.append("- Platforms: " + (", ".join(f"{p} {n}" for p, n in result.platform_counts) or "—"))
    out.append("- Bucketing: " + (
        "by follower tier" if result.bucketed
        else f"skipped (follower coverage {result.follower_coverage:.0%})"
    ))
    if result.unbucketed_posts:
        out.append(f"- Posts with no follower count, excluded from buckets: "
                   f"{result.unbucketed_posts}")
    out.append("")

    if not result.buckets:
        out += ["No scored posts at this rubric version. Nothing to regress.", ""]
        return _write(path, out)

    for bucket in result.buckets:
        out += [f"## Bucket: {bucket.name}", "", f"- Posts: {bucket.n}"]

        if bucket.skipped:
            out += [f"- **Skipped:** {bucket.skip_reason}. Not enough data yet.", ""]
        else:
            out.append(f"- Train R²: {bucket.train_r2:.3f} · Test R²: {bucket.test_r2:.3f}")
            if bucket.dropped_features:
                out.append("- Dropped (zero variance): " + ", ".join(bucket.dropped_features))
            out.append("")
            if bucket.no_pattern:
                out += [
                    f"**No pattern found.** Out-of-sample R² is {bucket.test_r2:.3f}; the rubric "
                    "does not predict engagement in this bucket. No ranked dimensions are "
                    "reported, because ranking noise would be astrology.",
                    "",
                ]
            else:
                out += ["Ranked by |correlation| — the dimension's own association with "
                        "engagement. Rows marked ⚠ have an OLS coefficient whose sign "
                        "contradicts that correlation, which means collinearity with another "
                        "dimension: trust the correlation, not the coefficient.", ""]
                out += ["| dimension | correlation | coefficient | |", "|---|---:|---:|:--|"]
                out += [f"| {r.feature} | {r.correlation:+.3f} | {r.coefficient:+.3f} | "
                        f"{'⚠' if r.unstable else ''} |"
                        for r in bucket.ranked]
                out.append("")

        out += _top_decile_section(bucket, posts, scored)

        if not bucket.skipped and not bucket.no_pattern:
            out += ["### What the top decile shares", ""]
            out += ["| dimension | top-decile mean | bucket mean |", "|---|---:|---:|"]
            out += [f"| {r.feature} | {r.top_decile_mean:.3f} | {r.bucket_mean:.3f} |"
                    for r in bucket.ranked]
            out.append("")

    return _write(path, out)


def _top_decile_section(bucket, posts, scored):
    out = [f"### Top decile ({len(bucket.top_decile_urls)} posts)", ""]
    header = ["url", "engagement", *QUESTIONS]
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "---|" * len(header))
    for url in bucket.top_decile_urls:
        post = posts[url]
        cells = [f"[link]({url})", str(post.engagement)]
        for q in QUESTIONS:
            value, label = scored[url][q]
            cells.append(label if q in CHOICE_QUESTIONS else f"{value:.2f}")
        out.append("| " + " | ".join(cells) + " |")
    out.append("")
    return out


def write_swipe(conn, corpus, path) -> Path:
    rows = conn.execute(
        """SELECT p.url, p.author, p.platform, p.text, p.posted_at, p.likes, p.comments,
                  p.reposts, s.reason, s.note, s.added_at
             FROM swipe s JOIN posts p ON p.url = s.url
            WHERE p.corpus = ?
            ORDER BY (p.likes + p.comments + p.reposts) DESC""",
        (corpus,),
    ).fetchall()
    scored = load_scores(conn, corpus)

    out = [f"# Swipe file — {corpus}", "", f"{len(rows)} posts.", ""]
    for r in rows:
        engagement = r["likes"] + r["comments"] + r["reposts"]
        out += [
            f"## {r['author']} — {engagement} engagement",
            "",
            f"{r['platform']} · {r['posted_at']} · [permalink]({r['url']}) · `{r['reason']}`",
            "",
        ]
        if r["note"]:
            out += [f"> **Note:** {r['note']}", ""]
        out += [r["text"], ""]
        line = score_line(scored.get(r["url"], {}))
        if line:
            out += [f"`{line}`", ""]
        out += ["---", ""]
    return _write(path, out)


def _write(path, lines) -> Path:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path
