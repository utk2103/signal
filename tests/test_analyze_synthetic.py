"""The regression has to find a signal that is definitely there and refuse one that is not.

Pure numpy, no checkpoint download: synthetic scores are written straight into the
`scores` table, so this runs in CI in seconds.
"""

from datetime import datetime, timedelta

import numpy as np

from social.analyze import MIN_BUCKET_POSTS, analyze
from social.models import Post, connect, insert_posts
from social.report import refresh_auto_swipe, write_pattern, write_swipe
from social.rubric import CHOICE_QUESTIONS, RUBRIC_VERSION, post_questions

QUESTIONS = post_questions()
CHOICE_LABELS = {q: sorted(QUESTIONS[q]["criteria"]) for q in CHOICE_QUESTIONS}
NUMERIC = [q for q in QUESTIONS if q not in CHOICE_QUESTIONS]
SCORE_MAX = {q: len(QUESTIONS[q]["criteria"]) - 1 for q in NUMERIC if QUESTIONS[q]["type"] == "score"}

# engagement = 2.0 + 1.2 * specificity + 1.5 * personal_stakes + noise, in log1p space.
SIGNAL = {"specificity": 1.2, "personal_stakes": 1.5}


def _draw(rng, question):
    if QUESTIONS[question]["type"] == "score":
        return float(rng.uniform(0.0, SCORE_MAX[question]))
    return float(rng.uniform(0.0, 1.0))


def build_corpus(conn, corpus, n, rng, signal, followers=None, base=2.0, noise=0.15):
    posts, score_rows = [], []
    scored_at = datetime(2026, 1, 1).isoformat()
    for i in range(n):
        values = {q: _draw(rng, q) for q in NUMERIC}
        y = base + sum(w * values[q] for q, w in signal.items()) + rng.normal(0, noise)
        url = f"https://example.com/{corpus}/{i}"
        posts.append(
            Post(
                url=url,
                platform="linkedin" if i % 2 else "x",
                corpus=corpus,
                author=f"author-{i % 7}",
                text=f"synthetic post {i}",
                posted_at=(datetime(2026, 1, 1) + timedelta(days=i)).isoformat(),
                likes=max(0, int(round(np.expm1(y)))),
                comments=0,
                reposts=0,
                author_followers=followers,
            )
        )
        for q, value in values.items():
            score_rows.append((url, q, RUBRIC_VERSION, "laya", value, None, 0.9, scored_at))
        for q, labels in CHOICE_LABELS.items():
            label = labels[int(rng.integers(len(labels)))]
            score_rows.append((url, q, RUBRIC_VERSION, "laya", float(rng.uniform(0.4, 1.0)),
                               label, 0.9, scored_at))

    insert_posts(conn, posts)
    with conn:
        conn.executemany("INSERT INTO scores VALUES (?,?,?,?,?,?,?,?)", score_rows)
    return posts


def test_recovers_the_two_planted_dimensions(tmp_path):
    conn = connect(tmp_path / "signal.sqlite")
    build_corpus(conn, "signal", 300, np.random.default_rng(1), SIGNAL)

    result = analyze(conn, "signal")
    assert not result.bucketed and result.confound_unmitigated
    (bucket,) = result.buckets
    assert bucket.name == "all"
    assert bucket.n == 300
    assert not bucket.skipped and not bucket.no_pattern
    assert bucket.test_r2 > 0.5

    top_two = {row.feature for row in bucket.ranked[:2]}
    assert top_two == set(SIGNAL)
    # The bigger planted weight times the bigger spread has to rank first.
    assert bucket.ranked[0].feature == "specificity"
    assert bucket.ranked[0].correlation > 0


def test_null_guard_fires_on_pure_noise(tmp_path):
    conn = connect(tmp_path / "noise.sqlite")
    build_corpus(conn, "noise", 300, np.random.default_rng(2), signal={}, noise=1.0)

    (bucket,) = analyze(conn, "noise").buckets
    assert not bucket.skipped
    assert bucket.no_pattern
    assert bucket.test_r2 < 0.05


def test_small_bucket_is_skipped_not_regressed(tmp_path):
    conn = connect(tmp_path / "buckets.sqlite")
    rng = np.random.default_rng(3)
    build_corpus(conn, "tiers", 30, rng, SIGNAL, followers=500)
    build_corpus(conn, "tiers2", 120, rng, SIGNAL, followers=5_000)
    # Same corpus tag for both tiers; rebuild under one name via a direct update.
    with conn:
        conn.execute("UPDATE posts SET corpus = 'tiers' WHERE corpus = 'tiers2'")

    result = analyze(conn, "tiers")
    assert result.bucketed and not result.confound_unmitigated
    by_name = {b.name: b for b in result.buckets}

    micro = by_name["micro"]
    assert micro.n == 30 < MIN_BUCKET_POSTS
    assert micro.skipped and micro.train_r2 is None and micro.ranked == ()
    assert "50" in micro.skip_reason

    small = by_name["small"]
    assert small.n == 120
    assert not small.skipped and not small.no_pattern
    assert {row.feature for row in small.ranked[:2]} == set(SIGNAL)


def test_reports_render(tmp_path):
    conn = connect(tmp_path / "report.sqlite")
    build_corpus(conn, "acme", 120, np.random.default_rng(4), SIGNAL)
    result = analyze(conn, "acme")

    assert refresh_auto_swipe(conn, result) == 12
    with conn:
        conn.execute(
            "INSERT INTO swipe VALUES (?,?,?,?)",
            ("https://example.com/acme/0", "2026-01-01", "manual", "kept by hand"),
        )
    # A refresh must not evict the manual row.
    refresh_auto_swipe(conn, result)
    manual = conn.execute("SELECT COUNT(*) FROM swipe WHERE reason = 'manual'").fetchone()[0]
    assert manual == 1

    pattern = write_pattern(conn, result, tmp_path / "PATTERN.md").read_text()
    assert "Confound unmitigated" in pattern
    assert "Test R²" in pattern
    assert "specificity" in pattern

    swipe = write_swipe(conn, "acme", tmp_path / "swipe.md").read_text()
    assert "kept by hand" in swipe
    assert "specificity" in swipe
