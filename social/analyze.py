"""Regress the rubric against engagement, within follower-tier buckets.

Two guards decide whether a bucket produces a number at all: fewer than 50 posts is not
regressed, and a test R2 under 0.05 is reported as "no pattern found" rather than a table.
"""

from collections import Counter
from dataclasses import dataclass, replace

import numpy as np

from .models import load_posts
from .rubric import CHOICE_QUESTIONS, RUBRIC_VERSION, post_questions

QUESTIONS = tuple(post_questions())
NUMERIC_QUESTIONS = tuple(q for q in QUESTIONS if q not in CHOICE_QUESTIONS)

# (name, lower bound inclusive, upper bound exclusive)
FOLLOWER_BUCKETS = (
    ("micro", 0, 1_000),
    ("small", 1_000, 10_000),
    ("mid", 10_000, 100_000),
    ("large", 100_000, None),
)

MIN_BUCKET_POSTS = 50
MIN_TEST_R2 = 0.05
TRAIN_FRACTION = 0.7
MIN_FOLLOWER_COVERAGE = 0.8


@dataclass(frozen=True)
class DimensionRow:
    feature: str
    coefficient: float
    correlation: float
    bucket_mean: float
    top_decile_mean: float
    unstable: bool = False


@dataclass(frozen=True)
class BucketResult:
    name: str
    n: int
    skipped: bool
    skip_reason: str | None
    no_pattern: bool
    train_r2: float | None
    test_r2: float | None
    ranked: tuple[DimensionRow, ...]
    dropped_features: tuple[str, ...]
    top_decile_urls: tuple[str, ...]


@dataclass(frozen=True)
class AnalysisResult:
    corpus: str
    rubric_version: str
    n_posts: int
    n_scored: int
    date_range: tuple[str, str] | None
    platform_counts: tuple[tuple[str, int], ...]
    bucketed: bool
    confound_unmitigated: bool
    follower_coverage: float
    unbucketed_posts: int
    buckets: tuple[BucketResult, ...]


def engagement_target(posts) -> np.ndarray:
    """log1p of raw likes + comments + reposts. Raw counts are the agency's decision."""
    return np.log1p(np.array([p.engagement for p in posts], dtype=float))


def top_decile(posts) -> list:
    """Top 10% by engagement, highest first. At least one post whenever there is any."""
    if not posts:
        return []
    k = max(1, round(len(posts) * 0.1))
    return sorted(posts, key=lambda p: p.engagement, reverse=True)[:k]


def follower_bucket(followers: int | None) -> str | None:
    if followers is None:
        return None
    for name, low, high in FOLLOWER_BUCKETS:
        if followers >= low and (high is None or followers < high):
            return name
    return None


def load_scores(conn, corpus, rubric_version=RUBRIC_VERSION, model="laya") -> dict:
    """{url: {question: (value, label)}} for one corpus, rubric version and engine.

    Engine is part of the key: mixing Laya and Jev rows into one regression would
    compare nothing meaningful."""
    rows = conn.execute(
        """SELECT s.url, s.question, s.value, s.label
             FROM scores s JOIN posts p ON p.url = s.url
            WHERE p.corpus = ? AND s.rubric_version = ? AND s.model = ?""",
        (corpus, rubric_version, model),
    )
    scored: dict[str, dict] = {}
    for r in rows:
        scored.setdefault(r["url"], {})[r["question"]] = (r["value"], r["label"])
    return scored


def _design_matrix(posts, scored):
    """Numeric questions enter directly; choice questions are one-hot with the most
    frequent label dropped, so coefficients read relative to the most common label."""
    names = list(NUMERIC_QUESTIONS)
    columns = [np.array([scored[p.url][q][0] for p in posts], dtype=float) for q in names]
    for q in CHOICE_QUESTIONS:
        labels = [scored[p.url][q][1] for p in posts]
        counts = Counter(labels)
        baseline = counts.most_common(1)[0][0]
        for label in sorted(counts):
            if label == baseline:
                continue
            names.append(f"{q}={label}")
            columns.append(np.array([float(x == label) for x in labels]))
    return names, np.column_stack(columns)


def _r2(y, pred) -> float:
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - float(((y - pred) ** 2).sum()) / ss_tot


def _pearson(x, y) -> float:
    xc, yc = x - x.mean(), y - y.mean()
    denom = np.sqrt(float(xc @ xc) * float(yc @ yc))
    return float(xc @ yc / denom) if denom > 0 else 0.0


def _regress(name, posts, names, X, y, seed):
    n = len(posts)
    decile = top_decile(posts)
    decile_urls = tuple(p.url for p in decile)
    if n < MIN_BUCKET_POSTS:
        return BucketResult(
            name=name, n=n, skipped=True,
            skip_reason=f"fewer than {MIN_BUCKET_POSTS} posts",
            no_pattern=False, train_r2=None, test_r2=None,
            ranked=(), dropped_features=(), top_decile_urls=decile_urls,
        )

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    cut = round(n * TRAIN_FRACTION)
    train, test = order[:cut], order[cut:]

    mean, std = X[train].mean(axis=0), X[train].std(axis=0)
    keep = std > 1e-12
    dropped = tuple(c for c, k in zip(names, keep) if not k)
    kept_names = [c for c, k in zip(names, keep) if k]
    Xs = (X[:, keep] - mean[keep]) / std[keep]

    A_train = np.column_stack([np.ones(len(train)), Xs[train]])
    coef, *_ = np.linalg.lstsq(A_train, y[train], rcond=None)
    A_test = np.column_stack([np.ones(len(test)), Xs[test]])
    train_r2 = _r2(y[train], A_train @ coef)
    test_r2 = _r2(y[test], A_test @ coef)

    decile_idx = [i for i, p in enumerate(posts) if p.url in set(decile_urls)]
    kept_X = X[:, keep]
    rows = [
        DimensionRow(
            feature=feature,
            coefficient=float(coef[j + 1]),
            correlation=_pearson(kept_X[:, j], y),
            bucket_mean=float(kept_X[:, j].mean()),
            top_decile_mean=float(kept_X[decile_idx, j].mean()) if decile_idx else 0.0,
        )
        for j, feature in enumerate(kept_names)
    ]
    # Rank by |correlation|, not |coefficient|. With one-hot hook/cta dummies the design
    # matrix is collinear, and a partial OLS coefficient can invert sign against the
    # dimension's own marginal association -- measured on a planted-signal corpus, the true
    # driver came back r=+0.72 with coefficient -1.02 and ranked 5th behind an artifact.
    # Marginal association is also the question an account manager is actually asking.
    rows = [
        replace(r, unstable=r.coefficient * r.correlation < 0 and abs(r.correlation) > 0.1)
        for r in rows
    ]
    rows.sort(key=lambda r: abs(r.correlation), reverse=True)

    return BucketResult(
        name=name, n=n, skipped=False, skip_reason=None,
        no_pattern=test_r2 < MIN_TEST_R2,
        train_r2=train_r2, test_r2=test_r2,
        ranked=tuple(rows), dropped_features=dropped, top_decile_urls=decile_urls,
    )


def analyze(conn, corpus, rubric_version=RUBRIC_VERSION, bucket=True, seed=0) -> AnalysisResult:
    all_posts = load_posts(conn, corpus)
    scored = load_scores(conn, corpus, rubric_version)
    needed = set(QUESTIONS)
    posts = [p for p in all_posts if needed <= scored.get(p.url, {}).keys()]

    platform_counts = tuple(sorted(Counter(p.platform for p in all_posts).items()))
    dates = [p.posted_at for p in all_posts]
    date_range = (min(dates), max(dates)) if dates else None

    if not posts:
        return AnalysisResult(
            corpus=corpus, rubric_version=rubric_version, n_posts=len(all_posts), n_scored=0,
            date_range=date_range, platform_counts=platform_counts,
            bucketed=False, confound_unmitigated=True, follower_coverage=0.0,
            unbucketed_posts=0, buckets=(),
        )

    coverage = sum(p.author_followers is not None for p in posts) / len(posts)
    bucketed = bucket and coverage >= MIN_FOLLOWER_COVERAGE

    names, X = _design_matrix(posts, scored)
    y = engagement_target(posts)

    if not bucketed:
        buckets = (_regress("all", posts, names, X, y, seed),)
        unbucketed = 0
    else:
        assigned = [follower_bucket(p.author_followers) for p in posts]
        unbucketed = sum(a is None for a in assigned)
        results = []
        for name, _, _ in FOLLOWER_BUCKETS:
            idx = [i for i, a in enumerate(assigned) if a == name]
            if not idx:
                continue
            results.append(
                _regress(name, [posts[i] for i in idx], names, X[idx], y[idx], seed)
            )
        buckets = tuple(results)

    return AnalysisResult(
        corpus=corpus, rubric_version=rubric_version,
        n_posts=len(all_posts), n_scored=len(posts),
        date_range=date_range, platform_counts=platform_counts,
        bucketed=bucketed, confound_unmitigated=not bucketed,
        follower_coverage=coverage, unbucketed_posts=unbucketed, buckets=buckets,
    )
