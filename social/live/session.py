"""The live loop: observe, dedup, score, paint, record.

Laya scores every post; Jev sees every Nth while budget remains. Both write to the same
`scores` table, tagged by engine, so the Phase 1 regression can later compare them -- on the
common subset, which is the only comparison that means anything.
"""

import json
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from laya_mlx.trex import backends

from ..models import insert_posts
from ..report import add_manual_swipe
from . import overlay
from .budget import Budget, Sampler, score_one
from .extract import ExtractionFailed, extract_raw, post_from_raw

QUEUE_MAX = 64


class Session:
    def __init__(self, browser, conn, corpus, *, platform="linkedin", model=None,
                 jev_every=5, jev_budget_usd=1.0, record=None):
        self.browser = browser
        self.page = browser.page
        self.conn = conn
        self.corpus = corpus
        self.platform = platform
        self.laya = backends.create("laya", model=model)
        self.budget = Budget(jev_budget_usd)
        self.sampler = Sampler(jev_every)
        self.jev = self._open_jev() if jev_every > 0 and jev_budget_usd > 0 else None
        self.record = self._open_record(record)

        self.seen = set()
        self.pending = deque(maxlen=QUEUE_MAX)
        self.stats = {"seen": 0, "scored": 0, "dropped": 0, "sponsored": 0,
                      "jev_scored": 0, "jev_errors": 0, "agree": 0, "compared": 0}
        self.latency = {"laya": deque(maxlen=200), "jev": deque(maxlen=200)}
        self.series = {k: deque(maxlen=200)
                       for k in ("laya", "jev", "lat_laya", "lat_jev", "agreement")}

    @staticmethod
    def _open_record(record):
        """`--session artifacts/live/session.jsonl` names a directory you have not made yet."""
        if not record:
            return None
        path = Path(record).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        return open(path, "a")

    def _open_jev(self):
        """Jev is optional: no key means a Laya-only session, announced, not fatal."""
        try:
            return backends.create("jev")
        except Exception as exc:
            print(f"[live] Jev disabled: {exc}", file=sys.stderr)
            return None

    def _harvest(self):
        """One JS evaluation, not two: extract_posts would drop ads before we can count them,
        and Post has nowhere to carry the flag, so we filter here and keep the tally."""
        posts = []
        for raw in extract_raw(self.page):
            if raw.get("sponsored"):
                self.stats["sponsored"] += 1
                continue
            try:
                posts.append(post_from_raw(raw, self.corpus, self.platform))
            except ValueError as exc:
                # One malformed card must not end a scroll session.
                print(f"[live] skipped a card: {exc}", file=sys.stderr)
        return posts

    def _enqueue(self, posts):
        for post in posts:
            if post.url in self.seen:
                continue
            self.seen.add(post.url)
            self.stats["seen"] += 1
            # maxlen deque drops the oldest silently, so count it ourselves: a post already
            # scrolled past is the right one to lose, but the user must see that it happened.
            if len(self.pending) == QUEUE_MAX:
                self.stats["dropped"] += 1
            self.pending.append(post)

    def _score(self, post):
        rows, ms, _ = score_one(self.laya, post, "laya")
        self.latency["laya"].append(ms)
        frame = {"laya": rows, "jev": None}

        if self.jev and self.sampler.take() and self.budget.allow():
            try:
                jrows, jms, tokens = score_one(self.jev, post, "jev")
                self.budget.charge(tokens)
                self.latency["jev"].append(jms)
                self.stats["jev_scored"] += 1
                frame["jev"] = jrows
                self._compare(rows, jrows)
            except Exception as exc:
                # A flaky hosted API must never cost the user a scroll session.
                self.stats["jev_errors"] += 1
                print(f"[live] Jev error on {post.url}: {exc}", file=sys.stderr)

        # The post row must exist before its scores: scores.url is a foreign key into posts.
        insert_posts(self.conn, [post])
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO scores VALUES (?,?,?,?,?,?,?,?)",
                rows + (frame["jev"] or []),
            )
        self.stats["scored"] += 1
        return frame

    def _compare(self, laya_rows, jev_rows):
        """Agreement on the choice questions -- the only ones with a comparable label."""
        lab = {r[1]: r[5] for r in laya_rows if r[5] is not None}
        jab = {r[1]: r[5] for r in jev_rows if r[5] is not None}
        for question, label in lab.items():
            if question in jab:
                self.stats["compared"] += 1
                self.stats["agree"] += label == jab[question]

    @staticmethod
    def _rubric(laya_rows, jev_rows):
        """rows -> {question: {"laya": cell, "jev": cell|None}}; cell carries value and label."""
        def cells(rows):
            return {r[1]: {"value": r[4], "label": r[5]} for r in rows or []}

        laya, jev = cells(laya_rows), cells(jev_rows)
        return {q: {"laya": cell, "jev": jev.get(q)} for q, cell in laya.items()}

    @staticmethod
    def _mean_value(rows):
        numeric = [r[4] for r in rows or [] if r[5] is None]
        return sum(numeric) / len(numeric) if numeric else 0.0

    def _median(self, key):
        values = sorted(self.latency[key])
        return values[len(values) // 2] if values else 0.0

    def _payload(self, post, frame):
        compared = self.stats["compared"]
        agreement = self.stats["agree"] / compared if compared else None
        self.series["laya"].append(self._mean_value(frame["laya"]))
        if frame["jev"]:
            self.series["jev"].append(self._mean_value(frame["jev"]))
        self.series["lat_laya"].append(self.latency["laya"][-1] if self.latency["laya"] else 0.0)
        if self.latency["jev"]:
            self.series["lat_jev"].append(self.latency["jev"][-1])
        if agreement is not None:
            self.series["agreement"].append(agreement)
        return {
            "post": {"urn": post.url, "author": post.author,
                     "permalink": post.url, "state": "scored"},
            "rubric": self._rubric(frame["laya"], frame["jev"]),
            "latency": {
                "laya": self.latency["laya"][-1] if self.latency["laya"] else None,
                "jev": self.latency["jev"][-1] if self.latency["jev"] else None,
            },
            "counters": {k: self.stats[k] for k in ("seen", "scored", "dropped", "sponsored")},
            "budget": {"spent": self.budget.spent_usd, "cap": self.budget.cap_usd,
                       "exhausted": self.budget.exhausted},
            "agreement": agreement,
            "note": None if self.jev else "Jev disabled",
            "series": {k: list(v) for k, v in self.series.items()},
        }

    def _save_clicked(self):
        for click in overlay.poll_clicks(self.page):
            url = click.get("urn") or click.get("url")
            if not url:
                continue
            try:
                add_manual_swipe(self.conn, url, click.get("note"))
                overlay.mark(self.page, url, "saved")
                print(f"[live] saved {url}", file=sys.stderr)
            except ValueError as exc:
                print(f"[live] could not save {url}: {exc}", file=sys.stderr)

    def tick(self):
        """One pass of the loop. Split out from run() so it can be driven by a test."""
        # Re-resolved every pass: you may open your feed in a new tab, or switch tabs.
        self.page = self.browser.active_page(self.platform)
        self.browser.settle()
        overlay.install(self.page)
        try:
            self._enqueue(self._harvest())
        except ExtractionFailed as exc:
            # The DOM changed. Silently harvesting nothing is the failure this stops.
            print(f"[live] EXTRACTION BROKEN: {exc}", file=sys.stderr)
            raise
        self._save_clicked()
        if not self.pending:
            return None
        post = self.pending.popleft()
        frame = self._score(post)
        overlay.update(self.page, self._payload(post, frame))
        overlay.mark(self.page, post.url, "both" if frame["jev"] else "laya")
        if self.record:
            self.record.write(json.dumps({
                "ts": datetime.now().astimezone().isoformat(),
                "url": post.url, "stats": dict(self.stats),
            }) + "\n")
        return post

    def run(self):
        # No "open your feed for you" here, ever: see the no-navigation rule in browser.py.
        print(f"[live] watching. Open your {self.platform} feed in this window yourself -- the "
              "tool never navigates, which is what keeps it from looking like automation. "
              "Then scroll; Ctrl-C to stop.", file=sys.stderr)
        try:
            while True:
                if self.tick() is None:
                    time.sleep(0.25)
        except KeyboardInterrupt:
            print("\n[live] stopping.", file=sys.stderr)
        finally:
            self.close()

    def close(self):
        if self.record:
            self.record.close()
        for backend in (self.laya, self.jev):
            if backend is not None:
                backend.close()
        print(f"[live] {self.stats['scored']} scored, {self.stats['dropped']} dropped, "
              f"Jev {self.stats['jev_scored']} posts / ${self.budget.spent_usd:.4f}",
              file=sys.stderr)
