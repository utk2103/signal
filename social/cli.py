"""laya-social: ingest a post corpus, score it with Laya, regress the rubric, report."""

import argparse
import os
import sys
from pathlib import Path

from . import analyze as analyze_mod
from . import report as report_mod
from .ingest.jsonfile import read_json_posts
from .models import connect, insert_posts
from .rubric import RUBRIC_VERSION

DEFAULT_DB = "posts.sqlite"


def cmd_ingest(args):
    conn = connect(args.db)
    if args.from_json:
        posts, errors = read_json_posts(args.from_json, args.corpus, platform=args.platform)
        for line in errors[:20]:
            print(f"skipped {line}", file=sys.stderr)
        if len(errors) > 20:
            print(f"... and {len(errors) - 20} more skipped rows", file=sys.stderr)
    else:
        # Imported here, not at module scope: apify pulls in httpx from the `social` extra,
        # and every other subcommand must keep working without it.
        from .ingest.apify import MissingApifyToken, fetch_linkedin, fetch_x

        fetch = fetch_linkedin if args.platform == "linkedin" else fetch_x
        targets = [t.strip() for t in Path(args.targets).read_text().split("\n") if t.strip()]
        try:
            posts = fetch(targets, args.corpus, actor=args.actor, max_posts=args.max_posts)
        except MissingApifyToken as exc:
            print(str(exc), file=sys.stderr)
            return 1
        errors = []
    written = insert_posts(conn, posts)
    print(f"ingested {written} posts into corpus {args.corpus!r} ({len(errors)} skipped)")
    return 0


def cmd_score(args):
    from .score import score_corpus

    conn = connect(args.db)
    score_corpus(conn, args.corpus, model=args.model, batch_size=args.batch_size)
    return 0


def cmd_report(args):
    conn = connect(args.db)
    result = analyze_mod.analyze(conn, args.corpus, bucket=args.bucket, seed=args.seed)
    if result.n_scored == 0:
        print(f"No posts scored at rubric {RUBRIC_VERSION}; run `laya-social score` first.",
              file=sys.stderr)
        return 1
    path = report_mod.write_pattern(conn, result, args.out)
    print(f"wrote {path} ({result.n_scored}/{result.n_posts} posts scored)")
    if result.confound_unmitigated:
        print("WARNING: follower counts too sparse to bucket; raw-engagement confound is "
              "unmitigated. See the top of the report.", file=sys.stderr)
    return 0


def cmd_swipe(args):
    conn = connect(args.db)
    if args.add:
        report_mod.add_manual_swipe(conn, args.add, args.note)
        print(f"starred {args.add}")
    if args.top_decile:
        result = analyze_mod.analyze(conn, args.corpus, bucket=args.bucket, seed=args.seed)
        n = report_mod.refresh_auto_swipe(conn, result)
        print(f"refreshed {n} auto top-decile entries")
    if args.export:
        path = report_mod.write_swipe(conn, args.corpus, args.export)
        print(f"wrote {path}")
    if not (args.add or args.top_decile or args.export):
        print("nothing to do: pass --add, --top-decile, or --export", file=sys.stderr)
        return 1
    return 0


def cmd_live(args):
    # Imported here, not at module scope: playwright comes from the `live` extra and every
    # other subcommand must keep working without it.
    from laya_mlx.trex.backends import read_env_file

    from .live.browser import Browser
    from .live.session import Session

    # Default `.env` is a convenience, not a requirement: absent is the normal Laya-only case.
    if Path(args.env_file).exists() and not os.environ.get("TYPESAFE_API_KEY"):
        key = read_env_file(args.env_file).get("TYPESAFE_API_KEY")
        if key:
            os.environ["TYPESAFE_API_KEY"] = key

    browser = Browser(args.profile)
    try:
        if args.action == "login":
            browser.login(args.platform)
            print(f"session saved to {args.profile}")
            return 0
        conn = connect(args.db)
        Session(
            browser, conn, args.corpus, platform=args.platform, model=args.model,
            jev_every=args.jev_every, jev_budget_usd=args.jev_budget_usd,
            record=args.session,
        ).run()
        return 0
    finally:
        browser.close()


def build_parser():
    parser = argparse.ArgumentParser(prog="laya-social", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def shared(p):
        p.add_argument("--corpus", required=True, help="client or niche tag")
        p.add_argument("--db", default=DEFAULT_DB, help=f"corpus path (default {DEFAULT_DB})")

    def split_args(p):
        p.add_argument("--no-bucket", dest="bucket", action="store_false",
                       help="skip follower-tier bucketing and regress the whole corpus")
        p.add_argument("--seed", type=int, default=0, help="train/test split seed")

    p = sub.add_parser("ingest", help="load posts from Apify or a local JSON/JSONL export")
    shared(p)
    p.add_argument("--platform", choices=("linkedin", "x"), default="linkedin")
    p.add_argument("--from-json", help="local JSON array or JSONL export; needs no Apify account")
    p.add_argument("--targets", help="file of profile URLs or handles, one per line (Apify)")
    p.add_argument("--actor", help="override the Apify actor id")
    p.add_argument("--max-posts", type=int, default=50, help="posts per target")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("score", help="score unscored posts with Laya")
    shared(p)
    p.add_argument("--model", default="aac6fef/laya-mlx")
    p.add_argument("--batch-size", type=int, default=32)
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("report", help="regress the rubric against engagement, write PATTERN.md")
    shared(p)
    split_args(p)
    p.add_argument("--out", default="PATTERN.md")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("swipe", help="maintain and export the local archive of good posts")
    shared(p)
    split_args(p)
    p.add_argument("--top-decile", action="store_true", help="refresh auto entries")
    p.add_argument("--add", help="star a post by url")
    p.add_argument("--note", help="why it is worth keeping (with --add)")
    p.add_argument("--export", help="write swipe.md to this path")
    p.set_defaults(func=cmd_swipe)

    p = sub.add_parser("live", help="score your feed as you scroll it yourself")
    p.add_argument("action", nargs="?", default="watch", choices=("watch", "login"),
                   help="'login' signs in once and saves the profile; default watches")
    p.add_argument("--corpus", default="live", help="client or niche tag")
    p.add_argument("--db", default=DEFAULT_DB, help=f"corpus path (default {DEFAULT_DB})")
    p.add_argument("--profile", default="~/.laya-social/chrome",
                   help="persistent browser profile directory")
    p.add_argument("--platform", choices=("linkedin", "x"), default="linkedin")
    p.add_argument("--jev-every", type=int, default=5,
                   help="send every Nth post to Jev; 0 disables Jev")
    p.add_argument("--jev-budget-usd", type=float, default=1.0,
                   help="hard per-session cap; Jev stops, Laya continues")
    p.add_argument("--session", help="write a JSONL session log here")
    p.add_argument("--env-file", default=".env",
                   help="read TYPESAFE_API_KEY from this .env when it is not already set")
    p.add_argument("--model", default="aac6fef/laya-mlx")
    p.set_defaults(func=cmd_live)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "ingest" and not args.from_json and not args.targets:
        print("ingest needs --from-json or --targets", file=sys.stderr)
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
