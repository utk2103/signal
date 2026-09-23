"""Laya and Jev play a local clone of the Chrome T-Rex game side by side, on the same course."""

import argparse
import json
import math
import os
import sys
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

from .backends import PROMPTS, create, read_env_file
from .brain import RemoteBrain
from .course import StagedCourse
from .director import CourseDirector
from .engine import FRAME_MS
from .pilot import Arena, Pilot
from .planner import snapshot
from .replay import Capture, Recorder, export

PLAYERS = ("laya", "jev")


def parse(argv=None):
    parser = argparse.ArgumentParser(
        prog="laya-trex", description=__doc__, epilog="Also: laya-trex export --help"
    )
    parser.add_argument(
        "--players",
        default="laya,jev",
        help="Comma-separated players, left to right: laya, jev (default: laya,jev)",
    )
    parser.add_argument(
        "--course",
        choices=("jev", "laya", "random"),
        default="jev",
        help="Who designs the obstacles (default: jev, the stronger designer in our test); "
        "random uses the original game's rules",
    )
    parser.add_argument(
        "--round-seconds",
        type=float,
        default=60,
        help="Round length in seconds (default: 60); 0 for endless survival",
    )
    parser.add_argument(
        "--rounds", type=int, default=0, help="Stop after this many rounds; 0 keeps playing"
    )
    parser.add_argument(
        "--course-style",
        choices=("staged", "original"),
        default="staged",
        help="Staged recovery gaps and themed phases, or original difficulty",
    )
    parser.add_argument("--no-replays", action="store_true", help="Hide the crash replay inset")
    parser.add_argument("--crash-log", help="Write two-second crash diagnostics as JSONL")
    parser.add_argument("--seed", type=int, default=7, help="Course seed shared by all players")
    parser.add_argument("--prompt", choices=PROMPTS, default="labeled")
    parser.add_argument(
        "--unassisted",
        action="store_true",
        help="Execute each model's first choice even when the planner marks it unsafe",
    )
    parser.add_argument(
        "--lockstep",
        type=int,
        metavar="FRAMES",
        help="Pause each game while its model answers, then play FRAMES frames per decision. "
        "Removes latency from play to compare decisions alone",
    )
    parser.add_argument("--laya-model", help="Local Laya checkpoint directory or cached Hub ID")
    parser.add_argument("--optimize", action="store_true", help="Compile Laya and reuse prefixes")
    parser.add_argument(
        "--laya-inflight",
        type=int,
        default=2,
        metavar="N",
        help="Questions Laya keeps in flight (default: 2): the next one is prepared while the "
        "GPU answers the current one, so the GPU never idles. 1 alternates the two",
    )
    parser.add_argument("--jev-model", default="jev-latest")
    parser.add_argument(
        "--jev-inflight",
        type=int,
        default=6,
        metavar="N",
        help="Requests Jev keeps in flight, asked a few frames apart (default: 6). A hosted API "
        "answers in parallel, so this gives Jev a turn every few frames instead of one per "
        "round trip; 1 asks one question at a time",
    )
    parser.add_argument(
        "--env-file",
        help="Read TYPESAFE_API_KEY from this .env file when it is not already set",
    )
    parser.add_argument("--headless", action="store_true", help="Play without a window")
    parser.add_argument("--duration", type=float, help="Stop after this many seconds of play")
    parser.add_argument("--report", help="Write the final comparison to this JSON file")
    parser.add_argument("--scale", type=float, help="Window scale (default: fit the screen)")
    parser.add_argument("--sound", action="store_true", help="Play the original sound effects")
    parser.add_argument("--snapshot", help="Save a PNG of the window when the run ends")
    parser.add_argument(
        "--record",
        metavar="FILE.jsonl",
        help="Record the run as data; render it later with: laya-trex export FILE.jsonl --output X.mp4",
    )
    parser.add_argument(
        "--record-seconds", type=float, help="Stop recording after this many seconds; keep playing"
    )
    args = parser.parse_args(argv)
    args.players = [p.strip() for p in args.players.split(",") if p.strip()]
    if not 1 <= len(args.players) <= 2 or any(p not in PLAYERS for p in args.players):
        parser.error("--players takes one or two of: laya, jev")
    if args.record_seconds is not None and (
        not args.record or not math.isfinite(args.record_seconds) or args.record_seconds <= 0
    ):
        parser.error("--record-seconds needs --record and a finite positive length")
    if len(set(args.players)) != len(args.players):
        parser.error("--players must not repeat a player")
    if not math.isfinite(args.round_seconds) or args.round_seconds < 0:
        parser.error("--round-seconds must be finite and nonnegative")
    if args.rounds < 0 or (args.rounds and not args.round_seconds):
        parser.error("--rounds needs a positive round length and a nonnegative count")
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration must be finite and positive")
    if args.lockstep is not None and args.lockstep < 1:
        parser.error("--lockstep must be at least 1 frame")
    if not 1 <= args.laya_inflight <= 3:
        parser.error("--laya-inflight takes 1 to 3; one GPU answers one question at a time")
    if not 1 <= args.jev_inflight <= 8:
        parser.error("--jev-inflight takes 1 to 8 (Jev allows 1,200 requests a minute)")
    return args


def load_key(args):
    if os.environ.get("TYPESAFE_API_KEY") or not args.env_file:
        return
    key = read_env_file(args.env_file).get("TYPESAFE_API_KEY")
    if key:
        os.environ["TYPESAFE_API_KEY"] = key


def build(args, log=print):
    load_key(args)
    options = {
        "laya": {
            "model": args.laya_model,
            "optimize": args.optimize,
            "inflight": 1 if args.lockstep else args.laya_inflight,
        },
        "jev": {"model": args.jev_model, "inflight": 1 if args.lockstep else args.jev_inflight},
    }
    brains = []
    resources = []
    try:
        for kind in args.players:
            log(f"Starting {kind}…")
            brain = RemoteBrain(kind, **options[kind])
            brains.append(brain)
            resources.append(brain)
        director = None
        if args.course != "random":
            # The designer runs beside the window; it works ahead, so its latency is hidden.
            log(f"Starting the {args.course} course designer…")
            designer = create(args.course, **{**options[args.course], "inflight": 1})
            resources.append(designer)
            director = CourseDirector(
                designer, args.seed, lookahead=12, staged=args.course_style == "staged"
            )
            # The director worker now owns and closes the designer backend.
            resources[-1] = director
        pilots = []
        for index, brain in enumerate(brains):
            pilot = Pilot(
                brain,
                guarded=not args.unassisted,
                prompt=args.prompt,
                lockstep=args.lockstep,
                seed=args.seed,
                course=director
                or (StagedCourse(args.seed) if args.course_style == "staged" else None),
            )
            pilots.append(pilot)
            # A constructed pilot owns its brain; close it only through the pilot.
            resources[index] = pilot
        return Arena(
            pilots,
            director=director,
            lockstep=args.lockstep,
            round_seconds=args.round_seconds,
            rounds=args.rounds,
        )
    except BaseException:
        # ExitStack attempts every close, even if an earlier cleanup raises.
        with ExitStack() as cleanup:
            for resource in resources:
                cleanup.callback(resource.close)
        raise


def warm_up(arena, log=print, timeout=60):
    """Measure each player's latency and let the designer get ahead before the start."""
    for pilot in arena.pilots:
        samples = pilot.brain.warm(snapshot(pilot.game, "run"), pilot.prompt)
        typical = sorted(samples)[len(samples) // 2] / FRAME_MS
        pilot.latency_frames = int(min(samples) / FRAME_MS)
        pilot.typical_frames = typical
        pilot.interval_frames = max(1.0, typical)
        pilot.jitter_frames = max(1.0, max(samples) / FRAME_MS + 1 - pilot.latency_frames)
        log(f"{pilot.brain.name}: {pilot.brain.detail}, warm answer {min(samples):.0f} ms")
    if arena.director:
        arena.director.demand(0, arena.director.lookahead)
        arena.director.demand(1, arena.director.lookahead)
        deadline = time.time() + timeout
        while not arena.director.ready(0, arena.director.lookahead) and time.time() < deadline:
            time.sleep(0.05)
        log(f"Course designer {arena.director.name}: {arena.director.designed} obstacles ready")


def run_headless(arena, args):
    capture = Capture(arena, args)
    recorder = (
        Recorder(args.record, capture.meta(), seconds=args.record_seconds) if args.record else None
    )
    arena.start()
    try:
        while not arena.finished and (
            args.duration is None or arena.frame * FRAME_MS / 1000 < args.duration
        ):
            if arena.advance() and recorder:
                recorder.write(capture.frame())
            for pilot in arena.pilots:
                pilot.game.drain_events()
            time.sleep(0.002)
    finally:
        if recorder:
            recorder.close()


def write_report(arena, args, path=None):
    report = arena.report()
    report["seed"] = args.seed
    report["course_style"] = args.course_style
    report["assistance"] = "none" if args.unassisted else "planner + arrival + emergency"
    report["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    crash_path = args.crash_log or (str(Path(path).with_suffix(".crashes.jsonl")) if path else None)
    if crash_path:
        target = Path(crash_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(json.dumps(event) + "\n" for event in arena.crash_events))
        report["crash_log"] = str(target)
    text = json.dumps(report, indent=2)
    if path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n")
    return text


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "export":
        return export(argv[1:])
    args = parse(argv)
    # Short interpreter-lock slices keep the window and the pilot threads responsive.
    sys.setswitchinterval(0.001)
    if args.headless and args.duration is None and not args.rounds:
        print("--headless needs --duration or --rounds", file=sys.stderr)
        return 2
    try:
        arena = build(args)
    except (RuntimeError, FileNotFoundError) as error:
        print(error, file=sys.stderr)
        return 1
    try:
        if args.headless:
            warm_up(arena)
            run_headless(arena, args)
        else:
            from .app import run_window

            run_window(arena, args, warm_up)
    except KeyboardInterrupt:
        pass
    finally:
        text = write_report(arena, args, args.report)
        print(text)
        if arena.director:
            arena.director.close()
        for pilot in arena.pilots:
            pilot.close()
    return 0
