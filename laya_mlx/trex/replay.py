"""Record a run as plain data, and render a recording to video at its original pace.

A recording holds what was really on screen: each game's drawable state, each model's
probabilities and timings, and wall-clock timestamps. The export draws those frames with
the same painter as the live window. It is a rendered replay, not a screen capture, and
the video says so.
"""

import argparse
import json
import os
import shutil
import subprocess
import time
from bisect import bisect_right
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from .course import phase
from .engine import CRASHED
from .pilot import Stats


def capture_game(game):
    t, night = game.trex, game.night
    return {
        "reveal": round(game.reveal, 1),
        "hs": list(game.horizon_source),
        "hx": list(game.horizon_x),
        "night": [
            round(night.opacity, 3),
            night.phase,
            round(night.x, 2),
            [[round(x, 2), y] for x, y in night.stars],
        ],
        "clouds": [[c.x, c.y] for c in game.clouds],
        "obs": [[o.kind.name, o.size, o.frame, o.x, o.y] for o in game.obstacles],
        "units": game.score_units,
        "vis": game.score_visible,
        "shown": game.shown_score,
        "hi": game.actual_distance(game.high_score) if game.high_score else 0,
        "trex": [t.x, t.y, t.sprite, bool(t.ducking and t.status != CRASHED)],
        "crashed": game.crashed,
        "rf": game.restart_frame,
        "inv": round(game.invert_fade, 3),
    }


class Capture:
    """Turns the live arena into one frame dict per call."""

    def __init__(self, arena, args):
        self.arena = arena
        self.args = args
        self.started = None
        self.seen = [0] * len(arena.pilots)
        self.replays = ReplayBuffer(len(arena.pilots))
        self.crashes_seen = 0

    def meta(self):
        arena, args = self.arena, self.args
        return {
            "type": "metadata",
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed": args.seed,
            "mode": f"lockstep, {args.lockstep} frames per decision"
            if args.lockstep
            else "real time",
            "prompt": args.prompt,
            "round_seconds": getattr(args, "round_seconds", 0),
            "course_style": getattr(args, "course_style", "original"),
            "replays": not getattr(args, "no_replays", False),
            "course": arena.director.name if arena.director else None,
            "players": [
                {
                    "name": p.brain.name,
                    "detail": p.brain.detail,
                    "guarded": p.guarded,
                    "paid": bool(p.brain.usd_per_token),
                }
                for p in arena.pilots
            ],
        }

    def panel(self, i, pilot):
        stats, last, game = pilot.stats, pilot.last, pilot.game
        new = stats.all_latency[self.seen[i] :]
        self.seen[i] = len(stats.all_latency)
        return {
            "p": last.probabilities if last else None,
            "exec": last.executed if last else None,
            "prop": last.proposed if last else None,
            "veto": bool(last and last.intervened),
            "score": game.score,
            "top": max(stats.scores + [game.score]),
            "deaths": game.deaths,
            "ms": Stats.percentile(list(stats.latency), 0.5),
            "model_ms": Stats.percentile(list(stats.inference), 0.5),
            "n": stats.decisions,
            "rate": round(stats.rate, 1),
            "saves": stats.interventions,
            "agree": stats.agreements / stats.decisions if stats.decisions else None,
            "cost": stats.cost,
            "err": stats.last_error,
            "new": [round(v, 1) for v in new],
            # Diagnostics: where the last answer's time went, and what the planner was told.
            "plan_ms": round(last.plan_ms, 1) if last else None,
            "robust": last.robust if last else None,
            "plan_states": last.plan_states if last else None,
            "last_model_ms": round(last.inference_ms, 1) if last else None,
            "timing": [
                pilot.latency_frames,
                pilot.jitter_frames,
                pilot.stagger,
                pilot.longest_wait,
            ],
            "discarded": stats.discarded,
            "decision": {
                "seq": last.seq,
                "view_frame": last.view_frame,
                "expected_first": last.expected_first,
                "latency_ms": last.latency_ms,
                "arrival_intervened": last.arrival_intervened,
            }
            if last
            else None,
            "spectator": pilot.spectator(self.arena.frame),
            "phase": phase(game)["name"]
            if getattr(self.args, "course_style", "original") == "staged"
            else "Original course",
        }

    def frame(self):
        arena = self.arena
        now = time.perf_counter()
        if self.started is None:
            self.started = now
        d = arena.director
        games = [capture_game(p.game) for p in arena.pilots]
        panels = [self.panel(i, p) for i, p in enumerate(arena.pilots)]
        clips = self.replays.capture(
            arena.frame, games, panels, [p.game.run_index for p in arena.pilots]
        )
        crashes = arena.crash_events[self.crashes_seen :]
        self.crashes_seen = len(arena.crash_events)
        return {
            "type": "frame",
            "t": round(now - self.started, 4),
            "f": arena.frame,
            "paused": arena.paused,
            "g": games,
            "s": panels,
            "match": arena.match.state(arena.frame) if arena.match else None,
            "replays": clips if not getattr(self.args, "no_replays", False) else [],
            "crashes": crashes,
            "c": {"designed": d.designed, "fallbacks": d.fallbacks, "errors": d.errors}
            if d
            else None,
        }


class ReplayBuffer:
    """Two seconds of game snapshots; emitted once on a crash, never pauses play."""

    def __init__(self, players):
        self.history = [deque(maxlen=121) for _ in range(players)]
        self.runs = [None] * players
        self.deaths = [0] * players
        self.last_frame = None

    def capture(self, frame, games, panels, runs):
        if frame == self.last_frame:
            return []
        self.last_frame = frame
        clips = []
        for i, (game, panel, run) in enumerate(zip(games, panels, runs)):
            history = self.history[i]
            if run != self.runs[i]:
                history.clear()
                self.runs[i] = run
            history.append(
                {
                    "frame": frame,
                    "g": game,
                    "action": panel.get("spectator", {}).get("last_action") or panel.get("exec"),
                    "decision": panel.get("decision"),
                    "spectator": panel.get("spectator", {}),
                }
            )
            while history and history[0]["frame"] < frame - 120:
                history.popleft()
            if panel["deaths"] > self.deaths[i] and history:
                clips.append(
                    {"player": i, "start": frame, "score": panel["score"], "frames": list(history)}
                )
            self.deaths[i] = panel["deaths"]
        return clips


class Recorder:
    def __init__(self, path, meta, seconds=None):
        self.seconds = seconds
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(f"{self.path} exists; recordings are never overwritten")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w")
        self.file.write(json.dumps(meta) + "\n")
        self.last = None

    def write(self, frame):
        if self.file.closed:
            return
        if frame["f"] != self.last:  # One entry per game frame.
            self.last = frame["f"]
            self.file.write(json.dumps(frame, separators=(",", ":")) + "\n")
            if self.seconds is not None and frame["t"] >= self.seconds:
                self.close()

    def close(self):
        self.file.close()


def load(path):
    meta, frames = None, []
    with Path(path).open() as file:
        for line in file:
            event = json.loads(line)
            if event["type"] == "metadata":
                meta = event
            else:
                frames.append(event)
    if meta is None or not frames:
        raise ValueError("A recording needs metadata and at least one frame")
    return meta, frames


def find_ffmpeg():
    """A system ffmpeg that actually runs, or the binary bundled with imageio-ffmpeg."""
    candidates = [shutil.which("ffmpeg")]
    try:
        import imageio_ffmpeg

        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except (ImportError, RuntimeError):
        pass
    for path in filter(None, candidates):
        try:
            if subprocess.run([path, "-version"], capture_output=True, timeout=10).returncode == 0:
                return path
        except (OSError, subprocess.TimeoutExpired):
            continue
    return None


def export(argv=None):
    parser = argparse.ArgumentParser(
        prog="laya-trex export", description="Render a recording to MP4 at its original pace."
    )
    parser.add_argument("recording")
    parser.add_argument("--output", required=True, help="MP4 file to write")
    parser.add_argument("--start", type=float, default=0.0, help="Seconds into the recording")
    parser.add_argument("--seconds", type=float, help="Clip length (default: to the end)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--poster", help="Also save the clip's last frame as a PNG")
    args = parser.parse_args(argv)
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        parser.error("MP4 export needs a working ffmpeg (on macOS: brew install ffmpeg)")
    output = Path(args.output)
    if output.exists():
        parser.error(f"{output} exists; outputs are never overwritten")
    meta, frames = load(args.recording)
    times = [frame["t"] for frame in frames]
    end = times[-1] if args.seconds is None else min(times[-1], args.start + args.seconds)
    if not 0 <= args.start < end:
        parser.error("The clip is empty; check --start and --seconds")

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    from .app import Painter

    pygame.init()
    pygame.display.set_mode((1, 1))
    painter = Painter("video", meta)
    surface = pygame.Surface(painter.size, pygame.SRCALPHA, 32)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen(
        [
            ffmpeg, "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{painter.size[0]}x{painter.size[1]}", "-r", str(args.fps), "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(output),
        ],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    fed = 0
    count = int((end - args.start) * args.fps)
    try:
        for n in range(count):
            at = args.start + n / args.fps
            index = max(0, bisect_right(times, at) - 1)
            while fed <= index:  # Charts need every earlier frame, including before the clip.
                painter.observe(frames[fed])
                fed += 1
            painter.draw(
                surface,
                frames[index],
                badge=f"RECORDED RUN · 1× · {at // 60:02.0f}:{at % 60:04.1f}",
            )
            encoder.stdin.write(pygame.image.tobytes(surface, "RGB"))
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("ffmpeg did not complete the MP4")
        if args.poster:
            pygame.image.save(surface, args.poster)
    finally:
        if encoder.poll() is None:
            encoder.kill()
        pygame.quit()
    sidecar = {
        "source": str(args.recording),
        "start": args.start,
        "seconds": round(end - args.start, 2),
        "fps": args.fps,
        "speed": "1x wall clock; each video frame shows the most recent recorded game frame",
        "kind": "rendered replay of recorded data, not a screen capture",
        "recording": {k: v for k, v in meta.items() if k != "type"},
    }
    output.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"Wrote {output} ({count} frames, {end - args.start:.1f} s)")
    return 0
