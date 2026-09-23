"""Drawing: each game from the original sprite sheet, with live stats and charts.

Everything is drawn from plain frame dicts (see replay.Capture), so the live window and
the video export share one painter.
"""

import math
import statistics
import threading
import time
from pathlib import Path

import pygame

from .engine import FRAME_MS, HEIGHT, HORIZON_Y, KINDS, MOON_PHASES, WIDTH
from .planner import ACTIONS
from .replay import Capture, Recorder

ASSETS = Path(__file__).parent / "assets"
PX = 2  # The canvas is drawn from the 2x sprite sheet, then scaled to fit.

# Positions in the 2x sheet, from the original sprite definitions.
SHEET = {
    "cactusSmall": (446, 2),
    "cactusLarge": (652, 2),
    "pterodactyl": (260, 2),
    "cloud": (166, 2),
    "horizon": (4, 104),
    "moon": (954, 2),
    "star": (1276, 2),
    "restart": (2, 130),
    "text": (1294, 2),
    "trex": (1678, 2),
}

PAGE = (255, 255, 255)
INK = (32, 33, 36)
TEXT = (95, 99, 104)
FAINT = (154, 160, 166)
RULE = (226, 228, 232)
TRACK = (241, 243, 244)
WARN = (197, 34, 31)
# Series colours follow the player, never its position. Validated for colour-blind
# separation and contrast on a white surface.
SERIES = {"Laya": (42, 120, 214), "Jev": (235, 104, 52)}
KEYS = {"jump": "JUMP  ↑", "duck": "DUCK  ↓", "run": "RUN"}


class GameView:
    """Draws one game exactly as the original lays it out, at 2x, then scales it."""

    def __init__(self, sheet, size):
        self.sheet = sheet
        self.size = size
        # An explicit 32-bit format: blits onto a display-format surface lose the sprites on macOS.
        self.canvas = pygame.Surface((WIDTH * PX, HEIGHT * PX), pygame.SRCALPHA, 32)

    def sprite(self, x, y, w, h, dest, alpha=None):
        if alpha is None:
            self.canvas.blit(self.sheet, (dest[0] * PX, dest[1] * PX), (x, y, w, h))
            return
        piece = self.sheet.subsurface((x, y, w, h)).copy()
        piece.set_alpha(int(255 * max(0.0, min(1.0, alpha))))
        self.canvas.blit(piece, (dest[0] * PX, dest[1] * PX))

    def render(self, g):
        canvas = self.canvas
        canvas.set_clip(None)
        canvas.fill(PAGE)
        # Before the intro ends the original shows only a dino-wide strip of canvas.
        canvas.set_clip((0, 0, int(g["reveal"] * PX), HEIGHT * PX))
        hx, hy = SHEET["horizon"]
        for i in (0, 1):
            self.sprite(hx + g["hs"][i], hy, WIDTH * PX, 12 * PX, (g["hx"][i], HORIZON_Y))
        opacity, phase, moon_x, stars = g["night"]
        if opacity > 0:
            sx, sy = SHEET["star"]
            for i, (x, y) in enumerate(stars):
                self.sprite(sx, sy + 9 * PX * i, 9 * PX, 9 * PX, (round(x), y), opacity)
            mx, my = SHEET["moon"]
            width = 40 if phase == 3 else 20
            self.sprite(
                mx + MOON_PHASES[phase] * PX, my, width * PX, 40 * PX, (round(moon_x), 30), opacity
            )
        for x, y in g["clouds"]:
            self.sprite(*SHEET["cloud"], 46 * PX, 14 * PX, (x, y))
        for name, size, frame, x, y in g["obs"]:
            kind = KINDS[name]
            w = kind.width * PX
            sx, sy = SHEET[name]
            sx += int((w * size) * (0.5 * (size - 1))) + w * frame
            self.sprite(sx, sy, w * size, kind.height * PX, (x, y))
        self.meter(g)
        tx, ty = SHEET["trex"]
        x, y, sprite, ducking = g["trex"]
        self.sprite(tx + sprite * PX, ty, (59 if ducking else 44) * PX, 47 * PX, (x, y))
        if g["crashed"]:
            tx, ty = SHEET["text"]
            self.sprite(tx, ty + 13 * PX, 191 * PX, 11 * PX, (205, 42))
            rx, ry = SHEET["restart"]
            self.sprite(rx + 36 * PX * g["rf"], ry, 36 * PX, 32 * PX, (WIDTH / 2 - 16, HEIGHT / 2))
        canvas.set_clip(None)
        if g["inv"] > 0:  # The page-wide CSS invert filter of the original.
            inverse = pygame.transform.invert(canvas)
            inverse.set_alpha(int(255 * g["inv"]))
            canvas.blit(inverse, (0, 0))
        if self.size == canvas.get_size():
            return canvas
        return pygame.transform.smoothscale(canvas, self.size)

    def glyph(self, value, x, alpha=None):
        tx, ty = SHEET["text"]
        # The original translates by y and then draws at y again, so digits sit at y = 10.
        self.sprite(tx + 10 * PX * value, ty, 10 * PX, 13 * PX, (x, 10), alpha)

    def meter(self, g):
        units = g["units"]
        x = WIDTH - 11 * (units + 1)
        if g["vis"]:
            for i, digit in enumerate(str(g["shown"]).zfill(units)[-units:]):
                self.glyph(int(digit), x + 11 * i)
        if g["hi"]:
            for i, char in enumerate("HI " + str(g["hi"]).zfill(units)[-units:]):
                if char != " ":
                    self.glyph(
                        {"H": 10, "I": 11}.get(char) or int(char), x - units * 20 + 11 * i, 0.8
                    )


def nice_ceiling(value):
    """The next clean axis maximum at or above value."""
    if value <= 0:
        return 1
    power = 10 ** math.floor(math.log10(value))
    return next(m * power for m in (1, 2, 2.5, 5, 10) if m * power >= value)


class Painter:
    """Lays out and draws a whole frame. `layout` is "window" or "video" (1920 x 1080)."""

    def __init__(self, layout, meta, scale=None):
        self.meta = meta
        self.video = layout == "video"
        self.k = 1.35 if self.video else 1.0
        self.names = [p["name"] for p in meta["players"]]
        n = len(self.names)
        if self.video:
            self.size = (1920, 1080)
            self.margin = 60
            game = (WIDTH * 2, HEIGHT * 2)
            self.rows = [
                {
                    "head": (60, 112 + i * 362),
                    "game": (60, 166 + i * 362),
                    "panel": (1300, 166 + i * 362, 560, 300),
                }
                for i in range(n)
            ]
            self.charts = (60, 838, 1800, 184)
            self.footer = 1050
        else:
            self.margin, gap = 28, 28
            game = (int(WIDTH * scale), int(HEIGHT * scale))
            width = 2 * self.margin + n * game[0] + (n - 1) * gap
            panel_top = 148 + game[1] + 22
            self.rows = [
                {
                    "head": (self.margin + i * (game[0] + gap), 84),
                    "game": (self.margin + i * (game[0] + gap), 148),
                    "panel": (self.margin + i * (game[0] + gap), panel_top, game[0], 250),
                }
                for i in range(n)
            ]
            self.charts = (self.margin, panel_top + 266, width - 2 * self.margin, 150)
            self.footer = self.charts[1] + self.charts[3] + 22
            self.size = (width, self.footer + 42)
        sheet = pygame.image.load(ASSETS / "sprite-2x.png").convert_alpha()
        self.views = [GameView(sheet, game) for _ in self.names]
        self.game_size = game
        sizes = {
            "hero": 40,
            "title": 22,
            "big": 26,
            "mid": 19,
            "body": 14,
            "small": 12,
            "label": 11,
        }
        bold = ("hero", "title", "big", "mid", "label")
        self.fonts = {
            name: self.font(round(size * (1 if name == "hero" else self.k)), name in bold)
            for name, size in sizes.items()
        }
        # Chart history, built from the frames observed so far.
        self.now = 0.0
        self.scores = [[] for _ in self.names]
        self.buckets = [{} for _ in self.names]
        self.last = None
        self.replay_clips = {}

    @staticmethod
    def font(size, bold=False):
        name = pygame.font.match_font("menlo", bold=bold) or pygame.font.match_font("monaco")
        return pygame.font.Font(name, size)

    def px(self, value):
        return round(value * self.k)

    # -- History for the charts ----------------------------------------------------------
    def observe(self, frame):
        t = frame["t"]
        self.now, self.last = t, frame
        for clip in frame.get("replays", []):
            self.replay_clips[clip["player"]] = clip
        for i, s in enumerate(frame["s"]):
            points = self.scores[i]
            # Four points a second, plus every drop, so a death stays a sharp edge.
            if not points or t - points[-1][0] >= 0.25 or s["score"] < points[-1][1]:
                points.append((t, s["score"]))
            for ms in s["new"]:
                self.buckets[i].setdefault(int(t * 2), []).append(ms)

    def latency_series(self, i):
        return [(b / 2 + 0.25, statistics.median(v)) for b, v in sorted(self.buckets[i].items())]

    # -- Primitives ----------------------------------------------------------------------
    def text(self, surface, value, font, color, pos, right=False):
        image = self.fonts[font].render(str(value), True, color)
        x, y = pos
        surface.blit(image, (x - image.get_width() if right else x, y))
        return image.get_width()

    def chip(self, surface, name, x, y, size=None):
        size = size or self.px(10)
        pygame.draw.rect(surface, SERIES.get(name, TEXT), (x, y, size, size), border_radius=2)
        return size

    # -- Frame ---------------------------------------------------------------------------
    def draw(self, surface, frame, badge=None):
        surface.fill(PAGE)
        if self.video:
            self.text(surface, "Signal  /  DINO ARENA", "title", INK, (60, 22))
            self.text(
                surface,
                "Local speed. Network decisions. One shared course.",
                "small",
                TEXT,
                (60, 57),
            )
            if badge:
                self.text(surface, badge, "small", FAINT, (1860, 28), right=True)
        self.match_banner(surface, frame)
        for i, row in enumerate(self.rows):
            self.header(surface, i, *row["head"])
            self.pipeline(surface, i, frame["s"][i], row)
            x, y = row["game"]
            surface.blit(self.views[i].render(frame["g"][i]), (x, y))
            pygame.draw.rect(
                surface, RULE, (x - 1, y - 1, self.game_size[0] + 2, self.game_size[1] + 2), 1
            )
            self.panel(surface, i, frame["s"][i], row["panel"])
            spectator = frame["s"][i].get("spectator", {})
            event = spectator.get("event")
            if event and frame["f"] - spectator.get("event_frame", -999) < 90:
                self.event_badge(surface, event, x + 12, y + self.game_size[1] - self.px(28), i)
        if not self.replay_band(surface, frame):
            self.chart_band(surface)
        self.foot(surface, frame)

    def headline(self, frame):
        times = [s["ms"] for s in frame["s"]]
        if len(times) != 2 or not all(times):
            return None
        fast = self.names[times.index(min(times))]
        return f"{fast} answers {max(times) / min(times):.0f}× faster"

    def header(self, surface, i, x, y):
        player = self.meta["players"][i]
        size = self.px(14)
        self.chip(surface, player["name"], x, y + self.px(7), size)
        w = self.text(surface, player["name"], "title", INK, (x + size + self.px(10), y))
        guard = "model + live shield" if player["guarded"] else "unassisted"
        detail = f"{'Local MLX' if not player['paid'] else 'Hosted API'} · {guard}"
        self.text(surface, detail, "small", TEXT, (x + size + w + self.px(24), y + self.px(9)))

    def fit_text(self, surface, value, font, color, pos, width):
        value = str(value)
        while value and self.fonts[font].size(value)[0] > width:
            value = value[:-2].rstrip() + "…" if len(value) > 2 else ""
        self.text(surface, value, font, color, pos)

    def match_banner(self, surface, frame):
        match = frame.get("match")
        x, y = self.margin, 80 if self.video else 14
        width = self.size[0] - 2 * self.margin
        if not match:
            self.text(
                surface, "ENDLESS SURVIVAL  ·  " + " vs ".join(self.names), "body", INK, (x, y)
            )
            return
        label = "MATCH COMPLETE" if match["finished"] else f"ROUND {match['round']}"
        clock = f"{match['remaining'] // 60:02}:{match['remaining'] % 60:02}"
        self.text(surface, f"{label}  {clock}", "mid", INK, (x, y))
        result = match.get("result")
        if result and (match["finished"] or frame["f"] - result["frame"] < 240):
            message = (
                f"{result['winner']} wins round {result['round']} ({result.get('reason', 'distance')})"
                if result["winner"]
                else f"Round {result['round']} tied"
            )
        else:
            message = (
                f"{match['leader']} leads by {match['lead']}"
                if match["leader"]
                else "Neck and neck"
            )
        if match["finished"]:
            message = (
                f"{match['champion']} wins the match" if match.get("champion") else "Match tied"
            )
        scores = "  ·  ".join(f"{name} {score}" for name, score in zip(self.names, match["points"]))
        wins = "—".join(str(win) for win in match["wins"])
        right = f"{scores}   |   MATCH {wins}"
        if self.video:
            self.text(surface, message, "body", TEXT, (x + 430, y + 4))
            self.text(surface, right, "body", INK, (self.size[0] - x, y + 4), right=True)
        else:
            self.text(surface, right, "body", INK, (self.size[0] - x, y + 3), right=True)
            self.text(
                surface,
                message + "  ·  distance wins; fewer live saves break ties",
                "small",
                TEXT,
                (x, y + 28),
            )
            pygame.draw.line(surface, RULE, (x, y + 55), (x + width, y + 55))

    def pipeline(self, surface, i, panel, row):
        s = panel.get("spectator", {})
        x, y = row["head"]
        thinking = s.get("thinking", 0)
        action = s.get("last_action") or panel.get("exec") or "—"
        text = f"THINKING {thinking}  →  ANSWER {(panel.get('prop') or '—').upper()}  →  ACTION {action.upper()}  ·  skipped {panel.get('discarded', 0)}"
        self.fit_text(
            surface,
            text,
            "label",
            SERIES.get(self.names[i], TEXT),
            (x, y + 29),
            self.game_size[0],
        )

    def event_badge(self, surface, text, x, y, i):
        font = self.fonts["small"]
        image = font.render(text, True, INK)
        rect = pygame.Rect(x, y, image.get_width() + 20, image.get_height() + 8)
        pygame.draw.rect(surface, PAGE, rect, border_radius=5)
        pygame.draw.rect(surface, SERIES.get(self.names[i], TEXT), rect, 1, border_radius=5)
        surface.blit(image, (x + 10, y + 4))

    def replay_band(self, surface, frame):
        active = [
            (i, clip)
            for i, clip in self.replay_clips.items()
            if 0 <= frame["f"] - clip["start"] < 240
        ]
        if not active:
            return False
        x, y, width, height = self.charts
        cell = width / max(2, len(active))
        for column, (i, clip) in enumerate(active):
            cx = x + column * cell
            self.text(
                surface,
                f"{self.names[i].upper()}  ·  CRASH REPLAY  ·  ½×  ·  SCORE {clip['score']}",
                "label",
                WARN,
                (cx, y),
            )
            frames = clip["frames"]
            at = frames[0]["frame"] + (frame["f"] - clip["start"]) / 2
            replay = next((f for f in reversed(frames) if f["frame"] <= at), frames[0])
            h = min(height - 48, (cell - 24) / 4)
            w = h * 4
            image = self.views[i].render(replay["g"])
            surface.blit(pygame.transform.smoothscale(image, (int(w), int(h))), (cx, y + 24))
            detail = f"Action: {(replay.get('action') or 'run').upper()}"
            decision = replay.get("decision")
            if decision:
                lo, hi = decision["expected_first"]
                detail += f" · answer {decision['latency_ms']:.0f}ms · planned {lo * FRAME_MS:.0f}–{hi * FRAME_MS:.0f}ms"
            self.fit_text(surface, detail, "small", TEXT, (cx, y + 26 + h), cell - 24)
        if len(active) == 1:
            cx = x + cell + 12
            self.text(surface, "LIVE RACE CONTINUES ABOVE", "label", INK, (cx, y + 10))
            clip = active[0][1]
            last = clip["frames"][-1].get("spectator", {})
            self.fit_text(
                surface,
                last.get("event") or "Reviewing the last two seconds",
                "body",
                TEXT,
                (cx, y + 42),
                cell - 24,
            )
            self.text(surface, "No pause. No extra thinking time.", "small", TEXT, (cx, y + 72))
        return True

    def panel(self, surface, i, s, rect):
        x, y, w, _ = rect
        k, name = self.px, self.names[i]
        self.text(surface, "NEXT MOVE · MODEL PROBABILITIES", "label", FAINT, (x, y))
        bar_x, bar_w = x + k(84), w - k(84) - k(118)
        for row, action in enumerate(ACTIONS):
            ry = y + k(22) + row * k(26)
            p = (s["p"] or {}).get(action, 0.0)
            chosen = s["exec"] == action
            self.text(surface, KEYS[action], "body", INK if chosen else TEXT, (x, ry))
            pygame.draw.rect(surface, TRACK, (bar_x, ry + k(3), bar_w, k(12)), border_radius=3)
            if p > 0:
                color = SERIES.get(name, TEXT) if chosen else FAINT
                pygame.draw.rect(
                    surface,
                    color,
                    (bar_x, ry + k(3), max(2, int(bar_w * p)), k(12)),
                    border_radius=3,
                )
            self.text(
                surface,
                f"{p * 100:3.0f}%",
                "body",
                INK if chosen else TEXT,
                (bar_x + bar_w + k(10), ry),
            )
            if s["veto"] and s["prop"] == action:
                self.text(surface, "vetoed", "small", WARN, (bar_x + bar_w + k(54), ry + k(2)))
            elif chosen and s["veto"]:
                self.text(surface, "SHIELD", "label", WARN, (bar_x + bar_w + k(54), ry + k(3)))
        minutes = max(self.now, 1) / 60
        if self.meta["players"][i]["paid"]:
            cost = (f"${s['cost']:.4f}", f"${s['cost'] / minutes:.4f} / min")
        else:
            cost = ("free", "on this Mac")
        answer = f"{s['ms']:.0f} ms" if s["ms"] is not None else "—"
        model = f"model {s['model_ms']:.0f} ms" if s["model_ms"] is not None else None
        tiles = [
            ("SCORE", f"{s['score']:,}", None),
            ("BEST", f"{s['top']:,}", None),
            ("DEATHS", s["deaths"], None),
            ("ANSWER TIME", answer, model),
            (
                "SURVIVAL",
                f"{s.get('spectator', {}).get('survival_frames', 0) / 60:.1f}s",
                s.get("phase", ""),
            ),
            (
                "BEST MOVE",
                f"{100 * s['agree']:.0f}%" if s["agree"] is not None else "—",
                None,
            ),
            (
                "LIVE SAVES",
                s.get("spectator", {}).get("emergency_saves", 0)
                + s.get("spectator", {}).get("arrival_saves", 0),
                f"{s['saves']} model vetoes",
            ),
            ("API COST", *cost),
        ]
        cell = w / 4
        for n, (label, value, sub) in enumerate(tiles):
            cx, cy = x + (n % 4) * cell, y + k(108) + (n // 4) * k(68)
            self.text(surface, label, "label", FAINT, (cx, cy))
            self.text(surface, value, "big" if n < 4 else "mid", INK, (cx, cy + k(15)))
            if sub:
                self.fit_text(
                    surface,
                    sub,
                    "small",
                    TEXT,
                    (cx, cy + k(15) + (k(30) if n < 4 else k(24))),
                    cell - k(8),
                )
        if s["err"]:
            self.text(surface, f"last error: {s['err'][:60]}", "small", WARN, (x, y + k(238)))

    def foot(self, surface, frame):
        meta, y = self.meta, self.footer
        pygame.draw.line(surface, RULE, (self.margin, y - 10), (self.size[0] - self.margin, y - 10))
        if meta["course"] and frame["c"]:
            c = frame["c"]
            course = f"Course designed by {meta['course']}: {c['designed']} obstacles so far"
            if filled := c["fallbacks"] + c["errors"]:
                course += f" ({filled} filled by the original random rule)"
        else:
            course = (
                "Course: staged random"
                if meta.get("course_style") == "staged"
                else "Course: the original random rules"
            )
        state = "PAUSED · " if frame["paused"] else ""
        line = f"{state}{course} · seed {meta['seed']}"
        self.text(surface, line, "small", TEXT, (self.margin, y))
        right = "made with laya-mlx" if self.video else "space pause · M sound · Q quit"
        self.text(surface, right, "small", FAINT, (self.size[0] - self.margin, y), right=True)

    # -- Charts --------------------------------------------------------------------------
    def chart_band(self, surface):
        x, y, w, h = self.charts
        gap = self.px(36)
        each = (w - 2 * gap) / 3
        self.line_chart(surface, (x, y, each, h), "SCORE OVER TIME", self.scores, "{:,.0f}")
        latency = [self.latency_series(i) for i in range(len(self.names))]
        self.line_chart(
            surface, (x + each + gap, y, each, h), "ANSWER TIME PER MOVE, MS", latency, "{:,.0f} ms"
        )
        self.head_to_head(surface, (x + 2 * (each + gap), y, each, h))

    def legend(self, surface, right, y):
        for name in reversed(self.names):
            right -= self.text(surface, name, "small", TEXT, (right, y), right=True) + self.px(6)
            right -= self.chip(surface, name, right - self.px(10), y + self.px(2)) + self.px(14)

    def line_chart(self, surface, rect, title, series, fmt):
        x, y, w, h = rect
        k = self.px
        self.text(surface, title, "label", FAINT, (x, y))
        self.legend(surface, x + w, y - k(1))
        left, right, top, bottom = x + k(40), x + w - k(74), y + k(26), y + h - k(18)
        t_max = max(20.0, self.now)
        v_max = nice_ceiling(max([v for points in series for _, v in points] + [1]))
        for fraction in (0, 0.5, 1):
            gy = bottom - (bottom - top) * fraction
            pygame.draw.line(surface, RULE, (left, gy), (right, gy))
            self.text(
                surface,
                f"{v_max * fraction:,.0f}",
                "small",
                FAINT,
                (left - k(8), gy - k(7)),
                right=True,
            )
        self.text(surface, "0 s", "small", FAINT, (left, bottom + k(3)))
        self.text(surface, f"{t_max:.0f} s", "small", FAINT, (right, bottom + k(3)), right=True)
        ends = []
        for name, points in zip(self.names, series):
            if not points:
                continue
            color = SERIES.get(name, TEXT)
            path = [
                (
                    left + (right - left) * min(t, t_max) / t_max,
                    bottom - (bottom - top) * min(v, v_max) / v_max,
                )
                for t, v in points
            ]
            if len(path) > 1:
                pygame.draw.lines(surface, color, False, path, k(2))
                pygame.draw.aalines(surface, color, False, path)
            ends.append([path[-1], color, fmt.format(points[-1][1])])
        # End labels are nudged apart rather than allowed to overlap.
        ends.sort(key=lambda e: e[0][1])
        label_y = [e[0][1] - k(7) for e in ends]
        for i in range(1, len(label_y)):
            label_y[i] = max(label_y[i], label_y[i - 1] + k(14))
        for (point, color, label), ly in zip(ends, label_y):
            pygame.draw.circle(surface, PAGE, point, k(4) + 2)  # Surface ring.
            pygame.draw.circle(surface, color, point, k(4))
            self.text(surface, label, "small", INK, (point[0] + k(10), ly))

    def bars(self, surface, x, y, w, title, values, fmt, v_max):
        k = self.px
        self.text(surface, title, "label", FAINT, (x, y))
        name_w, thickness = k(48), k(14)
        for row, (name, value) in enumerate(zip(self.names, values)):
            ry = y + k(19) + row * k(21)
            self.chip(surface, name, x, ry + k(3))
            self.text(surface, name, "small", TEXT, (x + k(16), ry))
            if value is None:
                continue
            x0 = x + k(16) + name_w
            length = max(3, (w - k(16) - name_w - k(84)) * min(value, v_max) / v_max)
            # Square at the baseline, rounded at the data end.
            pygame.draw.rect(
                surface,
                SERIES.get(name, TEXT),
                (x0, ry + k(1), length, thickness),
                border_top_right_radius=4,
                border_bottom_right_radius=4,
            )
            self.text(surface, fmt.format(value), "small", INK, (x0 + length + k(8), ry))

    def head_to_head(self, surface, rect):
        x, y, w, _ = rect
        if not self.last:
            return
        panels = self.last["s"]
        times = [s["ms"] for s in panels]
        known = [v for v in times if v]
        ceiling = nice_ceiling(max(known + [1]))
        self.bars(surface, x, y, w, "ANSWER TIME, MEDIAN", times, "{:,.0f} ms", ceiling)
        agree = [100 * s["agree"] if s["agree"] is not None else None for s in panels]
        self.bars(surface, x, y + self.px(72), w, "BEST MOVE PICKED", agree, "{:.0f}%", 100)


class Window:
    def __init__(self, arena, args):
        self.arena = arena
        self.args = args
        pygame.init()
        pygame.display.set_caption("Signal — T-Rex")
        self.capture = Capture(arena, args)
        n = len(arena.pilots)
        desktop = pygame.display.get_desktop_sizes()[0][0]
        fit = (desktop - 2 * 28 - (n - 1) * 28 - 40) / (n * WIDTH)
        scale = args.scale or max(0.8, min(2.0, int(fit * 20) / 20))
        pygame.display.set_mode((1, 1))  # Sprites need a display before they can be converted.
        self.painter = Painter("window", self.capture.meta(), scale)
        self.screen = pygame.display.set_mode(self.painter.size)
        self.recorder = (
            Recorder(args.record, self.capture.meta(), seconds=args.record_seconds)
            if args.record
            else None
        )
        self.sounds = {}
        if args.sound:
            pygame.mixer.init()
            names = {"press": "button-press", "hit": "hit", "score": "score-reached"}
            self.sounds = {k: pygame.mixer.Sound(ASSETS / f"{v}.ogg") for k, v in names.items()}
        self.clock = pygame.time.Clock()

    def loading(self, lines):
        painter = self.painter
        self.screen.fill(PAGE)
        painter.text(self.screen, "Signal — T-Rex", "title", INK, (painter.margin, 28))
        for i, line in enumerate(lines[-8:] or ["Starting…"]):
            painter.text(self.screen, line, "body", TEXT, (painter.margin, 84 + 24 * i))
        pygame.display.flip()

    def events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    return False
                if event.key == pygame.K_SPACE:
                    self.arena.paused = not self.arena.paused
                if event.key == pygame.K_m and self.sounds:
                    self.args.sound = not self.args.sound
        return True

    def run(self, warm_up):
        lines, failure = [], []

        def prepare():
            try:
                warm_up(self.arena, log=lines.append)
            except Exception as error:
                failure.append(error)

        worker = threading.Thread(target=prepare, daemon=True)
        worker.start()
        while worker.is_alive():
            if not self.events():
                return
            self.loading(lines)
            self.clock.tick(30)
        if failure:
            raise failure[0]
        arena = self.arena
        arena.start()
        finished_at = None
        while self.events():
            arena.advance()
            for pilot in arena.pilots:
                for name in pilot.game.drain_events():
                    if self.args.sound and name in self.sounds:
                        self.sounds[name].play()
            frame = self.capture.frame()
            self.painter.observe(frame)
            self.painter.draw(self.screen, frame)
            pygame.display.flip()
            if self.recorder:
                self.recorder.write(frame)
            if arena.finished:
                if finished_at is None:
                    finished_at = time.perf_counter()
                elif time.perf_counter() - finished_at >= 5:
                    break
            if self.args.duration and arena.frame * FRAME_MS / 1000 >= self.args.duration:
                break
            self.clock.tick(60)
        if self.args.snapshot:
            Path(self.args.snapshot).parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(self.screen, self.args.snapshot)

    def close(self):
        if self.recorder:
            self.recorder.close()
        pygame.quit()


def run_window(arena, args, warm_up):
    window = Window(arena, args)
    try:
        window.run(warm_up)
    finally:
        window.close()
