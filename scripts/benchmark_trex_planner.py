"""Deterministic, model-free planner benchmark.

Run from the repository root: python -m scripts.benchmark_trex_planner
Compare output_sha256 across revisions before comparing timings. This exercises
single and staggered requests, but does not measure model or network latency.
"""

import hashlib
import json
import statistics
import time
from dataclasses import asdict

from laya_mlx.trex.engine import Game
from laya_mlx.trex.planner import Planner, snapshot


def main():
    game = Game(17)
    game.invincible = True
    game.press_jump()
    samples = []
    for frame in range(4000):
        if frame % 39 == 0:
            game.press_jump()
        game.step()
        if game.obstacles and frame % 31 == 0:
            samples.append(snapshot(game, "run"))
    planner = Planner()
    results, times = [], []
    for timing, period in [((1, 3), None), ((1, 3), 2), ((18, 24), 6)]:
        for snap in samples:
            start = time.perf_counter()
            plan = planner.plan(snap, timing, timing, period)
            times.append((time.perf_counter() - start) * 1000)
            results.append(asdict(plan))
    print(
        json.dumps(
            {
                "calls": len(times),
                "total_ms": round(sum(times), 2),
                "median_ms": round(statistics.median(times), 3),
                "p95_ms": round(sorted(times)[int(len(times) * 0.95)], 3),
                "output_sha256": hashlib.sha256(
                    json.dumps(results, sort_keys=True).encode()
                ).hexdigest(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
