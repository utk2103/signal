"""A recording can end without stopping or limiting the live game."""

import json

from laya_mlx.trex.cli import parse
from laya_mlx.trex.replay import Recorder


def test_recording_closes_at_endpoint_and_ignores_later_play(tmp_path):
    path = tmp_path / "clip.jsonl"
    recorder = Recorder(path, {"type": "metadata"}, seconds=300)
    for f, t in enumerate((0, 299.9, 300.01, 301, 400)):
        recorder.write({"type": "frame", "f": f, "t": t})
    assert recorder.file.closed
    recorder.close()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["t"] for r in rows[1:]] == [0, 299.9, 300.01]


def test_record_limit_does_not_set_game_duration_or_round_limit():
    args = parse(["--round-seconds", "0", "--record", "clip.jsonl", "--record-seconds", "300"])
    assert args.duration is None and args.rounds == 0 and args.round_seconds == 0
    assert args.record_seconds == 300
