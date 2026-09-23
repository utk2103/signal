"""Resource cleanup when starting an arena fails partway through."""

import pytest

from laya_mlx.trex import cli


def test_build_closes_started_brains_when_next_player_fails(monkeypatch):
    closed = []

    class Brain:
        def __init__(self, kind, **options):
            if kind == "jev":
                raise RuntimeError("backend unavailable")

        def close(self):
            closed.append("laya")

    monkeypatch.setattr(cli, "RemoteBrain", Brain)
    with pytest.raises(RuntimeError, match="backend unavailable"):
        cli.build(cli.parse(["--course", "random"]), log=lambda message: None)
    assert closed == ["laya"]


@pytest.mark.parametrize("failure", ["designer", "director", "pilot", "arena", "success"])
def test_build_releases_resources_after_each_startup_failure(monkeypatch, failure):
    closed = []

    class Resource:
        def __init__(self, name):
            self.name = name

        def close(self):
            closed.append(self.name)

    class Brain(Resource):
        def __init__(self, kind, **options):
            super().__init__(kind)

    def create(*args, **kwargs):
        if failure == "designer":
            raise RuntimeError("startup failed")
        return Resource("designer")

    class Director(Resource):
        def __init__(self, designer, *args, **kwargs):
            if failure == "director":
                raise RuntimeError("startup failed")
            super().__init__("director")
            self.designer = designer

        def close(self):
            super().close()
            self.designer.close()

    class Pilot(Resource):
        def __init__(self, brain, **kwargs):
            if failure == "pilot" and brain.name == "jev":
                raise RuntimeError("startup failed")
            super().__init__(f"pilot-{brain.name}")
            self.brain = brain

        def close(self):
            super().close()
            self.brain.close()

    def arena(pilots, **kwargs):
        if failure == "success":
            return pilots
        raise RuntimeError("startup failed")

    monkeypatch.setattr(cli, "RemoteBrain", Brain)
    monkeypatch.setattr(cli, "create", create)
    monkeypatch.setattr(cli, "CourseDirector", Director)
    monkeypatch.setattr(cli, "Pilot", Pilot)
    monkeypatch.setattr(cli, "Arena", arena)
    if failure == "success":
        pilots = cli.build(cli.parse([]), log=lambda message: None)
        assert [pilot.brain.name for pilot in pilots] == ["laya", "jev"]
        assert closed == []
        return
    with pytest.raises(RuntimeError, match="startup failed"):
        cli.build(cli.parse([]), log=lambda message: None)
    expected = ["laya", "jev"]
    if failure != "designer":
        expected.append("designer")
    if failure in ("pilot", "arena"):
        expected.extend(["director", "pilot-laya"])
    if failure == "arena":
        expected.append("pilot-jev")
    assert sorted(closed) == sorted(expected)


def test_director_closes_backend_once_when_stopped():
    closed = []

    class Backend:
        def close(self):
            closed.append("backend")

    director = cli.CourseDirector(Backend(), seed=7)
    director.close()
    director.thread.join(timeout=1)
    assert not director.thread.is_alive()
    assert closed == ["backend"]
    director.close()
    assert closed == ["backend"]


def test_director_finishes_active_request_before_closing_backend(monkeypatch):
    from threading import Event

    started = Event()
    release = Event()
    closed = []

    class Backend:
        def ask(self):
            started.set()
            assert release.wait(timeout=5)

        def close(self):
            closed.append("backend")

    backend = Backend()

    class Ghost:
        obstacle_index = 0

        def step(self):
            backend.ask()

    director = cli.CourseDirector(backend, seed=7)
    monkeypatch.setattr(director, "ghost", lambda run: Ghost())
    try:
        director.demand(0, 0)
        assert started.wait(timeout=1)
        director.close()
        assert director.thread.is_alive()
        assert closed == []
    finally:
        release.set()
        director.close()
        director.thread.join(timeout=1)
    assert not director.thread.is_alive()
    assert closed == ["backend"]
