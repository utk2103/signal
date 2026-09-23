"""Decision models for the T-Rex autopilot: local Laya (MLX) and TypeSafe's Jev API.

Both receive the same state and the same Choice question, built from the planner's
labels, and return probabilities over jump, duck and run.
"""

import math
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .planner import ACTIONS

PROMPTS = ("labeled", "guided")
JEV_URL = "https://api.typesafe.ai"
JEV_USD_PER_TOKEN = 0.042 / 1_000_000  # jev-1.13 input price; output tokens are free.


def situation(plan):
    if plan.threat:
        ahead = f"{plan.threat[0].upper()}{plan.threat[1:]} ahead, {max(plan.distance, 0)} px away."
    elif plan.distance is not None and plan.ahead:
        ahead = f"{plan.ahead[0].upper()}{plan.ahead[1:]} ahead, {max(plan.distance, 0)} px away."
    else:
        ahead = "Nothing ahead."
    return f"The dino is in the air. {ahead}" if plan.airborne else ahead


def build_question(plan, prompt="labeled"):
    """The state and Choice question sent to either model. `labeled` never names the answer."""
    best, notes = plan.best, plan.notes
    facts = f"Dino runner game. {situation(plan)}"
    if prompt == "labeled":
        criteria = {
            a: f"Safe. {notes[a]}. Best."
            if a == best
            else f"Safe. {notes[a]}."
            if plan.safe[a]
            else f"Unsafe. {notes[a]}. Collision."
            for a in ACTIONS
        }
        instructions = "Choose the best safe action for the dinosaur."
    elif prompt == "guided":
        criteria = {
            a: f"Best: {notes[a]}, safe."
            if a == best
            else f"Safe: {notes[a]}."
            if plan.safe[a]
            else f"Collision: {notes[a]}."
            for a in ACTIONS
        }
        facts += f" Recommended action: {best}."
        instructions = (
            "Which action should the dinosaur take now? Pick the recommended safe action."
        )
    else:
        raise ValueError(f"prompt must be one of {PROMPTS}")
    return facts, {"action": {"type": "choice", "instructions": instructions, "criteria": criteria}}


@dataclass
class Answer:
    probabilities: dict
    inference_ms: float
    input_tokens: int
    model: str


def checked(probabilities):
    values = {a: float(probabilities.get(a, 0.0)) for a in ACTIONS}
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values.values()):
        raise ValueError(f"Model returned an invalid probability: {probabilities}")
    return values


def decide(backend, state, questions):
    """Ask the player question and return its checked action probabilities."""
    answers, elapsed, tokens = backend.ask(state, questions)
    return Answer(checked(answers["action"]["probabilities"]), elapsed, tokens, backend.model)


def hardware_name():
    from laya_mlx.snake.policy import hardware_name as name

    return name()


def prefer_performance_cores():
    """Ask macOS to schedule the calling thread as user-interactive work. Best effort."""
    try:
        import ctypes

        ctypes.CDLL("/usr/lib/libSystem.B.dylib").pthread_set_qos_class_self_np(0x21, 0)
    except (OSError, AttributeError):
        pass


class LayaBackend:
    """The local model. Every MLX call happens on one dedicated, high-priority thread.

    Callers on any thread queue their question and wait. MLX is driven from a single
    thread throughout (loading included), and that thread keeps its scheduling priority
    whatever the callers are doing. Questions can still be prepared while it works.
    """

    name = "Laya"
    usd_per_token = 0.0

    def __init__(self, model=None, optimize=False, wired_bytes=2 << 30, inflight=1):
        from laya_mlx.snake.policy import local_checkpoint

        self.path = local_checkpoint(model)
        self.model = self.path.name
        self.detail = f"local MLX on {hardware_name()} · {self.model}"
        if inflight > 1:
            self.detail += f" · {inflight} questions in flight"
        self.jobs = queue.SimpleQueue()
        ready = queue.SimpleQueue()
        self.thread = threading.Thread(
            target=self.serve, args=(optimize, wired_bytes, ready), name="laya-model", daemon=True
        )
        self.thread.start()
        if (error := ready.get()) is not None:
            raise error

    def serve(self, optimize, wired_bytes, ready):
        prefer_performance_cores()
        try:
            import mlx.core as mx

            from laya_mlx import Agent

            # Keep the weights resident. Under memory pressure macOS otherwise pages them out
            # between moves, and a 14 ms decision can take a quarter of a second.
            limit = mx.device_info()["max_recommended_working_set_size"]
            mx.set_wired_limit(min(wired_bytes, limit))
            agent = Agent(
                self.path,
                dtype="float16",
                device="gpu",
                batch_size=1,
                compile=optimize,
                pad_to_multiple=16 if optimize else None,
                cache_prompts=optimize,
            )
        except Exception as error:
            ready.put(error)
            return
        ready.put(None)
        while (job := self.jobs.get()) is not None:
            state, questions, reply = job
            try:
                started = time.perf_counter()
                output = agent.predict(state, questions)
                reply.put((output, (time.perf_counter() - started) * 1000))
            except Exception as error:
                reply.put(error)

    def ask(self, state, questions):
        reply = queue.SimpleQueue()
        self.jobs.put((state, questions, reply))
        result = reply.get()
        if isinstance(result, Exception):
            raise result
        output, elapsed = result
        return output["answers"], elapsed, output["usage"]["input_tokens"]

    def close(self):
        self.jobs.put(None)


def read_env_file(path):
    values = {}
    for line in Path(path).expanduser().read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip().removeprefix("export ")] = value.strip().strip("'\"")
    return values


class JevBackend:
    name = "Jev"
    usd_per_token = JEV_USD_PER_TOKEN

    def __init__(self, api_key=None, model="jev-latest", timeout=5.0, inflight=1):
        import httpx

        key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise RuntimeError(
                "Jev needs TYPESAFE_API_KEY. Export it, or pass --env-file PATH to a .env file "
                "that defines it."
            )
        # One persistent connection; no retries, since a late answer is useless in real time.
        self.client = httpx.Client(
            base_url=os.environ.get("TYPESAFE_BASE_URL", JEV_URL),
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=inflight + 1, max_keepalive_connections=inflight + 1
            ),
        )
        self.inflight = inflight
        self.request_model = model
        self.model = model
        self.detail = self.describe()

    def describe(self):
        text = f"TypeSafe API over the network · {self.model}"
        return text + (f" · {self.inflight} requests in flight" if self.inflight > 1 else "")

    def ask(self, state, questions):
        started = time.perf_counter()
        response = self.client.post(
            "/v1/systemone",
            json={"model": self.request_model, "state": state, "questions": questions},
        )
        elapsed = (time.perf_counter() - started) * 1000
        if response.status_code != 200:
            raise RuntimeError(f"Jev HTTP {response.status_code}: {response.text[:200]}")
        body = response.json()
        if self.model != body.get("model", self.model):
            self.model = body["model"]
            self.detail = self.describe()
        return body["answers"], elapsed, body.get("usage", {}).get("input_tokens", 0)

    def close(self):
        self.client.close()


def create(kind, *, model=None, optimize=False, inflight=1):
    """`inflight` is how many requests the player keeps going at once. A hosted API answers
    them in parallel. A local model answers one at a time, but a second question can be
    prepared meanwhile, which keeps the GPU from idling."""
    if kind == "laya":
        return LayaBackend(model, optimize=optimize, inflight=inflight)
    if kind == "jev":
        return JevBackend(model=model or "jev-latest", inflight=inflight)
    raise ValueError(f"Unknown backend {kind!r}; expected laya or jev")
