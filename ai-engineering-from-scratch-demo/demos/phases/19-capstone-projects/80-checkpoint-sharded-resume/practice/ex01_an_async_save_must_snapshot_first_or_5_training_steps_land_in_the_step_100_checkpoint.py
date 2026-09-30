"""Exercise 1 — an async save must snapshot first: without a clone, 5 steps of training land in the step-100 checkpoint and it still verifies.

    Add async write: kick off the save in a thread and let training continue. Block the next save until the previous one completes.

Reading of the exercise: `AsyncSaver` wraps the lesson's own `save_sharded`
in a `threading.Thread`; `save()` first joins the previous thread (the
barrier), then clones every tensor and hands the copy to a new thread. The
lesson's four demo ranks (`make_demo_state`, step 100) are trained with an
in-place Adam-style update. To make the overlap deterministic rather than
timed, the lesson's `_serialize_state` is wrapped so the first save cannot
serialize until the main thread has trained 5 steps and entered the second
save's barrier; the event log then shows who waited for whom.

**ANSWER: training runs 5 steps while save 1 is in flight, and save 2 waits
for it.** The log reads save1-start, 5 train steps, save2-waiting,
save1-done, save2-start, save2-done, and at most one save is ever in flight.
Both checkpoints then load through the lesson's `load_sharded` and match the
state at the step they were requested -- param, m and v on all 4 ranks
byte-equal -- although the live tensors had moved on by the time they were
written.

**FINDING: without the snapshot the checkpoint is torn and still passes
every check.** Hand the live dicts to the thread instead and the step-100
checkpoint holds step-105 parameters on 4/4 ranks, and each shard's own
`step` field says 105 while the manifest says 100. `load_sharded` accepts it:
the sha256 is taken over whatever bytes were written, so it proves the file
is intact, not that it is the state of the step it claims. The loader never
compares the shard's `step` with the manifest's.

Expected output: two PASS checks.
"""

from __future__ import annotations

import contextlib
import tempfile
import threading

from harness import parity, practice

try:
    import torch
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "80-checkpoint-sharded-resume"
KEYS = ("param_shard", "m_shard", "v_shard")


def snapshot(per_rank):
    return [{k: v.clone() if torch.is_tensor(v) else v for k, v in s.items()} for s in per_rank]


def train_step(per_rank):
    """In-place Adam-style step on every rank, as a real optimiser mutates its state."""
    for s in per_rank:
        g = s["param_shard"]
        s["m_shard"].mul_(0.9).add_(g, alpha=0.1)
        s["v_shard"].mul_(0.999).addcmul_(g, g, value=0.001)
        s["param_shard"].sub_(0.01 * s["m_shard"] / (s["v_shard"].sqrt() + 1e-8))
        s["step"] += 1


def async_saver(ref, log, copy=True):
    """save(per_rank, dir, step): join the previous save, then write in a thread."""
    inflight, barrier_entered = {"thread": None, "now": 0, "max": 0}, threading.Event()

    def run(data, dest, step):
        inflight["now"] += 1
        inflight["max"] = max(inflight["max"], inflight["now"])
        ref.save_sharded(data, dest, step)
        log.append(f"save{step}-done")
        inflight["now"] -= 1

    def save(per_rank, dest, step):
        if inflight["thread"] is not None:
            log.append(f"save{step}-waiting")
            barrier_entered.set()
            inflight["thread"].join()
        log.append(f"save{step}-start")  # logged before the thread runs, so the order is not a race
        inflight["thread"] = threading.Thread(target=run, args=(snapshot(per_rank) if copy else per_rank, dest, step))
        inflight["thread"].start()
    return save, inflight, barrier_entered


@contextlib.contextmanager
def gated(ref, barrier_entered):
    """The first serialize waits until the main thread has reached the next save."""
    original, first = ref._serialize_state, [True]

    def slow(state):
        if first[0]:
            first[0] = False
            barrier_entered.wait(timeout=30)
        return original(state)

    ref._serialize_state = slow
    try:
        yield
    finally:
        ref._serialize_state = original


def scenario(ref, copy):
    states, log = [ref.make_demo_state(r, 4) for r in range(4)], []
    save, inflight, barrier_entered = async_saver(ref, log, copy)
    with tempfile.TemporaryDirectory(prefix="ex01-") as tmp, gated(ref, barrier_entered):
        truth = {100: snapshot(states)}
        save(states, f"{tmp}/step_0100", 100)
        for _ in range(5):
            train_step(states)
            log.append("train")
        truth[105] = snapshot(states)
        save(states, f"{tmp}/step_0105", 105)
        inflight["thread"].join(timeout=30)
        loaded = {s: ref.load_sharded(f"{tmp}/step_{s:04d}", 4) for s in truth}
    return {"log": log, "max_inflight": inflight["max"], **compare(loaded, truth)}


def compare(loaded, truth):
    match = {s: all(torch.equal(loaded[s][1][r][k], truth[s][r][k]) for r in range(4) for k in KEYS) for s in truth}
    stale = sum(torch.equal(loaded[100][1][r]["param_shard"], truth[105][r]["param_shard"]) for r in range(4))
    return {"match": match, "stale_ranks": stale, "steps": (loaded[100][0].step, [s["step"] for s in loaded[100][1]])}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    return {"copy": scenario(ref, True), "live": scenario(ref, False)}


def verify(result):
    good, bad = result["copy"], result["live"]
    order = ["save100-start"] + ["train"] * 5 + ["save105-waiting", "save100-done", "save105-start", "save105-done"]
    return [
        practice.Check(
            "ANSWER: training runs 5 steps while save 1 is in flight, and save 2 waits for it",
            good["log"] == order and good["max_inflight"] == 1 and good["match"] == {100: True, 105: True},
            f"log {good['log']}; max saves in flight {good['max_inflight']}; byte-equal at requested step {good['match']}",
        ),
        practice.Check(
            "FINDING: without the snapshot the checkpoint is torn and still passes load_sharded",
            bad["log"] == order and bad["match"][100] is False and (bad["stale_ranks"], bad["steps"]) == (4, (100, [105] * 4)),
            f"step-100 checkpoint holds step-105 params on {bad['stale_ranks']}/4 ranks; manifest/shard steps {bad['steps']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
