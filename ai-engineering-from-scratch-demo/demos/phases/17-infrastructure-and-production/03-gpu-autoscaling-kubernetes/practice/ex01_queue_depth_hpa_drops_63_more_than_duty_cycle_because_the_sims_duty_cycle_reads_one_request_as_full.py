"""Exercise 1 — queue-depth HPA drops 63 more than duty-cycle, because the sim's duty cycle reads one request as full.

    Run `code/main.py`. Under a bursty workload, how many requests does naive
    duty-cycle HPA drop that queue-depth HPA catches? Where does the difference
    come from?

Reading of the exercise: "drop that queue-depth HPA catches" is a set
difference, so both strategies run on the same 743-request workload and the
dropped requests are compared one by one, not only counted. "Where does the
difference come from" is answered by tracing replicas per tick and by moving
the module's own knobs one at a time.

**ANSWER: none -- duty-cycle drops 1, queue-depth drops 64, and the one
duty-cycle drops queue-depth drops too.** Queue-depth HPA drops 63 requests
that duty-cycle HPA serves, and 0 the other way. The script prints "DUTY_CYCLE
drops requests because DCGM_FI_DEV_GPU_UTIL is a duty-cycle metric" under a
table that shows the reverse. The cost goes the other way as well: 266.2
idle GPU-minutes for duty-cycle against 49.5.

**FINDING: the difference is the spike onset and the scale-down rule.** The
sim's "utilization" is busy/ready read right after dispatch, so with one
replica a single request reads 100% > 70 and scales up. Duty-cycle is at 6
ready replicas when the spike starts at t = 600; queue-depth, which needs more
than 5 waiting, is still at 1, and 33 of its 64 drops arrive in 600-703s. After
that, queue-depth removes a replica on every tick the queue is empty, and a
new one takes 50 + 45 = 95s, so it saw-tooths; duty-cycle scales down only
below 20% and holds 14-16 replicas from t = 840 to the end of the spike. With MIN_WARM_REPLICAS = 6 queue-depth
drops 2; with instant nodes (NODE_PROVISION_SEC = 0) it still drops 44.

**FINDING: a real duty cycle never scales this sim at all.** A replica takes
one request per 15s tick and is busy 0.6 + 1.8 = 2.4s of it: 16% busy, below
the 20% scale-down line, so the > 70% branch can never fire. That is the
module run with MAX_REPLICAS = 1, and it drops 511 of 743. The lesson's claim
is true of the metric; the simulator's version of the metric is not it.

Structure: `run()` reruns the reference `simulate` with module knobs
overridden and restored; `trace()` reads replicas per tick with sys.settrace.
"""

from __future__ import annotations

import sys

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "03-gpu-autoscaling-kubernetes"


def load_ref():
    return parity.load_reference(PHASE, LESSON, "main")


def run(ref, strategy, **knobs):
    """(requests after the run, summary row) with module constants overridden."""
    saved = {k: getattr(ref, k) for k in knobs}
    for key, value in knobs.items():
        setattr(ref, key, value)
    try:
        reqs = [ref.Request(arrived_at=r.arrived_at) for r in ref.make_workload()]
        return reqs, ref.simulate(strategy, reqs)
    finally:
        for key, value in saved.items():
            setattr(ref, key, value)


def trace(ref, strategy):
    """Ready replicas at the end of each tick, read from simulate's locals."""
    ready = {}

    def local(frame, event, arg):
        if event == "line" and "peak_replicas" in frame.f_locals:
            ready[frame.f_locals["now"]] = frame.f_locals["replicas_ready"]
        return local

    previous = sys.gettrace()
    sys.settrace(lambda frame, e, a: local if frame.f_code.co_name == "simulate" else None)
    try:
        run(ref, strategy)
    finally:
        sys.settrace(previous)
    return ready


def dropped(reqs):
    return {i for i, r in enumerate(reqs) if r.dropped}


def solve():
    ref = load_ref()
    (dc, dc_row), (qd, qd_row) = run(ref, "DUTY_CYCLE"), run(ref, "QUEUE_DEPTH")
    return {
        "dc": dc_row,
        "qd": qd_row,
        "qd_only": len(dropped(qd) - dropped(dc)),
        "dc_only": len(dropped(dc) - dropped(qd)),
        "onset": sum(1 for i in dropped(qd) if 600 <= qd[i].arrived_at < 705),
        "ready_600": {s: trace(ref, s)[600.0] for s in ("DUTY_CYCLE", "QUEUE_DEPTH")},
        "warm6": run(ref, "QUEUE_DEPTH", MIN_WARM_REPLICAS=6)[1]["dropped"],
        "instant": run(ref, "QUEUE_DEPTH", NODE_PROVISION_SEC=0)[1]["dropped"],
        "busy_frac": (ref.REQUEST_PREFILL_SEC + ref.REQUEST_DECODE_SEC) / ref.HPA_TICK_SEC,
        "never_scales": run(ref, "DUTY_CYCLE", MAX_REPLICAS=1)[1],
    }


def verify(result):
    dc, qd, ready = result["dc"], result["qd"], result["ready_600"]
    frozen = result["never_scales"]
    return [
        practice.Check(
            "ANSWER: none -- duty-cycle drops 1, queue-depth drops 64, and the one "
            "duty-cycle drops queue-depth drops too",
            all(
                [
                    (dc["dropped"], qd["dropped"], result["qd_only"], result["dc_only"])
                    == (1, 64, 63, 0),
                    dc["idle_gpu_min"] > 5 * qd["idle_gpu_min"],
                ]
            ),
            f"of {dc['total']} requests duty-cycle drops {dc['dropped']}, queue-depth "
            f"{qd['dropped']}; queue-depth-only drops {result['qd_only']}, duty-cycle-only "
            f"{result['dc_only']}; idle GPU-min {dc['idle_gpu_min']} vs {qd['idle_gpu_min']}",
        ),
        practice.Check(
            "FINDING: the difference is the spike onset and the scale-down rule",
            all(
                [
                    ready == {"DUTY_CYCLE": 6, "QUEUE_DEPTH": 1},
                    result["onset"] == 33,
                    result["warm6"] == 2,
                    result["instant"] == 44,
                ]
            ),
            f"ready replicas at t=600 {ready}; {result['onset']} of {qd['dropped']} "
            f"queue-depth drops arrive in 600-703s; MIN_WARM_REPLICAS=6 -> "
            f"{result['warm6']}, NODE_PROVISION_SEC=0 -> {result['instant']}",
        ),
        practice.Check(
            "FINDING: a real duty cycle never scales this sim at all",
            all([round(result["busy_frac"], 2) == 0.16, frozen["dropped"] == 511]),
            f"a replica is busy {result['busy_frac']:.0%} of each tick, under the 20% "
            f"scale-down line; held at one replica duty-cycle drops {frozen['dropped']} "
            f"of {frozen['total']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
