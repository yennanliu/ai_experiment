"""Exercise 4 — per-example timeouts.

    Cap latency per example. Wrap the adapter call in a timeout; surface a separate `timeouts` column in the leaderboard.

Reading of the exercise: `run_task_capped` replaces the lesson's `run_task`
loop. Each call to `adapter.generate` runs in a daemon thread that is joined
with a cap. If the cap passes, every example in that call is scored as an
empty prediction and counted in `timeouts`. The scores go through the
lesson's `METRIC_FNS`, the board is written with the lesson's
`write_leaderboard`, and a `timeouts` field is then added to each task row.
The adapter under test is the lesson's `ToyAdapter`, except that it blocks
on an event that is never set for two prompts, `arith-01` and `code-02`. The
cap is 0.5 s, and the toy answers in microseconds, so the result does not
depend on timing. The same wrapper runs twice: once around each example,
and once around each batch of 4, which is `main.py`'s default.

**ANSWER: capped per example, the two hung prompts cost 1 timeout each.**
arithmetic and code-exec score 0.8, the other three tasks 1.0, the overall is
0.92, and the leaderboard JSON carries `timeouts` = 1, 1, 0, 0, 0.

**FINDING: wrapping the batched call costs the whole batch.** At batch size
4, each hung prompt times out the 4 examples that share its batch. That
makes 8 timeouts instead of 2, arithmetic and code-exec drop to 0.2, and the
overall is 0.68.

**FINDING: a timeout does not stop the call.** A Python thread cannot be
killed, so after the per-example run both hung calls are still alive. Any
cap needs the adapter to cooperate, or a separate process.

**FINDING: the adapter cap does not bound the metric.** `metric_code_exec`
runs the prediction in the harness's own process. A prediction
`while True: pass` never returns. Run in a subprocess, it is killed at its
5 s cap after the module has loaded, while a correct prediction there prints 1.0. In-process, it would
hang the lesson's runner for good.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import threading
import types

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "49-lm-eval-harness"
CAP_S = 0.5
HANG = {"compute: 7 - 3", "python: write a function f that squares its input"}
LOOP = "def f(x):\n    while True:\n        pass\n"
CHILD = (
    "import importlib.util, sys\n"
    "s = importlib.util.spec_from_file_location('m', sys.argv[1]); m = importlib.util.module_from_spec(s)\n"
    "sys.modules['m'] = m; s.loader.exec_module(m); print('started', flush=True)\n"
    "print(m.metric_code_exec(sys.argv[2], [], {'io_pairs': [[1, 2]]}))\n"
)


def hanging_adapter(ref, gate):
    toy = ref.ToyAdapter()

    def generate(prompts):
        if HANG & set(prompts):
            gate.wait()
        return toy.generate(prompts)

    return types.SimpleNamespace(name="toy.v1+hang", generate=generate)


def capped_call(adapter, prompts, threads):
    box = {}
    t = threading.Thread(target=lambda: box.setdefault("out", adapter.generate(prompts)), daemon=True)
    t.start(), t.join(CAP_S), threads.append(t)
    return box.get("out"), t.is_alive()


def run_task_capped(ref, name, examples, adapter, unit, threads):
    fn, total, timeouts = ref.METRIC_FNS[examples[0].metric], 0.0, 0
    for i in range(0, len(examples), unit):
        chunk = examples[i : i + unit]
        out, hung = capped_call(adapter, [ex.prompt for ex in chunk], threads)
        timeouts += len(chunk) if hung else 0
        total += sum(fn(p, ex.targets, ex.extras) for p, ex in zip(out or [""] * len(chunk), chunk))
    return ref.TaskResult(name, examples[0].metric, total / len(examples), round(total), len(examples)), timeouts


def board(ref, tasks, adapter, unit, path, threads):
    runs = {n: run_task_capped(ref, n, tasks[n], adapter, unit, threads) for n in sorted(tasks)}
    results = [res for res, _ in runs.values()]
    lb = ref.Leaderboard("leaderboard.v1", 0.0, sum(r.score for r in results) / len(results), results)
    ref.write_leaderboard(lb, path, adapter_name=adapter.name)
    payload = json.loads(path.read_text())
    for row in payload["tasks"]:
        row["timeouts"] = runs[row["task"]][1]
    path.write_text(json.dumps(payload, indent=2) + "\n")
    rows = json.loads(path.read_text())["tasks"]  # read back: the column is in the written file
    return round(payload["overall_score"], 3), {t["task"]: (round(t["score"], 3), t["timeouts"]) for t in rows}


def child(source, prediction, cap):
    try:
        done = subprocess.run([sys.executable, "-c", CHILD, source, prediction], capture_output=True, text=True, timeout=cap)
        return done.stdout.split()[-1]
    except subprocess.TimeoutExpired as exc:  # killed; 'started' proves it was the loop, not startup
        return "killed after start" if "started" in str(exc.stdout or "") else "killed before start"


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    tmp = pathlib.Path(tempfile.mkdtemp())
    ref.seed_fixture_tasks(tmp / "tasks")
    tasks, gate, threads = ref.load_all_tasks(tmp / "tasks"), threading.Event(), []
    per = board(ref, tasks, hanging_adapter(ref, gate), 1, tmp / "per.json", threads)
    alive = sum(t.is_alive() for t in threads)
    batch = board(ref, tasks, hanging_adapter(ref, gate), 4, tmp / "batch.json", threads)
    gate.set()
    source = str(parity.lesson_dir(PHASE, LESSON) / "code" / "main.py")
    loop, ok = child(source, LOOP, 5), child(source, "def f(x):\n    return x * 2\n", 60)
    return {"per": per, "batch": batch, "alive": alive, "loop": loop, "ok": ok}


def verify(result):
    r = result
    timeouts_col = lambda b: [v[1] for _, v in sorted(b[1].items())]  # noqa: E731
    return [
        practice.Check(
            "ANSWER: per-example cap: 1 timeout per hung prompt, overall 0.92, timeouts column in the JSON",
            r["per"][0] == 0.92 and timeouts_col(r["per"]) == [1, 1, 0, 0, 0]
            and (r["per"][1]["arithmetic"][0], r["per"][1]["code-exec"][0]) == (0.8, 0.8),
            f"overall {r['per'][0]}; (score, timeouts) {r['per'][1]}",
        ),
        practice.Check(
            "FINDING: capping the batched call times out the whole batch of 4",
            r["batch"][0] == 0.68 and timeouts_col(r["batch"]) == [4, 4, 0, 0, 0],
            f"overall {r['batch'][0]}; (score, timeouts) {r['batch'][1]}",
        ),
        practice.Check(
            "FINDING: a timed-out call keeps running; threads cannot be killed",
            r["alive"] == 2,
            f"{r['alive']} hung calls still alive after the per-example run",
        ),
        practice.Check(
            "FINDING: the adapter cap does not bound code_exec; an infinite loop needs a process kill",
            (r["loop"], r["ok"]) == ("killed after start", "1.0"),
            f"while-True prediction in a subprocess: {r['loop']}; correct prediction: {r['ok']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
