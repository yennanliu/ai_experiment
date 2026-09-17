"""Exercise 3 — 2.273x against 0.635x, and the chat task's optimal worker count is one.

    Compute the expected Hogwild! speedup for a 50k-token reasoning task with
    `p=0.8, c=500` and N=4 workers. Do the same for a 1k-token chat task with
    `p=0.3, c=200` and N=4. Why is one a win and the other a loss?

Reading of the exercise: both numbers come from the lesson's own
`expected_speedup`, and "why" is answered by separating the two terms it
contains -- Amdahl's parallel fraction, which is a property of the task, and the
`c * N` coordination overhead, which is a property of the deployment. Each is
then held fixed while the other varies, because the exercise's two cases differ
in both at once.

**ANSWER: 2.273x and 0.635x, and the chat task is slower than serial at every
N > 1.**

    task                  N=2      N=4      N=8    best N
    50k, p=0.8, c=500   1.613x   2.273x   2.632x     9
    1k,  p=0.3, c=200   0.800x   0.635x   0.428x     1

At `N = 1` the formula still charges `c * 1`, so it reports **0.833x** for a run
with nobody to coordinate with.

**MECHANISM: the overhead is `c * N` and the task is `T * (1 - p + p/N)`.** The
chat task's serial time is 1,000 units and its coordination cost at N=4 is
`200 * 4` = **800** -- 80% of the whole task before any work is done. The
reasoning task's is `500 * 4` = 2,000 against 50,000, **4%**.

**FINDING: neither `p` nor `c` alone decides it.** At the chat task's `p=0.3`
with the reasoning task's `T=50000`, N=4 gives **1.23x** -- a win. At the
reasoning task's `p=0.8` with the chat task's `T=1000` and `c=200`, N=4 gives
**0.83x** -- a loss. The variable that separates the two cases is `T`, which the
exercise's framing treats as scene-setting.

**FINDING: `expected_speedup` takes a fifth argument it never uses.** Its
signature is `(T_serial, p, c, N, steps_per_worker)` and the body is
`T_serial / (T_serial * ((1 - p) + p / N) + c * N)`. `steps_per_worker` appears
nowhere, so every caller must supply a number that cannot affect the result.

**MECHANISM: Amdahl caps the reasoning task at 5x and the chat task at 1.43x.**
`1 / (1 - p)` is the N to infinity limit with no overhead. The reasoning task
reaches 2.632x of its 5x at N=8; the chat task's ceiling is 1.43x and its
overhead consumes all of it before N=2.

Structure: `speedup` calls the lesson's own function; `best` scans for the
maximising worker count; `ceiling` is Amdahl's limit.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "22-async-hogwild-inference"
TASKS = {"50k reasoning": (50_000, 0.8, 500), "1k chat": (1_000, 0.3, 200)}
WORKERS = (2, 4, 8)
SCAN = range(1, 41)


def speedup(ref, task, workers):
    total, parallel_fraction, cost = task
    return ref.expected_speedup(total, parallel_fraction, cost, workers, 0)


def best(ref, task):
    return max(SCAN, key=lambda n: speedup(ref, task, n))


def ceiling(task):
    return 1 / (1 - task[1])


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = {}
    for name, task in TASKS.items():
        rows[name] = {"task": task, "best": best(ref, task), "ceiling": ceiling(task),
                      **{n: speedup(ref, task, n) for n in WORKERS}}
        rows[name]["best_value"] = speedup(ref, task, rows[name]["best"])
        rows[name]["overhead_share"] = task[2] * 4 / task[0]
    import inspect
    swapped = {
        "chat p, reasoning T": speedup(ref, (50_000, 0.3, 500), 4),
        "reasoning p, chat T": speedup(ref, (1_000, 0.8, 200), 4),
    }
    return {"rows": rows, "swapped": swapped,
            "params": list(inspect.signature(ref.expected_speedup).parameters),
            "body_uses_steps": "steps_per_worker" in
            inspect.getsource(ref.expected_speedup).split("return")[1]}


def column(rows, key, fmt):
    return ", ".join(f"{name} {format(row[key], fmt)}" for name, row in rows.items())


def verify(result):
    rows, swapped = result["rows"], result["swapped"]
    reasoning, chat = rows["50k reasoning"], rows["1k chat"]
    return [
        practice.Check(
            "ANSWER: 2.273x against 0.635x, and the chat task's best worker count is one",
            reasoning[4] > 2 > 1 > chat[4] and chat["best"] == 1,
            "at N=4 the two tasks score " + column(rows, 4, ".3f")
            + "x, and across N=2, 4 and 8 the reasoning task runs "
            + ", ".join(f"{reasoning[n]:.3f}" for n in WORKERS)
            + " while the chat task runs "
            + ", ".join(f"{chat[n]:.3f}" for n in WORKERS)
            + f". Scanning to 40 workers, the reasoning task peaks at N={reasoning['best']} with "
            f"{reasoning['best_value']:.3f}x and the chat task at N={chat['best']} with "
            f"{chat['best_value']:.3f}x -- no parallelism helps it at all",
        ),
        practice.Check(
            "MECHANISM: the overhead is c*N and the chat task's is 80% of the whole job",
            chat["overhead_share"] > 0.5 > reasoning["overhead_share"],
            f"expected_speedup is T / (T * (1 - p + p/N) + c * N). The chat task's serial time is "
            f"{chat['task'][0]:,} units and its coordination cost at N=4 is "
            f"{chat['task'][2]} x 4 = {chat['task'][2] * 4:,} -- "
            f"{chat['overhead_share']:.0%} of the whole task before any work is done. The "
            f"reasoning task's is {reasoning['task'][2] * 4:,} against "
            f"{reasoning['task'][0]:,}, {reasoning['overhead_share']:.0%}",
        ),
        practice.Check(
            "FINDING: neither p nor c alone decides it -- the variable is T",
            swapped["chat p, reasoning T"] > 1 > swapped["reasoning p, chat T"],
            f"at the chat task's p=0.3 with the reasoning task's T=50,000 and c=500, N=4 gives "
            f"{swapped['chat p, reasoning T']:.2f}x -- a win. At the reasoning task's p=0.8 with "
            f"the chat task's T=1,000 and c=200, N=4 gives "
            f"{swapped['reasoning p, chat T']:.2f}x -- a loss. The variable that separates the "
            "two cases is the one the exercise's framing treats as scene-setting",
        ),
        practice.Check(
            "MECHANISM: Amdahl caps them at 5x and 1.43x before any overhead",
            abs(reasoning["ceiling"] - 5.0) < 1e-9 and chat["ceiling"] < 1.5,
            "1 / (1 - p) is the N-to-infinity limit with no overhead: " + column(rows, "ceiling", ".2f")
            + f"x. The reasoning task reaches {reasoning[8]:.2f}x of its "
            f"{reasoning['ceiling']:.0f}x at N=8; the chat task's ceiling is "
            f"{chat['ceiling']:.2f}x and its overhead consumes all of it before N=2. And "
            f"expected_speedup takes a fifth argument, steps_per_worker, that its body never "
            f"uses -- {result['params']}, and the return line references it: "
            f"{result['body_uses_steps']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
