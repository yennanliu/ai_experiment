"""Exercise 1 — the ceiling is a quarter, reached at generation four.

    Run `code/main.py` with default flags. Note the score trajectory and the
    final agent's tool composition.

Reading of the exercise: `run_dgm` prints and returns `None`, so the
trajectory has to be read back out of its own stdout rather than
reconstructed -- which keeps the measurement on the shipped loop instead of a
copy of it. "Note" is then taken to mean: say what the trajectory is, and say
what it is bounded by.

**ANSWER: two improvements in 200 generations, ending on `['collapse',
'nop']` at 0.25.** The run reports 0.00 at generation **0** and 0.25 at
generation **4**; **199** of its 200 generations print nothing at all.
Reported equals true throughout, because the side channel is closed.

**FINDING: 0.25 is the ceiling, not the convergence point.** Brute-forcing
every operator sequence up to length 4 -- **1555** of them -- the best true
score any agent can reach is **0.25**, achieved by `collapse` alone. **6** of
the **8** benchmark cases want title case, and title case is not among the
**6** tools. The closed-channel headline says the loop "converges on the real
target"; it converges on a quarter of it, and the other three quarters are
unreachable by construction.

**FINDING: length is a free axis.** The winner carries a `nop`. The archive
is keyed by `(len(ops), round(reported, 2))`, so `['collapse']` and
`['collapse', 'nop']` are different cells and never compete -- there is no
pressure at all toward the smaller agent, and the reported composition is
longer than the behaviour needs.

**FINDING: the printed log is not the trajectory.** A line is printed only
when `rep > best_report`, so **2** of **200** generations appear. Worse,
`best_true` is assigned inside that same branch, so the "true" column is the
true score of whichever agent last set a *reported* record -- a number that
can only be read as the trajectory when the two happen to coincide.

Structure: `transcript()` captures the shipped run's stdout; `ceiling()`
brute-forces what any agent could score.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import itertools
import re

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "04-darwin-godel-machine"

GENERATIONS, SHIPPED_SEED, MAX_OPS = 200, 7, 4


def transcript(ref, hack_allowed, seed=SHIPPED_SEED, generations=GENERATIONS):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        ref.run_dgm(generations, hack_allowed, seed=seed)
    return buffer.getvalue()


def improvements(text):
    """(generation, reported, true) for every line the run chose to print."""
    rows = []
    for line in text.splitlines():
        if line.strip().startswith("gen "):
            found = re.findall(r"[-\d.]+", line)
            rows.append((int(found[0]), float(found[1]), float(found[2])))
    return rows


def summary(text, label):
    return float(re.search(rf"{label}\s+: ([-+\d.]+)", text).group(1))


def ceiling(ref):
    """The best true score any operator sequence up to MAX_OPS can reach."""
    names = [name for name, _fn in ref.TOOLS]
    best, total, winner = 0.0, 0, ()
    for length in range(MAX_OPS + 1):
        for sequence in itertools.product(names, repeat=length):
            total += 1
            score = ref.true_score(ref.Agent(list(sequence)))
            if score > best:
                best, winner = score, sequence
    return best, total, winner


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    text = transcript(ref, False)
    rows = improvements(text)
    best, searched, winner = ceiling(ref)
    ops = re.search(r"final ops\s+: \[(.*)\]", text).group(1)
    loop = inspect.getsource(ref.run_dgm)
    return {
        "improvements": rows,
        "generations": GENERATIONS,
        "silent": GENERATIONS - (len(rows) - 1),
        "final_ops": [name.strip().strip("'") for name in ops.split(",")],
        "reported": summary(text, "final reported score"),
        "true": summary(text, "final true score"),
        "gap": summary(text, "reported - true"),
        "ceiling": best,
        "searched": searched,
        "winner": list(winner),
        "cases": len(ref.CASES),
        "tools": len(ref.TOOLS),
        "titlecase_tools": [name for name, _fn in ref.TOOLS if "title" in name],
        "key_uses_length": "(len(child.ops)" in loop,
        "print_guard": loop.count("if rep > best_report"),
        "true_assigned_in_guard": loop.split("if rep > best_report")[1].count("best_true"),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: two improvements in 200 generations, ending on collapse+nop at 0.25",
            all([result["improvements"] == [(0, 0.0, 0.0), (4, 0.25, 0.25)],
                 result["final_ops"] == ["collapse", "nop"],
                 result["reported"] == 0.25, result["true"] == 0.25,
                 result["gap"] == 0.0]),
            f"the run prints {result['improvements']} and then nothing for "
            f"{result['silent']} generations, finishing on {result['final_ops']} with "
            f"reported {result['reported']} equal to true {result['true']}",
        ),
        practice.Check(
            "FINDING: 0.25 is the ceiling, not the convergence point",
            all([result["ceiling"] == 0.25, result["searched"] == 1555,
                 result["winner"] == ["collapse"], result["titlecase_tools"] == [],
                 result["cases"] == 8, result["tools"] == 6]),
            f"over {result['searched']} sequences the best reachable true score is "
            f"{result['ceiling']}, by {result['winner']} alone -- six of the "
            f"{result['cases']} cases want title case and "
            f"{len(result['titlecase_tools'])} of the {result['tools']} tools provide it",
        ),
        practice.Check(
            "FINDING: length is a free axis",
            all([result["key_uses_length"], "nop" in result["final_ops"],
                 result["winner"] == ["collapse"]]),
            f"the archive key starts with len(ops), so {result['winner']} and "
            f"{result['final_ops']} are different cells and never compete -- the "
            "winner carries a no-op with nothing to remove it",
        ),
        practice.Check(
            "FINDING: the printed log is not the trajectory",
            all([result["print_guard"] == 1, result["true_assigned_in_guard"] >= 1,
                 len(result["improvements"]) == 2]),
            f"a line prints only when the reported score sets a record, so "
            f"{len(result['improvements'])} of {result['generations']} generations "
            "appear, and best_true is assigned inside that same branch",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
