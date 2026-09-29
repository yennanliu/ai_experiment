"""Exercise 1 — a sixth task.

    Add a sixth task with a custom metric you write from scratch (BLEU-like overlap, BLEURT-like reference scoring, anything with a clear contract).

Reading of the exercise: the sixth task is `paraphrase`, five `restate:`
prompts with two reference restatements each, scored by `bleu2`: BLEU up to
bigrams, as defined by Papineni et al. 2002 (https://aclanthology.org/P02-1040.pdf,
read 2026-09-29): clipped n-gram counts over several references, a geometric
mean, and the brevity penalty exp(1 - r/c) against the best-match reference
length. The paper computes it per corpus. Here it is computed per example,
so the bigram precision gets add-one smoothing. The contract is written as checks: identical text
scores 1.0, empty scores 0.0, every score is in [0, 1], repeating one word
is clipped (0.25), and a one-word answer is cut by the brevity penalty
(0.007). The task is
written as a JSONL file next to the five seeded ones, the metric is added to
the lesson's `METRIC_FNS`, and `run_leaderboard` runs all six unchanged.

**ANSWER: `bleu2` below meets its contract, and the lesson's runner picks up
the sixth task with no change.** A prefix-stripping adapter scores 1.0 on
`paraphrase`, so the six-task toy overall stays 1.0.

**FINDING: the lesson's own `ToyAdapter` echoes any prompt it has no rule
for.** On `paraphrase` it answers with the prompt, `restate:` included. That
gives 0.843 on the task (listed as 4/5 correct with 0 exact), and its overall
drops from 1.0 to 0.974 even though nothing in the harness changed. The same echo trick scores 0.328 on the five
shipped tasks, and 1.0 on `generation`, because every generation target is a
word of its prompt.

**FINDING: the runner neither enforces the [0, 1] contract nor reports
partial credit honestly.** A metric that returns 1.5 gives a task score of
1.5 and an overall of 1.083. `correct` is `round(sum of scores)`: the echo
adapter matches 0 of 5 summaries exactly, yet is listed as 3/5 correct on
`summary`.
"""

from __future__ import annotations

import math
import pathlib
import re
import tempfile
import types
from collections import Counter

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "49-lm-eval-harness"
PARAPHRASE = [
    ("the cat sat on the mat", ["the cat sat on the mat", "a cat was sitting on the mat"]),
    ("the model scores every task", ["the model scores every task", "every task is scored by the model"]),
    ("rain fell over the hills", ["rain fell over the hills", "it rained over the hills"]),
    ("the harness writes a leaderboard", ["the harness writes a leaderboard", "a leaderboard is written by the harness"]),
    ("small tasks ship in the repo", ["small tasks ship in the repo", "the repo ships small tasks"]),
]


def _ngrams(toks, n):
    return Counter(tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))


def _precision(pred, refs, n, smooth):
    best = Counter()
    for r in refs:
        best |= _ngrams(r, n)
    got = _ngrams(pred, n)
    hits = sum(min(c, best[g]) for g, c in got.items())
    return (hits + smooth) / (sum(got.values()) + smooth)


def bleu2(prediction, targets, extras=None):
    """BLEU-2: geometric mean of clipped 1/2-gram precision, times the brevity penalty."""
    pred = re.findall(r"[a-z0-9]+", prediction.lower())
    refs = [re.findall(r"[a-z0-9]+", t.lower()) for t in targets]
    if not pred or not any(refs):
        return 0.0
    p1, p2 = _precision(pred, refs, 1, 0), _precision(pred, refs, 2, 1)
    if p1 == 0:
        return 0.0
    closest = min((len(r) for r in refs), key=lambda length: (abs(length - len(pred)), length))
    bp = 1.0 if len(pred) >= closest else math.exp(1 - closest / len(pred))
    return bp * math.sqrt(p1 * p2)


def adapter(name, fn):
    return types.SimpleNamespace(name=name, generate=lambda prompts: [fn(p) for p in prompts])


def board(ref, tasks, adp):
    b = ref.run_leaderboard(tasks, adp, batch_size=4)
    return round(b.overall_score, 3), {r.task: (round(r.score, 3), r.correct) for r in b.tasks}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    ref.METRIC_FNS["bleu2"] = bleu2
    tmp = pathlib.Path(tempfile.mkdtemp())
    ref.seed_fixture_tasks(tmp)
    five = ref.load_all_tasks(tmp)
    rows = [ref.Example(f"para-{i:02d}", f"restate: {s}", t, "bleu2") for i, (s, t) in enumerate(PARAPHRASE)]
    ref.write_task_jsonl(rows, tmp / "paraphrase.jsonl")
    six = ref.load_all_tasks(tmp)
    toy, echo = ref.ToyAdapter(), adapter("echo", lambda p: p)
    strip = adapter("strip", lambda p: p.split(":", 1)[1].strip() if p.startswith("restate:") else toy._answer(p))
    ref.METRIC_FNS["over"] = lambda p, t, e: 1.5
    over = {**five, "over": [ref.Example("o", "x", ["x"], "over")]}
    scores = [bleu2(toy._answer(r.prompt), r.targets) for r in rows] + [bleu2("a b c", ["x y"])]
    return {
        "contract": [bleu2("the cat sat", ["the cat sat"]), bleu2("", ["x"]), round(bleu2("the the the the", ["the cat"]), 3),
                     round(bleu2("cat", ["the cat sat on the mat"]), 3)],
        "in_range": all(0.0 <= s <= 1.0 for s in scores),
        "six_tasks": sorted(six), "strip": board(ref, six, strip), "toy": board(ref, six, toy),
        "echo5": board(ref, five, echo), "over": board(ref, over, toy)[0],
    }


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: bleu2 keeps its contract and the runner picks up the sixth task unchanged",
            r["contract"] == [1.0, 0.0, 0.25, 0.007] and r["in_range"] and len(r["six_tasks"]) == 6
            and r["strip"][0] == 1.0 and r["strip"][1]["paraphrase"] == (1.0, 5),
            f"identical/empty/repeat/one-word = {r['contract']}; tasks {r['six_tasks']}; "
            f"prefix-stripping adapter overall {r['strip'][0]}",
        ),
        practice.Check(
            "FINDING: ToyAdapter echoes unknown prompts, so a new task moves its overall",
            (r["toy"][0], r["toy"][1]["paraphrase"], r["echo5"][0], r["echo5"][1]["generation"][0])
            == (0.974, (0.843, 4), 0.328, 1.0),
            f"toy on paraphrase (score, correct) {r['toy'][1]['paraphrase']}, six-task overall {r['toy'][0]}; "
            f"pure echo on the five shipped tasks {r['echo5'][0]} (generation {r['echo5'][1]['generation'][0]})",
        ),
        practice.Check(
            "FINDING: the runner does not enforce [0, 1], and 'correct' is rounded partial credit",
            r["over"] == 1.083 and r["echo5"][1]["summary"] == (0.639, 3),
            f"a metric returning 1.5 gives overall {r['over']}; echo scores summary "
            f"{r['echo5'][1]['summary'][0]} and is listed {r['echo5'][1]['summary'][1]}/5 correct",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
