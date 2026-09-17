"""Exercise 1 — every model scores 1.00 on consistency, and none of them is reproducible.

    Add a "consistency" scorer that runs the same input through the model 5 times
    and measures how often the outputs match. Inconsistent answers on
    deterministic inputs reveal fragile prompts or high temperature settings.

Reading of the exercise: the scorer is built exactly as specified -- five calls,
count how often the outputs agree -- and run on all three of the lesson's own
demo models, including `demo_model_random`, which is the one it is meant to
catch. The same measurement is then repeated across fresh Python processes,
because "runs the same input through the model 5 times" leaves open whether the
five runs share an interpreter, and here that is the whole answer.

**ANSWER: 1.00 for all three models, on every prompt.** Five calls to
`demo_model_good`, `demo_model_bad` and `demo_model_random` return identical
strings every time. The first two are dictionary lookups; the third seeds numpy
with `hash(prompt)` before drawing, so it is a deterministic function of the
prompt too. The scorer finds nothing, correctly.

**FINDING: the same three models are not reproducible across processes.**
`np.random.seed(hash(prompt) % 2**31)` uses Python's `hash`, which is **salted
per interpreter** unless `PYTHONHASHSEED` is set. Six fresh processes give
several different answers to "What is 2 + 2?" -- `maybe`, `no`, `yes`,
`unknown`, `Paris`, `error` are all reachable -- and several different
perplexities for one fixed string, drawn afresh on every run.

**MECHANISM: the scorer looks in the one place the inconsistency is not.** Five
calls inside one interpreter share one hash seed, so they cannot disagree. The
exercise says an inconsistency scorer "reveals fragile prompts or high
temperature settings"; the fragility in this lesson is at process boundaries,
and a within-process scorer is blind to it by construction.

**FINDING: `token_log_probs_simulated` carries the same defect, and it feeds
the lesson's perplexity numbers.** Every perplexity the lesson prints is a
function of a salted hash, so rerunning the demo prints different figures, by
of the order of ten percent -- in an evaluation lesson, whose subject is
measuring things twice and comparing.

Structure: `consistency` is the scorer the exercise asks for; `across_processes`
re-runs one measurement in fresh interpreters, which is where the disagreement
lives.
"""

from __future__ import annotations

import os
import subprocess
import sys

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "10-evaluation"
RUNS, PROCESSES = 5, 6
PROMPTS = ("What is the capital of France?", "What is 2 + 2?", "Who wrote Hamlet?",
           "Name the largest planet.", "What year did World War 2 end?")
PROBE = "the quick brown fox jumps"
SNIPPET = (
    "import sys; sys.path.insert(0, %r);"
    "from harness import parity;"
    "m = parity.load_reference(%r, %r, 'main');"
    "print('%%.6f' %% m.perplexity(m.token_log_probs_simulated(%r)),"
    " m.demo_model_random('What is 2 + 2?'))"
)


def consistency(model_fn, prompt, runs=RUNS):
    """The scorer the exercise asks for: how often `runs` calls agree."""
    outputs = [model_fn(prompt) for _ in range(runs)]
    return sum(text == outputs[0] for text in outputs) / runs


def across_processes(root):
    """The same measurement in `PROCESSES` fresh interpreters, with no PYTHONHASHSEED."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}
    code = SNIPPET % (root, PHASE, LESSON, PROBE)
    return [subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, env=env, cwd=root).stdout.split()
            for _ in range(PROCESSES)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    models = {"good": ref.demo_model_good, "bad": ref.demo_model_bad,
              "random": ref.demo_model_random}
    rows = [row for row in across_processes(os.getcwd()) if len(row) == 2]
    return {
        "within": {name: [consistency(fn, prompt) for prompt in PROMPTS]
                   for name, fn in models.items()},
        "answers": sorted({row[1] for row in rows}),
        "perplexities": sorted(float(row[0]) for row in rows),
        "processes": len(rows),
    }


def verify(result):
    within, answers = result["within"], result["answers"]
    perplexities = result["perplexities"]
    spread = (perplexities[-1] / perplexities[0] - 1) if perplexities else 0.0
    return [
        practice.Check(
            f"ANSWER: consistency is 1.00 for all three models on all {len(PROMPTS)} prompts",
            all(score == 1.0 for scores in within.values() for score in scores),
            f"{RUNS} calls to each of {sorted(within)} return identical strings on every prompt. "
            "The first two are dictionary lookups and demo_model_random seeds numpy with "
            "hash(prompt) before drawing, so it is a deterministic function of the prompt too. "
            "The scorer finds nothing, and it is right",
        ),
        practice.Check(
            "FINDING: the same models give different answers in different processes",
            len(answers) > 1,
            f"across {result['processes']} fresh interpreters with PYTHONHASHSEED unset, "
            f"demo_model_random answers 'What is 2 + 2?' with {answers} -- "
            f"{len(answers)} different strings from a model the consistency scorer just called "
            "perfectly consistent. np.random.seed(hash(prompt) % 2**31) uses Python's hash, "
            "which is salted per interpreter",
        ),
        practice.Check(
            "FINDING: the lesson's own perplexity numbers change between runs",
            len(set(perplexities)) > 1,
            f"one fixed string, {PROBE!r}, gives {len(set(perplexities))} distinct perplexities "
            f"across {result['processes']} processes -- "
            + ", ".join(f"{value:.4f}" for value in perplexities)
            + f", a spread of {100 * spread:.0f}% on this draw and a different one on the next. "
            "token_log_probs_simulated seeds on the same salted hash, so every perplexity this "
            "lesson prints is different next time, in a lesson about measuring things twice",
        ),
        practice.Check(
            "MECHANISM: the scorer looks in the one place the inconsistency is not",
            all(score == 1.0 for scores in within.values() for score in scores)
            and len(answers) > 1,
            f"{RUNS} calls inside one interpreter share one hash seed, so they cannot disagree. "
            "The exercise says the scorer reveals fragile prompts or high temperature settings; "
            "the fragility here is at process boundaries, and a within-process scorer is blind "
            "to it by construction",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
