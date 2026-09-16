"""Exercise 5 — the same diff reads 1 improved / 0 regressed, 1 / 4, and 0 / 5.

    Build a "model diff" tool. Given eval results from two model versions,
    highlight which specific test cases improved, which regressed, and which
    stayed the same. This is the eval equivalent of a code diff -- essential for
    understanding whether a change helped or hurt.

Reading of the exercise: the tool is built as described -- per-case comparison
into improved, regressed and unchanged -- and run on the lesson's own
`EvalSuite` with all three of its scorers, because "which cases improved" is
only defined relative to a scorer and the lesson ships three. The two model
versions are the lesson's own `demo_model_bad` and a terse variant, a pair the
three scorers disagree about.

**FINDING: on the lesson's own two models no scorer finds a regression.**
`demo_model_good` against `demo_model_bad` is 7 improved under exact match and
F1, 4 improved and 3 unchanged under the judge -- and zero regressed under all
three. The scorers already disagree about how many cases moved, but the diff has
no contested case on it, so it cannot show what the tool is for.

**ANSWER: swap in a pair the scorers disagree about and one diff becomes
three.** The same change -- from the verbose model to a terse one -- reads:

    exact match   1 improved, 0 regressed, 6 unchanged
    token F1      1 improved, 4 regressed, 2 unchanged
    LLM judge     0 improved, 5 regressed, 2 unchanged

A pure win, a net loss, and a pure loss. Nothing about the models changed
between those three lines.

**MECHANISM: the scorers disagree about what happens when an answer gets
shorter.** `exact_match` is blind to a wrong answer becoming a different wrong
answer, so four cases that went from "Shakespeare" to "no" are **unchanged** to
it. `token_f1` sees the shared words disappear and calls them regressions. The
judge adds a length term, so it penalises the same four again and refuses to
credit the one case that became exactly right.

**FINDING: the exercise's own framing assumes what it asks you to build.**
"Essential for understanding whether a change helped or hurt" presumes a diff
has a direction. It has one per scorer, and choosing the scorer is the decision
the tool was supposed to inform.

Structure: `diff` is the tool; `terse` is the second model version, chosen so
the three shipped scorers disagree about it.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "10-evaluation"
METRICS = ("exact", "f1", "judge")
QA = (
    ("What is the capital of France?", "Paris"),
    ("What is 2 + 2?", "4"),
    ("Who wrote Hamlet?", "William Shakespeare"),
    ("What language is PyTorch written in?", "Python and C++"),
    ("What is the boiling point of water?", "100 degrees Celsius"),
    ("Name the largest planet.", "Jupiter"),
    ("What year did World War 2 end?", "1945"),
)


def terse(prompt):
    """The second version: exactly right once, tersely wrong everywhere else."""
    return {"Name the largest planet.": "Jupiter"}.get(prompt, "no")


def suite(ref):
    return ref.EvalSuite("demo", [ref.EvalCase(p, e) for p, e in QA],
                         {"exact": ref.exact_match, "f1": ref.token_f1,
                          "judge": ref.llm_judge_simulated})


def scores(ref, model_fn):
    return {row["input"]: row["scores"] for row in suite(ref).run(model_fn)}


def diff(before, after, metric):
    """The tool: per-case improved, regressed and unchanged under one scorer."""
    improved = sum(after[k][metric] > before[k][metric] for k in before)
    regressed = sum(after[k][metric] < before[k][metric] for k in before)
    return improved, regressed, len(before) - improved - regressed


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    bad = scores(ref, ref.demo_model_bad)
    good = scores(ref, ref.demo_model_good)
    short = scores(ref, terse)
    shifted = [k for k in bad
               if bad[k]["exact"] == short[k]["exact"] and short[k]["f1"] < bad[k]["f1"]]
    return {
        "cases": len(bad),
        "shipped": {metric: diff(bad, good, metric) for metric in METRICS},
        "contested": {metric: diff(bad, short, metric) for metric in METRICS},
        "blind": len(shifted),
        "example": (shifted[0] if shifted else "",
                    ref.demo_model_bad(shifted[0]) if shifted else "",
                    terse(shifted[0]) if shifted else ""),
    }


def verify(result):
    shipped, contested = result["shipped"], result["contested"]
    cases = result["cases"]
    prompt, was, now = result["example"]
    return [
        practice.Check(
            "FINDING: on the lesson's own two models every scorer reports zero regressions",
            all(regressed == 0 for _, regressed, _ in shipped.values()),
            "demo_model_good against demo_model_bad is "
            + ", ".join(f"{metric} {i} improved / {r} regressed / {u} unchanged"
                        for metric, (i, r, u) in shipped.items())
            + ". The three scorers already disagree about how many cases moved, but none of them "
            "finds a case that got worse -- so the diff has no contested case on it and cannot "
            "show what the tool is for",
        ),
        practice.Check(
            "ANSWER: on a contested pair, one diff becomes three",
            len({contested[m] for m in METRICS}) == len(METRICS),
            "the same change, from the verbose model to a terse one, reads "
            + "; ".join(f"{metric} {i} improved / {r} regressed / {u} unchanged"
                        for metric, (i, r, u) in contested.items())
            + ". A pure win, a net loss and a pure loss -- and nothing about the models changed "
            "between those three lines",
        ),
        practice.Check(
            "MECHANISM: exact_match is blind to a wrong answer becoming a different wrong answer",
            result["blind"] == contested["f1"][1] > 0,
            f"{result['blind']} of the {cases} cases are unchanged under exact_match and "
            f"regressions under token_f1 -- {prompt!r} went from {was!r} to {now!r}, both wrong, "
            "one sharing words with the expected answer and one not. exact_match cannot see the "
            "difference between two failures; token_f1 can, and the judge penalises the length "
            "drop on top",
        ),
        practice.Check(
            "FINDING: the exercise's framing assumes what it asks you to build",
            contested["exact"][1] == 0 < contested["judge"][1],
            "'essential for understanding whether a change helped or hurt' presumes a diff has a "
            f"direction. This one has {contested['exact'][1]} regressions under exact match and "
            f"{contested['judge'][1]} under the judge, on identical inputs. Choosing the scorer "
            "is the decision the tool was supposed to inform",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
