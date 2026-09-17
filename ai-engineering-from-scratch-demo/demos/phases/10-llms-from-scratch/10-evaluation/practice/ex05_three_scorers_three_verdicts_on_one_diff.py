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

Structure: `diff` is the tool and returns one record per case -- id, prompt,
before, after, verdict -- keyed by position so a repeated prompt stays two
cases; `counts` derives the summary from those records rather than counting
separately. `terse` is the second model version, chosen so the three shipped
scorers disagree about it.
"""

from __future__ import annotations

import collections

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
    """One row per case, in suite order, so a repeated prompt is still two cases."""
    return [(row["input"], row["scores"]) for row in suite(ref).run(model_fn)]


def verdict(was, now):
    return "improved" if now > was else ("regressed" if now < was else "unchanged")


def diff(before, after, metric):
    """The tool: one record per case, keyed by position so duplicate prompts stay distinct."""
    return [{"case": index, "prompt": prompt, "before": was[metric], "after": now[metric],
             "verdict": verdict(was[metric], now[metric])}
            for index, ((prompt, was), (_, now)) in enumerate(zip(before, after))]


def counts(records):
    """The summary, derived from the records rather than counted separately."""
    tally = collections.Counter(record["verdict"] for record in records)
    return tally["improved"], tally["regressed"], tally["unchanged"]


def blind_spots(contested):
    """Cases exact_match calls unchanged and token_f1 calls a regression."""
    by_case = {record["case"]: record for record in contested["exact"]}
    return [record for record in contested["f1"]
            if record["verdict"] == "regressed"
            and by_case[record["case"]]["verdict"] == "unchanged"]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    bad, good = scores(ref, ref.demo_model_bad), scores(ref, ref.demo_model_good)
    short = scores(ref, terse)
    contested = {metric: diff(bad, short, metric) for metric in METRICS}
    shifted = blind_spots(contested)
    first = shifted[0]["prompt"] if shifted else ""
    return {
        "cases": len(bad),
        "shipped": {metric: counts(diff(bad, good, metric)) for metric in METRICS},
        "records": contested,
        "contested": {metric: counts(records) for metric, records in contested.items()},
        "blind": len(shifted),
        "blind_cases": [record["case"] for record in shifted],
        "example": (first, ref.demo_model_bad(first) if first else "",
                    terse(first) if first else ""),
    }


def verify(result):
    shipped, contested = result["shipped"], result["contested"]
    cases, records = result["cases"], result["records"]
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
            f"cases {result['blind_cases']} -- {result['blind']} of the {cases} -- are unchanged "
            f"under exact_match and regressions under token_f1. Case "
            f"{records['f1'][result['blind_cases'][0]]['case']}, {prompt!r}, went from {was!r} to "
            f"{now!r}, both wrong, "
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
