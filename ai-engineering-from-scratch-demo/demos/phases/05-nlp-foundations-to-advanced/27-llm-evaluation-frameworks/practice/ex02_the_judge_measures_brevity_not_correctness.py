"""Exercise 2 — the judge measures brevity, not correctness.

    **Medium.** Hand-label 50 QA answers 0-1 for correctness. Score with G-Eval.
    Measure Spearman rho between judge and human.

Reading of the exercise: `deepeval` is absent, so `GEval` cannot be constructed;
the substitute is the lesson's own `g_eval_correctness`, which `docs/en.md` Step 5
offers as the stdlib stand-in for it. The 50 answers are invented here, built as
10 facts x 5 answer styles so the confound is controlled rather than accidental:
the gold string alone, a correct full sentence, a correct paraphrase, a wrong
value alone, and a wrong value in a full sentence. Human labels are 1, 1, 1, 0, 0
-- 30 correct against 20 incorrect.

Spearman rho between judge and human is **0.0331** at **p = 0.8196**. The lesson's
own calibration bar is "if rho < 0.7, your judge rubric needs work"; this judge is
not distinguishable from a coin. Sweeping `threshold` across the whole unit
interval does not rescue it -- the best value reachable is **0.4082** at
thresholds of 0.7 and above, and at 0.1 the correlation is **negative**, -0.1117.

The style means say what is actually being measured. Bare gold strings score
**1.00**, correct paraphrases **0.00**, and wrong values presented on their own
**0.80** -- eight of the ten one-digit-wrong answers are scored fully correct,
while every true restatement in different words is scored fully wrong. Correct
answers written as full sentences land at **0.30**.

The mechanism is the denominator. `g_eval_correctness` scores the fraction of the
*actual* answer's tokens that appear in the *expected* string, and the expected
string is three or four tokens long. Adding true, relevant words to a correct
answer lowers its score; deleting everything except the near-miss value raises it.
Judge score correlates with shortness at rho **0.5762** -- the inverse of the
length bias `docs/en.md` warns about under "Judge bias", and stronger than its
correlation with being right.

A one-line baseline the exercise never computes beats it outright: `expected in
actual` scores rho **0.6667** against the same labels, twenty times the judge's
correlation, with no rubric, no threshold and no model.

Structure: `FACTS` packs ten stem/gold/wrong/paraphrase rows; `styled` expands one
row into the five labelled answers; `dataset` builds all 50; `rho` wraps
`scipy.stats.spearmanr`; `sweep` re-scores the set at ten thresholds.
"""

from __future__ import annotations

import collections
import importlib.util

from scipy import stats

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "27-llm-evaluation-frameworks"

FACTS = """First iPhone released on|June 29, 2007|June 29, 2006|Apple shipped that phone in 2007
Eiffel Tower stands|330 metres tall|224 metres tall|The Paris landmark rises a third of a km
Everest summit reaches|8849 metres|8611 metres|The peak tops out just under nine km up
Berlin Wall fell in|November 1989|November 1979|Germany reopened its capital late that decade
Apollo 11 landed in|July 1969|July 1968|Armstrong walked on lunar soil that summer
Python 3.0 shipped in|December 2008|December 2005|The third major release arrived that winter
Water boils at|100 degrees celsius|100 degrees fahrenheit|It vapourises at the standard point
Amazon river empties into|the Atlantic Ocean|the Pacific Ocean|Its mouth opens east of Brazil
Curie won the physics Nobel in|1903|1911|She shared the award with Becquerel
Transformers were introduced in|2017|2014|The attention paper appeared that year"""
ROWS = tuple(tuple(line.split("|")) for line in FACTS.splitlines())
BAR = 0.7


def styled(stem, gold, wrong, para):
    """One fact as five labelled answers: gold, verbose, paraphrase, wrong, wrong-verbose."""
    return (("bare-gold", f"{gold}.", 1), ("verbose-right", f"{stem} {gold}.", 1),
            ("paraphrase", f"{para}.", 1), ("bare-wrong", f"{wrong}.", 0),
            ("verbose-wrong", f"{stem} {wrong}.", 0))


def dataset():
    """The 50 hand-labelled cases as (kind, answer, expected, human)."""
    return [(kind, text, row[1], label) for row in ROWS for kind, text, label in styled(*row)]


def rho(xs, ys):
    """Spearman correlation and its p-value, as plain floats."""
    result = stats.spearmanr(xs, ys)
    return float(result.statistic), float(result.pvalue)


def sweep(ref, cases, human):
    """Judge-human rho at ten thresholds across the unit interval."""
    out = {}
    for step in range(1, 11):
        threshold = step / 10
        scored = [ref.g_eval_correctness(a, e, threshold) for _, a, e, _ in cases]
        out[threshold] = round(rho(scored, human)[0], 4)
    return out


def by_style(cases, scores):
    """Mean judge score for each of the five answer styles."""
    buckets = collections.defaultdict(list)
    for (kind, _, _, _), score in zip(cases, scores):
        buckets[kind].append(score)
    return {k: round(sum(v) / len(v), 2) for k, v in buckets.items()}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    cases = dataset()
    human = [h for _, _, _, h in cases]
    judge = [ref.g_eval_correctness(a, e) for _, a, e, _ in cases]
    substring = [float(e.lower() in a.lower()) for _, a, e, _ in cases]
    brevity = [-len(ref.tokenize(a)) for _, a, _, _ in cases]
    return {
        "deepeval": importlib.util.find_spec("deepeval") is not None,
        "n": len(cases), "positives": sum(human),
        "rho": rho(judge, human), "substring_rho": rho(substring, human)[0],
        "brevity_rho": rho(judge, brevity)[0],
        "styles": by_style(cases, judge), "sweep": sweep(ref, cases, human),
        "grid": sorted({round(s, 4) for s in judge}),
        "expected_len": len(ref.tokenize(ROWS[0][1])),
    }


def verify(result):
    stat, pval = result["rho"]
    styles, sweep_values = result["styles"], result["sweep"]
    best = max(sweep_values.values())
    return [
        practice.Check(
            "ANSWER: rho = 0.0331 at p = 0.82 -- the judge is uncorrelated with the labels",
            not result["deepeval"] and abs(stat) < 0.1 and pval > 0.05,
            f"`deepeval` is absent, so `GEval` cannot be built and the substitute is the lesson's "
            f"own `g_eval_correctness`. Over {result['n']} hand-labelled answers "
            f"({result['positives']} correct) Spearman rho is {stat:.4f} at p = {pval:.4f}, "
            f"against the lesson's own bar of rho > {BAR}",
        ),
        practice.Check(
            "MECHANISM: it scores the fraction of the answer's tokens found in the gold string",
            styles["bare-wrong"] > styles["paraphrase"],
            f"the expected string is {result['expected_len']} tokens, so it is the denominator "
            f"that decides. Style means: bare gold {styles['bare-gold']:.2f}, correct sentence "
            f"{styles['verbose-right']:.2f}, correct paraphrase {styles['paraphrase']:.2f}, "
            f"one-digit-wrong value alone {styles['bare-wrong']:.2f}, wrong value in a sentence "
            f"{styles['verbose-wrong']:.2f}. True words lower the score; deleting them raises it",
        ),
        practice.Check(
            "FINDING: it tracks brevity better than it tracks correctness",
            result["brevity_rho"] > abs(stat),
            f"judge score against negative answer length gives rho {result['brevity_rho']:.4f}, "
            f"where judge against human gives {stat:.4f}. That is the inverse of the length bias "
            "`docs/en.md` warns about, and the thing it measures best",
        ),
        practice.Check(
            "FINDING: a one-line baseline the exercise never computes beats it",
            result["substring_rho"] > stat and result["substring_rho"] > best,
            f"`expected.lower() in actual.lower()` scores rho {result['substring_rho']:.4f} "
            f"against the same labels -- no rubric, no threshold, no model -- while the judge "
            f"scores {stat:.4f} and cannot reach {result['substring_rho']:.4f} at any threshold",
        ),
        practice.Check(
            "CONTROL: no threshold makes it calibrated",
            best < BAR and min(sweep_values.values()) < 0,
            f"sweeping `threshold` over the unit interval gives {sweep_values}: best "
            f"{best:.4f} at 0.7 and above, still short of {BAR}, and negative at 0.1. The shipped "
            f"default of 0.5 is not the problem, and the judge's range is the two-point grid "
            f"{result['grid']} because every answer here is one sentence",
        ),
        practice.Check(
            "CONTROL: the shipped judge never sees the question",
            result["n"] == 50 and len(ROWS) == 10,
            "`docs/en.md` Step 3 builds `GEval(name=..., criteria=..., evaluation_steps=...)` "
            "with `evaluation_params` including INPUT, but the shipped signature is "
            "`g_eval_correctness(actual, expected, threshold=0.5)` -- no question, no criteria, "
            f"no steps. The 50 cases are {len(ROWS)} facts x 5 styles, which is why the style "
            "means above are a controlled comparison rather than a sample of one",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
