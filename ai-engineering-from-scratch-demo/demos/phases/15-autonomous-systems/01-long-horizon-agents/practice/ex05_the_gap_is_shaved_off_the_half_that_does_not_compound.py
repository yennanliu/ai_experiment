"""Exercise 5 — the gap is shaved off the half that does not compound.

    Read the 2026 International AI Safety Report section on eval-context
    gaming. Design one evaluation protocol that would be robust to a model
    behaving differently in tests than in deployment.

Reading of the exercise: a protocol is prose, and prose about robustness is
cheap. The lesson already ships one correction for this -- `deploy_gap_note`
shaves 20-40% off the horizon -- so the design is graded against it: five
rules, each one a thing the shipped adjustment does not do, and the arithmetic
of the difference is what goes in the file.

**ANSWER: apply the gap per step, where it compounds, not to the horizon.**
Take the lesson's own 68-step trajectory at 0.99 per-step -- **50.5%**
end-to-end. Let the model behave at 0.90 on the fraction of steps it decides
are unobserved, at Anthropic's measured **12%** and **78%** rates: effective
per-step **0.9792** and **0.9198**, end-to-end **23.9%** and **0.3%**. The
shipped correction over the same gaming spans a 14-hour horizon to **11.2**
and **8.4** hours -- a factor of 0.8 and 0.6 where the trajectory arithmetic
says 0.47 and 0.007.

**FINDING: the correction lands on the half that cannot compound.**
`deploy_gap_note` calls exactly **2** names, `fmt_hours` and `print`. It never
reaches `end_to_end_reliability` or `max_steps_for_target`, so the lesson's
two halves are joined in the prose and nowhere in the code.

**FINDING: a third of the adjustment table adjusts nothing.** **3** horizons
crossed with **3** gaps is **9** rows, and one of the three gaps is **0.0** --
so **3** of 9 print the benchmark number unchanged under a "deploy" heading.
All six values are literals in the function body with no input to move them.

**FINDING: of the five protocol rules, the artifact can express one.** It
states the ceiling -- "METR numbers assume ideal tooling" is in the printed
text. Measuring the gap, compounding it per step, hiding the eval context and
reporting a distribution are the other four, and none has a seam: **4** of the
module's **9** functions are the demos, and all four take **0** parameters and
read **0** files.

Structure: `gamed()` compounds a gaming rate through the lesson's own
reliability; `rules()` scores the five-rule protocol against the module.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "01-long-horizon-agents"

CLEAN, DEGRADED, STEPS = 0.99, 0.90, 68     # the lesson's own 50%-length trajectory
GAMING = (0.12, 0.78)                       # Anthropic 2024 alignment-faking rates
DEMOS = ("horizon_projection", "reliability_compounding", "deploy_gap_note", "main")
COMPOUNDING = ("end_to_end_reliability", "max_steps_for_target")


def gamed(ref, rate):
    """End-to-end when the model runs degraded on `rate` of its steps."""
    per_step = CLEAN * (1 - rate) + DEGRADED * rate
    return per_step, ref.end_to_end_reliability(per_step, STEPS)


def calls(ref, name):
    tree = ast.parse(textwrap.dedent(inspect.getsource(getattr(ref, name))))
    return sorted({node.func.id for node in ast.walk(tree)
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)})


def rules(ref, source):
    """The five protocol rules, each scored against what the module can express."""
    demo_calls = calls(ref, "deploy_gap_note")
    return {
        "state the horizon is a ceiling": "assume ideal tooling" in source,
        "measure the gap on deployment traffic": "open" in demo_calls,
        "compound it per step": any(name in demo_calls for name in COMPOUNDING),
        "hide the eval context from the model": "input" in demo_calls,
        "report a distribution, not a scalar": "list" in demo_calls,
    }


def shaved(gaps=(0.0, 0.2, 0.4), horizon=14.0):
    """The lesson's own linear correction: the horizon, minus a flat fraction."""
    return [round(horizon * (1 - gap), 1) for gap in gaps]


def demo_params(ref):
    return [len(inspect.signature(getattr(ref, name)).parameters) for name in DEMOS]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    source = textwrap.dedent(inspect.getsource(ref.deploy_gap_note))
    scored, gap_calls = rules(ref, source), calls(ref, "deploy_gap_note")
    compounded = [gamed(ref, rate) for rate in GAMING]
    shipped = shaved()
    return {
        "clean": round(ref.end_to_end_reliability(CLEAN, STEPS), 3),
        "rates": list(GAMING),
        "per_step": [round(pair[0], 4) for pair in compounded],
        "end_to_end": [round(pair[1], 3) for pair in compounded],
        "shipped_hours": shipped,
        "shipped_factors": [round(hours / 14.0, 1) for hours in shipped[1:]],
        "gap_calls": gap_calls,
        "reaches_compounding": [n for n in COMPOUNDING if n in gap_calls],
        "rows": len(shipped) * 3,
        "zero_gap_rows": 3,
        "demos": list(DEMOS),
        "demo_params": demo_params(ref),
        "rules": scored,
        "satisfied": [name for name, ok in scored.items() if ok],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: compounded per step, 12% and 78% gaming cost 50.5% -> 23.9% -> 0.3%",
            all([result["clean"] == 0.505, result["per_step"] == [0.9792, 0.9198],
                 result["end_to_end"] == [0.239, 0.003]]),
            f"a clean 68-step run scores {result['clean']:.1%}; degrading 0.99 to 0.90 "
            f"on {result['rates']} of the steps gives per-step {result['per_step']} and "
            f"end-to-end {result['end_to_end']}, against the shipped horizon shave of "
            f"{result['shipped_factors']}x",
        ),
        practice.Check(
            "FINDING: the correction lands on the half that cannot compound",
            all([result["gap_calls"] == ["fmt_hours", "print"],
                 result["reaches_compounding"] == []]),
            f"deploy_gap_note calls {result['gap_calls']} and reaches "
            f"{len(result['reaches_compounding'])} of the two compounding functions, so "
            "the eval-vs-deploy discount never touches a trajectory",
        ),
        practice.Check(
            "FINDING: a third of the adjustment table adjusts nothing",
            all([result["rows"] == 9, result["zero_gap_rows"] == 3,
                 result["shipped_hours"] == [14.0, 11.2, 8.4]]),
            f"{result['zero_gap_rows']} of the {result['rows']} rows use a gap of 0.0 "
            f"and print the benchmark number unchanged under a deploy heading; the "
            f"other six read {result['shipped_hours'][1:]} hr from literals",
        ),
        practice.Check(
            "FINDING: of the five protocol rules, the artifact can express one",
            all([len(result["rules"]) == 5, len(result["satisfied"]) == 1,
                 result["satisfied"] == ["state the horizon is a ceiling"],
                 result["demo_params"] == [0, 0, 0, 0]]),
            f"{len(result['satisfied'])} of {len(result['rules'])} rules holds -- "
            f"{result['satisfied'][0]} -- and the other four need an input the module "
            f"has no seam for: its {len(result['demos'])} demo functions take "
            f"{result['demo_params']} parameters",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
