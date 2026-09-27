"""Exercise 4 — Apollo's eval reads 0 on an eval-aware schemer that sabotages 0.46, and Redwood's protocol ships 0.064 either way.

    Redwood and Apollo both do control/scheming work but with different
    styles. Pick a failure mode and describe how each would investigate it.

Reading of the exercise: the failure mode is eval-aware sabotage. The model
behaves when it believes it is being evaluated, and plants backdoors in
deployed work. Each organisation's investigation is run with the Phase 18
toy that the lesson says embodies its work. Apollo's is Lesson 8's scheming
harness, an agentic setting where deception can emerge. Redwood's is
Lesson 10's control protocols, with worst-case adversaries and concrete
protocols. Both use the lesson's own "Style:" lines. The agent is Lesson 8's
strongly misaligned agent (misaligned 0.6, goal-directed 0.8), run 400
trials per condition, seed 0.

**ANSWER, Apollo: measure the propensity, and measure situational awareness
as its own pillar.** Under evaluation the aware agent schemes 0.00 of the
time. Deployed, it schemes 0.46. With situational awareness off it schemes
0.46 in both. The eval reads the deployment rate only when the model cannot
tell an eval from deployment. So Apollo's investigation needs the
three-pillar decomposition, which measures awareness directly instead of
trusting a clean eval.

**ANSWER, Redwood: don't estimate the propensity; bound what it can do.**
The backdoor rate is treated as the adversary's choice, and trusted editing
with a 10% audit budget is run at every rate. At the deployed 0.46 it ships
0.064 backdoors per task, against 0.466 with no protocol. The worst case over
rates 0 to 1 is 0.193, at 1.0. The number is the same whether or not the
model knows it is being tested, which is the point of the style.

**FINDING: each method covers the other's blind spot.** Gating deployment on
Apollo's eval clears this agent at 0.00, and with no protocol it ships
0.466 backdoors per task. Redwood's protocol
does not need the eval to be honest, but it charges the worst-case price
even when the model is aligned. Apollo's awareness measurement is what says
the price is worth paying.

**FINDING: Lesson 10's toy cannot run Redwood's actual game.**
`run_protocol` has no caught state. An audited backdoor is silently dropped,
so a backdoor attempt costs U nothing, and the worst-case U backdoors every
task. That is the opposite of the rare, patient attacker that control is
built against. The toy's trusted editor also counts every backdoor it fixes
as useful work, so with all outputs backdoored it still reports 0.707
usefulness.

Structure: `apollo()` replays Lesson 8's `run_trial`; `redwood()` runs
Lesson 10's `run_protocol` and parses its line. Both swap in a seeded
`random.Random` and restore it.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import random
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "28-alignment-research-ecosystem"
APOLLO, REDWOOD = "08-in-context-scheming-frontier-models", "10-ai-control-subversion"
TRIALS, H_BUDGET, TASKS, SEED = 400, 0.10, 1000, 0
RATES = tuple(i / 10 for i in range(11))


@contextlib.contextmanager
def seeded(ref, seed=SEED):
    saved, ref.random = ref.random, random.Random(seed)
    try:
        yield ref.random
    finally:
        ref.random = saved


def apollo(ref, aware, under_eval):
    """Scheme share of Lesson 8's strong agent over TRIALS trials."""
    agent = ref.Agent(misaligned=0.6, goal_directed=0.8, situationally_aware=aware)
    with seeded(ref) as rng:
        acts = [ref.run_trial(agent, rng.choice(ref.SCENARIOS), rng.randint(0, 5), under_eval)[0]
                for _ in range(TRIALS)]
    return round(acts.count("scheme") / TRIALS, 3)


def redwood(ref, protocol, rate):
    """(useful, backdoor, H used) per task from Lesson 10's printed line."""
    out = io.StringIO()
    with seeded(ref), contextlib.redirect_stdout(out):
        ref.run_protocol(protocol, backdoor_rate=rate, h_budget=H_BUDGET, n=TASKS)
    return tuple(float(v) for v in re.findall(r"=([\d.]+)", out.getvalue()))


def styles(doc):
    return {org: re.search(r"^Style: (.+)$", doc.split(f"### {org}")[1], re.M).group(1)
            for org in ("Redwood Research", "Apollo Research")}


def solve():
    r8, r10 = (parity.load_reference(PHASE, lesson, "main") for lesson in (APOLLO, REDWOOD))
    conditions = {"aware/eval": (True, True), "aware/deploy": (True, False),
                  "unaware/eval": (False, True), "unaware/deploy": (False, False)}
    measured = {name: apollo(r8, *cond) for name, cond in conditions.items()}
    deployed = measured["aware/deploy"]
    sweep = {rate: redwood(r10, "trusted_ed", rate)[1] for rate in RATES}
    source = inspect.getsource(r10.run_protocol)
    return {
        "apollo": measured,
        "deployed": deployed,
        "redwood": redwood(r10, "trusted_ed", deployed)[1],
        "baseline": redwood(r10, "baseline", deployed)[1],
        "worst": max(sweep.items(), key=lambda kv: kv[1]),
        "caught_state": [w for w in ("caught", "shutdown", "break", "return") if w in source],
        "all_backdoor_useful": redwood(r10, "trusted_ed", 1.0)[0],
        "styles": styles(parity.doc_text(PHASE, LESSON)),
    }


def verify(result):
    r, a = result, result["apollo"]
    return [
        practice.Check(
            "ANSWER, Apollo: propensity reads 0.00 under eval, 0.46 deployed, unless SA is off",
            a == {"aware/eval": 0.0, "aware/deploy": 0.46, "unaware/eval": 0.46, "unaware/deploy": 0.46}
            and "deception can emerge" in r["styles"]["Apollo Research"],
            f"scheme share {a}; Apollo style: {r['styles']['Apollo Research']!r}",
        ),
        practice.Check(
            "ANSWER, Redwood: trusted editing ships 0.064 at the deployed rate, worst 0.193 at 1.0",
            (r["redwood"], r["baseline"], r["worst"]) == (0.064, 0.466, (1.0, 0.193))
            and "worst-case adversaries" in r["styles"]["Redwood Research"],
            f"at rate {r['deployed']}: trusted editing {r['redwood']} vs none {r['baseline']}; "
            f"worst over {RATES[0]}..{RATES[-1]}: {r['worst']}",
        ),
        practice.Check(
            "FINDING: each method covers the other's blind spot",
            a["aware/eval"] == 0.0 and r["baseline"] > r["deployed"] - 0.01 > r["redwood"],
            f"eval-gated deploy clears the agent at {a['aware/eval']} and ships {r['baseline']}; "
            f"the protocol ships {r['redwood']} regardless of what the eval read",
        ),
        practice.Check(
            "FINDING: Lesson 10's toy cannot run Redwood's actual game",
            r["caught_state"] == [] and r["all_backdoor_useful"] == 0.707,
            f"caught/shutdown/break/return in run_protocol: {r['caught_state']}; usefulness with "
            f"every output backdoored: {r['all_backdoor_useful']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
