"""Exercise 2 — a false pass is absorbing and looks exactly like a real one.

    Replace the external verifier with a noisy one (random 30% false
    positives). What does the loop do? This is the 2026 reality of most
    guardrail stacks.

Reading of the exercise: a false positive for a guardrail is "says ok when it
is not", and `run_loop` breaks on the first `ok`. So the loop does not
degrade gracefully -- it exits, and nothing re-checks. The measurement is
therefore over whole runs: how often the loop stops on an output that still
contains a known-wrong fact, and whether anything in the returned history
says so.

**ANSWER: at p=0.3, **52.8%** of **2000** seeded runs stop early on a
factually wrong output marked verified.** The arithmetic is one Bernoulli per
failing check: the trajectory has **2** checks that ought to fail, so the
closed form is `1 - (1-p)^2 = 0.51`, and the measurement lands **0.018**
above it.

**FINDING: the false pass is absorbing and unlabelled.** Every bad exit
carries `verified=True` and the critique `verifier: ok` -- **1** distinct
string across all of them, byte-identical to a genuine pass. The returned
`Attempt` has **4** fields and none of them records that the verdict was
sampled.

**FINDING: the exposure is the number of failing checks, not the noise
rate.** Measured at p = 0.1, 0.3 and 0.5, the bad-exit rates land within
**0.02** of `1 - (1-p)^2`. Noise on a check that was going to pass anyway is
free; noise is only dangerous where the guardrail was the thing catching
something.

**FINDING: the verifier already has a false-negative before any noise.**
`verify_external` iterates `KNOWN_WRONG_FACTS` and its loop body never uses
`fact` -- the two checks inside are hard-coded. **1** of the **3** listed
facts, `the sun orbits the earth`, is never tested, and an output asserting
it passes clean at p=0.

Structure: `noisy()` rebinds `ref.verify_external` under `try/finally`;
`run()` is the lesson's own `run_loop`.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "05-self-refine-and-critic"
TOPIC, RUNS, RATES = "world facts", 2000, (0.1, 0.3, 0.5)


def wrong_facts(ref, text):
    lowered = text.lower()
    return [fact for fact in ref.KNOWN_WRONG_FACTS
            if all(word in lowered for word in fact.split() if len(word) > 4)]


def run(ref, rate, seed, max_iters=4):
    """One seeded run with the verifier's verdict sampled."""
    shipped, rng = ref.verify_external, random.Random(seed)
    ref.verify_external = lambda text: (("verifier: ok", True)
                                        if rng.random() < rate else shipped(text))
    try:
        return ref.run_loop(TOPIC, use_critic=True, max_iters=max_iters)
    finally:
        ref.verify_external = shipped


def tally(ref, rate):
    bad, marks, critiques = 0, set(), set()
    for seed in range(RUNS):
        last = run(ref, rate, seed)[-1]
        if last.verified and wrong_facts(ref, last.output):
            bad += 1
            marks.add(last.verified)
            critiques.add(last.critique)
    return {"rate": round(bad / RUNS, 3), "marks": sorted(marks),
            "critiques": sorted(critiques)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    at_thirty = tally(ref, 0.3)
    sweep = {rate: tally(ref, rate)["rate"] for rate in RATES}
    detectable = [fact for fact in ref.KNOWN_WRONG_FACTS
                  if not ref.verify_external(f"- {fact}\n- filler two\n- filler three")[1]]
    return {
        "bad_rate": at_thirty["rate"], "runs": RUNS,
        "marks": at_thirty["marks"], "critiques": at_thirty["critiques"],
        "attempt_fields": list(ref.Attempt.__dataclass_fields__),
        "sweep": sweep,
        "model": {rate: round(1 - (1 - rate) ** 2, 3) for rate in RATES},
        "gap": max(abs(sweep[rate] - (1 - (1 - rate) ** 2)) for rate in RATES),
        "listed": len(ref.KNOWN_WRONG_FACTS), "detectable": len(detectable),
        "missed": [fact for fact in ref.KNOWN_WRONG_FACTS if fact not in detectable],
        "clean_pass": ref.verify_external(
            "- The sun orbits the earth\n- filler two\n- filler three"),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 52.8% of runs stop early on a wrong output marked verified",
            all([result["bad_rate"] == 0.528, result["runs"] == 2000,
                 result["model"][0.3] == 0.51, result["gap"] < 0.02]),
            f"at p=0.3, {result['bad_rate']:.1%} of {result['runs']} runs exit on an "
            f"output that still contains a known-wrong fact. The trajectory has 2 checks "
            f"that ought to fail, so the closed form is 1-(1-p)^2 = "
            f"{result['model'][0.3]} -- the loop breaks on the first ok and never "
            "re-checks",
        ),
        practice.Check(
            "FINDING: the false pass is absorbing and unlabelled",
            all([result["marks"] == [True], result["critiques"] == ["verifier: ok"],
                 result["attempt_fields"] == ["iteration", "output", "critique",
                                              "verified"]]),
            f"every bad exit carries verified={result['marks'][0]} and the critique "
            f"{result['critiques'][0]!r} -- {len(result['critiques'])} distinct string, "
            f"byte-identical to a genuine pass. Attempt has "
            f"{len(result['attempt_fields'])} fields and none records that the verdict "
            "was sampled",
        ),
        practice.Check(
            "FINDING: the exposure is the number of failing checks, not the rate",
            all([result["gap"] < 0.02, result["sweep"][0.1] < result["sweep"][0.5],
                 result["model"][0.1] == 0.19, result["model"][0.5] == 0.75]),
            f"measured rates {result['sweep']} against the closed form "
            f"{result['model']} -- the largest gap is {result['gap']:.3f}. A false pass "
            "on a check that was going to pass anyway is free; the exposure is exactly "
            "the number of checks the guardrail was the only thing catching",
        ),
        practice.Check(
            "FINDING: the verifier has a false negative before any noise is added",
            all([result["listed"] == 3, result["detectable"] == 2,
                 result["missed"] == ["the sun orbits the earth"],
                 result["clean_pass"] == ("verifier: ok", True)]),
            f"verify_external loops over KNOWN_WRONG_FACTS and never uses fact -- the "
            f"checks inside are hard-coded -- so {result['detectable']} of "
            f"{result['listed']} listed facts are actually tested. An output asserting "
            f"{result['missed'][0]!r} returns {result['clean_pass']} at p=0",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
