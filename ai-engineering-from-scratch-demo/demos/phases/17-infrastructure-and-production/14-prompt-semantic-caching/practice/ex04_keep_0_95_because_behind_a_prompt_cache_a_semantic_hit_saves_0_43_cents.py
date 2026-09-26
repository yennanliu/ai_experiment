"""Exercise 4 — keep 0.95, because behind a prompt cache a semantic hit saves 0.43 cents.

    Semantic cache at 0.95 threshold hits 20%. At 0.85 it hits 50% but you
    see incorrect cached responses. Pick the right threshold and justify.

Reading of the exercise: the choice is economic -- the 30 extra points of hits
0.85 buys are worth what one avoided LLM call saves, and cost what a wrong
answer costs times the fraction of those extra hits that are wrong. The saving
per hit is measured on the lesson's workload with the reference `simulate`
(L1 at 20% and 50%, L2 5-min), with and without the prompt cache under it.

**ANSWER: keep 0.95.** Behind the L2 cache a semantic hit saves $0.0043: the
200 output tokens ($0.0030) and an input already read at 0.1x. Going from 20%
to 50% hits cuts the workload's bill $4.75 -> $3.07, $1.67 over 389 extra
hits. 0.85 pays only if the fraction e of those extra hits that is wrong
satisfies e * E < $0.0043, where E is what one wrong answer costs you: at
E = $1 (a re-contact) that is e < 0.43%, one wrong answer in 232; at E = $10,
one in 2325. "You see incorrect cached responses" means e is already
visible -- far above that. Lower the threshold only in steps, with a labelled
eval of the band between thresholds, and stop where its error rate crosses
$0.0043 / E.

**FINDING: the prompt cache makes a semantic hit worth 3.8x less.** Without L2
a hit also skips 4529 fresh input tokens and saves $0.0163, so the tolerable
error rate is 3.8x higher. Stacking the two layers shrinks exactly the margin
a lower threshold spends.

**FINDING: `l1_threshold` does nothing in the reference.** `simulate` never
reads it; an L1 hit is a coin flip at `l1_hit_prob` and is always correct and
free. Thresholds 0.85 and 0.95 at the same hit probability give identical
bills, so the simulator cannot show the tradeoff this exercise is about.

Structure: `saving()` measures the bill cut per L1 hit; `tolerable()` is the
break-even error rate $saving / E.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "14-prompt-semantic-caching"
HIT = {0.95: 0.20, 0.85: 0.50}
WRONG_COST = (1.0, 10.0)


def run(ref, reqs, p, l2=True, threshold=0.95):
    cfg = ref.Config(l1_enabled=p > 0, l2_enabled=l2, parallel_penalty=False,
                     l1_threshold=threshold, l1_hit_prob=p, ttl="5min")
    return ref.simulate(reqs, cfg)


def saving(ref, reqs, l2):
    base = run(ref, reqs, 0.0, l2)["cost"]
    hi = run(ref, reqs, HIT[0.85], l2)
    return (base - hi["cost"]) / hi["l1_hits"]


def tolerable(per_hit, wrong_cost):
    return per_hit / wrong_cost


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    reqs = ref.make_workload()
    at = {t: run(ref, reqs, p) for t, p in HIT.items()}
    per_hit = saving(ref, reqs, l2=True)
    return {
        "bill": {t: round(r["cost"], 2) for t, r in at.items()},
        "extra_hits": at[0.85]["l1_hits"] - at[0.95]["l1_hits"],
        "per_hit": per_hit, "per_hit_no_l2": saving(ref, reqs, l2=False),
        "output_per_req": 200 / 1e6 * ref.BASE_OUTPUT,
        "tolerable": {e: tolerable(per_hit, e) for e in WRONG_COST},
        "same_bill": run(ref, reqs, 0.3, threshold=0.85) == run(ref, reqs, 0.3, threshold=0.95),
        "reads_threshold": "l1_threshold" in inspect.getsource(ref.simulate),
    }


def verify(result):
    tol = result["tolerable"]
    ratio = result["per_hit_no_l2"] / result["per_hit"]
    return [
        practice.Check(
            "ANSWER: keep 0.95",
            all([result["bill"] == {0.95: 4.75, 0.85: 3.07}, result["extra_hits"] == 389,
                 round(result["per_hit"], 4) == 0.0043, round(1 / tol[1.0]) == 232,
                 round(1 / tol[10.0]) == 2325]),
            f"bill {result['bill']}; each semantic hit saves ${result['per_hit']:.4f} "
            f"(${result['output_per_req']:.4f} of it output), so 0.85 pays only below "
            f"{tol[1.0]:.2%} wrong at $1 a wrong answer and {tol[10.0]:.3%} at $10",
        ),
        practice.Check(
            "FINDING: the prompt cache makes a semantic hit worth 3.8x less",
            round(result["per_hit_no_l2"], 4) == 0.0163 and round(ratio, 1) == 3.8,
            f"${result['per_hit_no_l2']:.4f} a hit without L2 against "
            f"${result['per_hit']:.4f} with it, {ratio:.1f}x",
        ),
        practice.Check(
            "FINDING: l1_threshold does nothing in the reference",
            result["same_bill"] and not result["reads_threshold"],
            "simulate() never reads l1_threshold; 0.85 and 0.95 at the same hit "
            "probability produce identical results",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
