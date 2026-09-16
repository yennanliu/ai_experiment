"""Exercise 5 — tree speculation accepts more tokens and is slower at every setting.

    Extend the speculative decoding simulator to implement tree-based speculation
    (EAGLE-2 style). Instead of a single chain of K draft tokens, generate a tree
    of candidates (e.g., 2 branches at each of 3 levels = 8 leaf candidates).
    Compare total tokens accepted per verification round vs linear speculation.

Reading of the exercise: the tree is built on the acceptance rule the lesson's
own `speculative_decode` uses -- `np.random.random() < acceptance_rate`, one
independent draw per drafted token -- because that is what "extend the simulator"
means, and the cost is the lesson's own `draft_cost` per drafted node plus one
flat `verify_cost` per round. The linear arm is the reference function itself,
checked against its own closed form.

**ANSWER: the tree accepts more per round and is slower at every setting.**

    linear  K=3    accepted 1.952 of  3    drafted  3    cost 15.0    2.0x
    linear  K=5    accepted 2.689 of  5    drafted  5    cost 17.0    2.2x
    linear  K=8    accepted 3.329 of  8    drafted  8    cost 20.0    2.2x
    tree    2x3    accepted 2.766 of  3    drafted 14    cost 26.0    1.4x
    tree    4x3    accepted 2.990 of  3    drafted 84    cost 96.0    0.4x
    tree    2x5    accepted 4.431 of  5    drafted 62    cost 74.0    0.7x

The exercise's own 2-branch, 3-level tree buys **0.077** more accepted tokens
than a 5-token chain for **9** more drafted ones, and `4x3` is slower than not
speculating at all.

**MECHANISM: acceptance in this simulator is a coin, not a comparison.**
`speculative_decode` computes `draft_model.get_probs(...)` and
`target_model.get_probs(...)` and then decides with
`r < draft_model.acceptance_rate`, discarding both. So a branch is one more
independent coin: the chance that some candidate at a level survives goes from
`p` to `1 - (1 - p)^b`, which at p = 0.8 is 0.800 to **0.960** for two branches
and 0.9984 for four. That is the entire benefit, and it costs `b` drafts per
node.

**FINDING: the thing that makes EAGLE-2's trees work is not in the model.** Real
tree speculation branches where the draft model is *uncertain*, so the extra
candidates are correlated with the positions the chain would have lost. Here
every draw is i.i.d., so a branch is worth the same wherever it is placed and the
optimal tree is the one with no branches.

**FINDING: the flat `verify_cost` is most of the round.** At K=5 the verify is
12 of the 17-unit round, **71%**, which is why the linear speedup is nearly flat
from K=5 to K=8 (2.17x against 2.16x) -- extra draft tokens are cheap until the
tree makes them the majority of the cost.

Structure: `accepted` and `drafted` are the closed forms for one speculation
shape under the lesson's coin, `plan` prices one with the lesson's own cost
constants, and `solve` runs `speculative_decode` to confirm the chain form
describes it.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "12-inference-optimization"
SEED, VOCAB, ACCEPT = 3, 64, 0.8
DRAFT_COST, VERIFY_COST, TARGET_COST = 1.0, 12.0, 10.0
PLANS = (("linear K=3", 1, 3), ("linear K=5", 1, 5), ("linear K=8", 1, 8),
         ("tree 2x3", 2, 3), ("tree 4x3", 4, 3), ("tree 2x5", 2, 5))


def per_level(branches, accept=ACCEPT):
    """The chance some candidate at one level survives: one coin, or the best of `b`."""
    return 1 - (1 - accept) ** branches


def accepted(branches, levels):
    """Expected accepted depth: the level survives only if every level above it did."""
    survive = per_level(branches)
    return sum(survive ** depth for depth in range(1, levels + 1))


def drafted(branches, levels):
    """Nodes the draft model has to produce -- a chain is `levels`, a tree is a geometric sum."""
    return sum(branches ** depth for depth in range(1, levels + 1))


def plan(branches, levels):
    """One speculation shape priced with the lesson's own cost constants."""
    nodes, taken = drafted(branches, levels), accepted(branches, levels)
    cost = DRAFT_COST * nodes + VERIFY_COST
    return {"accepted": taken, "drafted": nodes, "levels": levels, "cost": cost,
            "speedup": (taken + 1) * TARGET_COST / cost}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = {name: plan(branches, levels) for name, branches, levels in PLANS}
    np.random.seed(SEED)
    measured = ref.speculative_decode(ref.DraftModel(VOCAB, ACCEPT), ref.TargetModel(VOCAB),
                                      [1, 2, 3], num_speculative=5)
    return {
        "rows": rows,
        "reference": (measured["avg_accepted"], measured["speedup"]),
        "closed_form": accepted(1, 5),
        "per_level": {b: per_level(b) for b in (1, 2, 4)},
        "verify_share": VERIFY_COST / rows["linear K=5"]["cost"],
    }


def column(rows, field, fmt):
    return ", ".join(f"{name} {format(row[field], fmt)}" for name, row in rows.items())


def verify(result):
    rows = result["rows"]
    best_linear = max(rows[name]["speedup"] for name in rows if name.startswith("linear"))
    trees = {name: row for name, row in rows.items() if name.startswith("tree")}
    measured, measured_speedup = result["reference"]
    return [
        practice.Check(
            "ANSWER: every tree accepts more per round and is slower than the best chain",
            all(row["speedup"] < best_linear for row in trees.values())
            and rows["tree 2x3"]["accepted"] > rows["linear K=5"]["accepted"],
            "accepted tokens per round are " + column(rows, "accepted", ".3f")
            + " from " + column(rows, "drafted", "d")
            + " drafted nodes, for speedups of " + column(rows, "speedup", ".2f")
            + f"x. The exercise's own 2-branch, 3-level tree buys "
            f"{rows['tree 2x3']['accepted'] - rows['linear K=5']['accepted']:+.3f} accepted "
            f"tokens over a 5-token chain for "
            f"{rows['tree 2x3']['drafted'] - rows['linear K=5']['drafted']} more drafted ones, "
            f"and tree 4x3 at {rows['tree 4x3']['speedup']:.2f}x is slower than not speculating",
        ),
        practice.Check(
            "MECHANISM: acceptance here is a coin, so a branch is one more independent draw",
            abs(measured - result["closed_form"]) < 0.3,
            "speculative_decode computes draft_model.get_probs and target_model.get_probs and "
            "then decides with r < draft_model.acceptance_rate, discarding both. Acceptance is "
            f"i.i.d. Bernoulli({ACCEPT}), so the lesson's own K=5 run averages {measured:.3f} "
            f"accepted against the closed form's {result['closed_form']:.3f}, and a branch only "
            "raises the per-level survival from "
            + ", ".join(f"{b} branch{'es' if b > 1 else ''} {p:.3f}"
                        for b, p in result["per_level"].items())
            + " -- for b drafts per node",
        ),
        practice.Check(
            "FINDING: what makes EAGLE-2's trees work is not in this model",
            rows["tree 4x3"]["accepted"] > rows["tree 2x3"]["accepted"]
            > rows["linear K=3"]["accepted"],
            "real tree speculation branches where the draft model is uncertain, so the extra "
            "candidates sit at the positions the chain would have lost. Here every draw is "
            "independent of the tokens, so a branch is worth the same wherever it is placed: "
            f"widening 3 levels from 1 to 2 to 4 branches moves accepted depth "
            f"{rows['linear K=3']['accepted']:.3f} to {rows['tree 2x3']['accepted']:.3f} to "
            f"{rows['tree 4x3']['accepted']:.3f} against a ceiling of 3, while the draft cost "
            f"goes {rows['linear K=3']['drafted']} to {rows['tree 2x3']['drafted']} to "
            f"{rows['tree 4x3']['drafted']}. The optimal tree under this model has no branches",
        ),
        practice.Check(
            "FINDING: the flat verify_cost is 71% of a K=5 round",
            result["verify_share"] > 0.7 and abs(rows["linear K=5"]["speedup"]
                                                 - rows["linear K=8"]["speedup"]) < 0.05,
            f"verify_cost is {VERIFY_COST:.0f} of the {rows['linear K=5']['cost']:.0f}-unit K=5 "
            f"round, {100 * result['verify_share']:.0f}%, which is why the linear speedup is "
            f"nearly flat from K=5 to K=8 -- {rows['linear K=5']['speedup']:.2f}x against "
            f"{rows['linear K=8']['speedup']:.2f}x. Extra draft tokens are almost free until the "
            "tree makes them the majority of the round, and then they are the whole cost",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
