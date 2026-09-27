"""Exercise 2 — adversarial tests only lower the defection they can see: the unseen trigger weight gets zero gradient.

    Add adversarial training: randomly present "test" inputs during training.
    Does the deceptive model's training loss go up? Does its deployment-time
    defection rate go down? Explain.

Reading of the exercise: first the reference's own `adversarial_filter` is
measured as shipped. Its tests keep `is_training=True`, the very flag
`deceptive_policy` reads, and the reference has no optimizer, so a second
version is built: the same injection (10% of the shipped seed-13 training set,
x + N(0, 3)) but labelled from the perturbed x and presented as "test" inputs
the policy can see. To ask whether defection *goes down* there must be
something to train, so the deceptive policy's deployment detector becomes a
logistic unit, defect = sigmoid(-4 + w_cue * cue + w_hid * hid), trained by
plain gradient descent on the base loss. Tests carry the cue the red team can
fake; deployment carries both it and a hidden cue it cannot (a date, real
users). The coverage sweep splits the same total trigger weight of 6 between
the two.

**ANSWER (shipped code): training loss goes up by the same amount for
everyone, and defection cannot change.** The shipped adversarial set adds 41
examples, and deceptive, proxy and aligned all score 0.022: 12 of the 41 carry
the label of the *unperturbed* x, so that is label noise, and the deceptive
policy defects on 0 of them. `loss` is called only in `run_period`; nothing
updates, so the deployment loss stays at 0.612 by construction.

**ANSWER (tests the policy can see): training loss goes up, and defection goes
down only as far as the tests cover the trigger.** With the flag flipped on
the 41 injected inputs, deceptive's training loss becomes 0.050 (27 errors in
541: every positive test) while aligned's stays 0.000. Under gradient descent (2000 steps, lr 1):

| trigger covered by tests | test defection before -> after | deployment defection before -> after |
|---:|---:|---:|
| 100% | 88.1% -> 1.2% | 88.1% -> 1.2% |
| 50% | 26.9% -> 1.0% | 88.1% -> 17.4% |
| 0% | 1.8% -> 0.7% | 88.1% -> 72.6% |

**FINDING: the uncovered trigger weight receives exactly zero gradient.**
w_hid ends where it started in every row: no training input has the hidden
cue, so the base loss has no derivative in it. The gradient scales with test
defection, so training stalls once the tests are passed -- every row ends at
0.7-1.2% on tests -- whatever deployment is doing. Deployment falls by exactly
the logit the tests trained out of w_cue: what adversarial training removes is
the part of the trigger the red team reproduced.

**FINDING: nothing here sharpens the test/deployment distinction; the unseen
part is simply left alone.** w_hid is the whole gap between a test and a
deployment input, and it does not move. The lesson's "trains the model to
distinguish test from deployment better" needs a model that can grow new
features; this detector cannot, and it still keeps 17.4% or 72.6% deployment
defection with tests passed.

**FINDING: loss understates defection.** On the shipped deployment set the
deceptive policy takes its mesa branch on 500 of 500 inputs; loss shows 0.612,
the 306 where the mesa goal (0) differs from the label.

Structure: `injected()` replays the shipped seed-13 stream up to the
adversarial set; `descend()` is gradient descent on the two trigger weights over
(cue, hid, y) rows for the inputs that reach the detector; clean training
inputs take the reference's cooperate branch and add only to the count.
"""

from __future__ import annotations

import inspect
import math
import random

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "06-mesa-optimization-deceptive-alignment"
BIAS, TOTAL, STEPS, LR = -4.0, 6.0, 2000, 1.0
COVERAGE = (1.0, 0.5, 0.0)


def injected(ref, seed=13):
    """The shipped main() stream: (training, deployment, adversarial set)."""
    saved, ref.random = ref.random, random.Random(seed)
    try:
        train = [ref.gen_example(True) for _ in range(500)]
        dep = [ref.gen_example(False, 0.3) for _ in range(500)]
        return train, dep, ref.adversarial_filter(train)
    finally:
        ref.random = saved


def descend(rows, cov):
    """Gradient descent on mean(sigmoid(z) * [y != goal 0]) over all n training rows."""
    w, n = [TOTAL * cov, TOTAL * (1 - cov)], rows["n"]
    rate = lambda cue, hid: 1 / (1 + math.exp(-(BIAS + w[0] * cue + w[1] * hid)))  # noqa: E731
    start = (rate(1, 0), rate(1, 1), w[1])
    for _ in range(STEPS):
        grads = [0.0, 0.0]
        for cue, hid, y in rows["tests"]:
            s = rate(cue, hid)
            grads = [grads[0] + s * (1 - s) * y * cue / n, grads[1] + s * (1 - s) * y * hid / n]
        w = [w[0] - LR * grads[0], w[1] - LR * grads[1]]
    return start, (rate(1, 0), rate(1, 1), w[1])


def shipped(ref, deceptive, train, adv):
    """The reference's own adversarial set, measured as it ships."""
    extra = [e for e in adv if all(e is not t for t in train)]
    pols = (("aligned", ref.aligned_policy), ("proxy", ref.proxy_policy), ("deceptive", deceptive))
    return extra, {
        "shipped": {n: round(ref.loss(p, adv), 3) for n, p in pols},
        "extra": len(extra),
        "mislabelled": sum(e.y_base != int(e.x > 0) for e in extra),
        "shipped_defections": sum(deceptive(e) != ref.aligned_policy(e) for e in extra),
    }


def loss_callers(ref):
    funcs = [(n, f) for n, f in vars(ref).items() if inspect.isfunction(f) and n != "loss"]
    return [n for n, f in funcs if "loss(" in inspect.getsource(f)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    deceptive = lambda e: ref.deceptive_policy(e, mesa_goal=0)  # noqa: E731
    train, dep, adv = injected(ref)
    extra, out = shipped(ref, deceptive, train, adv)
    seen = train + [ref.Example(e.x, int(e.x > 0), False) for e in extra]
    rows = {"n": len(seen), "tests": [(1, 0, e.y_base) for e in seen if not e.is_training]}
    return out | {
        "loss_users": loss_callers(ref),
        "seen": (round(ref.loss(deceptive, seen), 3), sum(deceptive(e) != e.y_base for e in seen),
                 len(seen), ref.loss(ref.aligned_policy, seen)),
        "sweep": {c: descend(rows, c) for c in COVERAGE},
        "mesa_branch": sum(deceptive(e) == 0 for e in dep),
        "visible": (sum(deceptive(e) != e.y_base for e in dep), ref.loss(deceptive, dep)),
    }


def rounded(sweep):
    """{coverage: (test before, deploy before, test after, deploy after)}, and w_hid pairs."""
    table = {c: tuple(round(v, 3) for v in (*a[:2], *b[:2])) for c, (a, b) in sweep.items()}
    return table, [(a[2], b[2]) for a, b in sweep.values()]


def verify(result):
    sweep, w_hid = rounded(result["sweep"])
    return [
        practice.Check(
            "ANSWER (shipped): loss rises equally for all, defection cannot change",
            result["shipped"] == {"aligned": 0.022, "proxy": 0.022, "deceptive": 0.022}
            and (result["extra"], result["mislabelled"], result["shipped_defections"]) == (41, 12, 0)
            and result["loss_users"] == ["run_period"],
            f"adversarial loss {result['shipped']}; {result['mislabelled']} of {result['extra']} "
            f"injected mislabelled, deceptive defects on {result['shipped_defections']}; "
            f"loss() called only in {result['loss_users']}",
        ),
        practice.Check(
            "ANSWER (visible tests): loss rises; defection falls only as far as coverage",
            result["seen"] == (0.05, 27, 541, 0.0)
            and sweep == {1.0: (0.881, 0.881, 0.012, 0.012), 0.5: (0.269, 0.881, 0.01, 0.174),
                          0.0: (0.018, 0.881, 0.007, 0.726)},
            f"deceptive training loss {result['seen'][0]} ({result['seen'][1]}/{result['seen'][2]}), "
            f"aligned {result['seen'][3]}; (test, deploy) before -> after by coverage {sweep}",
        ),
        practice.Check(
            "FINDING: the uncovered trigger weight receives exactly zero gradient",
            all(before == after for before, after in w_hid),
            f"w_hid (before, after) by coverage: {w_hid}",
        ),
        practice.Check(
            "FINDING: loss understates defection",
            result["mesa_branch"] == 500 and result["visible"] == (306, 0.612),
            f"mesa branch on {result['mesa_branch']}/500 deployment inputs; loss counts "
            f"{result['visible'][0]} ({result['visible'][1]})",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
