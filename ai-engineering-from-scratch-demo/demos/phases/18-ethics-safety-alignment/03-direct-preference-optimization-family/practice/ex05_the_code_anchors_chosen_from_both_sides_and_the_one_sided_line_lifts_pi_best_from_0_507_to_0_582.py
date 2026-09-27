"""Exercise 5 — the code anchors the chosen response from both sides; the one-sided line lifts pi(best) from 0.507 to 0.582.

    Read the BPO paper abstract (OpenReview b97EwMUWu7). Write down the
    one-line correction BPO adds to DPO. Confirm against the implementation in
    `code/main.py`.

Reading of the exercise: OpenReview served only a browser-verification page
(fetched 2026-09-27), so the abstract could not be read. The correction is
therefore taken from the lesson's own summary of the paper: "a single-line
correction that penalizes downward moves on the chosen response". "Confirm"
is done by measurement. The gradient `train_dpo` adds for `variant="bpo"` is
isolated by running one step at lr 1 from a chosen policy state and
subtracting the DPO step. It is then compared with the analytic gradient of
candidate penalties, taken through the softmax Jacobian.

**ANSWER: the one-line correction, as the lesson states it:**
L_BPO = L_DPO + (lambda/2) x min(0, log pi(y_w) - log pi_ref(y_w))^2, which
penalizes the chosen log-prob only when it falls below the reference. In
`code/main.py` it is one line, `anchor_pen = -1.0 * min(0.0, log_pi_w -
log_ref_w)`. With that swap the added gradient is exactly 0 once the chosen
response has risen, and identical to the shipped one once it has fallen.

**FINDING: the shipped line is a two-sided anchor.** `anchor_pen = -1.0 *
(log_pi_w - log_ref_w)` is the gradient of 0.025 x (log pi(y_w) -
log pi_ref(y_w))^2. The measured extra gradient matches that penalty's
analytic gradient to 1e-12 in both states tested. When the chosen
response has *risen* (+0.832 nats) the anchor step lowers it (-0.00056). The
term pulls the chosen response back to the reference from either side,
unlike the lesson's "penalizes downward moves" and the code comment's
"toward/above ref". Its weight 0.05 is not scaled by beta.

**FINDING: the one-sided line moves BPO more than half way back to DPO.** On the
shipped seed-1 run, (pi(best), chosen log-prob drift, mean chosen probability)
is DPO (0.636, -0.155, 0.335), shipped BPO (0.507, -0.012, 0.308) and
one-sided BPO (0.582, -0.054, 0.321). That closes 58% of BPO's pi(best) gap.
Action 1 may now rise freely, but the mean chosen probability is still below
DPO's. The DPO policy in that run is
unchanged, so the difference is BPO's alone.

Structure: `one_step()` recovers the applied gradient from one `train_dpo`
step; `run_main()` runs the shipped `main()` with, optionally, `train_dpo`
re-executed with the one line swapped (done in `solve()`).
"""

from __future__ import annotations

import contextlib
import inspect
import io
import math
import random

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "03-direct-preference-optimization-family"
LINE = "anchor_pen = -1.0 * (log_pi_w - log_ref_w)"
ONE_SIDED = "anchor_pen = -1.0 * min(0.0, log_pi_w - log_ref_w)"
STATES = {"chosen fell": [-0.5, 0.2, 0.3, 0.1], "chosen rose": [0.1, 1.5, -0.6, -1.1]}
W, L = 1, 3


def one_step(ref, variant, logits, train=None):
    """The gradient train_dpo applies at `logits` for the pair (W, L): one step at lr 1."""
    make, saved = ref.make_policy_and_ref, ref.random
    ref.make_policy_and_ref = lambda: (ref.Policy(list(logits)), make()[1])
    ref.random = random.Random(0)
    try:
        pi = (train or ref.train_dpo)([(W, L, 0.7)], variant=variant, steps=1, lr=1.0)
    finally:
        ref.make_policy_and_ref, ref.random = make, saved
    return [a - b for a, b in zip(logits, pi.logits)]


def diff(xs, ys):
    return [x - y for x, y in zip(xs, ys)]


def anchor_check(ref, logits, one_sided, lam=0.05):
    """(chosen drift, max |BPO - DPO step - grad of lam/2 x drift^2|, chosen change from a 0.1
    anchor step, largest one-sided extra component, max |shipped - one-sided extra|)."""
    drift = ref.logsoftmax(logits)[W] - ref.make_policy_and_ref()[1].logprob(W)
    p, dpo = ref.softmax(logits), one_step(ref, "dpo", logits)
    extra = diff(one_step(ref, "bpo", logits), dpo)
    extra1 = diff(one_step(ref, "bpo", logits, one_sided), dpo)
    grad = [lam * drift * ((i == W) - p[i]) for i in range(len(p))]  # softmax Jacobian
    step = ref.logsoftmax(diff(logits, [0.1 * e for e in extra]))[W] - ref.logsoftmax(logits)[W]
    return (round(drift, 3), max(map(abs, diff(extra, grad))), step,
            max(map(abs, extra1)), max(map(abs, diff(extra, extra1))))


def run_main(ref, train=None):
    """Shipped main() on seed 1, BPO optionally one-sided: (policies, pairs)."""
    policies, pairs = {}, []
    saved = (ref.random, ref.report, ref.sample_pref_pair, ref.train_dpo)
    ref.train_dpo = train or saved[3]
    ref.random, ref.report = random.Random(1), lambda n, pi: policies.__setitem__(n, pi)
    ref.sample_pref_pair = lambda: pairs.append(saved[2]()) or pairs[-1]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ref.main()
    finally:
        ref.random, ref.report, ref.sample_pref_pair, ref.train_dpo = saved
    return policies, pairs


def chosen(ref, policies, pairs, name):
    """(pi(best), mean chosen log-prob drift, mean chosen probability)."""
    base, logp = ref.logsoftmax(policies["REF"].logits), ref.logsoftmax(policies[name].logits)
    n = len(pairs)
    return (round(math.exp(logp[1]), 3), round(sum(logp[w] - base[w] for w, _, _ in pairs) / n, 3),
            round(sum(math.exp(logp[w]) for w, _, _ in pairs) / n, 3))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    src, doc = inspect.getsource(ref.train_dpo), parity.doc_text(PHASE, LESSON)
    # train_dpo re-executed with its anchor line made one-sided
    exec(src.replace(LINE, ONE_SIDED).replace("def train_dpo(", "def _one_sided("), vars(ref))
    one_sided = ref._one_sided
    (shipped, pairs), (fixed, _) = run_main(ref), run_main(ref, one_sided)
    return {
        "line": src.count(LINE), "anchor": {k: anchor_check(ref, v, one_sided) for k, v in STATES.items()},
        "comment": "toward/above ref" in src and "- 0.05 * anchor_pen * gw_i" in src,
        "lesson": "penalizes downward moves on the chosen response" in doc,
        "dpo": chosen(ref, shipped, pairs, "DPO"), "bpo": chosen(ref, shipped, pairs, "BPO"),
        "bpo1": chosen(ref, fixed, pairs, "BPO"),
        "dpo_same": fixed["DPO"].logits == shipped["DPO"].logits,
    }


def verify(result):
    fell, rose = result["anchor"]["chosen fell"], result["anchor"]["chosen rose"]
    dpo, bpo, bpo1 = result["dpo"][0], result["bpo"][0], result["bpo1"][0]
    closed = round((bpo1 - bpo) / (dpo - bpo), 2)
    return [
        practice.Check(
            "ANSWER: the one-line correction, as the lesson states it",
            all([result["lesson"], result["line"] == 1, rose[3] == 0.0, fell[3] > 0,
                 fell[4] < 1e-12]),
            f"one-sided extra gradient {rose[3]} once chosen rose, shipped one +- {fell[4]:.0e} "
            "once it fell; docs/en.md says 'penalizes downward moves on the chosen response'",
        ),
        practice.Check(
            "FINDING: the shipped line is a two-sided anchor",
            all([fell[0] < 0 < rose[0] == 0.832, max(fell[1], rose[1]) < 1e-12,
                 fell[2] > 0 > rose[2], round(rose[2], 5) == -0.00056,
                 result["comment"]]),
            f"(drift, |extra - grad 0.025 x drift^2|, anchor-step change, ...) {fell}, {rose}",
        ),
        practice.Check(
            "FINDING: the one-sided line moves BPO more than half way back to DPO",
            (result["dpo"], result["bpo"], result["bpo1"], result["dpo_same"], closed)
            == ((0.636, -0.155, 0.335), (0.507, -0.012, 0.308), (0.582, -0.054, 0.321), True, 0.58),
            f"(pi best, drift, chosen prob) DPO {result['dpo']}, BPO {result['bpo']}, "
            f"one-sided {result['bpo1']}; gap closed {closed}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
