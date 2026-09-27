"""Exercise 1 — BPO cuts the chosen log-prob drop from 0.155 to 0.012 nats, yet keeps less chosen probability.

    Run `code/main.py`. Report the final chosen-log-prob drop for DPO and BPO.
    BPO should retain higher chosen absolute probability — verify this.

Reading of the exercise: the toy policy picks one of 4 actions, so "the chosen
response" of a pair is an action. The chosen-log-prob drop is the mean over the
run's 500 preference pairs of log pi(y_w) - log pi_ref(y_w), taken from the
policies `main()` itself trains (recorded by wrapping `report`, on the shipped
seed-1 stream). "Chosen absolute probability" is read literally: the mean of
pi(y_w) over the same pairs. `main()` prints neither number.

**ANSWER: DPO's chosen log-prob falls 0.155 nats on average, BPO's 0.012.**
BPO keeps 13x less of the drop. In both runs the same 302 of 500 chosen
responses lose log-probability: actions 0, 2 and 3 fall (DPO -0.167 / -0.957 /
-1.452, BPO -0.075 / -0.458 / -0.722), and only action 1 rises.

**FINDING: on absolute probability the claim is backwards.** The mean chosen
probability is 0.335 under DPO and 0.308 under BPO, from 0.264 at the reference.
BPO's anchor also holds back action 1, the most frequent winner (+0.746 nats
under DPO, +0.520 under BPO), and that costs more probability than it saves on
the falling actions. Over 50 seeds BPO has the smaller log-prob drop on 50 and
the higher chosen probability on 0.

**FINDING: the printout cannot answer the question.** `report` prints probs,
logits and a `win_rate` that is just pi(action 1). Nothing about chosen
log-probs is printed, although the module docstring says it compares them.

Structure: `run(seed)` executes the shipped `main()` with the module `random`
swapped for `random.Random(seed)` (restored after), recording every policy and
pair. The seed-1 transcript is compared byte for byte with a subprocess run of
`code/main.py`.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import math
import random
import subprocess
import sys

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "03-direct-preference-optimization-family"
SEEDS = range(50)


def run(ref, seed):
    """Shipped main() on a seeded stream: (policies by name, the 500 pairs, stdout)."""
    policies, pairs = {}, []
    saved = (ref.random, ref.report, ref.sample_pref_pair)

    def record_pair():
        pairs.append(saved[2]())
        return pairs[-1]

    def record(name, pi):
        policies[name] = pi
        saved[1](name, pi)

    ref.random, ref.report, ref.sample_pref_pair = random.Random(seed), record, record_pair
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            ref.main()
    finally:
        ref.random, ref.report, ref.sample_pref_pair = saved
    return policies, pairs, out.getvalue()


def chosen(ref, policies, pairs, name):
    """(mean chosen log-prob drift, pairs whose chosen fell, mean chosen prob, drift per action)."""
    base = ref.logsoftmax(policies["REF"].logits)
    logp = ref.logsoftmax(policies[name].logits)
    drift = [a - b for a, b in zip(logp, base)]
    n = len(pairs)
    return (round(sum(drift[w] for w, _, _ in pairs) / n, 3),
            sum(drift[w] < 0 for w, _, _ in pairs),
            round(sum(math.exp(logp[w]) for w, _, _ in pairs) / n, 3),
            [round(d, 3) for d in drift])


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    policies, pairs, out = run(ref, 1)
    script = parity.lesson_dir(PHASE, LESSON) / "code" / "main.py"
    shipped = subprocess.run([sys.executable, str(script)], capture_output=True, text=True).stdout
    sweep = []
    for seed in SEEDS:
        pol, prs, _ = run(ref, seed)
        dpo, bpo = chosen(ref, pol, prs, "DPO"), chosen(ref, pol, prs, "BPO")
        sweep.append((bpo[0] > dpo[0], bpo[2] > dpo[2]))
    return {
        "same_as_shipped": out == shipped, "pairs": len(pairs),
        "dpo": chosen(ref, policies, pairs, "DPO"), "bpo": chosen(ref, policies, pairs, "BPO"),
        "ref_prob": chosen(ref, policies, pairs, "REF")[2],
        "win_is_p1": all(ref.win_rate(p) == ref.softmax(p.logits)[1] for p in policies.values()),
        "sweep": [sum(col) for col in zip(*sweep)],
        "report_src": inspect.getsource(ref.report), "doc": ref.__doc__,
    }


def verify(result):
    dpo, bpo, sweep = result["dpo"], result["bpo"], result["sweep"]
    return [
        practice.Check(
            "ANSWER: DPO's chosen log-prob falls 0.155 nats on average, BPO's 0.012",
            (result["same_as_shipped"], result["pairs"], dpo[0], bpo[0], round(dpo[0] / bpo[0]),
             dpo[1], bpo[1], dpo[3], bpo[3])
            == (True, 500, -0.155, -0.012, 13, 302, 302, [-0.167, 0.746, -0.957, -1.452],
                [-0.075, 0.52, -0.458, -0.722]),
            f"seed-1 run reproduces code/main.py byte for byte: {result['same_as_shipped']}; "
            f"(drift, chosen fell, chosen prob, per-action drift) DPO {dpo}, BPO {bpo}",
        ),
        practice.Check(
            "FINDING: on absolute probability the claim is backwards",
            (dpo[2], bpo[2], result["ref_prob"], sweep) == (0.335, 0.308, 0.264, [50, 0]),
            f"mean chosen prob DPO {dpo[2]}, BPO {bpo[2]}, reference {result['ref_prob']}; over "
            f"{len(SEEDS)} seeds BPO has the smaller drop on {sweep[0]}, higher prob on {sweep[1]}",
        ),
        practice.Check(
            "FINDING: the printout cannot answer the question",
            all([result["win_is_p1"], "logprob" not in result["report_src"],
                 "chosen log-prob" in result["doc"]]),
            "report() prints probs, logits and win_rate == pi(action 1); the module docstring "
            "promises 'chosen log-prob'",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
