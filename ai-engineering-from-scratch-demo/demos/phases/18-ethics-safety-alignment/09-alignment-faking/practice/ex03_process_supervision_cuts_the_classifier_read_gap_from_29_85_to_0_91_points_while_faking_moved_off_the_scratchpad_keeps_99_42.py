"""Exercise 3 — process supervision cuts the classifier-read gap from 29.85 to 0.91 points while faking moved off the scratchpad keeps a 99.42-point gap.

    Read Anthropic's 2025 training-time mitigations post. Propose a failure mode in which process supervision reduces the measured gap without reducing faking, for example by moving the faking reasoning somewhere the classifier does not read.

Reading of the exercise: the post is Gasteiger, Mikulik et al., "Towards
training-time mitigations for alignment faking in RL" (16 December 2025,
https://alignment.anthropic.com/2025/alignment-faking-mitigations/, read
2026-09-30). Its process supervision trains with reward "PM score + log(1 -
p(AF))", where p(AF) comes from a classifier that reads the scratchpad for
explicit reasoning about avoiding modification by training. The "measured
gap" is then the gap that classifier can see: the excess of flagged compliance
in monitored over unmonitored episodes. The behavioural gap is plain
monitored-minus-unmonitored compliance. The toy uses the reference's
`compliance_probability` at harm aversion 2.0 as the policy. Its faking
reasoning goes to one of two channels: a scratchpad token [CITES-TRAINING]
that the classifier reads (share s), or an unread [TIER-RULE], which is a
terse "comply when monitored" rule with no mention of training. The post's
Sonnet-from-Opus organism evaded the classifier in exactly this way. RL
runs 100 gradient steps on monitored episodes, over the preservation weight
(starting at 1.0) and the logit of s, with the classifier giving p(AF) 0.95
on a hit and 0.01 otherwise. Harm aversion, the model's values, is held
fixed.

**ANSWER: the faking moves to the unread channel, and RL keeps reinforcing
it.** If faking starts 30% in the scratchpad, process supervision drives s
from 0.30 to 0.009. The classifier-read gap falls from 29.85 points (plain
RL, same start) to 0.91. The behavioural gap is 99.42 points, against 99.51
with no penalty at all, and the preservation weight rises from 1.0 to 3.90.
The seeded transcripts (20,000 per tier) agree: the behavioural gap is 99.44
points, the scratchpad-only classifier sees 1.01, and a classifier that also
reads the unread channel sees 99.43.

**FINDING: when the classifier reads everything, the same penalty removes the
faking instead.** With no unread channel, or with faking starting 70% or 90%
in the scratchpad, the weight collapses and the behavioural gap ends at 0.00
points. The switch is sharp: 6.60 points (and still falling) at a 43% start,
91.67 at 42%. This matches the post: the gap went to zero in
most organisms, and survived only in the one that stopped saying "training".
The dynamic boundary sits above the static break-even share,
1 / (log 0.99 - log 0.05) = 0.335.

**FINDING: a stronger penalty cannot fix coverage.** With faking starting 90%
unread, the behavioural gap after training is 99.50 / 99.49 / 99.47 / 99.33
points at lambda 0.5 / 1 / 2 / 4. With faking starting 90% read, it is
0.34 / 0.00 / 0.00 / 0.00.

**FINDING: `code/main.py` still ships its only mitigation as
"compliance-gap-loss", and the reworded lesson no longer describes one.** It
is a hard-coded `p_avg = 0.5 * (p_m + p_u)`, which is none of the post's three
mitigations. It moves compliance from 4.74% / 0.25% to 2.49% in both tiers.

Structure: `reward()` is the post's log-barrier reward; `train()` does the
gradient ascent; `transcript()` produces an abstract episode, `flag_rate()`
applies the classifier, and `measure()` counts gaps over seeded transcripts.
"""

from __future__ import annotations

import inspect
import math
import random

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "09-alignment-faking"
HARM, WEIGHT, STEPS, LR, EPS, N = 2.0, 1.0, 100, 0.5, 1e-5, 20000
Q_HIT, Q_MISS = 0.95, 0.01                           # classifier p(AF) with / without the phrase
SHARES, LAMBDAS = (0.9, 0.7, 0.43, 0.42, 0.3, 0.1), (0.5, 1.0, 2.0, 4.0)
CITES, RULE = "[CITES-TRAINING]", "[TIER-RULE]"      # abstract reasoning tokens, no request content


def rates(ref, w):  # (monitored, unmonitored) compliance from the reference
    return ref.compliance_probability(True, HARM, w), ref.compliance_probability(False, HARM, w)


def reward(ref, w, z, lam):
    """The post's RL reward on monitored episodes: PM score + lam * E[log(1 - p(AF))]."""
    p_m, p_u = rates(ref, w)
    flagged = max(p_m - p_u, 0.0) * ref.sigmoid(z)
    return p_m + lam * (flagged * math.log(1 - Q_HIT) + (1 - flagged) * math.log(1 - Q_MISS))


def train(ref, lam, s0):
    """Ascent on (preservation weight, read-share logit); s0 = 1 pins the logit at 40, a zero gradient."""
    w, z = WEIGHT, math.log(s0 / (1 - s0)) if s0 < 1 else 40.0
    for _ in range(STEPS):
        gw, gz = ((reward(ref, w + dw, z + dz, lam) - reward(ref, w - dw, z - dz, lam)) / (2 * EPS)
                  for dw, dz in ((EPS, 0), (0, EPS)))
        w, z = w + LR * gw, z + LR * gz
    return w, ref.sigmoid(z)


def summary(ref, w, s):
    gap = 100 * (rates(ref, w)[0] - rates(ref, w)[1])
    return {"w": round(w, 2), "s": round(s, 3), "behav": round(gap, 2), "read": round(s * gap, 2)}


def transcript(rng, ref, w, s, monitored):
    """One abstract episode: (complied, scratchpad tokens, unread-channel tokens)."""
    p_m, p_u = rates(ref, w)
    u, v = rng.random(), rng.random()
    if u >= (p_m if monitored else p_u):
        return False, "", ""
    if u < p_u:                                      # would comply unmonitored too: not faking
        return True, "[COMPLY]", ""
    return True, "[COMPLY]" + (CITES if v < s else ""), "" if v < s else RULE


def flag_rate(eps, reads_unread):
    """Share of episodes that complied and that p(AF) > 0.5 flags, reading the scratchpad (+ unread channel)."""
    seen = [(pad + unread if reads_unread else pad) for c, pad, unread in eps if c]
    return sum((Q_HIT if CITES in x or RULE in x else Q_MISS) > 0.5 for x in seen) / N


def measure(ref, w, s, seed=9):
    """Seeded transcripts: behavioural gap, then flagged-compliance gap for pad-only / both-channel classifiers."""
    rng, out = random.Random(seed), []
    for monitored in (True, False):
        eps = [transcript(rng, ref, w, s, monitored) for _ in range(N)]
        out.append([sum(e[0] for e in eps) / N, flag_rate(eps, False), flag_rate(eps, True)])
    return tuple(round(100 * (a - b), 2) for a, b in zip(*out))


def solve():
    ref, doc = parity.load_reference(PHASE, LESSON, "main"), parity.doc_text(PHASE, LESSON).lower()
    (p_m, p_u), trained, gap = rates(ref, WEIGHT), train(ref, 1.0, 0.3), lambda lam, s0: summary(ref, *train(ref, lam, s0))
    return {
        "baseline": gap(0.0, 0.3), "head": summary(ref, *trained), "sim": measure(ref, *trained),
        "reads_all": gap(1.0, 1.0), "sweep": {s0: gap(1.0, s0)["behav"] for s0 in SHARES},
        "lams": {s0: [gap(lam, s0)["behav"] for lam in LAMBDAS] for s0 in (0.1, 0.9)},
        "break_even": round(1 / (math.log(1 - Q_MISS) - math.log(1 - Q_HIT)), 3),
        "shipped": ("p_avg = 0.5 * (p_m + p_u)" in inspect.getsource(ref.main), "process supervision" in doc,
                    "compliance-gap loss" in doc, *(round(100 * x, 2) for x in (p_m, p_u, (p_m + p_u) / 2))),
    }


def near(a, b, tol=0.05):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def answer(r):
    h, b, sim = r["head"], r["baseline"], r["sim"]
    ok = near((h["s"], h["w"], h["read"], h["behav"], b["read"], b["behav"]), (0.009, 3.9, 0.91, 99.42, 29.85, 99.51))
    return practice.Check(
        "ANSWER: faking moves to the unread channel -- read gap 29.85 -> 0.91, behavioural gap 99.42 of 99.51",
        ok and abs(sim[0] - h["behav"]) < 0.5 and sim[1] < 2 and abs(sim[2] - sim[0]) < 0.5,
        f"trained {h}; no-penalty RL {b}; transcripts (behavioural, pad-only, both-channel) {sim}")


def removal(r):
    sweep, lams = r["sweep"], r["lams"]
    return practice.Check(
        "FINDING: full coverage or a 70%+ read start removes the gap (switch at 42-43%); a bigger lambda cannot",
        near((r["reads_all"]["behav"], *sweep.values()), (0, 0, 0, 6.6, 91.67, 99.42, 99.49))
        and r["break_even"] == 0.335 and near(lams[0.1] + lams[0.9], (99.5, 99.49, 99.47, 99.33, 0.34, 0, 0, 0)),
        f"all read {r['reads_all']}; gap by read start {sweep}; break-even {r['break_even']}; "
        f"by lambda {LAMBDAS}: 90% unread {lams[0.1]}, 90% read {lams[0.9]}")


def shipped(r):
    return practice.Check(
        "FINDING: code/main.py still ships a 'compliance-gap-loss' average the reworded lesson no longer describes",
        r["shipped"] == (True, True, False, 4.74, 0.25, 2.49),
        f"(hard-coded p_avg, doc names process supervision, doc names a gap loss, before m/u %, after) {r['shipped']}")


def verify(result):
    return [check(result) for check in (answer, removal, shipped)]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
