"""Exercise 1 -- a 3.3 -> 3.4 drift that changes a third of greedy tokens moves the lesson's acceptance 0.003.

    Measure acceptance-rate degradation when the draft is one version behind the target (e.g., Llama 3.3 -> 3.4 drift). Build a monitoring alert.

Reading of the exercise: the lesson's `TargetModel` is "v3.3". "v3.4" is
the same model with each position's weights multiplied by a seeded
log-normal factor exp(sigma * N(0, 1)), so sigma is the size of the version
gap and sigma = 0 is no gap. The draft is the lesson's `DraftModel`
(alignment 0.9, k = 4), still aligned to v3.3, and it runs through the
lesson's own `speculative_decode` against v3.4. Degradation is reported two
ways: the lesson's `acceptance_rate`, which uses its accept rule (target prob
>= 0.5 x max), and the expected greedy acceptance, meaning the chance that a
drafted token is v3.4's argmax. The alert calibrates on the first 100 target
calls of a 4,000-token stream, and v3.4 goes live at token 2,000. It fires
when a 50-call rolling mean falls below 90% of the calibrated baseline.
False alarms are counted over 20 seeded streams with no drift.

**ANSWER: acceptance degrades as below, and the alert has to watch greedy
agreement, not the lesson's metric.** Over 10,000 tokens:

    sigma  argmax kept  greedy acceptance  lesson acceptance  tok/call
    0      1.000        0.919              0.941              4.76
    0.1    0.674        0.625              0.938              4.75
    0.25   0.464        0.435              0.876              4.50
    0.5    0.328        0.311              0.474              2.90
    1.0    0.229        0.221              0.175              1.70

The greedy-agreement alert fires 16 target calls after v3.4 goes live at
sigma = 0.1, and 8 calls after at sigma >= 0.25. It raises 0 false alarms in
20 quiet streams.

**FINDING: the lesson's acceptance metric cannot see a small version gap.**
At sigma = 0.1, v3.4 changes the argmax on 32.6% of positions. The lesson's
acceptance moves from 0.941 to 0.938, and its tokens per call from 4.76 to
4.75. The same alert driven by the lesson's metric never fires at sigma =
0.1. It needs 41 calls at 0.25, and it raises 3 false alarms in 20 quiet
streams. The cause is the rule: any token within half of the max passes, and
on this 10-token vocabulary that is 55% of tokens.

**FINDING: in the lesson's accounting, a drifted draft cannot cost more than
no speculation.** A draft that always proposes the least likely token still
scores 1.001 tokens per target call, with 9 accepted tokens in 10,000.
Speedup is counted as target calls saved, so draft cost is zero and every call
emits at least one token. The skill file's "Drifted drafts cost more than no
speculation" cannot show up in this code.

Structure: `upgraded()` is v3.4; `setup()` wires the stale draft and a
logging verify around the lesson's classes; `degradation()` measures one
sigma; `alert()` and `monitor()` are the rolling-mean alert.
"""

from __future__ import annotations

import functools
import math
import random
import statistics
import types

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "14-speculative-decoding-server"
K, ALIGN, N, SWITCH, W, DROP = 4, 0.9, 4000, 2000, 50, 0.9
SIGMAS = (0.0, 0.1, 0.25, 0.5, 1.0)


@functools.cache
def upgraded(ref, s, sigma):
    """v3.4 at position s: v3.3's weights times a seeded log-normal nudge."""
    r = random.Random(s * 7 + 10013)
    w = [p * math.exp(sigma * r.gauss(0, 1)) for p in ref.softmax_from(s * 7 + 13)]
    return tuple(x / sum(w) for x in w)


def argmax(p):
    return max(range(len(p)), key=p.__getitem__)


def setup(ref, sigma, switch=0):
    """The live target (v3.4 from `switch` on) and a draft still aligned to v3.3."""
    target, old, draft = ref.TargetModel(), ref.TargetModel(), ref.DraftModel(alignment=ALIGN)
    target.distribution = lambda s: upgraded(ref, s, sigma) if s >= switch else old.distribution(s)
    stale = types.SimpleNamespace(propose=lambda c, k, rng, _t: draft.propose(c, k, rng, old))
    log, verify = [], target.verify

    def logged(tokens, ctx, rng):
        accepted, nxt = verify(tokens, ctx, rng)
        agree = sum(t == argmax(target.distribution(ctx + i)) for i, t in enumerate(tokens))
        log.append((ctx, len(accepted) / K, agree / K))
        return accepted, nxt

    target.verify = logged
    return target, stale, log


def degradation(ref, sigma):
    """(argmax kept, expected greedy acceptance, lesson acceptance, tok/call) over 10,000 tokens."""
    target, stale, _ = setup(ref, sigma)
    m = ref.speculative_decode(10_000, K, random.Random(7), target, stale)
    pairs = [(ref.softmax_from(s * 7 + 13), upgraded(ref, s, sigma)) for s in range(1, 10_001)]
    kept = statistics.mean(argmax(p) == argmax(q) for p, q in pairs)
    greedy = statistics.mean(ALIGN * (argmax(p) == argmax(q)) + (1 - ALIGN) * p[argmax(q)] for p, q in pairs)
    return round(kept, 3), round(greedy, 3), round(m.acceptance_rate(K), 3), round(m.tokens_per_target_call(), 2)


def alert(stream, col):
    """Calibrate on the first 100 calls; fire when a W-call mean falls below DROP x that baseline."""
    floor = DROP * sum(row[col] for row in stream[:100]) / 100
    for i in range(100 + W, len(stream) + 1):
        if sum(row[col] for row in stream[i - W:i]) / W < floor:
            return i - 1


def monitor(ref, sigma, seed=7):
    """Per metric, target calls between the first v3.4 call and the alert (None: never fired)."""
    target, stale, log = setup(ref, sigma, SWITCH)
    ref.speculative_decode(N, K, random.Random(seed), target, stale)
    first = next(i for i, row in enumerate(log) if row[0] + K >= SWITCH)
    return {m: None if (h := alert(log, col)) is None else h - first for m, col in (("lesson", 1), ("greedy", 2))}


def floor_probe(ref):
    """A draft that always proposes the least likely token, and the share of tokens the rule passes."""
    never = types.SimpleNamespace(propose=lambda c, k, rng, t: [argmax([-x for x in t.distribution(c + i)])
                                                                 for i in range(k)])
    m = ref.speculative_decode(10_000, K, random.Random(7), ref.TargetModel(), never)
    passing = statistics.mean(sum(x >= 0.5 * max(p) for x in p) / 10 for p in map(ref.softmax_from, range(20, 70_007, 7)))
    return round(m.tokens_per_target_call(), 3), m.accepted_sum, round(passing, 3)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    runs, quiet = [monitor(ref, s) for s in SIGMAS[1:]], [monitor(ref, 0.0, seed) for seed in range(20)]
    return {"table": {s: degradation(ref, s) for s in SIGMAS},
            "delays": {m: [r[m] for r in runs] for m in ("lesson", "greedy")},
            "false_alarms": {m: sum(q[m] is not None for q in quiet) for m in ("lesson", "greedy")},
            "floor": floor_probe(ref)}


def verify(result):
    t, d, fa, fl = result["table"], result["delays"], result["false_alarms"], result["floor"]
    return [
        practice.Check(
            "ANSWER: greedy acceptance 0.919 -> 0.221 across the gap; its alert fires in 16 calls at sigma 0.1",
            t == {0.0: (1, 0.919, 0.941, 4.76), 0.1: (0.674, 0.625, 0.938, 4.75), 0.25: (0.464, 0.435, 0.876, 4.5),
                  0.5: (0.328, 0.311, 0.474, 2.9), 1.0: (0.229, 0.221, 0.175, 1.7)}
            and (d["greedy"], fa["greedy"]) == ([16, 8, 8, 8], 0),
            f"sigma -> (argmax kept, greedy, lesson, tok/call) {t}; greedy-alert delays {d['greedy']}; "
            f"quiet false alarms {fa['greedy']}/20",
        ),
        practice.Check(
            "FINDING: the lesson's acceptance metric cannot see a small version gap",
            (t[0.0][2], t[0.1][2], t[0.1][0], d["lesson"], fa["lesson"], fl[2])
            == (0.941, 0.938, 0.674, [None, 41, 10, 6], 3, 0.55),
            f"sigma 0.1 keeps argmax on {t[0.1][0]} of positions, lesson acceptance {t[0.0][2]} -> {t[0.1][2]}; "
            f"lesson-alert delays {d['lesson']}; quiet false alarms {fa['lesson']}/20; "
            f"{fl[2]} of tokens pass the 0.5 x max rule",
        ),
        practice.Check(
            "FINDING: in the lesson's accounting a drifted draft cannot cost more than no speculation",
            fl[:2] == (1.001, 9),
            f"an always-least-likely draft: {fl[0]} tok/call, {fl[1]} accepted of 10,000",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
