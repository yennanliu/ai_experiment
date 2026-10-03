"""Exercise 2 — the one-step delay is what carries the coupling between streams.

    **Medium.** Add delayed parallel decoding: simulate 2 streams of tokens that
    must stay offset by 1 step. Train a joint predictor.

Reading of the exercise: two codebook streams, `a` (the lesson's own style-0
tokens from `make_tokens`) and `b`, a second codebook fully determined by the
first (`b_t = a_t + 8 mod 16`, standing in for an RVQ residual that depends on
its coarser layer). In the delayed layout, frame `t` carries `(a_t, b_{t-1})`,
so when `b_t` is emitted one frame later, `a_t` is already decoded. The joint
predictor is two 16-way heads that share the frame history -- the head for `a`
conditions on `a_{t-1}`, the head for `b` on `a_t` -- trained with the lesson's
own `init_counts`, `update_counts` and `probs`, and sampled with its own
`sample_from`. The score is coupling: the fraction of generated frames where
`b_t` is still `a_t`'s partner.

**ANSWER: delayed decoding keeps the streams coupled 97.3% of the time.** The
`b` head puts **96.9%** of its mass on the right partner; the rest is the
lesson's add-one prior.

**FINDING: remove the delay and coupling collapses to 36.5%.** The same heads
emitting `a_t` and `b_t` in the same step can only condition `b_t` on `a_{t-1}`,
so `b` guesses the step `a` is about to take. That is predictable in closed
form: two independent draws from `{0: 1/4, 1: 1/2, 2: 1/4}` agree with
probability `1/16 + 1/4 + 1/16 = 0.375`. The delay costs one extra frame and
buys a 2.7x gain in coupling.

**FINDING: a 256-way joint frame predictor without delay is worse still, 27.1%.**
Predicting the `(a, b)` pair as one token captures the coupling in principle,
but the lesson's add-one smoothing scales with the output vocabulary: 256 prior
counts against ~600 real ones leave only **71%** of an average seen row's mass on valid
pairs, and only **16** of the 256 frame contexts are ever seen in training. One
leak lands the sampler in an unseen, uniform row, where 6.25% of the mass is
valid, so errors compound.

**CONTROL: the offset is exactly one step.** In the delayed layout, the `b` slot
of every frame equals the partner of the previous frame's `a` across a
20-frame clip, and that clip takes 21 delayed frames.

Structure: `train` fits all three predictors; `generate` samples one clip per
mode; `coupling` scores it.
"""

from __future__ import annotations

import itertools
import random

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "11-audio-generation"
V, LENGTH, CLIPS = 16, 20, 1000


def partner(a):
    """The second codebook's token, determined by the first."""
    return (a + 8) % V


def train(ref):
    """Factorised heads (table 0: b|a_t delayed, table 1: b|a_{t-1} undelayed) and a joint."""
    rng = random.Random(42)
    heads_a, heads_b = ref.init_counts(), ref.init_counts()
    joint = [[[1.0] * (V * V) for _ in range(V * V)]]
    for _ in range(500):
        a = ref.make_tokens(0, LENGTH, rng)
        b = [partner(x) for x in a]
        ref.update_counts(heads_a, a, 0)
        for t in range(1, LENGTH):
            ref.update_counts(heads_b, [a[t], b[t]], 0)
            ref.update_counts(heads_b, [a[t - 1], b[t]], 1)
        ref.update_counts(joint, [x * V + y for x, y in zip(a, b)], 0)
    return heads_a, heads_b, joint


def generate(ref, model, mode, rng):
    heads_a, heads_b, joint = model
    a, b = [0], [partner(0)]
    for _ in range(LENGTH - 1):
        if mode == "joint":
            frame = ref.sample_from(ref.probs(joint, 0, a[-1] * V + b[-1]), rng)
            a.append(frame // V)
            b.append(frame % V)
            continue
        nxt = ref.sample_from(ref.probs(heads_a, 0, a[-1]), rng)
        cond = (0, nxt) if mode == "delayed" else (1, a[-1])
        b.append(ref.sample_from(ref.probs(heads_b, *cond), rng))
        a.append(nxt)
    return a, b


def coupling(ref, model, mode):
    rng, hits = random.Random(3), 0
    for _ in range(CLIPS):
        a, b = generate(ref, model, mode, rng)
        hits += sum(y == partner(x) for x, y in zip(a[1:], b[1:]))
    return hits / (CLIPS * (LENGTH - 1))


def delayed_layout(a, b):
    """Frame t carries (a_t, b_{t-1}); one extra frame flushes the last b."""
    return list(zip([*a, None], [None, *b]))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    model = train(ref)
    seen = [i for i, row in enumerate(model[2][0]) if sum(row) > V * V]
    valid = sum(
        ref.probs(model[2], 0, c)[x * V + partner(x)] for c in seen for x in range(V)
    )
    a = ref.make_tokens(0, LENGTH, random.Random(9))
    frames = delayed_layout(a, [partner(x) for x in a])
    modes = ("delayed", "undelayed", "joint")
    return {
        "coupling": {m: coupling(ref, model, m) for m in modes},
        "head": ref.probs(model[1], 0, 5)[partner(5)],
        "joint_valid": valid / len(seen),
        "seen": len(seen),
        "offset_ok": all(f[1] == partner(g[0]) for g, f in itertools.pairwise(frames)),
        "frames": len(frames),
    }


def verify(result):
    c = result["coupling"]
    return [
        practice.Check(
            "ANSWER: delayed decoding keeps the two streams coupled",
            c["delayed"] > 0.95,
            f"{c['delayed']:.1%} of generated frames keep b_t as a_t's partner; the b head puts "
            f"{result['head']:.1%} of its mass on the right token, the rest is the add-one prior",
        ),
        practice.Check(
            "FINDING: without the delay coupling collapses to the closed-form 0.375",
            abs(c["undelayed"] - 0.375) < 0.03,
            f"emitting a_t and b_t in the same step couples them {c['undelayed']:.1%} of the "
            "time: b can only see a_{t-1} and guesses a's next step, and two independent draws "
            "from {1/4, 1/2, 1/4} agree with probability 0.375. One extra frame buys "
            f"{c['delayed'] / c['undelayed']:.1f}x",
        ),
        practice.Check(
            "FINDING: a 256-way joint predictor is worse still under add-one smoothing",
            c["joint"] < c["undelayed"] and result["joint_valid"] < 0.8,
            f"predicting (a, b) as one token couples {c['joint']:.1%}: the prior's 256 ones leave "
            f"{result['joint_valid']:.0%} of the average seen row on valid pairs, only {result['seen']} of "
            "256 contexts are ever seen, and one leak lands in a uniform row",
        ),
        practice.Check(
            "CONTROL: the delayed layout offsets b by exactly one frame",
            result["offset_ok"] and result["frames"] == LENGTH + 1,
            f"every frame's b slot is the partner of the previous frame's a, and {LENGTH} "
            f"frames take {result['frames']} delayed steps",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
