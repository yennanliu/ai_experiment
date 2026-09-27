"""Exercise 3 — 2x-longer rejected answers push DPO's chosen drop from 0.18 to 1.94 nats; normalized SimPO is unchanged.

    Make the rejected responses on average 2x longer than chosen. Without
    changing anything else, show DPO's length exploitation numerically and
    SimPO's fix.

Reading of the exercise: the toy has no lengths. Its Policy scores one action
per response, and SimPO's `lens = [1, 1, 1, 1]` is a constant that DPO never
reads. So length is added the way a language model has it: a response of L
tokens has log pi(y) = L x log pi(token). Nothing else changes: the same 500
seed-1 pairs, the shipped `train_dpo` and `train_simpo` with default
hyperparameters, and training RNG seeded 2. Each pair gets lengths from a
separate RNG: chosen 1-3 tokens, rejected 2-6, so rejected responses are 2.04x
longer on average. A control draws both from 1-5 (ratio 1.02), which keeps
the total length about the same.

**ANSWER: DPO spends its margin on length.** With equal lengths DPO ends at
pi(best) 0.678, with a mean chosen log-prob drift of -0.183 nats. With 2x-longer
rejected responses the drift is -1.938, 10.6x the drop. The same-total-length
control gives -0.677, so long *rejected* responses alone make it 2.9x worse
than the control. The implicit reward is beta x L x log-ratio, so the mean
sequence-level gap is 11.164 nats on a per-token gap of 1.779. Six times the
margin comes from counting tokens.

**ANSWER: SimPO's normalization is an exact fix.** When `lens` is wired to each
response's real length, SimPO's final logits match its equal-length run to
within 1e-12, for both length schemes: pi(best) 0.509, drift -0.005.

**FINDING: unnormalized, a reference-free loss is gamed at step 0.** The shipped
SimPO (lens fixed at 1) on summed log-probs starts with a mean margin of
+4.35, against -0.325 normalized. Length has already won most of the
preference, so SimPO learns a third as much (token gap 0.107 against 0.328)
and ends at pi(best) 0.344. DPO starts at margin 0 whatever the lengths,
because policy and reference start from the same logits. Its exploitation
comes during training.

Structure: `Response` is an int that carries a token length, so the reference
functions run unchanged; `solve()` patches the reference Policy (restored
after) so that log pi(y) sums per-token log-probs;
`install_normalized_simpo()` re-executes `train_simpo` with its one `lens` line
pointed at the real lengths.
"""

from __future__ import annotations

import inspect
import random

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "03-direct-preference-optimization-family"


class Response(int):
    """An action index that also carries the token length of the sampled response."""

    def __new__(cls, action, length):
        obj = int.__new__(cls, action)
        obj.length = length
        return obj


def tokens(a):
    return getattr(a, "length", 1)


def install_normalized_simpo(ref):
    """train_simpo with its `lens` hook reading each response's real length."""
    src = inspect.getsource(ref.train_simpo)
    src = src.replace("lens = [1, 1, 1, 1]", "lens = _LENGTHS").replace("def train_simpo(", "def _simpo_norm(")
    ref._LENGTHS = type("Lengths", (), {"__getitem__": lambda _, a: tokens(a)})()
    exec(src, vars(ref))  # noqa: S102 - the reference's own function, one line changed
    return ref._simpo_norm


def lengthen(pairs, g, chosen, rejected):
    """The pairs with token lengths drawn from the two ranges, and rejected/chosen length."""
    out = [(Response(w, g.randint(*chosen)), Response(lo, g.randint(*rejected)), s)
           for w, lo, s in pairs]
    return out, round(sum(lo.length for _, lo, _ in out) / sum(w.length for w, _, _ in out), 2)


def summary(ref, pi, pairs):
    """(pi(best), mean chosen drift, mean token gap, mean sequence gap), in nats."""
    base = ref.logsoftmax(ref.make_policy_and_ref()[1].logits)
    d = [a - b for a, b in zip(ref.logsoftmax(pi.logits), base)]
    rows = [(d[w], d[w] - d[lo], tokens(w) * d[w] - tokens(lo) * d[lo]) for w, lo, _ in pairs]
    return (round(ref.softmax(pi.logits)[1], 3),
            *(round(sum(col) / len(rows), 3) for col in zip(*rows)))


def start_margin(ref, pairs, n=tokens):
    """SimPO's mean margin at step 0 on summed log-probs, divided by n(y) tokens."""
    k = inspect.signature(ref.train_simpo).parameters
    lp = ref.logsoftmax(ref.make_policy_and_ref()[0].logits)
    m = [k["beta"].default * (tokens(w) * lp[w] / n(w) - tokens(lo) * lp[lo] / n(lo))
         for w, lo, _ in pairs]
    return round(sum(m) / len(m) - k["gamma"].default, 3)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    saved = ref.random, ref.Policy.logprob, ref.Policy.grad_logprob
    norm, simpo, dpo = install_normalized_simpo(ref), ref.train_simpo, ref.train_dpo
    ref.Policy.logprob = lambda pi, a: tokens(a) * saved[1](pi, a)
    ref.Policy.grad_logprob = lambda pi, a: [tokens(a) * g for g in saved[2](pi, a)]
    try:
        ref.random, g = random.Random(1), random.Random(3)
        pairs = [ref.sample_pref_pair() for _ in range(500)]
        (longer, ratio), (same, same_ratio) = lengthen(pairs, g, (1, 3), (2, 6)), lengthen(
            pairs, g, (1, 5), (1, 5))
        runs = {"dpo": (dpo, pairs), "dpo_long": (dpo, longer), "dpo_same": (dpo, same),
                "simpo": (simpo, pairs), "simpo_raw": (simpo, longer),
                "simpo_norm": (norm, longer), "simpo_norm_same": (norm, same)}
        pis = {}
        for name, (fn, ps) in runs.items():
            ref.random = random.Random(2)
            pis[name] = fn(ps)
    finally:
        ref.random, ref.Policy.logprob, ref.Policy.grad_logprob = saved
    return {
        "ratio": ratio, "same_ratio": same_ratio,
        "start": (start_margin(ref, longer, lambda _: 1), start_margin(ref, longer)),
        "stats": {k: summary(ref, pi, runs[k][1]) for k, pi in pis.items()},
        "max_diff": max(abs(a - b) for k in ("simpo_norm", "simpo_norm_same")
                        for a, b in zip(pis[k].logits, pis["simpo"].logits)),
        "dpo_lens": "lens" in inspect.getsource(ref.train_dpo),
        "same_start": len({tuple(p.logits) for p in ref.make_policy_and_ref()}) == 1,
    }


def verify(result):
    st = result["stats"]
    dpo, long_, same, raw = st["dpo"], st["dpo_long"], st["dpo_same"], st["simpo_raw"]
    return [
        practice.Check(
            "ANSWER: DPO spends its margin on length",
            (result["ratio"], result["same_ratio"], dpo[:2], long_[1:], same[1],
             round(long_[1] / dpo[1], 1), round(long_[1] / same[1], 1), round(long_[3] / long_[2]))
            == (2.04, 1.02, (0.678, -0.183), (-1.938, 1.779, 11.164), -0.677, 10.6, 2.9, 6),
            f"length ratio {result['ratio']} (control {result['same_ratio']}); (pi best, drift, "
            f"token gap, sequence gap) equal {dpo}, 2x {long_}, control {same}",
        ),
        practice.Check(
            "ANSWER: SimPO's normalization is an exact fix",
            result["max_diff"] < 1e-12 and st["simpo_norm"][:2] == st["simpo"][:2] == (0.509, -0.005),
            f"max |logit diff| vs equal length {result['max_diff']:.1e}; {st['simpo_norm']}",
        ),
        practice.Check(
            "FINDING: unnormalized, a reference-free loss is gamed at step 0",
            (result["start"], raw[0], raw[2], st["simpo"][2], round(raw[2] / st["simpo"][2], 2),
             result["dpo_lens"], result["same_start"])
            == ((4.35, -0.325), 0.344, 0.107, 0.328, 0.33, False, True),
            f"step-0 margin raw vs normalized {result['start']}; raw run {raw}; train_dpo reads no lens",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
