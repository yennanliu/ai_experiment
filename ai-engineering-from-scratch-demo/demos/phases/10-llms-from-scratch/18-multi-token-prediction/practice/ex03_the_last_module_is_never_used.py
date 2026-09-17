"""Exercise 3 — `modules[D-1]` is applied and discarded, so at D=1 the MTP module has no effect at all.

    Implement D=2 in the toy: add a second MTP module that takes h^(1) and
    predicts `t_{i+2}`. Verify the joint loss and the parameter accounting match
    the DeepSeek paper's equations 19-21.

Reading of the exercise: `mtp_loss` already takes a list of modules and already
loops `k` from 1 to `D`, so "implement D=2" is a matter of passing two modules --
which is what `main` does. The verification is therefore the interesting half,
and it is done by replacing each module with a completely different one and
checking which losses move.

**ANSWER: D=2 works, and the second module contributes nothing.** Replacing
`modules[1]` with a module built from a different seed leaves both per-depth
losses **bit-identical** -- `3.5129, 3.2347` either way. Replacing `modules[0]`
moves the depth-2 loss from 3.2347 to 3.3433 and leaves depth 1 alone.

**MECHANISM: the loop scores before it advances.** For each `k`, `mtp_loss`
computes `shared_head_logits(h_prev, E)` and *then* sets
`h_prev = mtp_forward(h_prev, ..., modules[k-1])`. So the depth-`k` loss is read
off `h^(k-1)`, module `k-1` is applied afterwards, and the last module's output
is never scored. At D=1 that means **the MTP module is not in the loss**: two
completely different modules give the same 3.579055.

**FINDING: this is off by one against equations 19-21.** DeepSeek's equation 20
is `p_{i+k} = OutHead(h_i^(k))` -- the depth-`k` prediction comes from the depth-
`k` module's *output*. The toy predicts `t_{i+k}` from `h^(k-1)`, so depth 1 is
the backbone's own next-token head with no module involved, and depth 2 is
module 1's. Every depth is shifted one module earlier than the paper's.

**FINDING: the parameter accounting is for `D` modules and the loss uses `D-1`.**
`count_parameters` charges `D * per_mtp`; `mtp_loss` reads `D-1` of them. At the
toy's D=2 that is 2 modules charged and 1 used, and at the D=1 the exercise's
own parameter table uses throughout, it is **1 charged and 0 used** -- so the
overhead in Exercise 2 buys nothing that appears in the loss in Exercise 1.

Structure: `losses` runs the lesson's own `mtp_loss` with one module swapped out;
`swap` builds a replacement module from a different seed.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "18-multi-token-prediction"
VOCAB, HIDDEN, FF, SEQ, DEPTHS, LAM = 32, 8, 16, 12, 2, 0.3
SEED, NOISE_SEED, NOISE, OTHER = 23, 42, 0.15, 999


def setup(ref):
    rng = random.Random(SEED)
    embeddings = ref.rand_matrix(VOCAB, HIDDEN, rng, scale=0.2)
    tokens = [rng.randrange(VOCAB) for _ in range(SEQ)]
    modules = [ref.make_mtp_module(HIDDEN, FF, rng) for _ in range(DEPTHS)]
    noise = random.Random(NOISE_SEED)
    hidden = [ref.rms_norm(ref.add(embeddings[tokens[i]],
                                   [noise.gauss(0, NOISE) for _ in range(HIDDEN)]))
              for i in range(SEQ)]
    return embeddings, tokens, modules, hidden


def swap(ref, modules, index, seed=OTHER):
    """The same module list with one entry replaced by an unrelated module."""
    replaced = list(modules)
    replaced[index] = ref.make_mtp_module(HIDDEN, FF, random.Random(seed))
    return replaced


def losses(ref, hidden, tokens, modules, embeddings):
    return ref.mtp_loss(hidden, tokens, modules, embeddings, LAM)[1]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    embeddings, tokens, modules, hidden = setup(ref)
    base = losses(ref, hidden, tokens, modules, embeddings)
    one_module = [ref.make_mtp_module(HIDDEN, FF, random.Random(seed)) for seed in (7, 77)]
    return {
        "base": base,
        "swap_last": losses(ref, hidden, tokens, swap(ref, modules, -1), embeddings),
        "swap_first": losses(ref, hidden, tokens, swap(ref, modules, 0), embeddings),
        "depth_one": [losses(ref, hidden, tokens, [module], embeddings)[0]
                      for module in one_module],
        "charged": ref.count_parameters(VOCAB, HIDDEN, FF, 2, DEPTHS).mtp_total,
        "per_module": ref.count_parameters(VOCAB, HIDDEN, FF, 2, 1).per_mtp,
        "depths": DEPTHS,
    }


def verify(result):
    base, last, first = result["base"], result["swap_last"], result["swap_first"]
    one, two = result["depth_one"]
    return [
        practice.Check(
            "ANSWER: D=2 works and the second module contributes nothing",
            base == last and base != first,
            f"the per-depth losses are {base[0]:.4f}, {base[1]:.4f}. Replacing modules[1] with a "
            f"module built from a different seed gives {last[0]:.4f}, {last[1]:.4f} -- "
            f"bit-identical. Replacing modules[0] gives {first[0]:.4f}, {first[1]:.4f}, moving "
            f"depth 2 by {first[1] - base[1]:+.4f} and leaving depth 1 untouched. Only the first "
            f"of the {result['depths']} modules is in the loss",
        ),
        practice.Check(
            "MECHANISM: the loop scores before it advances, so the last module is discarded",
            one == two,
            "for each k, mtp_loss computes shared_head_logits(h_prev, E) and then sets "
            "h_prev = mtp_forward(h_prev, ..., modules[k-1]), so the depth-k loss is read off "
            f"h^(k-1) and module k-1 is applied afterwards. At D=1 that means the module is not "
            f"in the loss at all: two completely unrelated modules both give {one:.6f}",
        ),
        practice.Check(
            "FINDING: this is off by one against equations 19-21",
            base[0] == last[0] == first[0],
            "DeepSeek's equation 20 is p_(i+k) = OutHead(h_i^(k)) -- the depth-k prediction comes "
            "from the depth-k module's output. The toy predicts t_(i+k) from h^(k-1), so depth 1 "
            f"is the backbone's own next-token head with no module involved, which is why all "
            f"three arms above report {base[0]:.4f} at depth 1 whatever modules they were given. "
            "Every depth is shifted one module earlier than the paper's",
        ),
        practice.Check(
            "FINDING: the accounting charges D modules and the loss reads D-1",
            result["charged"] == result["depths"] * result["per_module"],
            f"count_parameters charges D * per_mtp -- {result['depths']} x "
            f"{result['per_module']:,} = {result['charged']:,} for the toy -- and mtp_loss reads "
            f"{result['depths'] - 1} of them. At the D=1 the exercise's own parameter table uses "
            "throughout, that is 1 module charged and 0 used, so the overhead priced in "
            "Exercise 2 buys nothing that appears in the loss in Exercise 1",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
