"""Exercise 2 — the interaction is exactly additive, and nothing here can make it otherwise.

    **Medium.** Train two separate LoRAs on two target transforms. Load them
    together and show their additive interaction. When does the interaction break
    linearity?

Reading of the exercise: two adapters are fitted independently on the same frozen
`W`, each to its own target delta, using the lesson's own `lora_forward` and the
same gradient `train_lora` uses. "Load them together" is then read literally --
`W + B1 A1 + B2 A2` -- and compared against applying each adapter separately and
summing what they added.

**ANSWER: it never breaks.** The two adapters are merged into one by
concatenating their factors -- `A = [A1; A2]`, `B = [B1 | B2]`, so
`B A = B1 A1 + B2 A2` -- and pushed through the lesson's own `lora_forward` in a
single call. Over PROBES random inputs that merged call and the sum of the two
separate contributions agree to **8.9e-16**. There is no weight, no scale and
no input at which the two interact.

**FINDING: that is a property of the code, not a result about LoRA.**
`lora_forward` computes `(W + alpha * B A) x`, which is **affine in the
adapters**: stacking two of them gives `(W + a1 B1 A1 + a2 B2 A2) x` by the
distributive law alone. Linearity cannot break because there is no nonlinearity
between the adapters to break it. The lesson's entire module contains no
activation function -- `matmul_mat_vec`, `outer`, `zeros` and `randn_matrix` are
the only operations, and all four are linear.

**FINDING: the exercise is describing real models, where the answer is different.**
In a transformer the adapted projections sit either side of softmax attention and
a nonlinear FFN, so two adapters that each shift a layer's output compose through
those nonlinearities and their effects stop adding. The question "when does the
interaction break linearity" is a good one; this code cannot be asked it, and
the honest answer is to say which of the two is missing.

**FINDING: the scaling `alpha` is exactly a scalar on the delta.** Doubling
`alpha` doubles the adapter's contribution to **1e-15**, at every probe, so the
"weight" knobs of a real stack are here a linear mixing coefficient and nothing
more.

**CONTROL: each adapter reproduces its own target alone.** Both target deltas
are built rank-2 so a rank-4 adapter can fit them exactly, and fitted separately
each matches its own target to better than **1e-8** mean squared error. The
additivity above holds between two adapters that each work, not two that both do
nothing.

Structure: `fit` trains one adapter with the lesson's own gradient; `stacked`
applies both at once; `separate` applies each and sums the contributions.
"""

from __future__ import annotations

import statistics

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "08-controlnet-lora-conditioning"
DIM, RANK, STEPS, RATE, PROBES = 6, 4, 4_000, 0.01, 300


def descend(a_mat, b_mat, err, proj, x):
    """The lesson's own LoRA gradient step, on both factors."""
    for i in range(DIM):
        for k in range(RANK):
            b_mat[i][k] -= RATE * err[i] * proj[k]
    for k in range(RANK):
        pull = sum(err[i] * b_mat[i][k] for i in range(DIM))
        for j in range(DIM):
            a_mat[k][j] -= RATE * pull * x[j]


def fit(ref, random, frozen, target, seed):
    """One adapter, trained by the gradient the lesson's own train_lora uses."""
    rng = random.Random(seed)
    a_mat = ref.randn_matrix(RANK, DIM, rng, scale=0.2)
    b_mat = [[0.0] * RANK for _ in range(DIM)]
    for _ in range(STEPS):
        x = [rng.gauss(0, 1) for _ in range(DIM)]
        want = ref.matmul_mat_vec(target, x)
        err = [p - t for p, t in zip(ref.lora_forward(frozen, a_mat, b_mat, x), want)]
        descend(a_mat, b_mat, err, ref.matmul_mat_vec(a_mat, x), x)
    return a_mat, b_mat


def contribution(ref, pair, x, alpha=1.0):
    """What one adapter adds to the frozen output: alpha * B A x."""
    return [alpha * v for v in ref.matmul_mat_vec(pair[1], ref.matmul_mat_vec(pair[0], x))]


def merge(first, second):
    """One adapter whose product is B1 A1 + B2 A2, by concatenating the factors."""
    return first[0] + second[0], [row_a + row_b for row_a, row_b in zip(first[1], second[1])]


def orthonormal(rng, count):
    """`count` orthonormal vectors, by Gram-Schmidt on random draws."""
    frame = []
    for _ in range(count):
        vec = [rng.gauss(0, 1) for _ in range(DIM)]
        for prior in frame:
            overlap = sum(a * b for a, b in zip(vec, prior))
            vec = [a - overlap * b for a, b in zip(vec, prior)]
        norm = sum(a * a for a in vec) ** 0.5
        frame.append([a / norm for a in vec])
    return frame


def low_rank(rng, scale):
    """A rank-2 delta, so a rank-RANK adapter can fit it exactly."""
    left, right = orthonormal(rng, 2), orthonormal(rng, 2)
    return [[scale * (left[0][i] * right[0][j] + 0.6 * left[1][i] * right[1][j])
             for j in range(DIM)] for i in range(DIM)]


def probe(ref, frozen, first, second, merged, count=PROBES, seed=99):
    """(worst merge gap, worst alpha-scaling gap) over random inputs."""
    rng, gap, scaled = __import__("random").Random(seed), 0.0, 0.0
    for _ in range(count):
        x = [rng.gauss(0, 1) for _ in range(DIM)]
        base = ref.matmul_mat_vec(frozen, x)
        apart = [b + u + v for b, u, v in
                 zip(base, contribution(ref, first, x), contribution(ref, second, x))]
        together = ref.lora_forward(frozen, merged[0], merged[1], x)
        gap = max(gap, max(abs(a - b) for a, b in zip(apart, together)))
        scaled = max(scaled, max(abs(2 * a - b) for a, b in
                                 zip(contribution(ref, first, x),
                                     contribution(ref, first, x, 2.0))))
    return gap, scaled


def accuracy(ref, frozen, pair, target, count=PROBES, seed=7):
    """Mean squared error of one adapter against its own target, alone."""
    rng, total = __import__("random").Random(seed), 0.0
    for _ in range(count):
        x = [rng.gauss(0, 1) for _ in range(DIM)]
        want = ref.matmul_mat_vec(target, x)
        got = ref.lora_forward(frozen, pair[0], pair[1], x)
        total += sum((p - q) ** 2 for p, q in zip(got, want))
    return total / count


def solve():
    import random
    ref = parity.load_reference(PHASE, LESSON, "main")
    rng = random.Random(3)
    frozen = ref.randn_matrix(DIM, DIM, rng, scale=0.5)
    deltas = [low_rank(rng, 0.8), low_rank(rng, 0.6)]
    targets = [[[frozen[i][j] + d[i][j] for j in range(DIM)] for i in range(DIM)] for d in deltas]
    first = fit(ref, random, frozen, targets[0], 5)
    second = fit(ref, random, frozen, targets[1], 6)
    gap, scaled = probe(ref, frozen, first, second, merge(first, second))
    return {
        "gap": gap, "scaling": scaled,
        "alone": [accuracy(ref, frozen, p, t) for p, t in zip((first, second), targets)],
        "linear_only": [n for n in ("tanh", "leaky", "sigmoid", "relu") if hasattr(ref, n)],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the interaction is exactly additive -- it never breaks",
            result["gap"] < 1e-12,
            f"the two adapters are merged by concatenating their factors, so B A = B1A1 + B2A2, "
            f"and pushed through the lesson's own lora_forward in one call. Over {PROBES} random "
            f"inputs that merged call and the sum of the two separate contributions agree to "
            f"{result['gap']:.1e}. There is no weight, no scale and no input at which they interact",
        ),
        practice.Check(
            "FINDING: that is a property of this code, not a result about LoRA",
            result["linear_only"] == [],
            f"lora_forward computes (W + alpha * B A) x, which is affine in the adapters, so "
            f"stacking gives (W + B1A1 + B2A2) x by the distributive law alone. The module "
            f"exposes {result['linear_only']} of tanh, leaky, sigmoid and relu: there is no "
            "nonlinearity anywhere between the adapters for linearity to break through",
        ),
        practice.Check(
            "FINDING: alpha is exactly a scalar on the delta",
            result["scaling"] < 1e-12,
            f"doubling alpha doubles the adapter's contribution to {result['scaling']:.1e} at "
            "every probe, so the weight knobs a real stack exposes are, here, a linear mixing "
            "coefficient and nothing more. In a real model they also change what the following "
            "nonlinearity sees, which is where the interaction the exercise asks about lives",
        ),
        practice.Check(
            "CONTROL: each adapter reproduces its own target alone",
            max(result["alone"]) < 1e-8,
            f"both target deltas are built rank-2 so a rank-{RANK} adapter can fit them exactly, "
            f"and fitted separately they match to {result['alone'][0]:.1e} and "
            f"{result['alone'][1]:.1e} mean squared error. The additivity above holds between two "
            "adapters that each work, not two that both do nothing",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
