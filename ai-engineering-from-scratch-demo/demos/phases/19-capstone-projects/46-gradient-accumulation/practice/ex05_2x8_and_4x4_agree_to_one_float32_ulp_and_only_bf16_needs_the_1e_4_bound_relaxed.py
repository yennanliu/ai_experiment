"""Exercise 5 -- 2x8 and 4x4 agree to 9.3e-9 in float32, so nothing needs relaxing; in bf16 they differ by one ulp, 4.9e-4.

    Modify the equivalence check to compare two different micro splits (2 by 8 vs 4 by 4) and explain any tolerance you need to relax.

Reading of the exercise: "2 by 8" is read as accum_steps x micro_batch, so
2 micro-batches of 8 against 4 of 4, on `equivalence_check`'s own fixture:
seed 7, 16 samples, the 32-48-8 GELU net, loss divided by the number of
chunks. 8 x 2 is added in case "2 by 8" was meant the other way round. The
lesson's `equivalence_check` compares a split with the full batch, not two
splits with each other. So it is run as shipped at accum 2 and 4, and the
split-vs-split gradient comparison is done directly with the lesson's
`make_model`, `synthetic_batch` and `zero_grads`. The comparison is then
repeated in float64 and bfloat16 to find out when the lesson's 1e-4 bound
would have to move. Exact float32 values below were measured on this
machine. The checks assert bounds, because BLAS summation order is
platform-dependent.

**ANSWER: no tolerance needs relaxing in float32.** The shipped check gives
1.3e-8 against the full batch at both accum 2 and accum 4. The two splits
differ from each other by 9.3e-9, and 8 x 2 differs from 4 x 4 by 7.5e-9.
The largest gradient entry is 0.0717, so 9.3e-9 is about 1.3e-7 relative,
one float32 rounding step (eps 1.19e-7). That is roughly 10,000x inside the
lesson's 1e-4. The only difference between the splits is the order in which
per-chunk gradients are summed into `param.grad`. The maths is identical,
because each chunk's mean loss divided by the number of chunks gives the
same per-sample weight 1/16. In float64 the gap is 2.1e-17.

**FINDING: the bound has to move with the dtype, not with the split.** In
bfloat16 the same 2 x 8 vs 4 x 4 comparison gives 4.9e-4 = 2^-11. That is
exactly one bf16 step at the gradient's magnitude (0.0717 is in [2^-4,
2^-3), and bf16 keeps 7 fraction bits). The lesson's 1e-4 would fail it,
and a bound of about 1e-3 (2 ulps, about 1.4% of max|g|) is needed. A
tolerance written as k * eps(dtype) * max|g| would carry over. The lesson's
fixed absolute 1e-4 does not.

Expected output: two PASS checks.
"""

from __future__ import annotations

from harness import parity, practice

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "46-gradient-accumulation"
SPLITS = {"2x8": 2, "4x4": 4, "8x2": 8}
DTYPES = {"float32": torch.float32, "float64": torch.float64, "bfloat16": torch.bfloat16}


def grads(ref, accum, dtype):
    """equivalence_check's fixture, accumulated over `accum` equal chunks."""
    gen = torch.Generator()
    gen.manual_seed(7)
    x, y = ref.synthetic_batch(16, 32, 8, gen)
    ref.seed_everything(7)
    model = ref.make_model(32, 48, 8).to(dtype)
    ref.zero_grads(model)
    for cx, cy in zip(torch.split(x.to(dtype), 16 // accum), torch.split(y, 16 // accum)):
        (nn.CrossEntropyLoss()(model(cx), cy) / accum).backward()
    return [p.grad.double() for p in model.parameters()]


def gap(a, b):
    return max(float((u - v).abs().max()) for u, v in zip(a, b))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    out = {"shipped": {a: ref.equivalence_check(accum_steps=a)["max_grad_diff"] for a in (2, 4)}}
    for name, dtype in DTYPES.items():
        g = {k: grads(ref, a, dtype) for k, a in SPLITS.items()}
        out[name] = {"2x8": gap(g["2x8"], g["4x4"]), "8x2": gap(g["8x2"], g["4x4"])}
    out["gmax"] = max(float(t.abs().max()) for t in grads(ref, 1, torch.float32))
    return out


def verify(result):
    f32, f64, bf16, gmax = result["float32"], result["float64"], result["bfloat16"], result["gmax"]
    eps32 = torch.finfo(torch.float32).eps
    return [
        practice.Check(
            "ANSWER: 2x8 vs 4x4 agree to ~1 float32 ulp, 10,000x inside the lesson's 1e-4 -- nothing to relax",
            all(v < 1e-7 for v in (*result["shipped"].values(), *f32.values()))
            and max(f32.values()) < 4 * eps32 * gmax and all(v < 1e-15 for v in f64.values())
            and abs(gmax - 0.0717) < 1e-4,
            f"shipped check vs full {result['shipped']}; float32 vs 4x4 {f32} (max|g| {gmax:.4f}, "
            f"eps*max|g| {eps32 * gmax:.2g}); float64 {f64}",
        ),
        practice.Check(
            "FINDING: in bfloat16 the same splits differ by one bf16 ulp, 4.9e-4, and 1e-4 must go to ~1e-3",
            1e-4 < bf16["2x8"] <= 2 * 2**-11 and 0 < bf16["8x2"] <= 2 * 2**-11,
            f"bfloat16 vs 4x4 {bf16}; one bf16 ulp at max|g| = 2^-11 = {2**-11:.3g}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
