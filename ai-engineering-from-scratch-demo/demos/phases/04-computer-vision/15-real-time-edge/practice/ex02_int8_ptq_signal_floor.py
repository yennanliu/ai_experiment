"""Exercise 2 — int8 ptq signal floor.

    **(Medium)** Apply post-training static quantisation to `mobilenet_v3_small`. Report FP32 vs INT8 latency and accuracy loss on a held-out subset of CIFAR-10 or similar.

Reading of the exercise: "accuracy loss on a held-out subset of CIFAR-10" asks
for two things this environment will not supply -- a downloaded dataset and
trained weights -- so the held-out set here is 32 fixed Gaussian images and the
score is top-1 *agreement* between the FP32 model and its INT8 copy, which is the
quantity accuracy loss proxies anyway. Running it that way exposes a failure the
exercise does not anticipate. A freshly built `mobilenet_v3_small` emits logits
of order 1e-10: its 34 BatchNorm layers still hold their initial
`running_mean=0, running_var=1`, which in `.eval()` makes every one of them an
identity, so nothing arrests the ~0.5 gain of the 19 `Hardswish` activations and
9 `Hardsigmoid` squeeze-excite gates, and mean activation magnitude falls seven
orders across the 13 feature stages. Calibrating a quantiser against that gives
an INT8 model whose output is pure quantisation noise -- 0.0 dB SQNR, agreement
at chance -- while every mechanical part of the pipeline works perfectly. The
control repairs it without a single gradient step, by running eight
training-mode batches so BatchNorm re-estimates its running statistics;
agreement then goes to 1.000. Two details of the lesson's `quantise_ptq` snippet
had to change. Its `backend="x86"` names an engine this arm64 host does not
have, so the backend is chosen from `torch.backends.quantized.supported_engines`
and reported; and its eager `prepare`/`convert` pair inserts no `QuantStub`,
which a plain torchvision model does not carry either, so the converted graph
would have no quantised input to feed its quantised convolutions. The FX
graph-mode API is used instead, on the same four steps. Warnings are recorded
rather than hidden, all raised inside torch: a DeprecationWarning that
`torch.ao.quantization is deprecated and will be removed in 2.10`, a UserWarning
that `torch.quantize_per_tensor ... and other quantized tensor creation
functions` are deprecated, and -- on a qnnpack host -- one C++ log line from the
dynamic arm, `qnnpack incorrectly ignores reduce_range`, which goes straight to
stderr rather than through `warnings` and so is not in the count. Latency is
machine-dependent, so it is reported and only loosely bounded; the weight
lattice, the byte counts and the gate counts are exact.

Structure: `arm` runs the lesson's four PTQ steps through `quantize_fx` and
returns agreement, SQNR and the FP32 logit spread; `signal_audit` walks the 13
`features` blocks recording mean absolute activation and inventories the
BatchNorm, gate, conv and linear modules; `lattice` pulls the first packed INT8
weight and rebuilds it from `int_repr` and the recorded scale in both float32 and
float64; the two module-level lambdas collect modules by class and count their
parameters. `solve` runs the two static arms with the BatchNorm re-estimation
between them, then a dynamic-quantisation arm for contrast, and serialises and
times all three in one loop. At 146 code lines this sits above D14's 120-line target, 4 clear of the
ceiling: six checks over three quantisation arms and a 13-stage activation trace.
A seventh probe that ran the docs' eager recipe verbatim and recorded where it
fails was measured during development and cut to stay under the ceiling; nothing
below rests on it.
"""

from __future__ import annotations

import copy
import io
import warnings

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "15-real-time-edge"

CLASSES, SHAPE, WARMUP, ITERS, CALIB, CALIB_BATCH = 10, (1, 3, 224, 224), 3, 9, 8, 4
HOLDOUT, BN_BATCHES, BN_BATCH = 32, 8, 16

modules_of = lambda model, cls: [m for m in model.modules() if isinstance(m, cls)]  # noqa: E731
numel_of = lambda modules: sum(p.numel() for m in modules for p in m.parameters())  # noqa: E731


def arm(torch, tq, qfx, model, data, backend) -> dict:
    calib, holdout = data
    prepared = qfx.prepare_fx(copy.deepcopy(model).eval(), tq.get_default_qconfig_mapping(backend), (calib[:1],))
    with torch.no_grad():
        for start in range(0, len(calib), CALIB_BATCH):
            prepared(calib[start:start + CALIB_BATCH])
        quantised = qfx.convert_fx(prepared)
        fp32, int8 = model(holdout), quantised(holdout)
    noise = (fp32 - int8).pow(2).mean()
    return {"quantised": quantised, "logit_std": fp32.std().item(),
            "agree": (fp32.argmax(1) == int8.argmax(1)).float().mean().item(),
            "sqnr": (10 * torch.log10(fp32.pow(2).mean() / noise)).item()}


def signal_audit(torch, model, sample) -> dict:
    names = [type(module).__name__ for module in model.modules()]
    norms, linears = modules_of(model, torch.nn.BatchNorm2d), modules_of(model, torch.nn.Linear)
    stages, hidden, small = [], sample, torch.randn(4096) * 1e-4
    with torch.no_grad():
        for block in model.features:
            hidden = block(hidden)
            stages.append(hidden.abs().mean().item())
        gain = torch.nn.functional.hardswish(small).abs().mean() / small.abs().mean()
    return {"stages": stages, "hs_gain": float(gain), "bn": len(norms), "convs": names.count("Conv2d"),
            "hardswish": names.count("Hardswish"), "hardsigmoid": names.count("Hardsigmoid"),
            "linears": len(linears), "lin_share": numel_of(linears) / numel_of([model]),
            "bn_default": all(bool((n.running_mean == 0).all()) for n in norms)}


def lattice(torch, quantised) -> dict:
    getters = (getattr(module, "weight", None) for module in quantised.modules())
    packed = next(get for get in getters if callable(get) and get().is_quantized)()
    ints, scale, zero, exact = packed.int_repr(), packed.q_scale(), packed.q_zero_point(), packed.dequantize()
    return {"levels": int(ints.unique().numel()), "lo": int(ints.min()), "hi": int(ints.max()),
            "dtype": str(packed.dtype), "scheme": str(packed.qscheme()).split(".")[-1], "scale": scale,
            "err32": float((exact - (ints.float() - zero) * scale).abs().max()),
            "err64": float((exact.double() - (ints.double() - zero) * scale).abs().max())}


def solve():
    try:
        import torch
        import torch.ao.quantization as tq
        from torch.ao.quantization import quantize_fx as qfx
        from torchvision.models import mobilenet_v3_small
    except ImportError as exc:                      # pragma: no cover - T1 needs torch+torchvision
        raise practice.Skip(f"needs torchvision: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    torch.manual_seed(0)
    engines = list(torch.backends.quantized.supported_engines)
    torch.backends.quantized.engine = backend = "x86" if "x86" in engines else engines[0]
    base = mobilenet_v3_small(weights=None, num_classes=CLASSES).eval()
    data = (torch.randn(CALIB, *SHAPE[1:]), torch.randn(HOLDOUT, *SHAPE[1:]))
    repaired, size, speed = copy.deepcopy(base).train(), {}, {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        raw = arm(torch, tq, qfx, base, data, backend)
        with torch.no_grad():                       # BatchNorm re-estimation: no optimiser, no gradient
            for _ in range(BN_BATCHES):
                repaired(torch.randn(BN_BATCH, *SHAPE[1:]))
        fixed = arm(torch, tq, qfx, repaired.eval(), data, backend)
        dynamic = tq.quantize_dynamic(copy.deepcopy(repaired), {torch.nn.Linear}, dtype=torch.qint8)
        for name, model in (("fp32", repaired), ("int8", fixed["quantised"]), ("dyn", dynamic)):
            torch.save(model.state_dict(), blob := io.BytesIO())
            size[name] = blob.getbuffer().nbytes
            speed[name] = ref.measure_latency(model, SHAPE, warmup=WARMUP, iters=ITERS)["p50_ms"]
    return {"raw": raw, "fixed": fixed, "engines": engines, "backend": backend, "bytes": size,
            "speeds": speed, "lattice": lattice(torch, fixed["quantised"]), **signal_audit(torch, base, data[1][:4]),
            "warnings": "; ".join(sorted({f"{w.category.__name__}: {str(w.message)[:52]}" for w in caught}))}


def verify(result):
    raw, fixed, lat, size, speed = (result[k] for k in ("raw", "fixed", "lattice", "bytes", "speeds"))
    shrink, stages = size["fp32"] / size["int8"], result["stages"]
    survival, gates = stages[-1] / stages[0], result["hardswish"] + result["hardsigmoid"]
    return [
        practice.Check(
            "ANSWER: INT8 is 3.8x smaller on disk and loses no top-1 agreement once the model has a signal",
            all([shrink > 3.5, fixed["agree"] == 1.0, speed["int8"] < speed["fp32"]]),
            f"`quantize_fx` static PTQ on `{result['backend']}` (chosen from {result['engines']}; the docs' "
            f"hardcoded x86 is not offered here): {size['fp32']:,} -> {size['int8']:,} bytes ({shrink:.2f}x), p50 "
            f"{speed['fp32']:.1f} -> {speed['int8']:.1f} ms ({speed['fp32'] / speed['int8']:.1f}x, machine-"
            f"dependent), agreement {fixed['agree']:.3f} over {HOLDOUT} images at SQNR {fixed['sqnr']:.1f} dB -- "
            f"agreement stands in for accuracy loss, CIFAR-10 and trained weights being unreachable"),
        practice.Check(
            "FINDING: quantise the model as built and the INT8 output is pure noise — 0.0 dB SQNR",
            all([raw["sqnr"] < 1.0, raw["agree"] < 0.5, raw["logit_std"] < 1e-8]),
            f"the same pipeline on the freshly built network scores agreement {raw['agree']:.3f} at "
            f"{raw['sqnr']:.2f} dB, because its FP32 logits have standard deviation {raw['logit_std']:.2e} -- the "
            f"step is wider than the signal. Nothing failed; a report omitting SQNR shows {shrink:.1f}x and no more"),
        practice.Check(
            "MECHANISM: untrained BatchNorm is the identity in eval, so 28 halving gates go unopposed",
            all([result["bn_default"], survival < 1e-5, 0.4 < result["hs_gain"] < 0.6]),
            f"all {result['bn']} BatchNorm2d layers still hold running_mean=0 (and running_var=1), an identity map "
            f"in `.eval()`, while the net has {result['hardswish']} Hardswish and {result['hardsigmoid']} "
            f"Hardsigmoid gates of measured small-signal gain {result['hs_gain']:.3f} each. Mean |activation| over "
            f"the 13 stages: {' '.join(f'{i}:{v:.1e}' for i, v in enumerate(stages))} -- a surviving fraction of "
            f"{survival:.1e}, within {survival / 0.5 ** gates:.0f}x of the {0.5 ** gates:.1e} {gates} halvings give"),
        practice.Check(
            "CONTROL: eight training-mode batches and no gradient step repair it — agreement 0.000 -> 1.000",
            all([fixed["logit_std"] > 1e4 * raw["logit_std"], fixed["sqnr"] > 20, raw["agree"] < fixed["agree"]]),
            f"re-estimating the BatchNorm statistics over {BN_BATCHES} forward passes of {BN_BATCH} images lifts "
            f"the FP32 logit spread {raw['logit_std']:.2e} -> {fixed['logit_std']:.2e}, the INT8 copy "
            f"{raw['sqnr']:.2f} -> {fixed['sqnr']:.1f} dB, agreement {raw['agree']:.3f} -> {fixed['agree']:.3f}"),
        practice.Check(
            "MECHANISM: the INT8 weights sit exactly on their scale lattice — the float32 rebuild is bitwise",
            all([lat["err32"] == 0.0, lat["levels"] <= 256, lat["lo"] >= -128, lat["hi"] <= 127]),
            f"the first packed weight is {lat['dtype']} under {lat['scheme']}, scale {lat['scale']:.3e}; rebuilding "
            f"it as `(int_repr - zero_point) * scale` reproduces `dequantize()` with max error {lat['err32']:.1f} in "
            f"float32 and {lat['err64']:.2e} in float64 -- float32 rounding, {lat['err64'] / lat['scale']:.1e} of a "
            f"step. Its integers span [{lat['lo']}, {lat['hi']}] over {lat['levels']} of the 256 values 8 bits allow"),
        practice.Check(
            "CONTROL: dynamic quantisation, the cheap option, compresses 1.4x and never touches a convolution",
            all([size["dyn"] > 2 * size["int8"], speed["dyn"] > 0.5 * speed["fp32"], result["linears"] == 2]),
            f"`quantize_dynamic(model, {{nn.Linear}})` on the same repaired weights reaches {size['dyn']:,} bytes "
            f"({size['fp32'] / size['dyn']:.2f}x, against {shrink:.2f}x static) at p50 {speed['dyn']:.1f} ms "
            f"({speed['fp32'] / speed['dyn']:.2f}x). It touches only the {result['linears']} nn.Linear layers, with "
            f"{result['lin_share']:.1%} of the parameters and almost none of the work, leaving all "
            f"{result['convs']} convolutions FP32 -- the docs' \"small speedup\" is here no speedup. Python "
            f"warnings recorded from torch across all three arms: {result['warnings']}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
