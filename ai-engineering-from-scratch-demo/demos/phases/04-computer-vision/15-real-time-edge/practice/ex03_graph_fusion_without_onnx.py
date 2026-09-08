"""Exercise 3 — graph fusion without onnx.

    **(Hard)** Export `convnext_tiny` to ONNX, run it through `onnxruntime` with the `CPUExecutionProvider`, and compare latency to the PyTorch eager baseline. Identify the first layer where ONNX Runtime is faster and explain why.

Reading of the exercise: the first half cannot be run here at all and the second
half is not well posed. Neither `onnx` nor `onnxscript` nor `onnxruntime` is
installed and none may be fetched, so both of `torch.onnx.export`'s paths refuse
-- the dynamo default for want of `onnxscript`, the legacy path for want of
`onnx` -- and the exact refusals are measured below rather than described. What
survives is the question underneath: what does a graph-optimising runtime do to
a model, and would it help this one? "The first layer where ONNX Runtime is
faster" presumes the two graphs have the same layers, which is exactly what a
fusing runtime destroys; the stand-in used here is torch's own freezer,
`torch.jit.freeze` plus `optimize_for_inference`, the one graph optimiser
available offline. It reproduces convnext_tiny's logits to well inside 1e-6 and
buys no meaningful speed, and counting the nodes shows why: it deletes 359
bookkeeping nodes and not one compute node, and splits `aten::linear` into
`aten::matmul` plus `aten::add`, so the optimised graph carries *more* compute
ops than the traced one. The fusion ONNX Runtime is usually credited for, folding
BatchNorm into the preceding convolution -- scaling the conv weight by
`gamma / sqrt(var + eps)` and rolling `beta - mean * that` into the bias -- cannot
fire on convnext_tiny at all: it has 22 convolutions and zero BatchNorm layers,
normalising with LayerNorm over a permuted (N,H,W,C) tensor instead. The control
does that fold on resnet18 with torch's own `fuse_conv_bn_eval`, where it is
exact. Six warnings are recorded rather than silenced, all raised inside torch:
that `torch.jit.trace` and `torch.jit.trace_method` are not supported on Python
3.14+, that `torch.jit.freeze` and `torch.jit.optimize_for_inference` are
deprecated in favour of `torch.compile`, and two from the legacy ONNX exporter
before it gives up. Latency is wall-clock and machine-dependent, so only a wide
bound is asserted on it; the node counts, module counts and fold error are exact.

Structure: `kinds` counts graph nodes by op and `census` counts modules by class
name; `onnx_attempts` runs both export paths and records what each raises;
`jit_probe` traces, freezes and optimises the model, compares outputs and times
both with the lesson's own `measure_latency`; `fold_model` walks a network
replacing every Conv2d->BatchNorm2d pair with the single equivalent convolution.
At 139 code lines this sits above D14's 120-line target, 11 clear of the ceiling:
six checks over two export paths, two graph snapshots and a fold on a second net.
"""

from __future__ import annotations

import collections
import copy
import importlib.util
import io
import warnings

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "15-real-time-edge"

RES, FOLD_RES, WARMUP, ITERS, OPSET, BN_BATCHES = 160, 64, 2, 7, 17, 4
PARTS = ("aten::permute", "aten::layer_norm", "aten::_convolution", "aten::gelu")
SHAPES = ("Conv2d", "BatchNorm2d", "LayerNorm", "LayerNorm2d", "Permute", "Linear")

kinds = lambda graph: collections.Counter(node.kind() for node in graph.nodes())  # noqa: E731
census = lambda model, name: sum(1 for m in model.modules() if type(m).__name__ == name)  # noqa: E731
moved = lambda a, b, parts: ", ".join(f"{p.split('::')[-1]} {a[p]}->{b[p]}" for p in parts)  # noqa: E731


def onnx_attempts(torch, model, sample) -> dict:
    outcomes = {}
    for label, extra in (("dynamo", {}), ("legacy", {"dynamo": False})):
        try:
            torch.onnx.export(model, (sample,), io.BytesIO(), opset_version=OPSET, **extra)
            outcomes[label] = "exported"
        except Exception as exc:                    # the exporter raises its own OnnxExporterError
            outcomes[label] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:58]}"
    return outcomes


def jit_probe(torch, ref, model, sample) -> dict:
    traced = torch.jit.trace(model, sample)
    frozen = torch.jit.optimize_for_inference(torch.jit.freeze(traced.eval()))
    with torch.no_grad():
        gap = (model(sample) - frozen(sample)).abs().max().item()
    return {"traced": kinds(traced.inlined_graph), "opt": kinds(frozen.graph), "gap": gap,
            "eager_p50": ref.measure_latency(model, sample.shape, warmup=WARMUP, iters=ITERS)["p50_ms"],
            "opt_p50": ref.measure_latency(frozen, sample.shape, warmup=WARMUP, iters=ITERS)["p50_ms"]}


def fold_model(fuse, nn, module) -> int:
    folded, children = 0, list(module.named_children())
    for index, (name, child) in enumerate(children):
        previous = children[index - 1] if index else (None, None)
        if isinstance(child, nn.BatchNorm2d) and isinstance(previous[1], nn.Conv2d):
            setattr(module, previous[0], fuse(previous[1], child))
            setattr(module, name, nn.Identity())
            folded += 1
        folded += fold_model(fuse, nn, child)
    return folded


def solve():
    try:
        import torch
        import torch.nn as nn
        from torch.nn.utils.fusion import fuse_conv_bn_eval
        from torchvision.models import convnext_tiny, resnet18
    except ImportError as exc:                      # pragma: no cover - T1 needs torch+torchvision
        raise practice.Skip(f"needs torchvision: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    torch.manual_seed(0)
    convnext, sample = convnext_tiny(weights=None, num_classes=10).eval(), torch.randn(1, 3, RES, RES)
    resnet, probe = resnet18(weights=None, num_classes=10).train(), torch.randn(2, 3, FOLD_RES, FOLD_RES)
    with torch.no_grad():                           # give BatchNorm real statistics to fold
        for _ in range(BN_BATCHES):
            resnet(torch.randn(8, 3, FOLD_RES, FOLD_RES))
    folded = copy.deepcopy(resnet.eval())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exports = onnx_attempts(torch, convnext, sample)
        absent = [n for n in ("onnx", "onnxscript", "onnxruntime") if importlib.util.find_spec(n) is None]
        jit = jit_probe(torch, ref, convnext, sample)
        pairs = fold_model(fuse_conv_bn_eval, nn, folded)
        with torch.no_grad():
            fold_gap = (resnet(probe) - folded(probe)).abs().max().item()
    return {"exports": exports, "absent": absent, "jit": jit, "pairs": pairs, "fold_gap": fold_gap,
            "before": sum(p.numel() for p in resnet.parameters()), "bn_left": census(folded, "BatchNorm2d"),
            "after": sum(p.numel() for p in folded.parameters()), "bn_resnet": census(resnet, "BatchNorm2d"),
            "modules": {name: census(convnext, name) for name in SHAPES},
            "warnings": "; ".join(sorted({str(w.message)[:52] for w in caught}))}


def verify(result):
    jit, traced, opt = result["jit"], result["jit"]["traced"], result["jit"]["opt"]
    modules, speedup = result["modules"], jit["eager_p50"] / jit["opt_p50"]
    return [
        practice.Check(
            "ANSWER: the exercise cannot be run here — both ONNX export paths refuse, and so does the runtime",
            all([result["exports"]["dynamo"] != "exported", result["exports"]["legacy"] != "exported",
                 "onnxruntime" in result["absent"]]),
            f"`torch.onnx.export` at opset {OPSET}, default dynamo path: {result['exports']['dynamo']}; with "
            f"`dynamo=False`: {result['exports']['legacy']}. `importlib.util.find_spec` finds no "
            f"{result['absent']}, and none of the three may be fetched -- so what follows is an offline stand-in"),
        practice.Check(
            "ANSWER: 'the first layer where ORT is faster' is not well posed — fusion does not preserve layers",
            all([sum(opt.values()) < sum(traced.values()) * 0.75, opt["aten::matmul"] == traced["aten::linear"]]),
            f"torch's own freezer takes the traced graph from {sum(traced.values())} nodes to {sum(opt.values())} "
            f"and `aten::linear` does not survive: all {traced['aten::linear']} become {opt['aten::matmul']} "
            f"`aten::matmul` plus {opt['aten::add']} `aten::add`, leaving no layer to line up against a module"),
        practice.Check(
            "FINDING: the one graph optimiser available here is numerically faithful and buys no real speed",
            all([jit["gap"] < 1e-5, speedup < 2.0]),
            f"`torch.jit.freeze` + `optimize_for_inference` on convnext_tiny at {RES}x{RES} reproduces the eager "
            f"logits to {jit['gap']:.2e} at p50 {jit['opt_p50']:.1f} ms against eager's {jit['eager_p50']:.1f} ms "
            f"({speedup:.2f}x, machine-dependent): 'compile it and it gets faster' is a hypothesis, not a law"),
        practice.Check(
            "MECHANISM: the fusion ORT is credited for cannot fire here — convnext_tiny has zero BatchNorm",
            all([modules["BatchNorm2d"] == 0, modules["Conv2d"] > 0, modules["Permute"] > modules["Conv2d"]]),
            f"module census of convnext_tiny: {', '.join(f'{k} {v}' for k, v in modules.items())} -- Conv+BN "
            f"folding has nothing to fold. The graph holds {traced['aten::permute']} `aten::permute` and "
            f"{traced['aten::layer_norm']} `aten::layer_norm` calls against {traced['aten::_convolution']} "
            f"convolutions: the (N,C,H,W) -> (N,H,W,C) shuffles LayerNorm and Linear need to work on the last axis"),
        practice.Check(
            "CONTROL: folded on resnet18 the same fusion is exact — 20 pairs, 4,800 parameters gone",
            all([result["pairs"] == result["bn_resnet"], result["fold_gap"] < 1e-5, result["bn_left"] == 0]),
            f"after re-estimating its running statistics over {BN_BATCHES} training-mode batches, resnet18's "
            f"{result['bn_resnet']} Conv2d->BatchNorm2d pairs fold through `fuse_conv_bn_eval` into single "
            f"convolutions: {result['pairs']} folded, {result['bn_left']} left, {result['before']:,} -> "
            f"{result['after']:,} parameters ({result['before'] - result['after']:,} gone), logits reproduced to "
            f"{result['fold_gap']:.2e}. The fusion is real; convnext_tiny is just not built for it"),
        practice.Check(
            "CONTROL: freezing deletes only bookkeeping — every compute op count survives it unchanged",
            all([traced[part] == opt[part] for part in PARTS]),
            f"of the {sum(traced.values()) - sum(opt.values())} nodes the freezer removes, none is arithmetic: "
            f"{moved(traced, opt, PARTS)}, while `prim::GetAttr` goes {traced['prim::GetAttr']}->"
            f"{opt['prim::GetAttr']} and `prim::ListConstruct` {traced['prim::ListConstruct']}->"
            f"{opt['prim::ListConstruct']}. Torch warnings recorded: {result['warnings']}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
