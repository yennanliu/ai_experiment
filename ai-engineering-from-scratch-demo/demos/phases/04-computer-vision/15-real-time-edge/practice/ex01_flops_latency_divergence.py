"""Exercise 1 — flops latency divergence.

    **(Easy)** Measure p50 latency for `resnet18`, `mobilenet_v3_small`, `efficientnet_v2_s`, and `convnext_tiny` at 224x224 on CPU. Report the table and identify which architecture has the best accuracy-per-ms.

Reading of the exercise: the latency half runs straight through the lesson's own
`measure_latency` on the four `weights=None` architectures, but the accuracy half
has no local answer. Downloading ImageNet weights is out of scope, so every model
timed below is randomly initialised, and two of the four emit the *same* class for
all 32 probe images -- there is no accuracy to divide by. What is available
offline is torchvision's own weight metadata: each `Weights` enum carries
`_metrics['ImageNet-1K']['acc@1']`, `num_params` and `_ops` in a plain `.meta`
dict, and the crop size those figures were measured at in `.transforms.keywords`,
none of which touches the network. So "accuracy-per-ms" here is
torchvision's published top-1 for the pretrained weights over latency measured
here on the identical architecture, and every check says so. That metadata also
audits the lesson's `flops_estimate`, which is wrong for exactly one of the four:
`linear_hook` adds `2 * in_features * out_features` once per `nn.Linear` call with
no factor for the spatial positions the layer is applied at. ConvNeXt does all of
its pointwise mixing in `nn.Linear` over a permuted (N, H, W, C) tensor, so the
estimate misses up to 3,136 positions per block and lands 14x low; restoring the
factor reproduces torchvision's published `_ops` for all four. Wall-clock is the
least portable thing here, so the timing checks assert only wide bounds and the
weight sits on the exact quantities.

Structure: `corrected_flops` is the lesson's estimator with the missing spatial
factor on `nn.Linear`, registered through the same forward hooks; `profile`
builds one architecture with its ImageNet head and records `parameter_count`,
both FLOP estimates (at 224 and at that weight entry's own crop size), p50/p95
from the lesson's `measure_latency`, torchvision's offline metadata and how many
distinct classes the random-init model predicts over a fixed 32-image probe;
`thread_sweep` re-times the two cheapest models at 1, 2 and 4 CPU threads and
restores 2 on the way out. The module-level lambdas are formatters and derived
ratios, kept out of `verify` so its branch count stays inside D14's limit. At 138
code lines this sits above D14's 120-line target, 12 clear of the ceiling: six
checks over four architectures, two FLOP estimators, an offline metadata audit
and a three-point thread sweep, with the reported table itself costing four.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "15-real-time-edge"

RESOLUTION, WARMUP, ITERS, PROBE, THREADS = 224, 3, 9, 32, (1, 2, 4)
SHAPE, NAMES = (1, 3, RESOLUTION, RESOLUTION), ("mobilenet_v3_small", "resnet18", "efficientnet_v2_s", "convnext_tiny")
SWEEP = ("mobilenet_v3_small", "resnet18")

per_ms_of = lambda t, key: {n: r[key] / r["p50"] for n, r in t.items()}          # noqa: E731
docerr_of = lambda t: {n: abs(r["fix_doc_g"] - 2 * r["doc_g"]) / (2 * r["doc_g"]) for n, r in t.items()}  # noqa: E731
ratios_of = lambda s: {n: s[n][SWEEP[1]] / s[n][SWEEP[0]] for n in s}            # noqa: E731
listing = lambda mapping, fmt: ", ".join(fmt(name, value) for name, value in mapping.items())  # noqa: E731
doc_row = lambda n, r: f"{n} {r['fix_doc_g'] / 2:.3f} vs {r['doc_g']}@{r['doc_res']}"  # noqa: E731
flop_gap = lambda t: t[SWEEP[1]]["fix224_g"] / t[SWEEP[0]]["fix224_g"]           # noqa: E731
rows = lambda t: "\n    ".join(                                                 # noqa: E731
    f"{n:19s}{r['params'] / 1e6:7.2f}M {r['ref_g']:7.3f} {r['fix224_g']:7.3f} {r['p50']:7.1f} "
    f"{r['p95']:7.1f} {r['top1']:6.2f} {r['top1'] / r['p50']:6.2f}" for n, r in t.items())
HEADER = f"{'model':19s}{'params':>8s} {'refGF':>7s} {'fixGF':>7s} {'p50ms':>7s} {'p95ms':>7s} {'top1':>6s} {'/ms':>6s}"


def corrected_flops(torch, nn, model, shape) -> int:
    total = [0]
    def conv_hook(module, _inp, out):
        c_out, c_in_per_group, kh, kw = module.weight.shape
        total[0] += 2 * c_in_per_group * c_out * kh * kw * out.shape[-2] * out.shape[-1]

    def linear_hook(module, _inp, out):
        total[0] += 2 * module.in_features * module.out_features * (out.numel() // out.shape[-1])

    hooks = [layer.register_forward_hook(conv_hook if isinstance(layer, nn.Conv2d) else linear_hook)
             for layer in model.modules() if isinstance(layer, (nn.Conv2d, nn.Linear))]
    model.eval()
    with torch.no_grad():
        model(torch.randn(shape))
    for hook in hooks:
        hook.remove()
    return total[0]


def profile(torch, nn, zoo, ref, name) -> dict:
    torch.manual_seed(0)
    model, entry = zoo.get_model(name, weights=None).eval(), zoo.get_model_weights(name).DEFAULT
    res = entry.transforms.keywords["crop_size"]
    torch.manual_seed(1)
    with torch.no_grad():
        classes = model(torch.randn(PROBE, *SHAPE[1:])).argmax(1).unique().numel()
    fix224 = corrected_flops(torch, nn, model, SHAPE) / 1e9
    latency = ref.measure_latency(model, SHAPE, warmup=WARMUP, iters=ITERS)
    return {"params": ref.parameter_count(model), "ref_g": ref.flops_estimate(model, SHAPE) / 1e9,
            "fix224_g": fix224, "classes": classes, "p50": latency["p50_ms"], "p95": latency["p95_ms"],
            "fix_doc_g": fix224 if res == RESOLUTION else corrected_flops(torch, nn, model, (1, 3, res, res)) / 1e9,
            "top1": entry.meta["_metrics"]["ImageNet-1K"]["acc@1"], "doc_g": entry.meta["_ops"],
            "doc_params": entry.meta["num_params"], "doc_res": res}


def thread_sweep(torch, zoo, ref) -> dict:
    table = {}
    for count in THREADS:
        torch.set_num_threads(count)
        table[count] = {n: ref.measure_latency(zoo.get_model(n, weights=None), SHAPE,
                                               warmup=WARMUP, iters=ITERS)["p50_ms"] for n in SWEEP}
    torch.set_num_threads(2)
    return table


def solve():
    try:
        import torch
        import torch.nn as nn
        from torchvision import models as zoo
    except ImportError as exc:                      # pragma: no cover - T1 needs torch+torchvision
        raise practice.Skip(f"needs torchvision: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    table = {name: profile(torch, nn, zoo, ref, name) for name in NAMES}
    return {"table": table, "sweep": thread_sweep(torch, zoo, ref),
            "defaults": ref.measure_latency(zoo.get_model("resnet18", weights=None), SHAPE),
            "split": next(n for n in range(2, 500) if int(n * 0.95) != n - 1)}


def verify(result):
    table, sweep, defaults = result["table"], result["sweep"], result["defaults"]
    per_ms, density, ratios = per_ms_of(table, "top1"), per_ms_of(table, "fix224_g"), ratios_of(sweep)
    doc_err, cnx = docerr_of(table), table["convnext_tiny"]
    best, worst = max(per_ms, key=per_ms.get), min(per_ms, key=per_ms.get)
    fast, slow = max(density, key=density.get), min(density, key=density.get)
    return [
        practice.Check(
            "ANSWER: only last place is stable — convnext_tiny trails the winner by more than 5x",
            worst == "convnext_tiny" and per_ms[best] > 5 * per_ms[worst],
            f"at {RESOLUTION}x{RESOLUTION} on CPU, 2 threads, machine-dependent wall-clock\n    {HEADER}\n    "
            f"{rows(table)}\n    top1 is torchvision's published ImageNet acc@1 for the pretrained weights, not "
            f"measured here; {best} leads at {per_ms[best]:.2f} points/ms, {worst} trails at {per_ms[worst]:.2f} "
            f"-- a {per_ms[best] / per_ms[worst]:.0f}x gap"),
        practice.Check(
            "ANSWER: no accuracy is measurable locally — two of four models predict one class for everything",
            [r["classes"] for r in table.values()].count(1) >= 2,
            f"with `weights=None` the four are randomly initialised; over one fixed batch of {PROBE} probe images "
            f"the number of distinct argmax classes out of 1,000 is "
            f"{listing(table, lambda n, r: f'{n} {r["classes"]}')} -- accuracy-per-ms must borrow published top-1"),
        practice.Check(
            "FINDING: FLOPs does not rank latency — GFLOPs delivered per ms spreads more than 5x",
            density[fast] > 5 * density[slow],
            f"corrected FLOP count over measured p50 gives {listing(density, lambda n, v: f'{n} {v:.3f}')} "
            f"GFLOP/ms, a {density[fast] / density[slow]:.0f}x spread on one host: {fast} converts FLOPs to time "
            f"best ({table[fast]['fix224_g']:.3f} GF) and {slow} worst ({table[slow]['fix224_g']:.3f} GF)"),
        practice.Check(
            "MECHANISM: `flops_estimate` misses the spatial factor on nn.Linear and runs 14x low on ConvNeXt",
            cnx["fix224_g"] > 10 * cnx["ref_g"] and max(doc_err.values()) < 0.01,
            f"`linear_hook` adds `2*in*out` once per call, ignoring the H*W positions a Linear over (N,H,W,C) is "
            f"applied at: convnext_tiny reads {cnx['ref_g']:.3f} GFLOPs against {cnx['fix224_g']:.3f} corrected, "
            f"{cnx['fix224_g'] / cnx['ref_g']:.2f}x. Halved to MACs at each entry's own crop size the fixed count "
            f"reproduces torchvision's `_ops` -- {listing(table, doc_row)} -- to a worst relative error of "
            f"{max(doc_err.values()):.1e}; `parameter_count` matches `num_params` "
            f"{sum(1 for r in table.values() if r['params'] == r['doc_params'])}/4"),
        practice.Check(
            "CONTROL: at the reference's own defaults p95 and p99 are the same number — both are the max",
            defaults["p95_ms"] == defaults["p99_ms"] and defaults["p95_ms"] >= defaults["p50_ms"],
            f"the helper defaults to iters=20 and indexes the sorted list at `int(20*0.95)` = 19, the last element, "
            f"while `p99_ms` is `times[-1]` = 19 too: resnet18 reports p50 {defaults['p50_ms']:.2f}, p95 "
            f"{defaults['p95_ms']:.2f}, p99 {defaults['p99_ms']:.2f} ms -- p100 until iters hits {result['split']}"),
        practice.Check(
            "CONTROL: mobilenet's 32x FLOP advantage never buys even 10x, at any thread count",
            max(ratios.values()) < 10.0,
            f"mobilenet_v3_small has {flop_gap(table):.0f}x fewer corrected FLOPs than resnet18; re-timed at 1, 2 "
            f"and 4 CPU threads their p50 is "
            f"{listing(sweep, lambda n, r: f'{n}t {r[SWEEP[0]]:.1f}/{r[SWEEP[1]]:.1f} ms ({ratios[n]:.2f}x)')} -- "
            f"the ratio moves with thread count alone, so 'which is faster' is a property of the host too"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
