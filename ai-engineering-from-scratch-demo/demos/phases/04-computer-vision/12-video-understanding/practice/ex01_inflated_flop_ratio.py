"""Exercise 1 — inflated flop ratio.

    **(Easy)** Compute FLOPs (approximate) for FramePool with T=8 vs an I3D-style 3D ResNet with T=8. Justify why 2D+pool is 3-5x cheaper.

Reading of the exercise: "an I3D-style 3D ResNet" has two defensible readings and
they do not agree, so both are measured. Reading (a) is the lesson's own
`inflate_2d_to_3d` applied to every Conv2d of the very ResNet-18 `FramePool`
wraps -- and that ratio is not "3-5x", it is exactly the temporal kernel, 3.0000
to four decimals at 64x64 and at 224x224, because the function hardcodes temporal
stride 1 and padding `time_kernel // 2` so T_out == T_in at every layer and each
conv's cost is scaled by `time_kernel` and by nothing else. The band the exercise
quotes is therefore not a range of outcomes but the range `time_kernel in
{3, 4, 5}`, and the lesson's own prose ("T/8 more FLOPs ... for temporal kernel of
3") would say 1x at T=8, which is measured false. Reading (b) is a real 3D
ResNet-18, torchvision's `r3d_18`, and it costs 5.61x -- above the band, and for a
reason that has nothing to do with time: it drops ResNet-18's stem max-pool, so
its first stage runs at four times the spatial positions. Put that max-pool back
and un-stride its time axis and it lands on 2.979x, the inflation's 3x. `r3d_18`
is built with `weights=None`: nothing is downloaded, the run starts from random
init, and a FLOP count does not depend on weight values anyway. FLOPs are counted
as multiply-accumulates (double them for the mul+add convention); ratios are
unaffected. Every shape but the timing run is propagated on torch's `meta`
device, so the 43 GMAC of inflated convolution is never actually executed and the
file costs ~5 s. `scan` hooks every conv
and linear, runs one tensor through, and totals one multiply-accumulate per
output element per input channel per kernel tap; `variants` measures r3d_18 as
shipped, then with ResNet-18's stem max-pool restored, then with its three
stride-2 time steps un-strided; `wall` times one forward after a warm-up.
"""

from __future__ import annotations

import math
import time

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "12-video-understanding"

T, CLASSES, SIZES, KERNELS, REPEATS = 8, 400, (64, 224), (3, 5), 2
DECOMP = 64                     # the size the r3d_18 decomposition is measured at
POOL = dict(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))   # ResNet-18's stem pool

giga = lambda mac, size, key: mac[size][key] / 1e9                       # noqa: E731 - a formatter
rat = lambda mac, size, key: mac[size][key] / mac[size]["pool"]          # noqa: E731 - a formatter
mac_count = lambda nn, mod, shape: (                                   # noqa: E731 - one MAC per
    shape[0] * mod.in_features * mod.out_features if isinstance(mod, nn.Linear)
    else math.prod(shape) * mod.in_channels * math.prod(mod.kernel_size))


def scan(torch, nn, model, shape, device="cpu") -> dict:
    rows = []
    handles = [layer.register_forward_hook(lambda mod, inp, out: rows.append(
        (mod, tuple(inp[0].shape), mac_count(nn, mod, tuple(out.shape)))))
        for layer in model.modules() if isinstance(layer, (nn.Conv2d, nn.Conv3d, nn.Linear))]
    with torch.no_grad():
        model(torch.zeros(shape, device=device))
    for handle in handles:
        handle.remove()
    conv = [row for row in rows if not isinstance(row[0], nn.Linear)]
    return {"conv": sum(row[2] for row in conv), "rows": conv,
            "params": sum(row[0].weight.numel() for row in conv)}


def inflated(torch, nn, ref, rows, kernel) -> dict:
    macs, params = 0, 0
    for module, shape_in, _cost in rows:
        conv3d = ref.inflate_2d_to_3d(module, time_kernel=kernel)
        params, meta = params + conv3d.weight.numel(), conv3d.to("meta")
        out = meta(torch.zeros(1, module.in_channels, T, shape_in[2], shape_in[3], device="meta"))
        macs += mac_count(nn, meta, tuple(out.shape))
    return {"conv": macs, "params": params}


def variants(torch, nn, factory, size) -> dict:
    model, shape = factory(weights=None, num_classes=CLASSES).to("meta"), (1, 3, T, size, size)
    out = {"shipped": scan(torch, nn, model, shape, "meta")["conv"]}
    model.stem = nn.Sequential(model.stem, nn.MaxPool3d(**POOL))
    out["pooled"] = scan(torch, nn, model, shape, "meta")["conv"]
    for module in model.modules():
        if isinstance(module, nn.Conv3d) and module.stride[0] == 2:
            module.stride = (1, *module.stride[1:])
    return {**out, "untimed": scan(torch, nn, model, shape, "meta")["conv"]}


def wall(torch, model, shape) -> float:
    sample = torch.randn(shape)
    with torch.no_grad():
        model.eval()(sample)
        start = time.perf_counter()
        for _ in range(REPEATS):
            model(sample)
    return (time.perf_counter() - start) / REPEATS


def solve():
    try:
        import torch
        import torch.nn as nn
        from torchvision.models.video import r3d_18
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    mac, params, big = {}, {}, SIZES[-1]
    for size in SIZES:
        pool = ref.FramePool(num_classes=CLASSES, pretrained=False)
        base = scan(torch, nn, pool, (1, T, 3, size, size))
        blown = {f"kt{k}": inflated(torch, nn, ref, base["rows"], k) for k in KERNELS}
        mac[size] = {"pool": base["conv"], **{n: v["conv"] for n, v in blown.items()},
                     **variants(torch, nn, r3d_18, size)}
        params = {"all": sum(p.numel() for p in pool.parameters()), "conv": base["params"],
                  "kt3": blown["kt3"]["params"], "r3d": sum(
                      p.numel() for p in r3d_18(weights=None, num_classes=CLASSES).parameters())}
    return {"mac": mac, "params": params, "seconds": {
        "pool": wall(torch, ref.FramePool(num_classes=CLASSES, pretrained=False),
                     (1, T, 3, big, big)),
        "r3d": wall(torch, r3d_18(weights=None, num_classes=CLASSES), (1, 3, T, big, big))}}


def verify(result):
    mac, params, seconds = result["mac"], result["params"], result["seconds"]
    small, big = SIZES[0], SIZES[-1]
    speed, share = seconds["r3d"] / seconds["pool"], params["kt3"] / params["conv"]
    return [
        practice.Check(
            "ANSWER: 3.0000x, not '3-5x' -- inflated ResNet-18 sits exactly on the band's lower edge",
            all(abs(rat(mac, size, "kt3") - 3.0) < 1e-4 for size in SIZES),
            f"FramePool at T={T} costs {giga(mac, small, 'pool'):.3f} GMAC at {small}x{small} and {giga(mac, big, 'pool'):.3f} GMAC at "
            f"{big}x{big} (double for the mul+add FLOP convention); the lesson's own inflate_2d_to_3d over all 20 Conv2d gives "
            f"{giga(mac, small, 'kt3'):.3f} / {giga(mac, big, 'kt3'):.3f} GMAC -- ratios {rat(mac, small, 'kt3'):.4f}, {rat(mac, big, 'kt3'):.4f}"),
        practice.Check(
            "MECHANISM: the ratio is the temporal kernel and nothing else -- kt=5 gives exactly 5.0000",
            all(abs(rat(mac, size, "kt5") - 5.0) < 1e-4 for size in SIZES),
            f"inflate_2d_to_3d hardcodes temporal stride 1 and padding kt//2, so T_out == T_in == {T} at every layer and each conv is scaled "
            f"by kt alone: kt=5 measures {rat(mac, small, 'kt5'):.4f} at {small}x{small}, {rat(mac, big, 'kt5'):.4f} at {big}x{big}. "
            "Independent of T, H, W, depth and width -- so '3-5x' is not a range of outcomes, it is kt in {3, 4, 5}"),
        practice.Check(
            "FINDING: a real 3D ResNet-18 costs 5.61x, above the band the exercise quotes",
            all(rat(mac, size, "shipped") > 5.0 for size in SIZES),
            f"torchvision r3d_18(weights=None) -- nothing downloaded, random init, and a FLOP count never reads weight values -- costs "
            f"{giga(mac, small, 'shipped'):.3f} GMAC at {small}x{small} and {giga(mac, big, 'shipped'):.3f} GMAC at {big}x{big}, i.e. "
            f"{rat(mac, small, 'shipped'):.4f}x and {rat(mac, big, 'shipped'):.4f}x over the same FramePool"),
        practice.Check(
            "MECHANISM: that extra factor is spatial, not temporal -- r3d_18 drops the stem max-pool",
            rat(mac, DECOMP, "pooled") < 2.0 and abs(rat(mac, DECOMP, "untimed") - 3.0) < 0.1,
            f"at {DECOMP}x{DECOMP}, putting ResNet-18's stem MaxPool back into r3d_18 takes it from {rat(mac, DECOMP, 'shipped'):.3f}x to "
            f"{rat(mac, DECOMP, 'pooled'):.3f}x -- a factor of {rat(mac, DECOMP, 'shipped') / rat(mac, DECOMP, 'pooled'):.2f} for layer1's "
            f"resolution alone -- and un-striding its three stride-2 time steps raises it to {rat(mac, DECOMP, 'untimed'):.3f}x, back onto "
            f"inflation's {rat(mac, DECOMP, 'kt3'):.3f}x"),
        practice.Check(
            "CONTROL: measured wall clock lands inside 3-5x, but not for the arithmetic's reason",
            2.0 < speed < 12.0,
            f"one {big}x{big} T={T} clip, {REPEATS} timed forwards after a warm-up on 2 threads: FramePool {seconds['pool'] * 1000:.0f} ms, "
            f"r3d_18 {seconds['r3d'] * 1000:.0f} ms, {speed:.2f}x against a MAC ratio of {rat(mac, big, 'shipped'):.2f}x -- memory traffic and "
            "kernel efficiency, machine-dependent, and why 'approximate FLOPs' answers this only loosely"),
        practice.Check(
            "CONTROL: capacity moves by the same factor, and the two 3D readings agree on it",
            abs(share - 3.0) < 1e-9 and 0.9 < params["r3d"] / params["kt3"] < 1.1,
            f"FramePool holds {params['all']:,} parameters, {params['conv']:,} of them in convolutions; inflation multiplies exactly that by "
            f"kt to {params['kt3']:,} ({share:.4f}x), and r3d_18 independently arrives at {params['r3d']:,} -- one number setting both"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
