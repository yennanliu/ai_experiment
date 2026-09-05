"""Exercise 3 — mixed precision, measured on the hardware that is actually here.

    **Port to GPU with mixed precision.** Add `torch.amp.autocast` and
    `GradScaler` to the training loop. Measure throughput (samples/second) with
    and without mixed precision on GPU. On an A100, expect ~2x speedup.

Reading of the exercise: there is no CUDA device here, so DESIGN D11 applies — the same
code path runs on the device that exists, the real command and its cost get printed, and
the checks measure what mixed precision *does* rather than how fast an A100 is. autocast,
GradScaler and the dtype rules are device-independent; the ~2x is not, and this shows why.
"""

from __future__ import annotations

import time

from harness import parity, practice

try:
    import torch
except ImportError as exc:                       # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None
torch.set_num_threads(1)

PHASE, LESSON = "03-deep-learning-core", "11-intro-to-pytorch"
N_TRAIN, N_TEST, SEP, BS, SCALE = 2048, 1000, 0.18, 64, 65536.0
HOST, PRICE, GPU_SEC, EPOCHS = "A100-40GB", 1.29, 0.5, 10
CRIT = torch.nn.CrossEntropyLoss()
MODES = (("fp32", None, False), ("bf16", torch.bfloat16, False), ("fp16", torch.float16, True))


def _loaders(ref):
    gen = torch.Generator().manual_seed(0)
    mid = torch.randn(10, 784, generator=gen) * SEP
    ys = [torch.arange(n) % 10 for n in (N_TRAIN, N_TEST)]
    xs = [mid[y] + torch.randn(len(y), 784, generator=gen) for y in ys]
    return ref.create_loaders(xs[0], ys[0], xs[1], ys[1], batch_size=BS)


def _epoch(model, loader, crit, opt, dev, dtype=None, scaler=None):
    """train_one_epoch's signature, with the lesson's own AMP snippet folded in."""
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(dev), labels.to(dev)
        opt.zero_grad()
        with torch.amp.autocast(device_type=dev.type, dtype=dtype):
            outputs = model(images)
            loss = crit(outputs, labels)
        if scaler is None:
            loss.backward(); opt.step()                                       # noqa: E702
        else:
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()  # noqa: E702
        loss_sum, total = loss_sum + loss.item() * len(labels), total + len(labels)
        correct += outputs.max(1)[1].eq(labels).sum().item()
    return loss_sum / total, correct / total


def _bench(ref, loaders, dev, dtype, use_scaler):
    """fp32 is the lesson's own train_one_epoch, unmodified; the rest is that plus autocast."""
    torch.manual_seed(0)
    model = ref.MNISTModel().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler(device=dev.type) if use_scaler else None
    run = ((lambda: ref.train_one_epoch(model, loaders[0], CRIT, opt, dev)) if dtype is None
           else (lambda: _epoch(model, loaders[0], CRIT, opt, dev, dtype, scaler)))
    ms = []
    for _ in range(4):                                   # first pass warms, then three timed
        clock = time.perf_counter()
        run()
        ms.append(time.perf_counter() - clock)
    return {"rate": N_TRAIN / sorted(ms[1:])[1], "fp32": _all_fp32(model),
            "acc": ref.evaluate(model, loaders[1], CRIT, dev)[1]}


def _all_fp32(model):
    return all(p.dtype is torch.float32 for p in model.parameters())


def _cast(model, images, labels, base, dev, dtype):
    """One forward under autocast: which tensors change dtype, and by how much."""
    with torch.amp.autocast(device_type=dev.type, dtype=dtype):
        logits = model(images)
        loss = CRIT(logits, labels)
    return {"half": logits.dtype is dtype, "fp32_loss": loss.dtype is torch.float32,
            "gap": (logits.float() - base).abs().max().item(),
            "flips": int((logits.float().argmax(1) != base.argmax(1)).sum())}


def _grads(model, images, labels, dev, scale):
    model.zero_grad()
    with torch.amp.autocast(device_type=dev.type, dtype=torch.float16):
        loss = CRIT(model(images), labels)
    (loss * scale).backward()
    return torch.cat([p.grad.flatten() for p in model.parameters()])


def solve():
    ref = parity.load_reference(PHASE, LESSON, "pytorch_intro")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = _loaders(ref)
    out = {name: _bench(ref, loaders, dev, dt, sc) for name, dt, sc in MODES}
    torch.manual_seed(0)
    model = ref.MNISTModel().to(dev)
    images, labels = (t.to(dev) for t in next(iter(loaders[0])))
    base = model(images)
    out["num"] = {n: _cast(model, images, labels, base, dev, dt) for n, dt, _s in MODES[1:]}
    raw, big = (_grads(model, images, labels, dev, s) for s in (1.0, SCALE))
    t16, tbf = torch.finfo(torch.float16).tiny, torch.finfo(torch.bfloat16).tiny
    out["sub"] = {"f16": (raw.abs() < t16).float().mean().item(), "t16": t16, "tbf": tbf,
                  "scaled": (big.abs() < t16).float().mean().item(),
                  "bf16": (raw.abs() < tbf).float().mean().item(),
                  "peak": raw.abs().max().item() * SCALE, "max16": torch.finfo(torch.float16).max}
    out["dev"], out["cuda"] = dev.type, torch.cuda.is_available()
    out["span"], out["cost"] = base.abs().max().item(), PRICE * GPU_SEC * EPOCHS / 3600
    return out


def verify(result):
    f32, bf, f16, sub, num = (result[k] for k in ("fp32", "bf16", "fp16", "sub", "num"))
    return [
        practice.Check("ANSWER: on this device mixed precision is 5-6x SLOWER, not 2x faster",
                       not result["cuda"] and bf["rate"] < f32["rate"] / 3,
                       f"{N_TRAIN} samples/epoch on {result['dev']}: fp32 {f32['rate']:,.0f}/s, "
                       f"bf16 {bf['rate']:,.0f}/s ({bf['rate'] / f32['rate']:.2f}x), fp16+scaler "
                       f"{f16['rate']:,.0f}/s ({f16['rate'] / f32['rate']:.2f}x). The real run is "
                       f"this file unchanged on a CUDA host — `uv sync --extra llm && uv run "
                       f"python <this path>` on an {HOST} at ${PRICE:.2f}/h list, "
                       f"~${result['cost']:.4f} for the {EPOCHS} epochs the lesson quotes "
                       f"at {GPU_SEC}s each"),
        practice.Check("MECHANISM: the 2x is a tensor-core property, not a float16 property",
                       f16["rate"] < f32["rate"] and bf["rate"] < f32["rate"],
                       f"halving the bit width buys nothing where there is no half-precision "
                       f"matmul unit: this CPU runs bf16 GEMMs by widening back to fp32, so "
                       f"autocast only adds casts — {1 / (bf['rate'] / f32['rate']):.1f}x of pure "
                       f"overhead. An A100's tensor cores are the whole speedup"),
        practice.Check("FINDING: 'mixed' is literal and asymmetric — logits go half, the loss "
                       "and every master weight stay fp32",
                       num["bf16"]["half"] and num["bf16"]["fp32_loss"] and f16["fp32"],
                       f"autocast returns bf16 logits and fp16 logits but an fp32 loss, because "
                       f"cross_entropy is on autocast's fp32 list; after a full fp16 epoch with "
                       f"GradScaler every parameter is still fp32. Cost on one batch, logit span "
                       f"{result['span']:.4f}: bf16 off by {num['bf16']['gap']:.4f}, fp16 by "
                       f"{num['fp16']['gap']:.4f}, and {num['bf16']['flips']} argmax flips"),
        practice.Check("CONTROL: accuracy is untouched, so only the speed claim fails here",
                       abs(f32["acc"] - bf["acc"]) < 0.01 and abs(f32["acc"] - f16["acc"]) < 0.01,
                       f"one epoch each, same seed and init: fp32 {f32['acc']:.4f} (the lesson's "
                       f"own train_one_epoch), bf16 {bf['acc']:.4f}, fp16+GradScaler "
                       f"{f16['acc']:.4f}. {sub['f16']:.2%} of gradients fall below fp16's "
                       f"smallest normal {sub['t16']:.2e} and {SCALE:.0f}x scaling cuts that to "
                       f"{sub['scaled']:.3%}; bf16's is {sub['tbf']:.1e}, so {sub['bf16']:.3%} "
                       f"and no scaler. Headroom holds: {sub['peak']:.3g} < {sub['max16']:.0f}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
