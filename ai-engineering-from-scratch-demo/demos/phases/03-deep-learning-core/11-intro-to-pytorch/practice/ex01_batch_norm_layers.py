"""Exercise 1 — batch norm against dropout, on equal terms.

    **Add batch normalization.** Insert `nn.BatchNorm1d` after each linear layer
    (before the activation). Compare test accuracy and training speed vs the
    dropout-only version. Batch norm should reach 98%+ in fewer epochs.

Reading of the exercise: MNIST arrives over HTTP, which a check must not depend on, so the
fixture is a seeded 10-class Gaussian mixture in 784-D — MNIST's shape and class count, no
network. That carries the comparative claim (fewer epochs?) but not the absolute 98%. Three
models run because the lesson's batch-norm model also deletes the dropout.
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
N_TRAIN, N_TEST, SEP, EPOCHS, KEYS = 1280, 2000, 0.18, 12, ("dropout", "bn_only", "bn_kept")


def _blobs():
    """Ten isotropic Gaussians in 784-D; SEP sets the Bayes ceiling."""
    gen = torch.Generator().manual_seed(0)
    mid = torch.randn(10, 784, generator=gen) * SEP
    ys = [torch.arange(n) % 10 for n in (N_TRAIN, N_TEST)]
    xs = [mid[y] + torch.randn(len(y), 784, generator=gen) for y in ys]
    bayes = (torch.cdist(xs[1], mid).argmin(1) == ys[1]).float().mean().item()
    return (xs[0], ys[0]), (xs[1], ys[1]), bayes


def _insert_bn(model):
    """The exercise verbatim: BatchNorm1d after each hidden Linear, dropout kept."""
    old, new = list(model.net), []
    for i, layer in enumerate(old):
        new.append(layer)
        if isinstance(layer, torch.nn.Linear) and i < len(old) - 1:
            new.append(torch.nn.BatchNorm1d(layer.out_features))
    model.net = torch.nn.Sequential(*new)
    return model


def _train(ref, make, loaders):
    torch.manual_seed(0)
    model, crit, dev = make(), torch.nn.CrossEntropyLoss(), torch.device("cpu")
    opt, acc, ms = torch.optim.Adam(model.parameters(), lr=1e-3), [], []
    for _ in range(EPOCHS):
        clock = time.perf_counter()
        ref.train_one_epoch(model, loaders[0], crit, opt, dev)
        ms.append(time.perf_counter() - clock)
        acc.append(ref.evaluate(model, loaders[1], crit, dev)[1])
    return {"first": acc[0], "top": max(acc), "ms": sorted(ms)[EPOCHS // 2] * 1000,
            "to98": next((i + 1 for i, a in enumerate(acc) if a >= 0.98), EPOCHS + 1),
            "par": sum(p.numel() for p in model.parameters()),
            "sd": sum(v.numel() for v in model.state_dict().values())}


def _tail_batch(ref, model, test):
    loader = ref.create_loaders(test[0][:1281], test[1][:1281], *[t[:64] for t in test], 64)[0]
    try:
        ref.train_one_epoch(model, loader, torch.nn.CrossEntropyLoss(),
                            torch.optim.Adam(model.parameters(), lr=1e-3), torch.device("cpu"))
    except ValueError as exc:
        return str(exc)
    return ""


def solve():
    ref = parity.load_reference(PHASE, LESSON, "pytorch_intro")
    train, test, bayes = _blobs()
    loaders = ref.create_loaders(*train, *test, batch_size=64)
    makes = (ref.MNISTModel, ref.MNISTModelWithBatchNorm, lambda: _insert_bn(ref.MNISTModel()))
    out = {"bayes": bayes, **{k: _train(ref, m, loaders) for k, m in zip(KEYS, makes)}}
    out["crash"] = _tail_batch(ref, ref.MNISTModelWithBatchNorm(), test)
    out["control"] = _tail_batch(ref, ref.MNISTModel(), test)
    return out


def verify(result):
    drop, bn, kept = (result[k] for k in KEYS)
    ratio = kept["ms"] / drop["ms"]
    return [
        practice.Check("ANSWER: batch norm costs accuracy and roughly matches the epoch cost",
                       drop["top"] > kept["top"] > bn["top"] > 0.97 and 0.8 < ratio < 1.5,
                       f"peak over {EPOCHS} epochs, {N_TRAIN}/{N_TEST} rows, Bayes "
                       f"{result['bayes']:.4f}: dropout {drop['top']:.4f} at {drop['ms']:.1f} ms; "
                       f"+batch norm {kept['top']:.4f} at {kept['ms']:.1f} ms ({ratio:.2f}x); "
                       f"lesson's model {bn['top']:.4f} at {bn['ms']:.1f} ms, minus its dropout"),
        practice.Check("FINDING: batch norm reaches 98% later, not sooner",
                       bn["to98"] > kept["to98"] > drop["to98"],
                       f"epochs to 98%: dropout {drop['to98']}, +batch norm {kept['to98']}, "
                       f"lesson's model {bn['to98']}. 'Fewer epochs' holds only for epoch 1, "
                       f"where batch norm leads {bn['first']:.4f} to {drop['first']:.4f}"),
        practice.Check("FINDING: parameters() undercounts the batch-norm checkpoint by 770",
                       bn["par"] - drop["par"] == 768 and bn["sd"] - bn["par"] == 770,
                       f"dropout {drop['par']:,} parameters = {drop['sd']:,} state_dict "
                       f"numbers; batch norm {bn['par']:,} against {bn['sd']:,}. +768 is "
                       f"2*(256+128) scales and shifts, the other 770 running stats — buffers "
                       f"parameters() misses and torch.save writes"),
        practice.Check("FINDING: a size-1 last batch kills batch norm inside train_one_epoch",
                       "Expected more than 1 value per channel" in result["crash"]
                       and result["control"] == "",
                       f"1281 rows at batch_size=64 ends on a batch of 1: ValueError "
                       f"{result['crash']!r}. CONTROL: the dropout model runs the same loader "
                       f"fine. 60000 % 64 = 32, so MNIST hides it and the missing drop_last "
                       f"in create_loaders stays a landmine"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
