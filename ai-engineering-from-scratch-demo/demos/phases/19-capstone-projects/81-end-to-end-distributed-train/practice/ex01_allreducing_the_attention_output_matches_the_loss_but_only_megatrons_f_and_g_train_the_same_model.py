"""Exercise 1 — allreducing the attention output matches the loss, but only Megatron's f+g trains the same.

    Add a tensor-parallel split of the attention head and verify the loss matches the single-rank baseline. Two ranks: half the heads per rank, allreduce of the attention output.

Reading of the exercise: two gloo ranks each keep half of the lesson's
MiniGPT heads in both blocks: their q/k/v rows of `qkv` and their columns
of `proj`. The partial attention outputs are summed with an allreduce.
Both ranks see the same batch, the lesson's 4 x 16 tokens. "The loss
matches" is checked twice: once at step 0 (forward only), and at every
step of 20 Adam steps at the lesson's LR, against the unsplit model trained
on one rank. Three backward rules are compared for the allreduce: the
exercise's recipe as worded (allreduce forward, gradient passed through),
`torch.distributed.nn.functional.all_reduce`, and Megatron's pair (`g` at
the output plus `f`, which allreduces the gradient at the input).

**ANSWER: with Megatron's f and g the loss matches the single-rank baseline
at all 20 steps to 4.8e-7, and the two ranks stay bit-identical.** Each rank
holds 4,096 of the 8,192 attention weights.

**FINDING: allreducing only the attention output matches the loss exactly
at step 0 but trains a different model.** The input to the attention layer
is replicated, so its gradient is a sum over both ranks' heads. Without `f`
each rank backpropagates only its own heads' share. The embedding gradient
is 0.7% low, the two copies of the "replicated" weights drift apart (their
losses differ by 3.8e-3 by step 20), and the curve leaves the baseline by
3.0e-3. torch's autograd-aware `all_reduce` errs the other way: it allreduces
the gradient as well, which doubles the output gradient. That gives an
embedding gradient 0.6% high and ranks 6.9e-3 apart. A step-0 loss check
passes on all three.

Structure: `launch()` starts this file once per rank as a subprocess with a
120 s timeout and kills stragglers; `worker()` trains the baseline and the
three split variants. Expected output: two PASS checks.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys

from harness import parity, practice

try:
    import torch
    import torch.distributed as dist
    import torch.distributed.nn.functional as dnn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "81-end-to-end-distributed-train"


def launch(ws, *args, timeout=120):
    """Run this file as `ws` gloo ranks in subprocesses; return each rank's JSON, kill stragglers."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        argv = [sys.executable, __file__, "--rank", str(ws), str(s.getsockname()[1]), *map(str, args)]
    pipe = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "text": True}
    procs = [subprocess.Popen(argv, env=os.environ | {"RANK": str(r)}, **pipe) for r in range(ws)]
    try:
        outs = [p.communicate(timeout=timeout) for p in procs]
    finally:
        for p in procs:
            p.kill()
    if any(p.returncode for p in procs):
        raise RuntimeError(f"a rank failed: {[err[-400:] for _, err in outs]}")
    return [json.loads(out.splitlines()[-1]) for out, _ in outs]


def summed(t):
    t = t.detach().clone()
    dist.all_reduce(t)
    return t


def f_op(x):
    """Megatron's `f`: identity forward on the replicated input, allreduce of its gradient backward."""
    y = x * 1
    y.register_hook(summed)
    return y


VARIANTS = {  # name -> (f at the attention input, g at the attention output)
    "output_allreduce_only": (lambda x: x, lambda x: x + (summed(x) - x).detach()),
    "dist_nn_all_reduce": (lambda x: x, dnn.all_reduce),
    "megatron_f_and_g": (f_op, lambda x: x + (summed(x) - x).detach()),
}


def shard_attention(attn, rank, ws, f, g):
    """Keep this rank's heads: their q/k/v rows of `qkv` and their columns of `proj`."""
    e, hd, local_h = attn.embed_dim, attn.head_dim, attn.num_heads // ws
    cols = torch.arange(rank * local_h * hd, (rank + 1) * local_h * hd)
    wqkv = attn.qkv.weight.detach()[torch.cat([j * e + cols for j in range(3)])].clone().requires_grad_()
    wproj = attn.proj.weight.detach()[:, cols].clone().requires_grad_()

    def forward(x):
        b, t, _ = x.shape
        q, k, v = (f(x) @ wqkv.T).reshape(b, t, 3, local_h, hd).unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        a = (q @ k.transpose(-2, -1)) / hd**0.5
        a = a.masked_fill(torch.triu(torch.ones(t, t), diagonal=1).bool(), float("-inf")).softmax(-1)
        return g((a @ v).transpose(1, 2).reshape(b, t, local_h * hd) @ wproj.T)

    attn.forward = forward
    return [wqkv, wproj]


def train(ref, rank, ws, variant):
    """The lesson's seeded MiniGPT, data and LR, 20 plain Adam steps; per-step losses, step-0 embedding grad."""
    torch.manual_seed(ref.SEED)
    model, blk = ref.MiniGPT(), ref.BATCH * (ref.SEQ_LEN + 1)
    shards = [w for b in model.blocks for w in shard_attention(b.attn, rank, ws, *variant)] if variant else []
    opt, corpus, losses, first = torch.optim.Adam([*model.parameters(), *shards], lr=ref.LR), None, [], None
    corpus = ref.make_corpus(ref.SEED + 7, ref.STEPS * blk).reshape(ref.STEPS, ref.BATCH, ref.SEQ_LEN + 1)
    for block in corpus:
        opt.zero_grad()
        loss = F.cross_entropy(model(block[:, :-1]).reshape(-1, ref.VOCAB), block[:, 1:].reshape(-1))
        loss.backward()
        first = first or model.tok_embed.weight.grad.norm().item()
        opt.step()
        losses.append(loss.item())
    return {"losses": losses, "embed_grad": first, "held": sum(w.numel() for w in shards)}


def worker(rank, ws, port):
    ref = parity.load_reference(PHASE, LESSON, "main")
    os.environ["GLOO_SOCKET_IFNAME"] = ref._loopback_iface()
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=int(rank), world_size=int(ws))
    out = {name: train(ref, int(rank), int(ws), v) for name, v in {"baseline": None, **VARIANTS}.items()}
    dist.destroy_process_group()
    return out


def compare(runs, base):
    gap = lambda a, b: max(abs(x - y) for x, y in zip(a["losses"], b["losses"]))  # noqa: E731
    return {"step0_gap": abs(runs[0]["losses"][0] - base["losses"][0]), "curve_gap": gap(runs[0], base),
            "rank_gap": gap(runs[0], runs[1]), "embed_ratio": runs[0]["embed_grad"] / base["embed_grad"]}


def solve():
    ranks = launch(2)
    return {"held": ranks[0]["megatron_f_and_g"]["held"], **{n: compare([r[n] for r in ranks], ranks[0]["baseline"]) for n in VARIANTS}}


def wrong(v, ratio):
    return v["step0_gap"] < 1e-5 and v["curve_gap"] > 1e-3 and v["rank_gap"] > 1e-3 and round(v["embed_ratio"], 3) == ratio


def verify(result):
    r, bad = result, ("output_allreduce_only", "dist_nn_all_reduce")
    show = {k: {m: f"{x:.4g}" for m, x in r[k].items()} for k in VARIANTS}
    good = r["megatron_f_and_g"]
    return [
        practice.Check(
            "ANSWER: with Megatron's f and g the loss matches the single-rank baseline at all 20 steps",
            good["step0_gap"] < 1e-5 and good["curve_gap"] < 1e-5 and good["rank_gap"] == 0.0 and r["held"] == 4096,
            f"f+g vs baseline {show['megatron_f_and_g']}; each rank holds {r['held']} of 8192 attention weights",
        ),
        practice.Check(
            "FINDING: allreducing only the attention output matches the loss at step 0 but trains a different model",
            wrong(r[bad[0]], 0.993) and wrong(r[bad[1]], 1.006),
            f"output allreduce only {show[bad[0]]}; torch.distributed.nn all_reduce {show[bad[1]]}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    if sys.argv[1:2] == ["--rank"]:
        raise SystemExit(print(json.dumps(worker(os.environ["RANK"], *sys.argv[2:]))))
    raise SystemExit(practice.selfcheck(globals()))
