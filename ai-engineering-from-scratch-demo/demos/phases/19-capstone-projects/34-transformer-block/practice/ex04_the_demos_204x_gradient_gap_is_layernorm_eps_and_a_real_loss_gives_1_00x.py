"""Exercise 4 — the variants differ, but the demo's 204x gradient gap comes from eps, and a real loss gives 1.00x.

    Build a sanity check that feeds a `(2, 16, 384)` tensor with `H=6` through both variants and asserts the forward outputs are different (for example, `not torch.allclose`) when weights are initialized identically and dropout is set to zero.

Reading of the exercise: "both variants" means one lesson `TransformerBlock`
with `pre_ln=True` and one with `pre_ln=False`, at d_model 384 and 6 heads.
Dropout is 0 in the config and both blocks run in eval mode. The post-LN block
gets the pre-LN block's `state_dict`. A check is only useful if it can fail,
so it also runs on two pre-LN blocks, where it must report "same".
Separately, the gap the lesson's demo draws between the two variants is
measured again.

**ANSWER: the outputs differ, and the check does fail on a true negative.**
With identical weights and a seed-0 input of (2, 16, 384), `torch.allclose` is
False: the outputs differ by up to 0.58 and by 0.07 on average. The same
check on two identical pre-LN blocks gives a difference of exactly 0.0 and
`allclose` True.

**FINDING: the demo's gradient gap measures LayerNorm's eps, not gradient
flow.** Run as shipped, the demo prints a pre-LN embedding gradient of
0.002258, a post-LN one of 0.000011, and a ratio of 203.62x. The doc calls
this "order of magnitude larger". But the demo's loss is `sum(final_ln(x)^2)`,
which equals B*T*D * var/(var + eps): 12,287.9 against 12,288, a constant.
The only gradient left comes from eps. Raise eps from 1e-5 to 1e-3 and the
pre-LN gradient grows about 100x, to 0.226. Give the same stacks a real
objective instead, cross-entropy through a fixed random 128-way head, and the
embedding gradients are 2.397 (pre-LN) and 2.392 (post-LN): a ratio of
1.00x.

**FINDING: the code never trains anything.** The doc promises a 12-layer
stack "at common learning rates" and says it "shows which one survives". The
demo builds 6 layers, runs one backward pass, and has no optimizer and no
learning rate.

Structure: `blocks()` builds the identically initialised pair; `stacks()`
rebuilds the demo's two stacks; `grads()` measures the embedding gradient
under either loss.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import re

from harness import parity, practice

try:
    import torch
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "34-transformer-block"


def cfg(ref, pre_ln, d_model=384, context_length=16):
    return ref.BlockConfig(d_model=d_model, num_heads=6, context_length=context_length,
                           attn_dropout=0.0, residual_dropout=0.0, pre_ln=pre_ln)


def blocks(ref, second_pre_ln):
    torch.manual_seed(0)
    first, second = ref.TransformerBlock(cfg(ref, True)), ref.TransformerBlock(cfg(ref, second_pre_ln))
    second.load_state_dict(first.state_dict())
    return first.eval(), second.eval()


def compare(ref, second_pre_ln):
    first, second = blocks(ref, second_pre_ln)
    x = torch.randn(2, 16, 384, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        a, b = first(x), second(x)
    diff = (a - b).abs()
    return {"allclose": torch.allclose(a, b), "max": diff.max().item(), "mean": diff.mean().item()}


def stacks(ref, eps=1e-5):
    """The demo's pair: seed 0, 6 blocks, d_model 192, post gets pre's weights, then its tokens."""
    torch.manual_seed(0)
    pre = ref.BlockStack(cfg(ref, True, 192, 64), depth=6)
    post = ref.BlockStack(cfg(ref, False, 192, 64), depth=6)
    post.load_state_dict(pre.state_dict())
    for norm in [m for s in (pre, post) for m in s.modules() if isinstance(m, ref.LayerNorm)]:
        norm.eps = eps
    return pre.eval(), post.eval(), torch.randint(0, 128, (2, 32))


def grads(ref, eps=1e-5, real_loss=False):
    pre, post, tokens = stacks(ref, eps)
    gen = torch.Generator().manual_seed(1)
    head, target = torch.randn(192, 128, generator=gen), torch.randint(0, 128, (64,), generator=gen)
    out = []
    for stack in (pre, post):
        if not real_loss:
            out.append(ref.gradient_norm_at_embedding(stack, tokens))
            continue
        stack.zero_grad(set_to_none=True)
        logits = (stack(tokens) @ head).view(-1, 128)
        torch.nn.functional.cross_entropy(logits, target).backward()
        out.append(stack.embed.weight.grad.norm().item())
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        ref.demo()
    pre, _, tokens = stacks(ref)
    with torch.no_grad():
        loss = pre(tokens).pow(2).sum().item()
    src, doc = inspect.getsource(ref), parity.doc_text(PHASE, LESSON)
    return {
        "differ": compare(ref, False), "same": compare(ref, True),
        "printed": [float(v) for v in re.findall(r"grad norm: ([\d.]+)", log.getvalue())],
        "ratio": float(re.search(r"ratio *: ([\d.]+)x", log.getvalue()).group(1)),
        "loss": loss, "btd": 2 * 32 * 192, "base": grads(ref), "eps3": grads(ref, 1e-3),
        "real": grads(ref, real_loss=True),
        "doc": ["order of magnitude larger" in doc, "common learning rates" in doc],
        "trains": ["optim" in src, "lr" in re.findall(r"\blr\b", src), "depth = 6" in src],
    }


def verify(result):
    r, d, s = result, result["differ"], result["same"]
    real = r["real"][0] / r["real"][1]
    return [
        practice.Check(
            "ANSWER: the outputs differ, and the check does fail on a true negative",
            all([not d["allclose"], 0.5 < d["max"] < 0.7, 0.05 < d["mean"] < 0.1,
                 s["allclose"], s["max"] == 0.0]),
            f"pre vs post: allclose {d['allclose']}, max |diff| {d['max']:.2f}, mean "
            f"{d['mean']:.2f}; pre vs pre: allclose {s['allclose']}, max {s['max']}",
        ),
        practice.Check(
            "FINDING: the demo's gradient gap measures LayerNorm's eps, not gradient flow",
            all([abs(r["printed"][0] - 0.002258) < 5e-5, r["printed"][1] < 5e-5,
                 150 < r["ratio"] < 260, abs(r["loss"] - r["btd"]) < 1, r["doc"][0],
                 50 < r["eps3"][0] / r["base"][0] < 200, 0.9 < real < 1.1]),
            f"demo prints {r['printed']} ({r['ratio']}x); loss {r['loss']:.1f} vs B*T*D "
            f"{r['btd']}; pre-LN grad at eps 1e-5 {r['base'][0]:.6f}, at 1e-3 "
            f"{r['eps3'][0]:.3f}; cross-entropy grads {r['real'][0]:.3f} / {r['real'][1]:.3f} "
            f"= {real:.2f}x",
        ),
        practice.Check(
            "FINDING: the code never trains anything",
            r["doc"][1] and r["trains"] == [False, False, True],
            f"doc says 'common learning rates': {r['doc'][1]}; main.py has optim / lr / "
            f"depth = 6: {r['trains']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
