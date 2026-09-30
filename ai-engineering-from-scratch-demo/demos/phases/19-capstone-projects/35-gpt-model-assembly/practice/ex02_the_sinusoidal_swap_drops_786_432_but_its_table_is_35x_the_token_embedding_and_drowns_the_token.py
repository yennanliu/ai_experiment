"""Exercise 2 -- the sinusoidal swap drops 786,432, but its table is 35x the token embedding and drowns the token.

    Replace the learned position embedding with a sinusoidal table computed at construction time. Confirm the model still forwards and the parameter count drops by 786,432.

Reading of the exercise: the lesson's 124M `GPTModel` is built (seed 0) and
its `pos_embed` module is swapped for one that holds the Vaswani et al.
sin/cos table, computed once in `__init__` as a buffer and indexed by the same
`position_ids` the reference `forward` already passes. The reference model
class is not edited. "Still forwards" means a batch of token ids gives logits
of shape (batch, seq, 50257). The count is the lesson's `count_parameters`.

**ANSWER: it forwards and drops exactly 786,432 = 1024 x 768.** The count goes
from 124,439,808 to 123,653,376, and a (1, 32) batch returns (1, 32, 50257)
logits.

**FINDING: the drop only appears if the table is a buffer.** The obvious
one-liner, `nn.Embedding.from_pretrained(table, freeze=True)`, forwards the
same way, but the lesson's `count_parameters` still counts it: the drop is 0.
A frozen weight is still an `nn.Parameter`, and the counter never checks
`requires_grad`.

**FINDING: at the lesson's initialisation the table drowns the token.** Each
sinusoidal row has norm sqrt(768 / 2) = 19.60, while `_init_weights` draws
token rows at std 0.02, norm 0.554: 35x smaller. Two different tokens at the
same position get input vectors with cosine 0.999. Swap only the last token
of 16 random sequences and the last-position logits keep a mean cosine above
0.99 with the table, against about 0.75 with the learned table, which is
initialised at the same 0.02 as the tokens. Attention Is All You Need scales
token embeddings by sqrt(d_model) for this reason ("In the embedding layers,
we multiply those weights by sqrt(d_model)", section 3.4,
https://arxiv.org/html/1706.03762v7, read 2026-09-29). The lesson's `forward`
does not, so a straight swap starts training from inputs that are almost
all position.

Structure: `table()` is the sin/cos table; `Sinusoidal` holds it as a buffer
(a module, because the reference `forward` calls `self.pos_embed(ids)`);
`swap_cosine()` measures how much the last token moves the output.
"""

from __future__ import annotations

import math

from harness import parity, practice

try:
    import torch
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "35-gpt-model-assembly"


def table(n_pos, d_model):
    """PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1) = cos(same)."""
    pos = torch.arange(n_pos, dtype=torch.float32)[:, None]
    freq = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(1e4) / d_model))
    out = torch.zeros(n_pos, d_model)
    out[:, 0::2], out[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)
    return out


class Sinusoidal(torch.nn.Module):
    def __init__(self, n_pos, d_model):
        super().__init__()
        self.register_buffer("table", table(n_pos, d_model))

    def forward(self, position_ids):
        return self.table[position_ids]


def swap_cosine(model, vocab):
    """Mean cosine of last-position logits when only the last token changes."""
    gen = torch.Generator().manual_seed(1)
    seq = torch.randint(0, vocab, (16, 32), generator=gen)
    alt = seq.clone()
    alt[:, -1] = torch.randint(0, vocab, (16,), generator=gen)
    with torch.no_grad():
        a, b = model(seq)[:, -1], model(alt)[:, -1]
    return torch.cosine_similarity(a, b, dim=-1).mean().item()


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    cfg = ref.GPTConfig()
    torch.manual_seed(0)
    model = ref.GPTModel(cfg).eval()
    before, learned_cos = ref.count_parameters(model), swap_cosine(model, cfg.vocab_size)
    frozen = torch.nn.Embedding.from_pretrained(table(cfg.context_length, cfg.d_model), freeze=True)
    model.pos_embed = frozen
    frozen_count = ref.count_parameters(model)
    model.pos_embed = Sinusoidal(cfg.context_length, cfg.d_model)
    with torch.no_grad():
        shape = tuple(model(torch.randint(0, cfg.vocab_size, (1, 32))).shape)
    tok = model.tok_embed.weight.detach()
    pos = model.pos_embed.table
    return {
        "before": before, "after": ref.count_parameters(model), "frozen": frozen_count,
        "shape": shape, "expected": cfg.context_length * cfg.d_model,
        "pos_norm": pos.norm(dim=1).mean().item(), "tok_norm": tok.norm(dim=1).mean().item(),
        "input_cos": torch.cosine_similarity(tok[1] + pos[5], tok[2] + pos[5], dim=0).item(),
        "learned_cos": learned_cos, "sin_cos": swap_cosine(model, cfg.vocab_size),
    }


def verify(result):
    r = result
    drop, ratio = r["before"] - r["after"], r["pos_norm"] / r["tok_norm"]
    return [
        practice.Check(
            "ANSWER: it forwards and drops exactly 786,432 = 1024 x 768",
            drop == r["expected"] == 786_432 and r["after"] == 123_653_376
            and r["shape"] == (1, 32, 50257),
            f"{r['before']:,} -> {r['after']:,} (drop {drop:,}); logits {r['shape']}",
        ),
        practice.Check(
            "FINDING: the drop only appears if the table is a buffer",
            r["before"] - r["frozen"] == 0,
            f"from_pretrained(freeze=True) drop: {r['before'] - r['frozen']:,}",
        ),
        practice.Check(
            "FINDING: at the lesson's initialisation the table drowns the token",
            round(r["pos_norm"], 2) == 19.60 and 34 < ratio < 36 and r["input_cos"] > 0.99
            and r["sin_cos"] > 0.99 and r["learned_cos"] < 0.85,
            f"row norms: sinusoidal {r['pos_norm']:.2f}, token {r['tok_norm']:.3f} ({ratio:.0f}x); "
            f"input cosine of two tokens at one position {r['input_cos']:.4f}; logits cosine after "
            f"swapping the last token: learned {r['learned_cos']:.3f}, sinusoidal {r['sin_cos']:.4f}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
