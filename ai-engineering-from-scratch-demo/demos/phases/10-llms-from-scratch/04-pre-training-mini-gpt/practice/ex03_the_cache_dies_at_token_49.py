"""Exercise 3 — the cache is exact, ~3x faster, and dead after 48 of the 200 tokens.

    Add a KV cache to the generation function. Store K and V tensors for each
    layer after the first forward pass, and reuse them for subsequent tokens.
    Measure the speedup: generate 200 tokens with and without the cache and
    compare wall-clock time.

Reading of the exercise: the model is the one the lesson trains --
`train_mini_gpt`'s defaults, so `max_seq_len` is 64 -- and "the generation
function" is `generate`'s loop, re-implemented with `argmax` in place of its
sampling so the cached and uncached arms can be compared token for token rather
than distribution for distribution.

**ANSWER: about 3x, and exact.** Inside the context window, 48 generated tokens
take roughly 40 ms uncached against 13 ms cached, and the two produce an
identical token sequence -- the cache is an optimisation and not an
approximation, which is the thing worth checking first.

**FINDING: 48 of the requested 200 tokens can use it.** `train_mini_gpt` passes
`max_seq_len=seq_len=64`, and `generate` slices `tokens[-seq_len:]`. A 16-token
prompt leaves **48** positions before the window starts sliding.

**MECHANISM: sliding the window invalidates every entry in the cache.** The
model's positions are *learned absolute* embeddings indexed `pos_embed[:seq_len]`
from zero. When the window slides by one, every surviving token's positional
embedding changes, so every cached K and V was computed from the wrong input.
The cache must be rebuilt from scratch at each of the remaining 152 steps, which
costs more than not caching. The exercise's measurement is available on 24% of
the generation it asks for and structurally impossible on the other 76%.

**FINDING: the fix is not a better cache.** Nothing about the cache is wrong;
absolute position is. RoPE and ALiBi make a token's representation depend on
*relative* offset, so a sliding window leaves cached entries valid. The choice
of position encoding is what decides whether a KV cache survives a long
generation, and this model made it before the cache was written.

Structure: `Cache` is the incremental decoder, one position per `step`; `greedy`
is `generate`'s loop with argmax substituted so the two arms are comparable.
"""

from __future__ import annotations

import contextlib
import io
import time

import numpy as np

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "04-pre-training-mini-gpt"
STEPS, SEED, WANTED = 500, 1, 200
PROMPT = list(b"Machine learning")
CORPUS = ("Machine learning is a subset of artificial intelligence. "
          "Deep learning uses neural networks with many layers. "
          "The transformer architecture relies on self-attention. "
          "Language models predict the next token in a sequence. ") * 10


class Cache:
    """Incremental decode: K and V kept per layer, one new position per `step`."""

    def __init__(self, model):
        self.model = model
        self.kv = [[None, None] for _ in model.blocks]
        self.pos = 0

    def step(self, token):
        model = self.model
        x = (model.embedding.token_embed[np.array([[token]])]
             + model.embedding.pos_embed[self.pos])
        for block, kv in zip(model.blocks, self.kv):
            x = x + self._attend(block, kv, block.ln1.forward(x))
            x = x + block.ffn.forward(block.ln2.forward(x))
        self.pos += 1
        return (model.ln_f.forward(x) @ model.embedding.token_embed.T)[0, -1, :]

    def _attend(self, block, kv, h):
        attn = block.attn
        shape = (1, 1, attn.num_heads, attn.head_dim)
        query = (h @ attn.W_q).reshape(shape).transpose(0, 2, 1, 3)
        for slot, weight in ((0, attn.W_k), (1, attn.W_v)):
            new = (h @ weight).reshape(shape).transpose(0, 2, 1, 3)
            kv[slot] = new if kv[slot] is None else np.concatenate([kv[slot], new], axis=2)
        scores = query @ kv[0].transpose(0, 1, 3, 2) / np.sqrt(attn.head_dim)
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights = weights / weights.sum(axis=-1, keepdims=True)
        merged = (weights @ kv[1]).transpose(0, 2, 1, 3)
        return merged.reshape(1, 1, self.model.embed_dim) @ attn.W_out


def greedy(model, count):
    """`generate`'s loop with argmax in place of sampling: full forward every step."""
    window = model.embedding.pos_embed.shape[0]
    tokens = list(PROMPT)
    for _ in range(count):
        context = np.array(tokens[-window:]).reshape(1, -1)
        tokens.append(int(np.argmax(model.forward(context)[0, -1, :])))
    return tokens


def cached(model, count):
    """The same loop, one position at a time, reusing K and V."""
    cache, tokens = Cache(model), list(PROMPT)
    for token in PROMPT[:-1]:
        cache.step(token)
    logits = cache.step(PROMPT[-1])
    for _ in range(count):
        tokens.append(int(np.argmax(logits)))
        logits = cache.step(tokens[-1])
    return tokens


def timed(fn, *args):
    start = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - start


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    np.random.seed(SEED)
    with contextlib.redirect_stdout(io.StringIO()):
        model = ref.train_mini_gpt(CORPUS, num_steps=STEPS)
    window = model.embedding.pos_embed.shape[0]
    inside = window - len(PROMPT)
    plain, plain_seconds = timed(greedy, model, inside)
    fast, fast_seconds = timed(cached, model, inside)
    return {
        "window": window,
        "inside": inside,
        "identical": plain == fast,
        "seconds": (plain_seconds, fast_seconds),
        "full_seconds": timed(greedy, model, WANTED)[1],
        "positions": model.embedding.pos_embed.shape,
    }


def verify(result):
    plain, fast = result["seconds"]
    inside, window = result["inside"], result["window"]
    return [
        practice.Check(
            "ANSWER: ~3x inside the window, and the cache is exact rather than approximate",
            result["identical"] and plain > 1.5 * fast,
            f"{inside} generated tokens take {plain * 1000:.0f} ms uncached and "
            f"{fast * 1000:.0f} ms cached, {plain / fast:.1f}x, and the two arms produce an "
            "identical token sequence. Exactness is the thing to check first: a cache that "
            "changed the output would be a different model, not a faster one",
        ),
        practice.Check(
            f"FINDING: only {result['inside']} of the {WANTED} tokens the exercise asks for can use it",
            inside < WANTED // 3 and window == result["positions"][0],
            f"train_mini_gpt passes max_seq_len=seq_len=64, so pos_embed is "
            f"{result['positions'][0]}x{result['positions'][1]} and generate slices "
            f"tokens[-{window}:]. A {len(PROMPT)}-token prompt leaves {inside} positions before "
            f"the window starts sliding, so the measurement the exercise asks for is available "
            f"on {100 * inside / WANTED:.0f}% of the generation it names",
        ),
        practice.Check(
            "MECHANISM: sliding the window invalidates every entry in the cache",
            result["positions"][0] == window < len(PROMPT) + WANTED,
            "positions here are learned absolute embeddings indexed pos_embed[:seq_len] from "
            "zero, so when the window slides by one every surviving token's positional "
            f"embedding changes and every cached K and V was computed from an input that no "
            f"longer exists. The cache has to be rebuilt at each of the remaining "
            f"{WANTED - inside} steps, which costs more than never caching -- the uncached "
            f"{WANTED}-token run takes {result['full_seconds'] * 1000:.0f} ms and a rebuilt "
            "cache would pay that plus the rebuild",
        ),
        practice.Check(
            "FINDING: the fix is a position scheme, not a better cache",
            plain / fast > 1.5,
            f"nothing about the cache is wrong -- it is exact and {plain / fast:.1f}x faster "
            "where it applies. Absolute position is what breaks it. RoPE and ALiBi make a "
            "token's representation depend on relative offset, so a sliding window leaves "
            "cached entries valid. The position encoding is what decides whether a KV cache "
            "survives a long generation, and this model made that choice before the cache "
            "was written",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
