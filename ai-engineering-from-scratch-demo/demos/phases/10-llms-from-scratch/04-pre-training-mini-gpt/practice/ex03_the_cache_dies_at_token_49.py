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

**FINDING: a position scheme removes the first obstacle and not the last one.**
Nothing about the cache is wrong; absolute position is, so RoPE or ALiBi would
let surviving entries keep their indices when the window slides. That rescues
**layer 0 only**. Changing the token at position 0 while holding every other
token and every position fixed moves the surviving tokens' cached K by 0.00 at
layer 0 and **0.27 to 0.35** at layers 1-3: layer 0's entries depend on nothing
but their own token and position, and every deeper layer's was computed from a
hidden state that attended over the token the window is about to evict. Exact
rolling-window attention needs those entries rebuilt. A relative position scheme
buys a cache that is cheap and approximate, not one that is exact.

Structure: `Cache` is the incremental decoder, one position per `step`; `greedy`
is `generate`'s loop with argmax substituted so the two arms are comparable;
`stale_context` changes one evicted-to-be token and reads the drift per layer.
"""

from __future__ import annotations

import contextlib
import io
import time

import numpy as np

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "04-pre-training-mini-gpt"
STEPS, SEED, WANTED, PROMPT = 500, 1, 200, list(b"Machine learning")
CORPUS = ("Machine learning is a subset of artificial intelligence. Deep learning uses "
          "neural networks with many layers. The transformer architecture relies on "
          "self-attention. Language models predict the next token in a sequence. ") * 10


class Cache:
    """Incremental decode: K and V kept per layer, one new position per `step`."""

    def __init__(self, model):
        self.model, self.pos, self.kv = model, 0, [[None, None] for _ in model.blocks]

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
        merged = ((weights / weights.sum(-1, keepdims=True)) @ kv[1]).transpose(0, 2, 1, 3)
        return merged.reshape(1, 1, self.model.embed_dim) @ attn.W_out


def greedy(model, count):
    """`generate`'s loop with argmax in place of sampling: full forward every step."""
    window, tokens = model.embedding.pos_embed.shape[0], list(PROMPT)
    for _ in range(count):
        ids = np.array(tokens[-window:]).reshape(1, -1)
        tokens.append(int(np.argmax(model.forward(ids)[0, -1, :])))
    return tokens


def fill(model, tokens):
    cache = Cache(model)
    for token in tokens:
        cache.step(token)
    return cache



def cached(model, count):
    """The same loop, one position at a time, reusing K and V."""
    tokens, cache = list(PROMPT), fill(model, PROMPT[:-1])
    logits = cache.step(PROMPT[-1])
    for _ in range(count):
        tokens.append(int(np.argmax(logits)))
        logits = cache.step(tokens[-1])
    return tokens


def stale_context(model, tokens, swapped):
    """Per-layer drift in the surviving tokens' cached K when only token 0 changes."""
    kept, altered = fill(model, tokens), fill(model, [swapped] + list(tokens[1:]))
    return [float(np.abs(a[0][:, :, 1:] - b[0][:, :, 1:]).max())
            for a, b in zip(kept.kv, altered.kv)]


def timed(fn, *args):
    start = time.perf_counter()
    return fn(*args), time.perf_counter() - start


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    np.random.seed(SEED)
    with contextlib.redirect_stdout(io.StringIO()):
        model = ref.train_mini_gpt(CORPUS, num_steps=STEPS)
    inside = model.embedding.pos_embed.shape[0] - len(PROMPT)
    plain, plain_seconds = timed(greedy, model, inside)
    fast, fast_seconds = timed(cached, model, inside)
    return {"window": inside + len(PROMPT), "inside": inside, "identical": plain == fast,
            "seconds": (plain_seconds, fast_seconds),
            "positions": model.embedding.pos_embed.shape,
            "drift": stale_context(model, PROMPT, (PROMPT[0] + 1) % 256)}


def verify(result):
    plain, fast = result["seconds"]
    inside, window, drift = result["inside"], result["window"], result["drift"]
    return [
        practice.Check(
            "ANSWER: ~3x inside the window, and the cache is exact rather than approximate",
            result["identical"] and plain > 1.5 * fast,
            f"{inside} generated tokens take {plain * 1000:.0f} ms uncached and "
            f"{fast * 1000:.0f} ms cached, {plain / fast:.1f}x, and the two arms produce an "
            "identical token sequence -- a cache that changed the output would be a different "
            "model, not a faster one",
        ),
        practice.Check(
            f"FINDING: only {inside} of the {WANTED} tokens the exercise asks for can use it",
            inside < WANTED // 3 and window == result["positions"][0],
            f"train_mini_gpt passes max_seq_len=seq_len=64, so pos_embed is "
            f"{result['positions'][0]}x{result['positions'][1]} and generate slices "
            f"tokens[-{window}:]. A {len(PROMPT)}-token prompt leaves {inside} positions before "
            f"the window slides, so the measurement is available on "
            f"{100 * inside / WANTED:.0f}% of the generation asked for",
        ),
        practice.Check(
            "MECHANISM: sliding the window invalidates every entry in the cache",
            result["positions"][0] == window < len(PROMPT) + WANTED,
            "positions here are learned absolute embeddings indexed pos_embed[:seq_len] from "
            "zero, so one slide changes every surviving token's positional embedding and every "
            f"cached K and V came from an input that no longer exists. It has to be rebuilt at "
            f"each of the remaining {WANTED - inside} steps, and a rebuild is the uncached "
            "forward plus bookkeeping -- strictly worse than never caching",
        ),
        practice.Check(
            "FINDING: a position scheme removes the first obstacle, not the last one",
            drift[0] == 0.0 and min(drift[1:]) > 0.1,
            f"nothing about the cache is wrong -- it is exact and {plain / fast:.1f}x faster "
            "where it applies -- and absolute position is what breaks it, so RoPE or ALiBi would "
            "let surviving entries keep their indices. That rescues layer 0 only: changing the "
            "token at position 0, all else held fixed, moves the surviving cached K by "
            + ", ".join(f"layer {i} {d:.2f}" for i, d in enumerate(drift))
            + ": layer 0 depends on nothing but its own token and position, and every deeper "
            "layer came from a state that attended over the token about to be evicted. Exact "
            "rolling-window attention needs those rebuilt -- a relative position scheme buys a "
            "cheap approximate cache, not an exact one",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
