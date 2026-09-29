"""Exercise 3 — cross-attention costs 5.4x self-attention, and 73% of that is the K/V projection, not Nt*Nv.

    Profile the cross-attention vs the self-attention layer at `Nt=64, Nv=576` (a 24x24 grid at higher resolution). The cross-attention cost is `Nt * Nv` and dominates at high image resolution.

Reading of the exercise: "profile" is read as counting the FLOPs of one
lesson `CausalSelfAttention` and one lesson `CrossAttention` inside a default
`DecoderBlock` (hidden 256, 8 heads), batch 1, `Nt=64` text tokens against
`Nv=576` image tokens. `torch.utils.flop_counter.FlopCounterMode` counts them,
which is exact and deterministic. Wall-clock medians are printed alongside
but not asserted. Cross-attention is profiled twice: as called by the decoder
(it projects the image into K and V itself) and with a precomputed `kv_cache`.
Then `Nv` is swept to find where cross-attention starts to cost more.

**ANSWER: cross-attention dominates, 205.5M FLOPs against 37.7M (5.44x).**
With the K/V cache it is still larger, 54.5M (1.44x). For reference, the
block's feed-forward costs 67.1M.

**FINDING: most of the cost is the K/V projection, not the `Nt * Nv` term.**
The `Nt * Nv` score and weighted-sum matmuls cost 4*Nt*Nv*D = 37.7M, only 18%
of the uncached cross-attention. Projecting 576 image tokens into K and V
costs 4*Nv*D^2 = 151.0M, which is 73%. That term scales with `Nv`, not with
`Nt * Nv`. The sweep puts the crossovers at exactly `Nv = Nt` = 64 without a
cache and at `Nv = D + Nt` = 320 with one.

**FINDING: the shipped KV cache saves nothing.** `VisionLanguageDecoder`
rebuilds the cache inside every `forward`, so a full decoder pass costs
1,275,068,416 FLOPs with `use_cache=True` and exactly the same with
`use_cache=False`. The doc says the image keys and values "are computed once
at the start of the decode" and reused for every step. Nothing in the code
keeps a cache across calls.

Structure: `flops()` wraps FlopCounterMode; `profile()` measures one
configuration; `solve()` sweeps `Nv` and runs the decoder both ways.
"""

from __future__ import annotations

import time

from harness import parity, practice

try:
    import torch
    from torch.utils.flop_counter import FlopCounterMode
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "61-cross-attention-fusion"
NT, NV, D = 64, 576, 256


def flops(fn):
    with FlopCounterMode(display=False) as counter, torch.no_grad():
        fn()
    return counter.get_total_flops()


def median_ms(fn, n=20):
    times = []
    with torch.no_grad():
        for _ in range(n):
            start = time.perf_counter()
            fn()
            times.append(time.perf_counter() - start)
    return round(sorted(times)[n // 2] * 1e3, 3)


def profile(ref, nv, timed=False):
    cfg = ref.DecoderConfig(vision_tokens=nv, max_text_len=NT)
    torch.manual_seed(0)
    block = ref.DecoderBlock(cfg).eval()
    gen = torch.Generator().manual_seed(0)
    x, memory = torch.randn(1, NT, D, generator=gen), torch.randn(1, nv, D, generator=gen)
    mask, cache = ref.causal_mask(NT), block.cross_attn.project_memory(memory)
    calls = {
        "self": lambda: block.self_attn(x, mask=mask),
        "cross": lambda: block.cross_attn(x, memory),
        "cached": lambda: block.cross_attn(x, memory, kv_cache=cache),
        "ffn": lambda: block.ffn(x),
        "kv_proj": lambda: block.cross_attn.project_memory(memory),
    }
    out = {name: flops(fn) for name, fn in calls.items()}
    if timed:
        out["ms"] = {name: median_ms(calls[name]) for name in ("self", "cross", "cached")}
    return out


def decoder_flops(ref):
    cfg = ref.DecoderConfig(vision_tokens=NV, max_text_len=NT)
    torch.manual_seed(0)
    dec = ref.VisionLanguageDecoder(cfg).eval()
    ids = torch.randint(0, cfg.text_vocab, (1, NT), generator=torch.Generator().manual_seed(0))
    memory = torch.randn(1, NV, D, generator=torch.Generator().manual_seed(1))
    return [flops(lambda c=c: dec(ids, memory, use_cache=c)) for c in (False, True)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    sweep = {nv: profile(ref, nv) for nv in (32, 64, 128, 320, 576)}
    even = [[nv for nv, p in sweep.items() if p[k] == p["self"]] for k in ("cross", "cached")]
    return {"main": profile(ref, NV, timed=True), "even": even, "decoder": decoder_flops(ref),
            "doc": "computed once at the start of the decode" in parity.doc_text(PHASE, LESSON)}


def verify(result):
    m, dec, (even_uncached, even_cached) = result["main"], result["decoder"], result["even"]
    nt_nv = 4 * NT * NV * D
    return [
        practice.Check(
            "ANSWER: cross-attention dominates, 205.5M FLOPs against 37.7M (5.44x)",
            (m["self"], m["cross"], m["cached"], m["ffn"])
            == (37_748_736, 205_520_896, 54_525_952, 67_108_864),
            f"self {m['self']:,}, cross {m['cross']:,} ({m['cross'] / m['self']:.2f}x), cached "
            f"cross {m['cached']:,} ({m['cached'] / m['self']:.2f}x), ffn {m['ffn']:,}; median "
            f"ms (not asserted) {m['ms']}",
        ),
        practice.Check(
            "FINDING: most of the cost is the K/V projection, not the Nt*Nv term",
            all([m["kv_proj"] == 4 * NV * D * D, m["cross"] - m["cached"] == m["kv_proj"],
                 round(nt_nv / m["cross"], 2) == 0.18, round(m["kv_proj"] / m["cross"], 2) == 0.73,
                 even_uncached == [NT], even_cached == [D + NT]]),
            f"Nt*Nv term {nt_nv:,} ({nt_nv / m['cross']:.0%}), kv projection {m['kv_proj']:,} "
            f"({m['kv_proj'] / m['cross']:.0%}); cross == self at Nv {even_uncached} uncached, "
            f"{even_cached} cached",
        ),
        practice.Check(
            "FINDING: the shipped KV cache saves nothing",
            dec[0] == dec[1] == 1_275_068_416 and result["doc"],
            f"decoder forward FLOPs use_cache=False {dec[0]:,}, use_cache=True {dec[1]:,}; doc "
            f"says 'computed once at the start of the decode': {result['doc']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
