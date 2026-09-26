"""Exercise 5 — the EAGLE-3 head for 70B is bigger than Llama 3.2 1B, but reads less per step.

    Compute the memory cost of the EAGLE-3 draft head for Llama 3.3 70B. How
    does it compare to running Llama 3.2 1B as a classic draft?

Reading of the exercise: "memory cost" is three numbers: weights resident,
weights read per draft step, and KV cache per token. They are computed from
the published configs. `yuhuili/EAGLE3-LLaMA3.3-Instruct-70B`, the EAGLE-3
authors' head, is `LlamaForCausalLM` with hidden 6144, intermediate 16384,
48 heads, 8 KV heads, 1 layer, target_hidden 8192, vocab 128256 and
draft_vocab 32000. Llama 3.2 1B has hidden 2048, intermediate 8192,
16 layers, 8 KV heads of dim 64, and tied embeddings. Llama 3.3 70B has
hidden 8192, intermediate 28672, 80 layers and 8 KV heads of dim 128. The
configs were read from Hugging Face on 2026-09-26. So were the per-tensor
byte counts of the head's `pytorch_model.bin`, taken from its zip central
directory; the head's parameter count is checked against those bytes.

**ANSWER: the head is 1.576B parameters, 3.15 GB in fp16, which is 1.28x
Llama 3.2 1B's 1.236B (2.47 GB in bf16).** Its parameters: an `fc` fusing
three 8192-wide target layers into 6144 (151M); one decoder layer whose
q/k/v read the 12288-wide concatenation of embedding and fused features
(440M); a 32000-token draft lm_head (197M); and a full-vocabulary embedding
table (788M). The config-derived sum matches the checkpoint's tensor bytes
exactly. Each draft is about 2% of the 141 GB bf16 target: the head is 2.2%
and the 1B is 1.75%.

**FINDING: half the head is an embedding table, so it reads 0.64x the 1B's
bytes per step.** The 788M-parameter embedding is a row lookup. What a draft
step streams is `fc`, the layer and the 32000-row head: 788M parameters,
1.58 GB. The 1B streams all 16 layers plus its tied 128256-row lm_head, which
is 1.236B parameters and 2.47 GB. One layer also means one sequential kernel
chain per draft token instead of 16.

**FINDING: KV cache is where they differ most, 8x.** The head keeps 4 KiB
per token, the 1B 32 KiB, and the 70B target 320 KiB. At 256 sequences of
8192 tokens that is 8 GiB, 64 GiB and 640 GiB: a 1.25% overhead on the target
against 10%.

**FINDING: the lesson's analyzer has no memory term at all.** `SpecPoint`
has four fields: alpha, k, verify_overhead and concurrency. Weights, KV and
draft forwards are not modelled separately; they hide inside
verify_overhead.

Structure: `params()` builds each model's tensors from its config; the byte
table is the measured checkpoint.
"""

from __future__ import annotations

import dataclasses

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "05-eagle3-speculative-decoding"
VOCAB, GB, KIB, GIB = 128256, 1e9, 1024, 1024**3
# tensor bytes in the head's pytorch_model.bin, read from its zip central directory, in
# order: d2t, t2d, q, k, v, o, gate, up, down, 4 norms, fc, embed_tokens, lm_head
CHECKPOINT_BYTES = [
    int(n)
    for n in (
        "256000 128256 150994944 25165824 25165824 75497472 201326592 201326592 "
        "201326592 12288 12288 12288 12288 301989888 1576009728 393216000"
    ).split()
]


def layer(h, inter, heads_kv, head_dim, q_in=None):
    q_in = q_in or h
    attn = q_in * h + 2 * q_in * heads_kv * head_dim + h * h
    return attn + 3 * h * inter + 2 * h


def params():
    """Parameter counts: (total, streamed per draft step)."""
    h, inter = 6144, 16384
    eagle = {
        "fc": 3 * 8192 * h,
        "layer": layer(h, inter, 8, 128, q_in=2 * h) + 2 * h,
        "lm_head": 32000 * h,
        "embed": VOCAB * h,
    }
    small = {"layers+norm": 16 * layer(2048, 8192, 8, 64) + 2048, "embed": VOCAB * 2048}
    target = 80 * layer(8192, 28672, 8, 128) + 8192 + 2 * VOCAB * 8192
    return eagle, small, target


def kv_per_token(layers, heads_kv, head_dim, dtype_bytes=2):
    return 2 * layers * heads_kv * head_dim * dtype_bytes


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    eagle, small, target = params()
    kv = {
        "eagle": kv_per_token(1, 8, 128),
        "1b": kv_per_token(16, 8, 64),
        "target": kv_per_token(80, 8, 128),
    }
    ctx = 256 * 8192
    return {
        "eagle": sum(eagle.values()),
        "eagle_parts": eagle,
        "small": sum(small.values()),
        "target_gb": 2 * target / GB,
        "fp16_matches": 2 * sum(eagle.values()) + 256000 + 128256
        == sum(CHECKPOINT_BYTES),
        "eagle_step": sum(eagle.values()) - eagle["embed"],
        "small_step": sum(small.values()),
        "kv_kib": {k: v / KIB for k, v in kv.items()},
        "kv_gib": {k: v * ctx / GIB for k, v in kv.items()},
        "fields": [f.name for f in dataclasses.fields(ref.SpecPoint)],
    }


def verify(result):
    e, s, t = result["eagle"], result["small"], result["target_gb"]
    step = result["eagle_step"] / result["small_step"]
    kv, gib = result["kv_kib"], result["kv_gib"]
    return [
        practice.Check(
            "ANSWER: the head is 1.576B parameters, 3.15 GB fp16, 1.28x Llama 3.2 1B's 1.236B",
            all(
                [
                    result["fp16_matches"],
                    round(e / 1e9, 3) == 1.576,
                    round(s / 1e9, 3) == 1.236,
                    round(e / s, 2) == 1.28,
                    round(2 * e / GB / t, 3) == 0.022,
                    round(2 * s / GB / t, 4) == 0.0175,
                ]
            ),
            f"parts {result['eagle_parts']}; fp16 {2 * e / GB:.2f} GB (matches checkpoint "
            f"tensor bytes: {result['fp16_matches']}) vs 1B {2 * s / GB:.2f} GB; target bf16 {t:.0f} GB",
        ),
        practice.Check(
            "FINDING: half the head is an embedding table, so it reads 0.64x the 1B's bytes per step",
            (round(result["eagle_parts"]["embed"] / e, 2), round(step, 2))
            == (0.5, 0.64),
            f"streamed per draft step: head {result['eagle_step'] / 1e6:.0f}M params, "
            f"1B {result['small_step'] / 1e6:.0f}M ({step:.2f}x)",
        ),
        practice.Check(
            "FINDING: KV cache is where they differ most, 8x",
            kv == {"eagle": 4.0, "1b": 32.0, "target": 320.0}
            and gib == {"eagle": 8.0, "1b": 64.0, "target": 640.0},
            f"KiB per token {kv}; at 256 x 8192 tokens GiB {gib}",
        ),
        practice.Check(
            "FINDING: the lesson's analyzer has no memory term at all",
            result["fields"] == ["alpha", "k", "verify_overhead", "concurrency"],
            f"SpecPoint fields {result['fields']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
