"""Exercise 4 — unify the API, not the model, because gpt-oss-20b is 13.8 GB and the Nano runs an 8B.

    Jetson AGX Orin runs gpt-oss-20b at 40 tok/s. Jetson Nano fits only a 3B.
    If your product targets both, how do you unify the inference stack?

Reading of the exercise: "unify" is decided by measurement -- which models
fit each device, and at what bandwidth ceiling -- and then by what the two
candidate models share. Both boards are modelled with the reference
`ceiling()`: AGX Orin is the reference's own 205 GB/s target; Orin Nano Super
is 8 GB at 102 GB/s (NVIDIA technical blog, "Jetson Orin Nano Developer Kit
Gets a Super Boost"). gpt-oss-20b's bytes are counted from its config.json
(24 layers, hidden 2880, 64 query / 8 KV heads of dim 64, 32 experts, 4
active, vocab 201088, MXFP4 experts, BF16 attention and lm_head); its
checkpoint is 13.76 GB (three safetensors files on the Hugging Face repo).

**ANSWER: one OpenAI-compatible API and one engine build, with a model chosen
per device from a manifest.** gpt-oss-20b's 13.76 GB checkpoint does not fit
the Nano's 8 GB. A 3B at INT4 fits both, at a ceiling of 56.5 tok/s on the
Nano and 113.5 on AGX Orin. So the product can ship one model everywhere,
evaluated once, or the largest fit per device -- gpt-oss-20b on AGX, an 8B
INT4 on the Nano -- behind the same endpoint, evaluated twice.
What is shared is the endpoint, the container and the manifest.

**FINDING: the reference ceiling cannot produce 40 tok/s for gpt-oss-20b; the
MoE active bytes can.** `ceiling()` on the whole checkpoint gives 14.9 tok/s
on AGX Orin. Counting only what a token reads -- attention, 4 of 32 experts,
the lm_head -- gives 3.60B parameters, matching the model card's "3.6B
active", and 3.70 GB: a 55.4 tok/s ceiling, so 40 tok/s is 72% of it. The
same count gives 20.9B total and 13.74 GB, matching the card and the files.

**FINDING: the Nano is not limited to a 3B.** NVIDIA's own table for the Orin
Nano Super runs Llama 3.1 8B at 19.14 tok/s and Gemma 2 9B at 9.21 (INT4, MLC
API). 8B INT4 is 4.52 GB, a 22.6 tok/s ceiling, so 19.14 is 85% of it; the
3B's 43.07 tok/s is 76% of 56.5.

**FINDING: the two models share no tokenizer, so the unifying layer is the
chat API, not the prompt.** gpt-oss uses a 201,088-token vocabulary and the
harmony chat format; Llama 3.2 uses 128,256 tokens and its own template.
Token budgets, prompt caching and stop sequences all have to be
per-model, behind `/v1/chat/completions`.

Structure: `gpt_oss_params()` counts parameters from the config;
`manifest()` picks the largest model that fits each device.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "12-edge-inference"
CFG = {"layers": 24, "hidden": 2880, "q_heads": 64, "kv_heads": 8, "head_dim": 64,
       "experts": 32, "active": 4, "ffn": 2880, "vocab": 201088}
CHECKPOINT_GB = (4_792_272_488 + 4_798_702_184 + 4_170_342_232) / 1e9
MXFP4_BPW, BF16_BPW = 4.25, 16
NANO = ("Jetson Orin Nano Super", 102, 8)  # name, GB/s, GB
NVIDIA_NANO = {"Llama 3.1 8B": 19.14, "Llama 3.2 3B": 43.07, "Gemma 2 9B": 9.21}
VOCAB = {"gpt-oss-20b": 201088, "Llama 3.2 3B": 128256}


def gpt_oss_params():
    """(dense, one expert, lm_head) parameter counts from config.json."""
    c = CFG
    q_out, kv_out = c["q_heads"] * c["head_dim"], c["kv_heads"] * c["head_dim"]
    attn = c["hidden"] * (2 * q_out + 2 * kv_out) * c["layers"]
    expert = 3 * c["hidden"] * c["ffn"]  # gate, up, down
    return attn, expert, c["hidden"] * c["vocab"]


def gpt_oss_bytes(active_only):
    attn, expert, head = gpt_oss_params()
    n = CFG["active"] if active_only else CFG["experts"]
    experts = expert * n * CFG["layers"]
    embed = 0 if active_only else head  # one embedding row per token is negligible
    return (experts * MXFP4_BPW + (attn + head + embed) * BF16_BPW) / 8 / 1e9


MODELS = {"gpt-oss-20b": CHECKPOINT_GB, "Llama 3.1 8B INT4": 8.03e9 * 4.5 / 8 / 1e9,
          "Llama 3.2 3B INT4": 3.21e9 * 4.5 / 8 / 1e9}


def manifest(memory_gb):
    return max((gb, m) for m, gb in MODELS.items() if gb < memory_gb)[1]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    agx = next(t for t in ref.TARGETS if t.name == "Jetson AGX Orin")
    nano = ref.Target(NANO[0], NANO[1], None, "8 GB")
    attn, expert, head = gpt_oss_params()
    ceil = {(d.name, m): ref.ceiling(d, gb) for d in (agx, nano) for m, gb in MODELS.items()}
    return {
        "ceil": ceil, "agx_dense": ref.ceiling(agx, CHECKPOINT_GB),
        "agx_active": ref.ceiling(agx, gpt_oss_bytes(True)),
        "active_params": attn + head + expert * CFG["active"] * CFG["layers"],
        "total_params": attn + 2 * head + expert * CFG["experts"] * CFG["layers"],
        "active_gb": gpt_oss_bytes(True), "total_gb": gpt_oss_bytes(False),
        "manifest": {"AGX Orin 64 GB": manifest(64), "Orin Nano 8 GB": manifest(NANO[2])},
    }


def verify(result):
    c, nano = result["ceil"], NANO[0]
    eff8 = NVIDIA_NANO["Llama 3.1 8B"] / c[(nano, "Llama 3.1 8B INT4")]
    eff3 = NVIDIA_NANO["Llama 3.2 3B"] / c[(nano, "Llama 3.2 3B INT4")]
    return [
        practice.Check(
            "ANSWER: one OpenAI-compatible API and one engine build, with a model chosen per device",
            all([CHECKPOINT_GB > NANO[2], round(c[(nano, "Llama 3.2 3B INT4")], 1) == 56.5,
                 round(c[("Jetson AGX Orin", "Llama 3.2 3B INT4")], 1) == 113.5]),
            f"checkpoint {CHECKPOINT_GB:.2f} GB vs {NANO[2]} GB; 3B INT4 ceiling "
            f"{c[(nano, 'Llama 3.2 3B INT4')]:.1f} on the Nano, "
            f"{c[('Jetson AGX Orin', 'Llama 3.2 3B INT4')]:.1f} on AGX; largest fit {result['manifest']}",
        ),
        practice.Check(
            "FINDING: the reference ceiling cannot produce 40 tok/s for gpt-oss-20b; "
            "the MoE active bytes can",
            all([result["agx_dense"] < 40 < result["agx_active"],
                 round(result["active_params"] / 1e9, 1) == 3.6,
                 round(result["total_params"] / 1e9, 1) == 20.9,
                 abs(result["total_gb"] - CHECKPOINT_GB) < 0.05]),
            f"whole checkpoint {result['agx_dense']:.1f} tok/s; active "
            f"{result['active_params'] / 1e9:.2f}B params, {result['active_gb']:.2f} GB, "
            f"{result['agx_active']:.1f} tok/s (40 = {40 / result['agx_active']:.0%}); total "
            f"{result['total_params'] / 1e9:.1f}B, {result['total_gb']:.2f} GB",
        ),
        practice.Check(
            "FINDING: the Nano is not limited to a 3B",
            result["manifest"]["Orin Nano 8 GB"] == "Llama 3.1 8B INT4" and eff8 < 1 and eff3 < 1,
            f"NVIDIA measures 8B at {NVIDIA_NANO['Llama 3.1 8B']} tok/s ({eff8:.0%} of "
            f"{c[(nano, 'Llama 3.1 8B INT4')]:.1f}) and 3B at {NVIDIA_NANO['Llama 3.2 3B']} "
            f"({eff3:.0%})",
        ),
        practice.Check(
            "FINDING: the two models share no tokenizer, so the unifying layer is the chat API",
            VOCAB["gpt-oss-20b"] == CFG["vocab"] != VOCAB["Llama 3.2 3B"],
            f"vocabularies {VOCAB}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
