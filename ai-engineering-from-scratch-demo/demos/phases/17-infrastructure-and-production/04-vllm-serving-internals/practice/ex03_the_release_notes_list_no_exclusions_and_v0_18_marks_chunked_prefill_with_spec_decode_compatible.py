"""Exercise 3 — the release notes list no exclusions, and v0.18 marks chunked prefill with spec decode compatible.

    Re-read the vLLM v0.18.0 release notes. Which combinations of flags are
    mutually exclusive? List them.

Reading of the exercise: three sources were read on 2026-09-26. The first is
the v0.18.0 GitHub release body (published 2026-03-20). The notes alone name
no exclusions, so the second is the compatibility matrix that v0.18.0 ships,
`docs/features/README.md` at that tag, whose own title is "mutually exclusive
features". The third is the speculative-decoding docs and `arg_utils.py` at
the same tag. The pairs are held in `MATRIX_NO`, and the check tests the
lesson's current text, and the toy's, against them.

**ANSWER: the release notes list none. The v0.18.0 matrix lists 27
incompatible pairs, 8 of them with speculative decoding.** Speculative
decoding (SD) is incompatible with LoRA, pooling, encoder-decoder models,
async output processing, multi-step, best-of, beam search and prompt embeds.
Chunked prefill (CP) is incompatible with encoder-decoder and multi-step.
Encoder-decoder also excludes prefix caching (APC), LoRA, async output,
multi-step and prompt embeds. Pooling excludes logprobs, prompt logprobs, async output,
multi-step, best-of, beam search and prompt embeds. Multi-step excludes LoRA,
best-of and beam search. Prompt embeds excludes prompt logprobs and
multimodal. Three pairs are partial: pooling with CP, pooling with APC, and
multimodal with LoRA. Separately, the SD docs list pipeline
parallelism as "not composible with speculative decoding as of
`vllm<=0.15.0`". The matrix still carries V0-era rows such as multi-step, so
treat it as the project's own statement rather than a V1 test result. Its
SD x LoRA cell is also stale: upstream's review cites vllm-project/vllm#21068
(merged 2025-11-08) for LoRA with spec decode on the V1 engine.

**CONTROL: the lesson and its skill file now agree with v0.18.0.** An
earlier lesson text said you "cannot combine `--enable-chunked-prefill` with
draft-model speculative decoding (`--speculative-model`)", and cited an
N-gram GPU exception. Two findings here refuted that: the matrix marks CP x SD
compatible, v0.18.0 registers only `--speculative-config`, and the N-gram
release line is about the async scheduler. Upstream a05d3925 rewrote the
section. It now says the v0.18.0 matrix marks speculative decoding compatible
with chunked prefill and prefix caching, and it names the two SD-docs limits:
pipeline parallelism through v0.15.0 and draft models through v0.10.0. The
check tests each of those against the constants, and tests that neither the
doc nor the skill file still names `--speculative-model` or the N-gram
exception. The skill file now names `--speculative-config`. The toy still has
no speculative mode: `simulate_continuous` takes only `(reqs, chunked)`.

Structure: the sourced text lives in the constants; `solve()` reads the
lesson doc, its skill file and the toy's signature against them.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "04-vllm-serving-internals"
RELEASE = {  # github.com/vllm-project/vllm/releases/tag/v0.18.0, fetched 2026-09-26
    "exclusion_statements": (),
    "ngram": "NGram speculative decoding now runs on GPU and is compatible with the "
             "async scheduler",
}
MATRIX_NO = (  # docs/features/README.md @ v0.18.0, Feature x Feature, each ❌ cell
    ("SD", "LoRA"), ("pooling", "SD"), ("enc-dec", "CP"), ("enc-dec", "APC"),
    ("enc-dec", "LoRA"), ("enc-dec", "SD"), ("logP", "pooling"), ("prmpt logP", "pooling"),
    ("async output", "SD"), ("async output", "pooling"), ("async output", "enc-dec"),
    ("multi-step", "CP"), ("multi-step", "LoRA"), ("multi-step", "SD"),
    ("multi-step", "pooling"), ("multi-step", "enc-dec"), ("best-of", "SD"),
    ("best-of", "pooling"), ("best-of", "multi-step"), ("beam-search", "SD"),
    ("beam-search", "pooling"), ("beam-search", "multi-step"), ("prompt-embeds", "SD"),
    ("prompt-embeds", "pooling"), ("prompt-embeds", "enc-dec"),
    ("prompt-embeds", "prmpt logP"), ("prompt-embeds", "mm"),
)
MATRIX_PARTIAL = (("pooling", "CP"), ("pooling", "APC"), ("mm", "LoRA"))
SD_DOCS = {"pipeline parallel": "vllm<=0.15.0", "draft model unsupported": "vllm<=0.10.0"}
ARG_FLAGS = ("--speculative-config",)  # vllm/engine/arg_utils.py @ v0.18.0


def partners(feature):
    return sorted({a if b == feature else b for a, b in MATRIX_NO if feature in (a, b)})


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    doc = parity.doc_text(PHASE, LESSON)
    skill = (parity.lesson_dir(PHASE, LESSON) / "outputs"
             / "skill-vllm-scheduler-reader.md").read_text(encoding="utf-8")
    return {
        "pairs": len(MATRIX_NO), "sd": partners("SD"), "cp": partners("CP"),
        "doc_matrix": "matrix marks speculative decoding as compatible with chunked "
                      "prefill and prefix caching" in doc,
        "cp_apc_ok": not any({"SD", f} in [set(p) for p in MATRIX_NO] for f in ("CP", "APC")),
        "doc_limits": [v.replace("vllm<=", "v") for v in SD_DOCS.values()
                       if v.replace("vllm<=", "through v") in doc],
        "skill_flag": "`--speculative-config`" in skill,
        "stale": [t for t in ("--speculative-model", "N-gram GPU", "cannot combine")
                  if t in doc or t in skill],
        "toy_params": list(inspect.signature(ref.simulate_continuous).parameters),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the release notes list none; the v0.18.0 matrix lists 27 pairs, "
            "8 with speculative decoding",
            all([RELEASE["exclusion_statements"] == (), result["pairs"] == 27,
                 len(result["sd"]) == 8, result["cp"] == ["enc-dec", "multi-step"]]),
            f"{result['pairs']} ❌ pairs plus {len(MATRIX_PARTIAL)} partial; SD excludes "
            f"{result['sd']}, CP excludes {result['cp']}; SD docs add pipeline parallelism "
            f"({SD_DOCS['pipeline parallel']}); the release's N-gram line reads "
            f"'{RELEASE['ngram']}'",
        ),
        practice.Check(
            "CONTROL: the lesson and its skill file now agree with v0.18.0's matrix",
            all([result["doc_matrix"], result["cp_apc_ok"], len(result["doc_limits"]) == 2,
                 result["skill_flag"], result["stale"] == [],
                 "--speculative-model" not in ARG_FLAGS,
                 result["toy_params"] == ["reqs", "chunked"]]),
            "upstream a05d3925 dropped the chunked-prefill gotcha: the doc says the matrix "
            f"marks SD compatible with CP and APC (neither pair is ❌), names the limits "
            f"{result['doc_limits']}, the skill file uses {list(ARG_FLAGS)}, stale phrases "
            f"left: {result['stale']}; the toy still takes only {result['toy_params']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
