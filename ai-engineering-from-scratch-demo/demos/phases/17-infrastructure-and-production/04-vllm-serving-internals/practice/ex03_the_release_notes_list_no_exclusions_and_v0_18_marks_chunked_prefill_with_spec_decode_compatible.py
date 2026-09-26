"""Exercise 3 — the release notes list no exclusions, and v0.18 marks chunked prefill with spec decode compatible.

    Re-read the vLLM v0.18.0 release notes. Which combinations of flags are
    mutually exclusive? List them.

Reading of the exercise: three sources were read on 2026-09-26. The first is
the v0.18.0 GitHub release body (published 2026-03-20). The notes alone name
no exclusions, so the second is the compatibility matrix that v0.18.0 ships,
`docs/features/README.md` at that tag, whose own title is "mutually exclusive
features". The third is the speculative-decoding docs and `arg_utils.py` at
the same tag. The pairs are held in `MATRIX_NO`, and the check tests the
lesson's own claim, and the toy's, against them.

**ANSWER: the release notes list none. The v0.18.0 matrix lists 27
incompatible pairs, 8 of them with speculative decoding.** Speculative
decoding (SD) is incompatible with LoRA, pooling, encoder-decoder models,
async output processing, multi-step, best-of, beam search and prompt embeds.
Chunked prefill (CP) is incompatible with encoder-decoder and multi-step.
Encoder-decoder also excludes prefix caching (APC), LoRA, async output and
prompt embeds. Pooling excludes logprobs, prompt logprobs, async output,
multi-step, best-of, beam search and prompt embeds. Multi-step excludes LoRA,
best-of and beam search. Prompt embeds excludes prompt logprobs and
multimodal. Three pairs are partial: pooling with CP, pooling with APC, and
multimodal with LoRA. Separately, the SD docs list pipeline
parallelism as "not composible with speculative decoding as of
`vllm<=0.15.0`". The matrix still carries V0-era rows such as multi-step, so
treat it as the project's own statement rather than a V1 test result.

**FINDING: the lesson's v0.18.0 gotcha is the opposite of v0.18.0's matrix.**
The lesson says you "cannot combine `--enable-chunked-prefill` with
draft-model speculative decoding". The skill file in `outputs/` hard-rejects
that pairing. v0.18.0's matrix marks CP x SD as compatible. Its SD docs say
draft models were unsupported only in `vllm<=0.10.0`. The toy scheduler has no
speculative mode to test either way: `simulate_continuous` takes only
`(reqs, chunked)`.

**FINDING: `--speculative-model` is not a v0.18.0 flag, and the N-gram line is
about the async scheduler.** v0.18.0's `arg_utils.py` registers
`--speculative-config` and no `--speculative-model`. The one release-note
sentence behind the lesson's "documented exception" reads "NGram speculative
decoding now runs on GPU and is compatible with the async scheduler". That
is the async scheduler, not chunked prefill.

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
        "cp_sd_excluded": {"CP", "SD"} in [set(p) for p in MATRIX_NO],
        "doc_gotcha": "cannot combine `--enable-chunked-prefill` with draft-model" in doc,
        "skill_gotcha": "`--enable-chunked-prefill` + `--speculative-model` combination as "
                        "a hard incompatibility" in skill,
        "toy_params": list(inspect.signature(ref.simulate_continuous).parameters),
        "doc_flag": "--speculative-model" in doc, "doc_ngram": "N-gram GPU" in doc,
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
            f"({SD_DOCS['pipeline parallel']})",
        ),
        practice.Check(
            "FINDING: the lesson's v0.18.0 gotcha is the opposite of v0.18.0's matrix",
            all([result["doc_gotcha"], result["skill_gotcha"], not result["cp_sd_excluded"],
                 result["toy_params"] == ["reqs", "chunked"]]),
            "the lesson and its skill file forbid chunked prefill with draft-model spec "
            f"decode; the matrix marks CP x SD compatible, the SD docs limit the draft-model "
            f"gap to {SD_DOCS['draft model unsupported']}, and the toy takes only "
            f"{result['toy_params']}",
        ),
        practice.Check(
            "FINDING: --speculative-model is not a v0.18.0 flag, and the N-gram line is "
            "about the async scheduler",
            result["doc_flag"] and result["doc_ngram"]
            and "--speculative-model" not in ARG_FLAGS and "async scheduler" in RELEASE["ngram"],
            f"v0.18.0 registers {list(ARG_FLAGS)}; the release says '{RELEASE['ngram']}'",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
