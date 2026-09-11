"""Exercise 3 — neither end of the projector fits the components named.

    **Hard.** Build a minimal audio-captioning baseline: BEATs encoder + 2-layer
    projector + frozen Llama-3.2-1B. Fine-tune only the projector on AudioCaps.
    Compare to SALMONN on Clotho-AQA.

Reading of the exercise: `transformers`, `torch`, `peft`, `datasets` and
`soundfile` are all absent, so no fine-tune runs. The shapes can be checked
exactly, and they are what the exercise turns on: the projector the lesson hands
you does not connect the two components the exercise names, at either end.

The doc's Step 2 projector is `Linear(1280, 4096) -> GELU -> Linear(4096, 4096)`.
BEATs emits **768** dimensions and Llama-3.2-1B's hidden size is **2048**. So
`audio_dim` is 1.67x too wide and `llm_dim` is exactly **2x** too wide:

| projector | parameters |
|---|---:|
| the doc's, 1280 -> 4096 -> 4096 | **22.03 M** |
| the exercise's own parts, 768 -> 2048 -> 2048 | **5.77 M** |
| `code/main.py`, one layer, no bias | 5.24 M |

The doc's specification is **3.82x** the size of the one the exercise's components
actually need, and `code/main.py` ships a single layer where the exercise asks for
two (Exercise 1).

**Nothing checks a shape.** Passing a 768-dimensional BEATs frame to `projector`
at its default `audio_dim=1280` raises **`IndexError`** -- it indexes `f[j]` past
the end of the frame. Passing `audio_dim=768` works and returns **4096**-wide
vectors, which a 2048-wide Llama cannot consume, with no error at all. One
mismatch crashes, the other does not, and the silent one is the one that would
reach training.

**The comparison has no shared metric.** The exercise fine-tunes on **AudioCaps**,
which is audio captioning scored with CIDEr or SPIDEr, and compares the result to
SALMONN on **Clotho-AQA**, which is audio question answering scored with accuracy.
Different task, different corpus, different unit -- there is no number the two
sides both produce.

What the design does buy is the thing the exercise is really about: a correctly
shaped projector is **0.4670%** of a frozen 1.24 B-parameter Llama-3.2-1B, so
fp32 Adam state is **69.3 MB** against **14.8 GB** for a full fine-tune.

Structure: `two_layer` counts the doc's projector shape at any pair of widths;
`shape_error` reports what happens when a BEATs frame meets the default
`audio_dim`; `spec_text` reads the exercise back out of the reference doc to
check which corpora it names.
"""

from __future__ import annotations

import importlib.util

from harness import coverage, parity, practice

PHASE, LESSON = "06-speech-and-audio", "10-audio-language-models"
WHISPER_DIM, DOC_LLM_DIM = 1280, 4096
BEATS_DIM, LLAMA_DIM, LLAMA_PARAMS = 768, 2048, 1_235_814_400
ADAM_BYTES = 12
CORPORA = ("AudioCaps", "Clotho-AQA")
ABSENT = ("transformers", "torch", "peft", "datasets", "soundfile")


def two_layer(audio, llm):
    """The doc's Step 2 shape: Linear -> GELU -> Linear, both carrying a bias."""
    return audio * llm + llm + llm * llm + llm


def one_layer(audio, llm):
    """What `projector` builds: one weight matrix, no bias."""
    return audio * llm


def shape_error(ref, width=BEATS_DIM):
    """What `projector` does with a BEATs-width frame at its default `audio_dim`."""
    try:
        ref.projector([[0.01] * width])
    except Exception as exc:
        return type(exc).__name__
    return "accepted"


def emitted_width(ref, width=BEATS_DIM):
    """The width `projector` returns when told the encoder's real dimension."""
    return len(ref.projector([[0.01] * width], audio_dim=width)[0])


def spec_text(pack_index=3):
    """The exercise as upstream writes it, so the corpora it names are read not typed."""
    return coverage.exercise_block(parity.doc_text(PHASE, LESSON, "en"))[pack_index - 1]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    fitted = two_layer(BEATS_DIM, LLAMA_DIM)
    text = spec_text()
    return {
        "absent": [m for m in ABSENT if importlib.util.find_spec(m) is None],
        "doc": two_layer(WHISPER_DIM, DOC_LLM_DIM), "fitted": fitted,
        "shipped": one_layer(WHISPER_DIM, DOC_LLM_DIM),
        "default_error": shape_error(ref), "emitted": emitted_width(ref),
        "named": [name for name in CORPORA if name in text],
        "share": fitted / LLAMA_PARAMS,
        "projector_state": fitted * ADAM_BYTES, "full_state": LLAMA_PARAMS * ADAM_BYTES,
    }


def verify(result):
    return [
        practice.Check(
            "CONTROL: nothing can fine-tune, and the shapes do not need it",
            len(result["absent"]) == len(ABSENT),
            f"find_spec is None for {result['absent']} and neither AudioCaps nor Clotho-AQA is "
            "here. Every number below is arithmetic over the widths the doc and the exercise "
            "each name, and none of it needs a GPU to be wrong",
        ),
        practice.Check(
            "ANSWER: the doc's projector is 2x too wide at the LLM end and 1.67x at the audio end",
            result["doc"] > 3.5 * result["fitted"],
            f"the doc's Step 2 is Linear({WHISPER_DIM}, {DOC_LLM_DIM}) -> GELU -> "
            f"Linear({DOC_LLM_DIM}, {DOC_LLM_DIM}), {result['doc'] / 1e6:.2f} M parameters, "
            f"while BEATs emits {BEATS_DIM} and Llama-3.2-1B's hidden size is {LLAMA_DIM}: the "
            f"projector those two need is {result['fitted'] / 1e6:.2f} M, "
            f"{result['doc'] / result['fitted']:.2f}x smaller",
        ),
        practice.Check(
            "MECHANISM: one mismatch crashes and the other passes silently",
            result["default_error"] == "IndexError" and result["emitted"] == DOC_LLM_DIM,
            f"a {BEATS_DIM}-wide BEATs frame at the default audio_dim={WHISPER_DIM} raises "
            f"{result['default_error']}, because the loop indexes f[j] past the end of the "
            f"frame; told audio_dim={BEATS_DIM} it works and returns {result['emitted']}-wide "
            f"vectors that a {LLAMA_DIM}-wide Llama cannot consume, with no error. The silent "
            "one is the one that reaches training",
        ),
        practice.Check(
            "FINDING: the comparison has no shared metric",
            result["named"] == list(CORPORA),
            f"the exercise text names {result['named']}: it fine-tunes on the first, which is "
            "audio captioning scored with CIDEr or SPIDEr, and compares to SALMONN on the "
            "second, which is audio question answering scored with accuracy. Different task, "
            "different corpus, different unit -- no number both sides produce",
        ),
        practice.Check(
            "MECHANISM: a correctly shaped projector is 0.47% of the frozen model",
            result["share"] < 0.005 and result["full_state"] / result["projector_state"] > 200,
            f"{result['fitted'] / 1e6:.2f} M against Llama-3.2-1B's "
            f"{LLAMA_PARAMS / 1e9:.2f} B is {result['share'] * 100:.4f}%, so fp32 Adam at "
            f"{ADAM_BYTES} bytes per trainable parameter is "
            f"{result['projector_state'] / 1e6:.1f} MB of optimizer state against "
            f"{result['full_state'] / 1e9:.1f} GB for a full fine-tune",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
