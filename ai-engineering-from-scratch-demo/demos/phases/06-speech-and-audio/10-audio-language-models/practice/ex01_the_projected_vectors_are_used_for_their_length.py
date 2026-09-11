"""Exercise 1 — the projected vectors are used for their length.

    **Easy.** Run `code/main.py` to see a toy projector pattern + fake LALM
    routing of (audio-embedding, text-tokens) → output tokens.

Reading of the exercise: the routing is the thing to watch, because it is the
only place where the three components are supposed to meet. The encoder half is
right -- `fake_audio_encoder(3.0)` returns **150 frames of 1280** at exactly
**50 frames per second**, which is Whisper-large's encoder rate and width. What
happens after it is not.

**`main()` projects 8 of the 150 frames** -- 5.33% of the clip -- and that is a
compute budget, not a choice: the pure-Python matmul runs at roughly **0.22 s per
frame**, so the whole 3-second clip would take about **34 s**. The demo prints
"(first 8 frames)" and moves on.

**The 32,768 numbers it computes are then used for one thing: `len`.** Step 3
reads

    interleaved = interleave_with_text(list(range(len(projected))), text_tokens)

so the audio side of the sequence is the integers `0..7`. Every `AUDIO` payload
in the interleaved list is an `int`; the 4096-dimensional vectors the projector
produced are discarded on the line after they are printed. The one seam the
lesson exists to show -- projected audio entering the LLM's embedding space --
is the seam that is not connected.

**And `interleave_with_text` does not interleave.** It returns
`[("AUDIO", a) for a in ...] + [("TEXT", t) for t in ...]`: across the 13-item
sequence there is **1** transition between kinds, where an interleaving would
have 12.

Two smaller things. The projector in `code/main.py` is **23.8%** of the projector
the doc's own Step 2 prints -- one `Linear` instead of two, ReLU instead of GELU,
no bias, **5.24 M against 22.03 M** parameters. And it calls `random.seed(1)` on
the global module, so a caller's own random stream does not survive the call.

Structure: `encode` runs the lesson's encoder; `project_sample` times the
truncated projection; `kinds` reads the interleaved sequence back; `doc_params`
and `code_params` count the two projector specifications.
"""

from __future__ import annotations

import random
import time

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "10-audio-language-models"
SECONDS, AUDIO_DIM, LLM_DIM, SHOWN = 3.0, 1280, 4096, 8
TEXT_TOKENS = (2345, 1098, 7, 9821, 65)
WHISPER_RATE = 50


def project_sample(ref, features, count=SHOWN):
    """(projected rows, seconds) for the slice `main()` actually projects."""
    start = time.perf_counter()
    rows = ref.projector(features[:count])
    return rows, time.perf_counter() - start


def kinds(sequence):
    return [kind for kind, _ in sequence]


def transitions(labels):
    """How many times the sequence changes between AUDIO and TEXT."""
    return sum(1 for left, right in zip(labels, labels[1:]) if left != right)


def doc_params(audio=AUDIO_DIM, llm=LLM_DIM):
    """The doc's Step 2 projector: Linear -> GELU -> Linear, both with bias."""
    return audio * llm + llm + llm * llm + llm


def code_params(audio=AUDIO_DIM, llm=LLM_DIM):
    """What `projector` builds: one weight matrix, no bias, no second layer."""
    return audio * llm


def rng_survives(ref):
    """Does a caller's global random stream survive a call to `projector`?"""
    random.seed(999)
    before = random.random()
    random.seed(999)
    ref.projector([[0.0] * 4], audio_dim=4, llm_dim=4)
    return before == random.random()


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    features = ref.fake_audio_encoder(SECONDS)
    rows, seconds = project_sample(ref, features)
    sequence = ref.interleave_with_text(list(range(len(rows))), list(TEXT_TOKENS))
    labels = kinds(sequence)
    return {
        "frames": len(features), "dim": len(features[0]), "rate": len(features) / SECONDS,
        "shown": len(rows), "width": len(rows[0]), "numbers": len(rows) * len(rows[0]),
        "seconds": seconds, "full_seconds": seconds / len(rows) * len(features),
        "payloads": sorted({type(value).__name__ for kind, value in sequence
                            if kind == "AUDIO"}),
        "length": len(sequence), "transitions": transitions(labels),
        "doc": doc_params(), "code": code_params(), "rng_ok": rng_survives(ref),
    }


def verify(result):
    share = result["shown"] / result["frames"]
    return [
        practice.Check(
            "ANSWER: 150 frames at 50 per second, of which main() projects 8",
            result["rate"] == WHISPER_RATE and result["shown"] == SHOWN,
            f"`fake_audio_encoder({SECONDS})` returns {result['frames']} frames of "
            f"{result['dim']} at {result['rate']:.0f} per second -- Whisper-large's encoder "
            f"rate and width -- and `main()` projects {result['shown']} of them, {share:.2%} of "
            f"the clip, into {result['width']} dimensions",
        ),
        practice.Check(
            "MECHANISM: the truncation is a compute budget, not a choice",
            result["full_seconds"] > 20,
            f"the pure-Python matmul runs at {result['seconds'] / result['shown']:.3f} s per "
            f"frame, so all {result['frames']} frames would take "
            f"{result['full_seconds']:.1f} s against the {result['seconds']:.2f} s the demo "
            "spends. The printed '(first 8 frames)' is the whole reason the number is 8",
        ),
        practice.Check(
            "FINDING: the projected vectors are used only for their length",
            result["payloads"] == ["int"],
            f"Step 3 calls `interleave_with_text(list(range(len(projected))), text_tokens)`, so "
            f"every AUDIO payload is an {result['payloads'][0]} index and the "
            f"{result['numbers']} projected numbers are discarded on the line after they are "
            "printed. The one seam the lesson exists to show is the one left unconnected",
        ),
        practice.Check(
            "FINDING: `interleave_with_text` concatenates",
            result["transitions"] == 1,
            f"it returns the audio list followed by the text list, so the {result['length']}-item "
            f"sequence changes kind {result['transitions']} time where an interleaving would "
            f"change {result['length'] - 1}",
        ),
        practice.Check(
            "FINDING: the shipped projector is 23.8% of the one the doc prints",
            result["code"] < result["doc"] / 4,
            f"the doc's Step 2 is `up(act(down(x)))` -- two Linears with bias and a GELU, "
            f"{result['doc'] / 1e6:.2f} M parameters -- and `projector` builds one weight "
            f"matrix with no bias and a ReLU, {result['code'] / 1e6:.2f} M, "
            f"{result['code'] / result['doc']:.1%} of it",
        ),
        practice.Check(
            "CONTROL: `projector` reseeds the global RNG",
            not result["rng_ok"],
            "its first statement is `random.seed(1)`, so a caller's own stream does not survive "
            "the call. Seeding the module RNG to 999, calling `projector`, and drawing again "
            "does not reproduce the value drawn before it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
