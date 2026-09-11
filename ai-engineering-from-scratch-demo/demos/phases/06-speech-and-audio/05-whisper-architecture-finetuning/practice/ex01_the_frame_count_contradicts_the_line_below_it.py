"""Exercise 1 — the frame count contradicts the line printed below it.

    **Easy.** Run `code/main.py`. It tokenizes a Whisper-style prompt, computes
    decoded shape budgets, and prints the chunk schedule for a 10-minute clip.

Reading of the exercise: running it is one command, so the exercise is read as
"run it and check the three things it prints". The prompts are right. The other
two are arithmetic, and both are off by a stated amount.

**Step 2 disagrees with its own next line.** `encoder_frames(30.0)` returns
**2998**, and the sentence immediately below reads "Whisper zero-pads all inputs
to 30 s -> 3000 frames". The sentence is right: Whisper's mel front end centres
its frames, so the count is `seconds * sr / hop` exactly -- 100 per second, 3000
for 30 s. The function uses the un-centred `1 + (samples - 400)//hop` and comes
back **two frames short at every duration**: 98 for 1 s, 998 for 10 s, 2998 for
30 s. Two frames is 20 ms, and the file prints both numbers four lines apart.

**Step 4's Turbo row is produced by overwriting the decoder with an encoder.**
The loop carries `if name == "Turbo": dec = enc`, so the printed "dec 78.7 M" is
four *encoder* blocks -- self-attention and MLP -- for a stack that in a decoder
also carries cross-attention. Restoring it:

| | encoder | decoder | embed | total | published |
|---|---:|---:|---:|---:|---:|
| Turbo, as printed | 629.3 | **78.7** | 70.2 | **778.2 M** | 809 M |
| Turbo, decoder counted as a decoder | 629.3 | **104.9** | 70.2 | **804.4 M** | 809 M |

The hack costs 26.2 M parameters and moves the estimate from **3.8% low to 0.6%
low**. One line of the same loop already computes the right thing and throws it
away.

Two smaller things in the same function. `transformer_params` takes `n_heads` and
never reads it -- head count does not change parameter count, so the argument is
inert. And its embedding term charges `3000 * d_model` rows of audio positional
embedding, while Step 2's own closing line says the encoder sees **1500** tokens
after the stride-2 convolution: 1.92 M rows charged as 3.84 M at `d_model=1280`.

Structure: `centred_frames` is the count Whisper's front end actually produces;
`turbo` rebuilds Step 4's row with and without the overwrite; `prompts` collects
the three Step 1 prompts.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "05-whisper-architecture-finetuning"
SR, HOP, N_FFT = 16000, 160, 400
TURBO = {"layers": 4, "d_model": 1280, "d_ff": 5120, "heads": 20, "vocab": 51865}
BIG_LAYERS, ENCODER_TOKENS, PADDED = 32, 1500, 30.0
PUBLISHED = {"Turbo": 809.0, "Large-v3": 1550.0}
DURATIONS = (1.0, 10.0, 30.0)


def centred_frames(seconds, sr=SR, hop=HOP):
    """What a centred mel front end returns: `seconds * sr / hop`, exactly."""
    return int(seconds * sr / hop)


def turbo(ref):
    """Step 4's Turbo row, as printed and with the decoder counted as a decoder."""
    small = ref.transformer_params(TURBO["layers"], TURBO["d_model"], TURBO["d_ff"],
                                   TURBO["heads"], TURBO["vocab"])
    big = ref.transformer_params(BIG_LAYERS, TURBO["d_model"], TURBO["d_ff"],
                                 TURBO["heads"], TURBO["vocab"])
    encoder, embed = big[0], big[2]
    return {"encoder": encoder, "printed_decoder": small[0], "real_decoder": small[1],
            "embed": embed, "printed": encoder + small[0] + embed,
            "fixed": encoder + small[1] + embed,
            "large": sum(big)}


def prompts(ref):
    return {"en": ref.build_prompt("en", "transcribe", False),
            "fr": ref.build_prompt("fr", "translate", False),
            "ja": ref.build_prompt("ja", "transcribe", True)}


def uses_heads(ref):
    """`n_heads` is a parameter of `transformer_params` that nothing inside it reads."""
    body = inspect.getsource(ref.transformer_params).split(":", 1)[1]
    return "n_heads" in body


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = turbo(ref)
    audio_rows = 3000 * TURBO["d_model"]
    return {
        "frames": {s: (ref.encoder_frames(s), centred_frames(s)) for s in DURATIONS},
        "turbo": rows, "prompts": prompts(ref), "heads_used": uses_heads(ref),
        "audio_rows": audio_rows, "real_rows": ENCODER_TOKENS * TURBO["d_model"],
        "printed_error": abs(rows["printed"] / 1e6 - PUBLISHED["Turbo"]) / PUBLISHED["Turbo"],
        "fixed_error": abs(rows["fixed"] / 1e6 - PUBLISHED["Turbo"]) / PUBLISHED["Turbo"],
    }


def verify(result):
    frames, rows = result["frames"], result["turbo"]
    gaps = {s: want - got for s, (got, want) in frames.items()}
    return [
        practice.Check(
            "ANSWER: the three prompts are right, and only the prompts are",
            result["prompts"]["en"][:2] == ["<|startoftranscript|>", "<|en|>"]
            and "<|notimestamps|>" not in result["prompts"]["ja"],
            f"EN transcribe is {' '.join(result['prompts']['en'])} and the timestamped Japanese "
            f"prompt correctly drops `<|notimestamps|>`, leaving "
            f"{len(result['prompts']['ja'])} tokens against {len(result['prompts']['en'])}",
        ),
        practice.Check(
            "FINDING: `encoder_frames` is two frames short of the line printed below it",
            set(gaps.values()) == {2},
            f"Step 2 prints {frames[PADDED][0]} frames for {PADDED} s and then says Whisper pads "
            f"to {centred_frames(PADDED)}; a centred front end gives seconds*sr/hop exactly, so "
            f"the shortfall is {set(gaps.values())} frames at every duration -- "
            f"{[frames[s][0] for s in DURATIONS]} against "
            f"{[frames[s][1] for s in DURATIONS]}. Two frames is 20 ms",
        ),
        practice.Check(
            "FINDING: Step 4's Turbo decoder is four encoder blocks, not four decoder blocks",
            rows["printed_decoder"] < rows["real_decoder"],
            f"`if name == 'Turbo': dec = enc` prints {rows['printed_decoder'] / 1e6:.1f} M where "
            f"four decoder layers, which also carry cross-attention, are "
            f"{rows['real_decoder'] / 1e6:.1f} M. The loop computes the right number one line "
            f"earlier and discards it",
        ),
        practice.Check(
            "ANSWER: restoring it moves the estimate from 3.8% low to 0.6% low",
            result["fixed_error"] < result["printed_error"] / 5,
            f"{rows['printed'] / 1e6:.1f} M printed against a published "
            f"{PUBLISHED['Turbo']:.0f} M is {result['printed_error'] * 100:.1f}% low; "
            f"{rows['fixed'] / 1e6:.1f} M is {result['fixed_error'] * 100:.1f}%. For comparison "
            f"the unhacked Large-v3 row prints {rows['large'] / 1e6:.1f} M against "
            f"{PUBLISHED['Large-v3']:.0f} M",
        ),
        practice.Check(
            "CONTROL: `n_heads` is inert, and the audio positional table is double-counted",
            not result["heads_used"] and result["audio_rows"] == 2 * result["real_rows"],
            f"`transformer_params` takes `n_heads` and never reads it -- head count does not "
            f"change parameter count -- and charges 3000 rows of audio positional embedding "
            f"({result['audio_rows'] / 1e6:.2f} M) where Step 2's own closing line says the "
            f"encoder sees {ENCODER_TOKENS} tokens after the stride-2 conv "
            f"({result['real_rows'] / 1e6:.2f} M)",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
