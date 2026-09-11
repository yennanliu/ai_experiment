"""Exercise 1 — two streams that never touch.

    **Easy.** Run `code/main.py`. It simulates the two-stream + inner-monologue
    architecture symbolically.

Reading of the exercise: "simulates the two-stream + inner-monologue
architecture" is a claim about what the loop models, so the check is whether the
three streams the doc describes are coupled the way the doc says. They are not.
The loop runs -- 25 frames, `25 x 8` user codebooks, `25 x 8` Moshi codebooks, 25
text tokens -- and every arrow between the streams is missing.

**The user stream is one token vector repeated 25 times.** `fake_mimi_encode`
seeds its RNG on `int(mean|x| * 1000)`, and the mean absolute value of a
0.15-amplitude sine is `0.15 * 2/pi = 0.09549` whatever its frequency. Every
frame of `simulate_user_speech` therefore seeds on **95**, so the 25 frames
produce **1** distinct token vector between them. The frequency sweep that
`simulate_user_speech` goes to the trouble of building -- 220 Hz rising by 20 Hz
a frame -- is erased by the encoder.

**The user stream never reaches either generator.** `depth_transformer` seeds on
`len(context_user_mimi) + len(context_moshi_mimi)` and
`inner_monologue_next_token` returns `f"tok_{len(text_so_far)}"`. Both read
lengths; neither reads a value. Replacing the user audio with **silence**, or
with **Gaussian noise**, leaves `moshi_mimi` and `moshi_text` byte-identical --
so the simulation has two streams and zero coupling, and `t` is the only input
Moshi has.

**The inner monologue is a parameter that is never read.** `context_text` is the
first argument of `depth_transformer` and does not appear in its body. The doc's
central claim for the design -- "force it to emit text tokens alongside audio
... this improves semantic coherence" -- is the one arrow the simulation drops.

**And the depth transformer is not one.** The doc says the 8 codebooks "have
inter-codebook dependencies" and are predicted "sequentially" by a small
transformer; the function draws 8 independent `randint`s from a single RNG in a
list comprehension. There is no conditioning of codebook `k` on codebook `k-1`.
The same `main()` also prints its realtime target as `target: &lt; 80 ms`, an
HTML escape that survived into a Python string literal.

Structure: `run` drives one pass of the loop over a given audio stream and
returns the three streams; `flat_stream` and `noise_stream` are the two controls;
`reads` reports whether a function's body mentions one of its own parameters.
"""

from __future__ import annotations

import inspect
import math
import random

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "15-streaming-speech-to-speech-moshi-hibiki"
FRAMES, AMPLITUDE = 25, 0.15


def run(ref, stream):
    """One pass of `main()`'s loop: (user mimi, moshi mimi, moshi text)."""
    user, moshi, text = [], [], []
    for chunk in stream:
        user.append(ref.fake_mimi_encode(chunk))
        text.append(ref.inner_monologue_next_token(text, user))
        moshi.append(ref.depth_transformer(text, user, moshi))
    return user, moshi, text


def flat_stream(ref, frames=FRAMES):
    width = int(ref.SAMPLE_RATE * ref.FRAME_MS / 1000)
    return [[0.0] * width for _ in range(frames)]


def noise_stream(ref, frames=FRAMES, seed=7):
    rnd = random.Random(seed)
    width = int(ref.SAMPLE_RATE * ref.FRAME_MS / 1000)
    return [[rnd.gauss(0, 0.5) for _ in range(width)] for _ in range(frames)]


def reads(function, parameter):
    """Does the body of `function` mention `parameter` at all?"""
    return parameter in inspect.getsource(function).split(":", 1)[1]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    audio = ref.simulate_user_speech(FRAMES)
    spoken = run(ref, audio)
    silent = run(ref, flat_stream(ref))
    noisy = run(ref, noise_stream(ref))
    seeds = {int(sum(abs(x) for x in chunk) / len(chunk) * 1000) for chunk in audio}
    codebook_source = inspect.getsource(ref.depth_transformer)
    return {
        "frames": len(spoken[0]), "codebooks": ref.CODEBOOKS,
        "text": len(spoken[2]), "seeds": len(seeds), "seed": sorted(seeds),
        "mean_abs": AMPLITUDE * 2 / math.pi,
        "vectors": len({tuple(v) for v in spoken[0]}),
        "silent_match": (spoken[1], spoken[2]) == (silent[1], silent[2]),
        "noisy_match": (spoken[1], spoken[2]) == (noisy[1], noisy[2]),
        "user_differs": spoken[0] != silent[0],
        "text_reads_user": reads(ref.inner_monologue_next_token, "user_mimi_stream"),
        "depth_reads_text": reads(ref.depth_transformer, "context_text"),
        "comprehension": "for" in codebook_source.split("return")[1].split("]")[0],
        "escaped": "&lt;" in inspect.getsource(ref.main),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the loop runs and produces all three streams",
            result["frames"] == result["text"] == FRAMES and result["codebooks"] == 8,
            f"{result['frames']} frames of {result['codebooks']} codebooks on each audio "
            f"stream and {result['text']} inner-monologue tokens, over "
            f"{FRAMES * 80} ms of simulated audio at {1000 / 80:.1f} Hz",
        ),
        practice.Check(
            "FINDING: all 25 user frames encode to one token vector",
            result["seeds"] == 1 and result["vectors"] == 1,
            f"`fake_mimi_encode` seeds on int(mean|x| * 1000), and the mean absolute value of "
            f"a {AMPLITUDE} sine is {AMPLITUDE} * 2/pi = {result['mean_abs']:.5f} whatever its "
            f"frequency, so every frame seeds on {result['seed']} and the {FRAMES} frames yield "
            f"{result['vectors']} distinct vector. The 220 Hz sweep is erased by the encoder",
        ),
        practice.Check(
            "FINDING: silence and noise produce the same Moshi streams as speech",
            result["silent_match"] and result["noisy_match"] and result["user_differs"],
            "`depth_transformer` seeds on len(user) + len(moshi) and "
            "`inner_monologue_next_token` returns f'tok_{len(text_so_far)}' -- both read "
            "lengths, neither reads a value. Swapping the audio for silence or for Gaussian "
            "noise leaves `moshi_mimi` and `moshi_text` byte-identical, though `user_mimi` "
            "does change. The frame index is the only input Moshi has",
        ),
        practice.Check(
            "FINDING: the inner monologue is a parameter that is never read",
            not result["depth_reads_text"] and not result["text_reads_user"],
            "`context_text` is the first argument of `depth_transformer` and does not appear "
            "in its body; `user_mimi_stream` is the second argument of "
            "`inner_monologue_next_token` and does not appear in its. The doc's central claim "
            "-- emit text alongside audio to improve coherence -- is the arrow that is dropped",
        ),
        practice.Check(
            "CONTROL: the depth transformer predicts the codebooks independently",
            result["comprehension"] and result["escaped"],
            f"the doc says the {result['codebooks']} codebooks 'have inter-codebook "
            "dependencies' and are predicted 'sequentially'; the body draws them in one list "
            "comprehension from a single RNG, with no conditioning of codebook k on k-1. The "
            "same `main()` prints its realtime target as 'target: &lt; 80 ms' -- an HTML "
            "escape that survived into a string literal",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
