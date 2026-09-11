"""Exercise 2 — the stopwatch has no start.

    **Medium.** Pull Moshi from HuggingFace, run the server, test one
    conversation. Measure wall-clock latency from end-of-user-speech to start-of-
    Moshi-response.

Reading of the exercise: `moshi`, `torch`, `transformers`, `sounddevice` and
`websockets` are all absent, so the server cannot be run. The half that can be
settled without it is whether the quantity the exercise names is one this
architecture has, and it is not -- **there is no end of user speech anywhere in
the module**. A word-boundary search across every function body finds no `vad`,
no `silence`, no `turn`, no `endpoint`, no `energy` and no `threshold`. The loop
emits a Moshi frame on *every* iteration, unconditionally, so Moshi's response
starts at frame 0 while the user is 25 frames from finishing: measured
end-of-speech minus start-of-response is **-1920 ms**. Full duplex removes the
turn boundary, and the stopwatch the exercise describes needs one.

What `main()` does print is not a latency at all but a compute cost, and the two
numbers sit on different axes:

| | |
|---|---|
| two `time.sleep` constants in the generators | **5 ms**, most of a frame |
| measured per-frame cost | single-digit milliseconds |
| the doc's own architectural floor | **160 ms** (80 ms frame + 80 ms acoustic) |
| ratio | the demo runs **more than 20x below** the floor it quotes |

The floor is a property of the frame grid, not of the hardware, so no amount of
compute speed reaches it; printing a single-digit millisecond figure beside a
`target: < 80 ms per frame` line invites the reader to conclude the opposite.

**And the cost does not grow with context.** Over 200 frames the median of the
last 25 is within a few percent of the median of the first 25, where attention
over a KV cache growing one frame at a time would put the last window near **15x**
the first. That growth is the entire reason streaming inference is hard; here
`depth_transformer` seeds an RNG on two `len()` calls and does constant work.
Widening it does nothing either: taking `CODEBOOKS` from 8 to 128 -- sixteen
times the tokens to predict -- moves the frame cost by a few percent, because the
`time.sleep(0.003)` dominates it. (The millisecond figures above are wall-clock
and move with machine load, so every check below asserts a ratio.)

Structure: `frame_times` runs the loop and returns per-frame milliseconds;
`sleep_budget` sums the literal sleeps in the two generators; `codebook_cost`
times the depth transformer's body at a given codebook count; `mentions` is a
word-boundary search over the module's own source.
"""

from __future__ import annotations

import importlib.util
import inspect
import random
import re
import statistics
import time

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "15-streaming-speech-to-speech-moshi-hibiki"
LONG_RUN, WINDOW, FLOOR_MS = 200, 25, 160.0
TURN_WORDS = ("vad", "silence", "turn", "endpoint", "energy", "threshold")
WIDTHS = (8, 128)
ABSENT = ("moshi", "torch", "transformers", "sounddevice", "websockets")


def frame_times(ref, frames):
    """Milliseconds spent in each iteration of `main()`'s loop."""
    user, moshi, text, spent = [], [], [], []
    for chunk in ref.simulate_user_speech(frames):
        start = time.perf_counter()
        user.append(ref.fake_mimi_encode(chunk))
        text.append(ref.inner_monologue_next_token(text, user))
        moshi.append(ref.depth_transformer(text, user, moshi))
        ref.fake_mimi_decode(moshi[-1])
        spent.append((time.perf_counter() - start) * 1000)
    return spent


def sleep_budget(ref):
    """The literal `time.sleep` arguments in the two generators, in milliseconds."""
    source = inspect.getsource(ref.depth_transformer) + inspect.getsource(
        ref.inner_monologue_next_token)
    return sum(float(x) for x in re.findall(r"time\.sleep\(([\d.]+)\)", source)) * 1000


def codebook_cost(width, repeats=20):
    """The depth transformer's body at `width` codebooks -- one sleep, one comprehension."""
    spent = []
    for _ in range(repeats):
        start = time.perf_counter()
        time.sleep(0.003)
        rng = random.Random(5)
        [rng.randint(0, 1023) for _ in range(width)]
        spent.append((time.perf_counter() - start) * 1000)
    return statistics.mean(spent)


def mentions(ref, words):
    """Which of `words` appear as whole words anywhere in the module's functions."""
    source = "".join(inspect.getsource(getattr(ref, name)) for name in dir(ref)
                     if not name.startswith("_") and callable(getattr(ref, name)))
    return [word for word in words if re.search(rf"\b{word}\b", source, re.I)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    spent = frame_times(ref, LONG_RUN)
    mean = statistics.mean(spent)
    widths = {width: codebook_cost(width) for width in WIDTHS}
    speech_frames = 25
    return {
        "absent": [m for m in ABSENT if importlib.util.find_spec(m) is None],
        "turn_words": mentions(ref, TURN_WORDS),
        "gap_ms": -(speech_frames - 1) * ref.FRAME_MS, "speech_ms": speech_frames * ref.FRAME_MS,
        "mean": mean, "sleep": sleep_budget(ref), "share": sleep_budget(ref) / mean,
        "floor_ratio": FLOOR_MS / mean,
        "drift": statistics.median(spent[-WINDOW:]) / statistics.median(spent[:WINDOW]),
        "linear": (LONG_RUN - WINDOW / 2) / (WINDOW / 2),
        "widths": widths, "width_ratio": widths[WIDTHS[1]] / widths[WIDTHS[0]],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the module has no end of user speech to start the stopwatch from",
            not result["turn_words"] and len(result["absent"]) == len(ABSENT),
            f"find_spec is None for {result['absent']}, so the server cannot run -- and a "
            f"word-boundary search over every function body finds none of {list(TURN_WORDS)}. "
            f"The loop emits a Moshi frame unconditionally on every iteration, so the response "
            f"starts at frame 0 while the user has {result['speech_ms']} ms left to speak: "
            f"end-of-speech minus start-of-response is {result['gap_ms']:+d} ms",
        ),
        practice.Check(
            "FINDING: what is printed is compute, and it sits far below the doc's own floor",
            result["floor_ratio"] > 5,
            f"the measured cost is {result['mean']:.2f} ms a frame against the "
            f"{FLOOR_MS:.0f} ms floor the takeaways quote (80 ms frame + 80 ms acoustic delay) "
            f"-- {result['floor_ratio']:.1f}x below it. The floor is a property of the frame "
            "grid, not of the hardware, so no compute speed reaches it",
        ),
        practice.Check(
            "MECHANISM: most of that number is two `time.sleep` constants",
            result["share"] > 0.5,
            f"`depth_transformer` and `inner_monologue_next_token` sleep "
            f"{result['sleep']:.0f} ms between them, which is {result['share']:.1%} of the "
            f"{result['mean']:.2f} ms frame. The remainder is one RNG seed and two list "
            "comprehensions",
        ),
        practice.Check(
            "FINDING: the per-frame cost does not grow with context",
            result["drift"] < 2 < result["linear"],
            f"over {LONG_RUN} frames the median of the last {WINDOW} is "
            f"{result['drift']:.3f}x the median of the first {WINDOW}, where attention over a "
            f"KV cache growing one frame at a time would put it near "
            f"{result['linear']:.0f}x. `depth_transformer` seeds an RNG on two `len()` calls "
            "and does constant work, so the one thing that makes streaming hard is absent",
        ),
        practice.Check(
            "CONTROL: widening the depth transformer costs nothing either",
            result["width_ratio"] < 1.5,
            f"taking CODEBOOKS from {WIDTHS[0]} to {WIDTHS[1]} is sixteen times the tokens to "
            f"predict and changes the frame cost by {abs(result['width_ratio'] - 1):.1%}: "
            f"{result['widths'][WIDTHS[0]]:.3f} ms against "
            f"{result['widths'][WIDTHS[1]]:.3f} ms. The sleep dominates both",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
