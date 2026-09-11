"""Exercise 3 — twenty utterances, and two different quantities.

    **Hard.** Take your Lesson 12 pipeline agent and compare P50 latency vs Moshi
    on 20 matched test utterances. Write up when a pipeline architecturally wins
    anyway.

Reading of the exercise: both modules exist and both print a number in
milliseconds, so the comparison can be run -- and running it is how you find that
**the two numbers are not the same quantity**. Lesson 12 reports "user-perceived
(to first audio)", a turn-level delay that begins when the user stops speaking.
Lesson 15 reports the cost of processing one 80 ms frame. Put side by side over
20 matched utterances they read **~804 ms against ~7 ms**, a ratio of order **100x**,
and the ratio says nothing about either architecture: one number contains a
400 ms end-of-speech wait and two model calls, the other contains a `sleep(0.003)`.

**The pipeline's latency is constants.** `streaming_stt` returns the literal
`"set a timer for five minutes"` for any input; `llm_with_tools` sleeps 0.12 s
and branches on a substring of that constant; `streaming_tts` sleeps 0.10 s for
one word and for two hundred; the end-of-speech wait is the literal `400`. The
only input-dependent term in the whole pipeline is `streaming_stt`'s
`len(utterance)/sr * 0.05`, which is why 20 utterances of 1-2 s spread over about
**1.5%** relative standard deviation. Twenty matched utterances are twenty
measurements of one constant.

**And twenty is what fixes what a P50 can say.** A median has no standard error
to quote, but it has an exact distribution-free interval: for `n = 20` the
tightest symmetric 95% confidence interval for the median is the **6th to the
15th order statistic**, coverage **0.9586**. A P50 from 20 samples is that band,
not a point.

What that band costs, measured by simulation on log-normal latencies:

| true ratio between the two systems | detected at n=20 vs 20, sigma=0.20 | sigma=0.35 |
|---:|---:|---:|
| 1.10x | **30%** | **13%** |
| 1.25x | 91% | **51%** |
| 1.50x | 100% | 92% |

The doc's own cheatsheet puts Moshi at 200-300 ms and the managed pipelines at
300-500 ms -- overlapping ranges whose ratio could be anywhere from 1.0x to 2.5x.
At the low end of that, 20 utterances is a coin flip.

Structure: `pipeline_turn` runs Lesson 12's measurable stages once; `frame_cost`
runs one Lesson 15 frame; `median_interval` gives the order-statistic coverage;
`detect_rate` is the power simulation.
"""

from __future__ import annotations

import math
import random
import statistics
import time

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "15-streaming-speech-to-speech-moshi-hibiki"
PIPELINE_LESSON = "12-voice-assistant-pipeline"
N, SR, END_OF_SPEECH_MS = 20, 16000, 400.0
RATIOS, SIGMAS, TRIALS = (1.10, 1.25, 1.50), (0.20, 0.35), 600


def pipeline_turn(pipe, samples):
    """Lesson 12's user-perceived delay: its 400 ms wait plus STT, LLM and TTS TTFA."""
    start = time.perf_counter()
    text = pipe.streaming_stt([0.0] * samples)
    reply = pipe.llm_with_tools(text)
    for call in reply["tool_calls"]:
        pipe.dispatch_tool(call["name"], call["args"])
    pipe.streaming_tts(reply["text"])
    return (time.perf_counter() - start) * 1000 + END_OF_SPEECH_MS


def frame_cost(duplex, frames=N):
    """Lesson 15's per-frame cost, the number its `main()` actually prints."""
    user, moshi, text, spent = [], [], [], []
    for chunk in duplex.simulate_user_speech(frames):
        start = time.perf_counter()
        user.append(duplex.fake_mimi_encode(chunk))
        text.append(duplex.inner_monologue_next_token(text, user))
        moshi.append(duplex.depth_transformer(text, user, moshi))
        spent.append((time.perf_counter() - start) * 1000)
    return spent


def median_interval(n=N):
    """The tightest symmetric distribution-free 95% interval for a median of n samples."""
    best = (1, n, 1.0)
    for low in range(1, n // 2 + 1):
        high = n + 1 - low
        coverage = sum(math.comb(n, k) for k in range(low, high)) * 0.5**n
        if coverage >= 0.95:
            best = (low, high, coverage)
    return best


def rank_sum_p(left, right):
    """Two-sided Mann-Whitney p, normal approximation -- no scipy needed."""
    merged = sorted([(v, 0) for v in left] + [(v, 1) for v in right])
    ranks = sum(i + 1 for i, (_, side) in enumerate(merged) if side == 0)
    statistic = ranks - len(left) * (len(left) + 1) / 2
    centre = len(left) * len(right) / 2
    spread = math.sqrt(len(left) * len(right) * (len(left) + len(right) + 1) / 12)
    return 2 * (1 - 0.5 * (1 + math.erf(abs(statistic - centre) / spread / math.sqrt(2))))


def detect_rate(rnd, ratio, sigma, n=N, trials=TRIALS):
    """How often n-vs-n calls a true `ratio` difference significant at 5%."""
    hits = 0
    for _ in range(trials):
        left = [math.exp(rnd.gauss(0, sigma)) for _ in range(n)]
        right = [ratio * math.exp(rnd.gauss(0, sigma)) for _ in range(n)]
        hits += rank_sum_p(left, right) < 0.05
    return hits / trials


def solve():
    duplex = parity.load_reference(PHASE, LESSON, "main")
    pipe = parity.load_reference(PHASE, PIPELINE_LESSON, "main")
    rnd = random.Random(0)
    lengths = [int(SR * rnd.uniform(1.0, 2.0)) for _ in range(N)]
    turns = [pipeline_turn(pipe, n) for n in lengths]
    frames = frame_cost(duplex)
    low, high, coverage = median_interval()
    power = {(ratio, sigma): detect_rate(random.Random(1), ratio, sigma)
             for sigma in SIGMAS for ratio in RATIOS}
    return {
        "turn_p50": statistics.median(turns), "frame_p50": statistics.median(frames),
        "ratio": statistics.median(turns) / statistics.median(frames),
        "rel_sd": statistics.pstdev(turns) / statistics.mean(turns),
        "same_text": pipe.streaming_stt([0.0] * 10) == pipe.streaming_stt([0.5] * 40000),
        "band": (low, high), "coverage": coverage, "n": N,
        "spread": sorted(turns)[high - 1] - sorted(turns)[low - 1],
        "power": power,
    }


def verify(result):
    power, low, high = result["power"], *result["band"]
    return [
        practice.Check(
            "ANSWER: the two numbers are not the same quantity",
            result["ratio"] > 20,
            f"Lesson 12's user-perceived delay to first audio has a P50 of "
            f"{result['turn_p50']:.0f} ms over {N} matched utterances; Lesson 15's per-frame "
            f"cost has a P50 of {result['frame_p50']:.2f} ms -- a ratio of "
            f"{result['ratio']:.0f}x. One contains a {END_OF_SPEECH_MS:.0f} ms end-of-speech "
            "wait and two model calls, the other a sleep(0.003)",
        ),
        practice.Check(
            "MECHANISM: the pipeline's latency is constants, so 20 utterances measure one",
            result["same_text"] and result["rel_sd"] < 0.05,
            f"`streaming_stt` returns the same transcript for a 10-sample and a 40000-sample "
            f"buffer, `llm_with_tools` sleeps 0.12 s and `streaming_tts` 0.10 s regardless of "
            f"their arguments, and the end-of-speech wait is the literal "
            f"{END_OF_SPEECH_MS:.0f}. The only input-dependent term is STT's len/sr * 0.05, so "
            f"{N} utterances of 1-2 s spread {result['rel_sd']:.2%} relative standard deviation",
        ),
        practice.Check(
            "FINDING: at n=20 a P50 is a band between the 6th and 15th order statistic",
            (low, high) == (6, 15) and result["coverage"] > 0.95,
            f"a median has no standard error, but it has an exact distribution-free interval: "
            f"for n={result['n']} the tightest symmetric 95% interval runs from the "
            f"{low}th to the {high}th smallest sample, coverage {result['coverage']:.4f}. On "
            f"this pipeline that band is {result['spread']:.0f} ms wide",
        ),
        practice.Check(
            "FINDING: twenty utterances cannot resolve a ten percent difference",
            power[(1.10, 0.35)] < 0.3 < power[(1.50, 0.35)],
            f"simulated on log-normal latencies, {N}-vs-{N} calls a true 1.10x difference at "
            f"sigma 0.20 / 0.35 only {power[(1.10, 0.20)]:.0%} / {power[(1.10, 0.35)]:.0%} of "
            f"the time, 1.25x {power[(1.25, 0.20)]:.0%} / {power[(1.25, 0.35)]:.0%}, and 1.50x "
            f"{power[(1.50, 0.20)]:.0%} / {power[(1.50, 0.35)]:.0%}. The doc's own cheatsheet "
            "puts Moshi at 200-300 ms and managed pipelines at 300-500 ms, overlapping ranges",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
