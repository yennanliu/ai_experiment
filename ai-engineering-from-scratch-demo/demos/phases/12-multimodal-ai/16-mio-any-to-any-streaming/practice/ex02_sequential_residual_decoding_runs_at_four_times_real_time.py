"""Exercise 2 — sequential residual decoding runs at four times real time.

    SpeechTokenizer residual-VQ uses 8 codebooks. Propose why parallel-decoding
    the residual levels is necessary (vs sequential) and what latency savings it
    brings.

Reading of the exercise: "necessary" and "what savings" are two different
questions and only the second is about latency, so both are answered. The saving
comes from the lesson's own trace with the residual term replaced by seven extra
decode steps; the necessity comes from the real-time factor, which is a frame
rate the lesson does not state and so is declared here.

**ANSWER: it saves 250 ms, 48% of the sequential total.** The lesson prices
"residual-VQ layers 1..7" at **30 ms** for all seven together. Decoded one at a
time, each level needs its own pass at the first-token cost -- 7 x 40 ms -- so
the total goes **270 -> 520 ms**, from the lesson's conversational band into its
acceptable one.

**ANSWER: and necessity is a different argument, made in real-time factor.** At
a stated 12.5 Hz frame rate every 80 ms of audio needs one set of 8 codes.
Parallel decoding spends one 40 ms pass per frame -- a real-time factor of
**0.5**, comfortably ahead. Sequential spends eight, **320 ms** per 80 ms frame,
an RTF of **4.0**: the decoder falls three seconds behind for every second
spoken, and the gap never closes.

**FINDING: the frame rate decides it, and the lesson does not name one.** At
50 Hz -- SpeechTokenizer's own rate -- even *parallel* decoding is an RTF of
**2.0** and cannot stream either. The architecture question the exercise asks
has a threshold, and the threshold is a number the latency model leaves out.

**FINDING: the 30 ms the lesson charges is not a model pass.** Seven residual
levels in 30 ms is 4.3 ms each, well under the 40 ms first-token cost -- so the
figure implies a small depth head, not the main transformer, run over all levels
at once. That is the actual design being priced, and the label "layers 1..7"
hides it.

Structure: `parallel_total` is the lesson's own trace, `sequential_total`
replaces its residual term with per-level decode steps, and `rtf` converts a
per-frame cost into a real-time factor.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "16-mio-any-to-any-streaming"
CODEBOOKS, RESIDUAL_LEVELS = 8, 7
FRAME_RATES = (50.0, 12.5)          # SpeechTokenizer's own, and Moshi-class
SECONDS, SIZE = 2.0, 8


def trace(ref):
    return ref.streaming_decode_latency(SECONDS, SIZE)


def parallel_total(ref):
    return round(sum(step.ms for step in trace(ref)))


def sequential_total(ref):
    """The same trace with the residual term replaced by per-level decode steps."""
    rows = trace(ref)
    residual, per_token = rows[3].ms, rows[2].ms
    return round(sum(step.ms for step in rows) - residual + RESIDUAL_LEVELS * per_token)


def rtf(per_frame_ms, frame_rate):
    return round(per_frame_ms / (1000.0 / frame_rate), 2)


def verdict(ms):
    return "conversational" if ms < 500 else ("acceptable" if ms < 800 else "sluggish")


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = trace(ref)
    residual, per_token = rows[3].ms, rows[2].ms
    fast, slow = parallel_total(ref), sequential_total(ref)
    return {
        "residual_ms": residual, "per_token_ms": per_token,
        "levels": RESIDUAL_LEVELS, "codebooks": CODEBOOKS,
        "parallel": fast, "sequential": slow,
        "saving": slow - fast, "saving_pct": round((slow - fast) / slow * 100, 1),
        "verdicts": (verdict(fast), verdict(slow)),
        "rtf": {rate: (rtf(per_token, rate), rtf(CODEBOOKS * per_token, rate))
                for rate in FRAME_RATES},
        "per_level_ms": round(residual / RESIDUAL_LEVELS, 1),
        "cheaper_than_a_pass": residual / RESIDUAL_LEVELS < per_token,
        "head_ratio": round(per_token / (residual / RESIDUAL_LEVELS), 1),
    }


def verify(result):
    rtfs = result["rtf"]
    return [
        practice.Check(
            "ANSWER: it saves 250 ms, 48% of the sequential total",
            all([result["parallel"] == 270, result["sequential"] == 520,
                 result["saving"] == 250, result["saving_pct"] == 48.1,
                 result["verdicts"] == ("conversational", "acceptable")]),
            f"the lesson prices {result['levels']} residual levels together at "
            f"{result['residual_ms']} ms; one pass each at the "
            f"{result['per_token_ms']} ms first-token cost takes the total "
            f"{result['parallel']} -> {result['sequential']} ms -- a saving of "
            f"{result['saving']}, {result['saving_pct']}%, and the difference between "
            f"{result['verdicts'][0]} and {result['verdicts'][1]} on the lesson's own scale",
        ),
        practice.Check(
            "ANSWER: necessity is a real-time-factor argument, not a latency one",
            all([rtfs[12.5] == (0.5, 4.0),
                 rtfs[12.5][1] > 1.0 > rtfs[12.5][0]]),
            f"at a stated 12.5 Hz frame rate every 80 ms of audio needs one set of "
            f"{result['codebooks']} codes. Parallel spends one {result['per_token_ms']} ms "
            f"pass -- RTF {rtfs[12.5][0]} -- and sequential spends "
            f"{result['codebooks']}, {result['codebooks'] * result['per_token_ms']:.0f} ms "
            f"per frame, RTF {rtfs[12.5][1]}. The decoder falls three seconds behind per "
            "second spoken, and the gap never closes",
        ),
        practice.Check(
            "FINDING: the frame rate decides it, and the lesson does not name one",
            all([rtfs[50.0] == (2.0, 16.0), rtfs[50.0][0] > 1.0]),
            f"at 50 Hz -- SpeechTokenizer's own rate -- even parallel decoding is RTF "
            f"{rtfs[50.0][0]} and cannot stream either, against {rtfs[12.5][0]} at 12.5 Hz. "
            "The architecture question has a threshold, and the threshold is a number the "
            "latency model leaves out",
        ),
        practice.Check(
            "FINDING: the 30 ms the lesson charges is not a model pass",
            all([result["per_level_ms"] == 4.3, result["cheaper_than_a_pass"],
                 result["head_ratio"] == 9.3]),
            f"{result['levels']} levels in {result['residual_ms']} ms is "
            f"{result['per_level_ms']} ms each, {result['head_ratio']}x cheaper than the "
            f"{result['per_token_ms']} ms first-token cost. The figure implies a small depth "
            "head run over all levels at once, not the main transformer -- which is the "
            "design being priced, and the label 'layers 1..7' hides it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
