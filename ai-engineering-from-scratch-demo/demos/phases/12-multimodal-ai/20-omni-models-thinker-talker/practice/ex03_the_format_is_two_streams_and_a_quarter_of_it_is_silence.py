"""Exercise 3 — the format is two streams, and a quarter of it is silence.

    Full-duplex support requires the model to emit audio while listening.
    Propose a training data format that teaches this.

Reading of the exercise: the format is proposed and then priced, because the
reason full-duplex data is rare is not that nobody thought of the format -- it is
that the format cannot be produced from the recordings everyone already has. The
arithmetic uses a stated 12.5 Hz frame rate with 8 residual codebooks, the same
configuration Lesson 12.16 costs out.

**ANSWER: two aligned token streams at one frame rate, with silence as an
explicit token.** Every frame carries a user token and an assistant token, both
present at every instant, so "speaking while listening" is representable rather
than special-cased. A turn boundary is then nothing -- it is a run of silence
ending in one stream, not a delimiter.

**FINDING: it costs 62x a transcript.** A ten-minute conversation is 7,500 frames
x 2 streams x 8 codebooks = **120,000** tokens, against **1,950** text tokens for
the same exchange at 150 words a minute. Full-duplex training data is expensive
per second in a way turn-based data is not.

**FINDING: a quarter of it is (silence, silence).** At a stated 50% speaking
share per party and independent streams, both are silent in **25%** of frames and
exactly one speaks in 50%. So a quarter of the tokens teach nothing and half
teach turn-taking; the **25%** where both speak is the entire full-duplex
signal, and it is the smallest bucket.

**FINDING: and that bucket is 0% of any turn-based corpus.** Half-duplex
recordings are segmented by turn, so overlapping speech is either discarded or
was never captured on separate channels. The format is not the hard part -- a
corpus that contains the phenomenon is, which is why the answer to "propose a
format" is really "propose a recording setup".

Structure: `stream_tokens` prices one conversation in the format, `buckets`
splits frames by who is speaking under a stated overlap model, and
`TRANSCRIPT_RATE` is the half-duplex baseline it is compared against.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "20-omni-models-thinker-talker"
FRAME_RATE, CODEBOOKS, STREAMS = 12.5, 8, 2
MINUTES = 10
WORDS_PER_MINUTE, TOKENS_PER_WORD = 150, 1.3
SPEAKING_SHARE = 0.5
FIELDS = ("frame", "user_tokens", "assistant_tokens", "user_text", "assistant_text")


def frames(minutes=MINUTES, rate=FRAME_RATE):
    return int(minutes * 60 * rate)


def stream_tokens(minutes=MINUTES):
    return frames(minutes) * STREAMS * CODEBOOKS


def transcript_tokens(minutes=MINUTES):
    return int(minutes * WORDS_PER_MINUTE * TOKENS_PER_WORD)


def buckets(share=SPEAKING_SHARE):
    """Frame shares under independent streams: both silent, one speaking, both speaking."""
    return {"both silent": round((1 - share) ** 2 * 100),
            "exactly one": round(2 * share * (1 - share) * 100),
            "both speaking": round(share * share * 100)}


def solve():
    parity.load_reference(PHASE, LESSON, "main")
    split = buckets()
    return {
        "fields": FIELDS, "streams": STREAMS, "rate": FRAME_RATE,
        "frames": frames(), "tokens": stream_tokens(),
        "transcript": transcript_tokens(),
        "ratio": round(stream_tokens() / transcript_tokens()),
        "per_second": stream_tokens() // (MINUTES * 60),
        "buckets": split,
        "signal_share": split["both speaking"],
        "smallest": min(split, key=split.get),
        "wasted": split["both silent"],
        "turn_based_overlap": 0,          # half-duplex capture, by construction
        "turn_based_loss": split["both speaking"],
        "channels_needed": STREAMS,
    }


def verify(result):
    split = result["buckets"]
    return [
        practice.Check(
            "ANSWER: two aligned token streams at one frame rate, silence made explicit",
            all([result["streams"] == 2, result["rate"] == 12.5,
                 result["frames"] == 7500, len(result["fields"]) == 5]),
            f"every frame carries {result['fields']} at {result['rate']} Hz, so both parties "
            f"are present at every instant and speaking-while-listening is representable "
            f"rather than special-cased. A ten-minute conversation is {result['frames']:,} "
            "frames, and a turn boundary is a run of silence ending rather than a delimiter",
        ),
        practice.Check(
            "FINDING: it costs 62x a transcript",
            all([result["tokens"] == 120_000, result["transcript"] == 1950,
                 result["ratio"] == 62, result["per_second"] == 200]),
            f"{result['frames']:,} frames x {STREAMS} streams x {CODEBOOKS} codebooks is "
            f"{result['tokens']:,} tokens -- {result['per_second']} a second -- against "
            f"{result['transcript']:,} text tokens for the same exchange at "
            f"{WORDS_PER_MINUTE} words a minute. {result['ratio']}x, and the ratio is fixed "
            "by the frame rate rather than by the conversation",
        ),
        practice.Check(
            "FINDING: a quarter of it is (silence, silence)",
            all([split == {"both silent": 25, "exactly one": 50, "both speaking": 25},
                 result["wasted"] == 25, result["signal_share"] == 25]),
            f"at a stated {SPEAKING_SHARE:.0%} speaking share per party with independent "
            f"streams the frame split is {split}%. A quarter teaches nothing, half teaches "
            f"turn-taking, and the {result['signal_share']}% where both speak is the entire "
            "full-duplex signal",
        ),
        practice.Check(
            "FINDING: and that bucket is 0% of any turn-based corpus",
            all([result["turn_based_overlap"] == 0,
                 result["turn_based_loss"] == result["signal_share"],
                 result["turn_based_loss"] == 25, result["signal_share"] > 0]),
            f"half-duplex recordings are segmented by turn, so overlapping speech is "
            f"discarded or was never captured on separate channels -- "
            f"{result['turn_based_overlap']}% of a turn-based corpus contains the "
            f"phenomenon. That is exactly the {result['turn_based_loss']}% of frames this "
            f"format exists to carry, measured off the same split, so a turn-based corpus "
            f"cannot supply any of its training signal. The format needs "
            f"{result['channels_needed']} channels recorded simultaneously: 'propose a "
            "format' is really 'propose a recording setup'",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
