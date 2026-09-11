"""Exercise 1 — the decode is never scored.

    **Easy.** Run `code/main.py`. It greedily decodes a hand-crafted CTC output
    and computes WER against a reference.

Reading of the exercise: the sentence describes one pipeline -- decode, then
score the decode -- and `main()` runs two disconnected halves. Step 2 decodes
`'hello world'` and prints it. Step 5 computes WER for five **hand-typed**
hypotheses against a different reference, `'hello world this is a test'`. No line
in the file ever passes a decoder output to `wer`. Doing it is the answer:

    wer('hello world', ctc_greedy(probs)) == 0.000

Three things fall out of running the other decoder beside it.

**On clean frames the lesson's beam is worse than greedy.** `'helo world'`
against `'hello world'` -- **WER 0.500 against 0.000**. `build_frame_probs` puts
a blank between every character, so greedy is already exact and a beam can only
tie or lose. The file names the cause in a printed note and leaves it (it is
Exercise 2's job).

**Step 4 is captioned "corrupt logits; beam should beat greedy", and it does
not.** `'hlhce llolo wolnrld'` and `'hlhce lolo wolnrld'` differ by one dropped
`l` and score **1.500 each** -- identical, and both worse than the 1.000 an empty
hypothesis would score, because `wer` divides by the reference length and is
unbounded above.

**`corrupt` does not produce probabilities.** It subtracts up to 0.6 from entries
that start at 0.02, so the minimum entry after Step 4 is **-0.58**. Row sums stay
at 1.0 -- it moves mass rather than adding it -- so nothing looks wrong until
`ctc_beam` clamps every negative to `1e-10` and silently decodes a vector that
holds more than its mass. `ctc_greedy` takes an argmax and never notices.

And the "hand-crafted CTC output" is not random at all: `build_frame_probs` calls
`random.seed(0)` and then draws nothing. Re-seeding to 999 reproduces all 44
frames byte for byte, and `one_hot_like`'s `noise` is a constant.

Structure: `decode_and_score` runs one decoder and scores it with the lesson's
own `wer`; `frame_stats` reports the minimum entry and the set of row sums of a
frame list; `deterministic` re-runs the builder under a different seed.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "04-speech-recognition-asr"
TARGET = "hello world"
CHARS_PER, BLANK_RUNS = 3, 1
SWAPS, STRENGTH, WIDTH = 6, 0.6, 16


def decode_and_score(ref, decoder, frames, target=TARGET, **kwargs):
    """(decode, WER) -- the pairing the exercise describes and `main()` never makes."""
    text = decoder(frames, **kwargs)
    return text, ref.wer(target, text)


def frame_stats(frames):
    """What the frame vectors actually are: smallest entry, and the sums they claim."""
    return {"min": min(min(row) for row in frames),
            "sums": sorted({round(sum(row), 6) for row in frames})}


def deterministic(ref):
    """`build_frame_probs` seeds the RNG and draws nothing -- so the seed cannot matter."""
    first = ref.build_frame_probs(TARGET, CHARS_PER, BLANK_RUNS)
    random.seed(999)
    return first == ref.build_frame_probs(TARGET, CHARS_PER, BLANK_RUNS), len(first)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    clean = ref.build_frame_probs(TARGET, CHARS_PER, BLANK_RUNS)
    corrupt = ref.corrupt(clean, SWAPS, STRENGTH)
    same, frames = deterministic(ref)
    return {
        "greedy": decode_and_score(ref, ref.ctc_greedy, clean),
        "beam": decode_and_score(ref, ref.ctc_beam, clean, beam_width=8),
        "bad_greedy": decode_and_score(ref, ref.ctc_greedy, corrupt),
        "bad_beam": decode_and_score(ref, ref.ctc_beam, corrupt, beam_width=WIDTH),
        "empty": ref.wer(TARGET, ""),
        "clean_stats": frame_stats(clean), "corrupt_stats": frame_stats(corrupt),
        "reseeded": same, "frames": frames,
    }


def verify(result):
    greedy, beam = result["greedy"], result["beam"]
    bad_greedy, bad_beam = result["bad_greedy"], result["bad_beam"]
    stats = result["corrupt_stats"]
    return [
        practice.Check(
            "ANSWER: the greedy decode scores WER 0.000, a number the lesson never computes",
            greedy == (TARGET, 0.0),
            f"`ctc_greedy` returns {greedy[0]!r} and `wer({TARGET!r}, ...)` is "
            f"{greedy[1]:.3f}. `main()` never makes that call: Step 2 prints a decode and Step 5 "
            "scores five hand-typed strings against a different reference",
        ),
        practice.Check(
            "FINDING: on clean frames the lesson's beam is worse than greedy",
            beam[1] > greedy[1],
            f"beam {beam[0]!r} at WER {beam[1]:.3f} against greedy's {greedy[1]:.3f}. "
            f"`build_frame_probs` puts a blank between every character, so greedy is already "
            "exact and a beam can only tie or lose here",
        ),
        practice.Check(
            "FINDING: Step 4 says the beam should beat greedy; the two tie at WER 1.500",
            bad_beam[1] == bad_greedy[1] > result["empty"],
            f"greedy {bad_greedy[0]!r} and beam {bad_beam[0]!r} differ by one dropped letter and "
            f"score {bad_greedy[1]:.3f} each -- both worse than the {result['empty']:.3f} an "
            "empty hypothesis scores, since `wer` divides by the reference length only",
        ),
        practice.Check(
            "MECHANISM: `corrupt` leaves the simplex while the row sums still read 1.0",
            stats["min"] < 0 and stats["sums"] == result["clean_stats"]["sums"],
            f"subtracting {STRENGTH} from entries that start at 0.02 puts the minimum at "
            f"{stats['min']:.2f}, while the sums stay {stats['sums']} because mass is moved "
            f"rather than added. `ctc_beam` clamps each negative to 1e-10 and decodes a vector "
            "holding more than its own mass; `ctc_greedy`'s argmax never sees it",
        ),
        practice.Check(
            "CONTROL: the 'hand-crafted' frames are fully deterministic",
            result["reseeded"] and result["frames"] == 44,
            f"`build_frame_probs` calls `random.seed(0)` and then draws nothing, so re-seeding "
            f"to 999 reproduces all {result['frames']} frames byte for byte, and "
            "`one_hot_like`'s `noise` argument is a constant offset rather than noise",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
