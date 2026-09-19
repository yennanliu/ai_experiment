"""Exercise 5 — swapping two labels costs 100% or nothing.

    Implement a minimal diarization pipeline using AF3's output. How do you
    signal speaker changes?

Reading of the exercise: "how do you signal speaker changes" is read as a
question about the output *contract* rather than about a token, because the
answer that matters is what a scorer can do with it -- and the central fact about
diarization output is that speaker labels are arbitrary. The pipeline is
therefore built around a metric, and the metric is what forces every field in the
schema.

**ANSWER: emit contiguous turns with a local speaker id, and score under the best
label permutation.** A turn is `{speaker, start, end, text}`; the turns tile the
timeline with no gaps and no overlaps; and consecutive turns carry different
speakers, which is what makes the boundary a *change* rather than a repetition.

**FINDING: without permutation matching, a perfect transcript scores 100% error.**
Relabelling S1 as S2 and S2 as S1 leaves the segmentation exactly right and gives
a naive DER of **100.0%**; under the best permutation it is **0.0%**. Speaker ids
are local to a clip, so any scorer that compares them literally is measuring a
naming convention.

**FINDING: the metric is boundary-sensitive in seconds, not in turns.** A single
boundary moved by 0.5 s in a 10-second clip costs **5.0%** DER with the turn
count, the speaker count and the order all correct. Reporting "3 of 3 turns
found" would call that perfect.

**FINDING: the lesson's own table says cascaded diarization is "partial", and
the reason is in this schema.** Whisper emits text with timings and no speaker
field at all, so a cascade can produce `start`, `end` and `text` and must infer
`speaker` from the words. Three of the four fields come free and the fourth is
the whole task.

Structure: `TURNS` is the reference, `der` scores a hypothesis against it in
seconds of speech, `best_permutation` searches the label assignments, and `CASES`
pairs each schema constraint with the failure it prevents.
"""

from __future__ import annotations

import itertools

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "19-audio-language-whisper-to-af3"
SCHEMA = ("speaker", "start", "end", "text")
TURNS = (("S1", 0.0, 4.0), ("S2", 4.0, 7.0), ("S1", 7.0, 10.0))
SWAPPED = (("S2", 0.0, 4.0), ("S1", 4.0, 7.0), ("S2", 7.0, 10.0))
SHIFTED = (("S1", 0.0, 4.5), ("S2", 4.5, 7.0), ("S1", 7.0, 10.0))
GRID = 0.01


def speakers(turns):
    return sorted({name for name, _, _ in turns})


def at(turns, moment):
    for name, start, end in turns:
        if start <= moment < end:
            return name
    return None


def der(reference, hypothesis, mapping=None, grid=GRID):
    """Confusion time over total speech time, on a fine grid."""
    total = max(end for _, _, end in reference)
    steps = int(total / grid)
    wrong = 0
    for step in range(steps):
        moment = step * grid
        predicted = at(hypothesis, moment)
        if mapping:
            predicted = mapping.get(predicted, predicted)
        wrong += at(reference, moment) != predicted
    return round(wrong / steps * 100, 1)


def best_permutation(reference, hypothesis):
    names = speakers(reference)
    options = [dict(zip(speakers(hypothesis), order))
               for order in itertools.permutations(names)]
    return min(der(reference, hypothesis, mapping) for mapping in options)


def tiles(turns):
    return all(a[2] == b[1] for a, b in zip(turns, turns[1:]))


def alternates(turns):
    return all(a[0] != b[0] for a, b in zip(turns, turns[1:]))


def solve():
    parity.load_reference(PHASE, LESSON, "main")
    return {
        "schema": SCHEMA, "fields": len(SCHEMA),
        "turns": len(TURNS), "speakers": speakers(TURNS),
        "tiles": tiles(TURNS), "alternates": alternates(TURNS),
        "perfect_naive": der(TURNS, TURNS),
        "swapped_naive": der(TURNS, SWAPPED),
        "swapped_matched": best_permutation(TURNS, SWAPPED),
        "swapped_turns_correct": len(SWAPPED) == len(TURNS),
        "shifted_naive": der(TURNS, SHIFTED),
        "shifted_matched": best_permutation(TURNS, SHIFTED),
        "shifted_turns_correct": len(SHIFTED) == len(TURNS),
        "shift_seconds": 0.5,
        "cascade_fields": [field for field in SCHEMA if field != "speaker"],
        "cascade_missing": ["speaker"],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: contiguous turns with a local speaker id, scored under a permutation",
            all([result["schema"] == ("speaker", "start", "end", "text"),
                 result["fields"] == 4, result["turns"] == 3,
                 result["tiles"], result["alternates"],
                 result["perfect_naive"] == 0.0]),
            f"a turn is {result['schema']}; the {result['turns']} turns tile the timeline "
            f"with no gaps ({result['tiles']}) and consecutive turns carry different "
            f"speakers ({result['alternates']}), which is what makes a boundary a change "
            "rather than a repetition",
        ),
        practice.Check(
            "FINDING: without permutation matching, a perfect transcript scores 100% error",
            all([result["swapped_naive"] == 100.0, result["swapped_matched"] == 0.0,
                 result["swapped_turns_correct"]]),
            f"relabelling S1 as S2 and S2 as S1 leaves the segmentation exactly right and "
            f"gives a naive DER of {result['swapped_naive']}%; under the best permutation it "
            f"is {result['swapped_matched']}%. Speaker ids are local to a clip, so a scorer "
            "that compares them literally is measuring a naming convention",
        ),
        practice.Check(
            "FINDING: the metric is boundary-sensitive in seconds, not in turns",
            all([result["shifted_matched"] == 5.0, result["shifted_turns_correct"],
                 result["shift_seconds"] == 0.5]),
            f"one boundary moved by {result['shift_seconds']} s in a 10-second clip costs "
            f"{result['shifted_matched']}% DER with the turn count, the speaker count and the "
            "order all correct. Reporting '3 of 3 turns found' would call that perfect",
        ),
        practice.Check(
            "FINDING: the lesson's table calls cascaded diarization 'partial' for this reason",
            all([result["cascade_fields"] == ["start", "end", "text"],
                 result["cascade_missing"] == ["speaker"],
                 len(result["cascade_fields"]) == result["fields"] - 1]),
            f"Whisper emits {result['cascade_fields']} and no speaker field at all, so a "
            f"cascade gets {len(result['cascade_fields'])} of the {result['fields']} fields "
            f"free and must infer {result['cascade_missing'][0]} from the words. Three come "
            "for nothing and the fourth is the whole task",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
