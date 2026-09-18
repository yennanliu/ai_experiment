"""Exercise 2 — steps() returns a hundred times the total if you forget.

    InternVL3 reports 40% text / 35% interleaved / 20% caption / 5% video. If
    your target task is video-heavy, propose a new ratio and argue why the base
    model still needs substantial text and caption data.

Reading of the exercise: the proposed ratio is run through the lesson's own
`CorpusMix` so the argument is made in training steps rather than percentages,
and the argument for keeping text is taken from the lesson's own
post-hoc-vs-native table -- which prices exactly the skills a text floor
protects -- rather than asserted.

**ANSWER: 25 / 25 / 15 / 35.** Video goes from 5% to 35%, **7x**: 25,000
training steps to 175,000 out of 500,000.

**FINDING: the argument for the text floor is two rows of the lesson's own
table.** Post-hoc training is priced there at "-2 to -8" MMLU and "-3 to -10"
GSM8K, and those are precisely the skills the 40% text share exists to hold. The
proposal cuts text by **37.5%**, so it is spending against a risk the lesson has
already quantified -- which is the honest way to argue for it, rather than
claiming the risk is absent.

**FINDING: raising video shrinks the bucket video depends on.** Interleaved is
the only share that teaches binding one entity across images, and a video clip
is an interleaved sequence with a clock. The proposal grows sequence-shaped data
from 40% to **60%** while cutting the non-video half of it by **28.6%** -- more
sequences, less of the thing that teaches what a sequence is.

**FINDING: `steps()` returns 100x the total if `normalize()` was not called.**
`CorpusMix(40, 35, 20, 5).steps(500_000)` sums to **50,000,000** against the
500,000 requested, because the fields are still percentages. The class has a
method that must be called first, no way to tell whether it was, and the failure
is a silent factor of 100.

Structure: `plan` normalises a mix and returns its step allocation, `SHIPPED`
and `PROPOSED` are the two ratios, and `unnormalised` is the same call without
the required `normalize()`.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "10-internvl3-native-multimodal"
SHIPPED = (40, 35, 20, 5)
PROPOSED = (25, 25, 15, 35)
TOTAL_STEPS = 500_000
BUCKETS = ("text", "interleaved", "caption", "video")
REGRESSIONS = {"MMLU": (2, 8), "GSM8K": (3, 10)}


def plan(ref, shares, total=TOTAL_STEPS):
    mix = ref.CorpusMix(*shares)
    mix.normalize()
    return mix.steps(total)


def unnormalised(ref, shares, total=TOTAL_STEPS):
    """The same call with the required normalize() left out."""
    return ref.CorpusMix(*shares).steps(total)


def sequence_share(shares):
    """Interleaved plus video -- the data that is a sequence of images."""
    return (shares[1] + shares[3]) / sum(shares)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    shipped, proposed = plan(ref, SHIPPED), plan(ref, PROPOSED)
    skipped = unnormalised(ref, SHIPPED)
    return {
        "shipped": shipped, "proposed": proposed,
        "shipped_total": sum(shipped.values()), "proposed_total": sum(proposed.values()),
        "video_growth": proposed["video"] // shipped["video"],
        "text_cut_pct": round((PROPOSED[0] / SHIPPED[0] - 1) * 100, 1),
        "interleaved_cut_pct": round((PROPOSED[1] / SHIPPED[1] - 1) * 100, 1),
        "sequence": [round(sequence_share(SHIPPED) * 100), round(sequence_share(PROPOSED) * 100)],
        "regressions": REGRESSIONS,
        "skipped": skipped, "skipped_total": sum(skipped.values()),
        "blowup": sum(skipped.values()) // TOTAL_STEPS,
    }


def verify(result):
    shipped, proposed = result["shipped"], result["proposed"]
    return [
        practice.Check(
            "ANSWER: 25 / 25 / 15 / 35 -- video goes 5% to 35%, seven times the steps",
            all([shipped == {"text": 200_000, "interleaved": 175_000,
                             "caption": 100_000, "video": 25_000},
                 proposed == {"text": 125_000, "interleaved": 125_000,
                              "caption": 75_000, "video": 175_000},
                 result["video_growth"] == 7,
                 result["proposed_total"] == result["shipped_total"] == TOTAL_STEPS]),
            f"the shipped {SHIPPED} mix allocates {shipped} of {TOTAL_STEPS:,} steps; the "
            f"proposed {PROPOSED} allocates {proposed}. Video grows "
            f"{result['video_growth']}x and the total is unchanged",
        ),
        practice.Check(
            "FINDING: the argument for the text floor is two rows of the lesson's own table",
            all([result["regressions"] == {"MMLU": (2, 8), "GSM8K": (3, 10)},
                 result["text_cut_pct"] == -37.5]),
            f"post-hoc training is priced in the same file at {result['regressions']['MMLU']} "
            f"MMLU and {result['regressions']['GSM8K']} GSM8K -- exactly the skills a text "
            f"floor holds. The proposal cuts text by {abs(result['text_cut_pct'])}%, so it "
            "spends against a risk the lesson has already quantified rather than one it "
            "denies",
        ),
        practice.Check(
            "FINDING: raising video shrinks the bucket video depends on",
            all([result["sequence"] == [40, 60], result["interleaved_cut_pct"] == -28.6]),
            f"interleaved is the only share that teaches binding an entity across images, "
            f"and a video clip is an interleaved sequence with a clock. Sequence-shaped data "
            f"grows {result['sequence'][0]}% -> {result['sequence'][1]}% while its non-video "
            f"half falls {abs(result['interleaved_cut_pct'])}% -- more sequences, less of "
            "what teaches a sequence",
        ),
        practice.Check(
            "FINDING: steps() returns 100x the total if normalize() was not called",
            all([result["skipped_total"] == 50_000_000, result["blowup"] == 100,
                 result["skipped"]["text"] == 20_000_000]),
            f"CorpusMix{SHIPPED}.steps({TOTAL_STEPS:,}) without normalize() returns "
            f"{result['skipped']} -- {result['skipped_total']:,} steps against the "
            f"{TOTAL_STEPS:,} requested, a factor of {result['blowup']}, because the fields "
            "are still percentages. A method that must be called first, no way to tell "
            "whether it was, and a silent factor of 100",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
