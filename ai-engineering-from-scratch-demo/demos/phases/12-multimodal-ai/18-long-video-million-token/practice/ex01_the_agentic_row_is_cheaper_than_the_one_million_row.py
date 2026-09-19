"""Exercise 1 — the agentic row is cheaper than the one-million row.

    A 45-minute lecture at 1 FPS, 81 tokens per frame. Total tokens? Fits in
    which models' contexts?

Reading of the exercise: the count comes from the lesson's own `tokens`, and
"fits in which" is answered by checking every power-of-two context rather than by
naming one -- which then makes it possible to check the lesson's own `fits in`
column, the only part of its budget table that is not derivable from the other
three.

**ANSWER: 218,700 tokens.** It fits a **256k** context at **83.4%** full, and
everything above. It does not fit 128k (**1.67x** over) or a 200k context
(**1.09x**) -- so a 45-minute lecture misses Claude-class context by 9%.

**FINDING: three of the five sized labels are loose.**
The smallest power-of-two context that actually holds each row is 8k, 32k, 64k,
256k, **512k**, 1M and 256k, against labels of "32k+", "32k", "128k", "256k",
"1M / LongVILA", "Gemini 2.5 only" and "agentic retrieval". The 1-hour row is
labelled 1M and fits 512k.

**FINDING: and the row labelled "agentic retrieval" is the cheaper of the two.**
A 2-hour video at 32 tokens a frame is **230,400** tokens; a 1-hour video at 81
is **291,600**. The table routes the smaller one to an agent and the larger one
to a million-token model, so its own recommendation column is not ordered by its
own token column.

**FINDING: the 45-minute case is between two rows and the interpolation is
wrong.** The table's 30-minute row says 256k and its 60-minute row says 1M; the
true answer for 45 minutes is 256k, which is the *lower* of the two neighbours
because the 60-minute label was already 2x loose.

Structure: `tokens` is the lesson's own function, `smallest_fit` walks the
context ladder, and `TABLE` transcribes the seven rows with their labels.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "18-long-video-million-token"
LECTURE_SECONDS, FPS, PER_FRAME = 2700, 1, 81
CONTEXTS = (8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576)
CLAUDE_CLASS = 200_000
TABLE = ((60, 1, 81, "32k+"), (300, 1, 81, "32k"), (300, 2, 81, "128k"),
         (1800, 1, 81, "256k"), (3600, 1, 81, "1M / LongVILA"),
         (7200, 1, 81, "Gemini 2.5 only"), (7200, 1, 32, "agentic retrieval"))


def smallest_fit(count, ladder=CONTEXTS):
    return next((size for size in ladder if count <= size), None)


def name(size):
    return f"{size // 1024}k"


def names_a_context(label):
    """Only five of the seven labels name a context size at all."""
    return label[0].isdigit()


def priced(ref):
    """Every table row with its token count, its smallest fitting context, and its label."""
    return [(seconds, ref.tokens(seconds, fps, per_frame),
             name(smallest_fit(ref.tokens(seconds, fps, per_frame))), label)
            for seconds, fps, per_frame, label in TABLE]


def loose(rows):
    return sum(1 for (_, _, fit, label) in rows
               if names_a_context(label) and fit not in label)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    lecture = ref.tokens(LECTURE_SECONDS, FPS, PER_FRAME)
    rows = priced(ref)
    agentic = next(count for _, count, _, label in rows if "agentic" in label)
    million = next(count for _, count, _, label in rows if "1M" in label)
    return {
        "lecture": lecture, "fits": name(smallest_fit(lecture)),
        "share_pct": round(lecture / 262144 * 100, 1),
        "over_128k": round(lecture / 131072, 2),
        "over_claude": round(lecture / CLAUDE_CLASS, 2),
        "counts": [count for _, count, _, _ in rows],
        "smallest": [fit for _, _, fit, _ in rows],
        "labels": [label for _, _, _, label in rows],
        "sized_rows": sum(names_a_context(label) for _, _, _, label in rows),
        "loose": loose(rows),
        "agentic": agentic, "million": million,
        "misordered": agentic < million,
        "neighbours": [label for seconds, _, _, label in rows
                       if seconds in (1800, 3600)],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 218,700 tokens -- fits 256k at 83.4%, misses 200k by 9%",
            all([result["lecture"] == 218_700, result["fits"] == "256k",
                 result["share_pct"] == 83.4, result["over_128k"] == 1.67,
                 result["over_claude"] == 1.09]),
            f"{LECTURE_SECONDS // 60} minutes at {FPS} FPS and {PER_FRAME} tokens a frame is "
            f"{result['lecture']:,} tokens -- {result['share_pct']}% of a 256k context, "
            f"{result['over_128k']}x a 128k one and {result['over_claude']}x a "
            f"{CLAUDE_CLASS:,} one. A 45-minute lecture misses Claude-class context by 9%",
        ),
        practice.Check(
            "FINDING: the lesson's fits-in column is loose in three of seven rows",
            all([result["smallest"] == ["8k", "32k", "64k", "256k", "512k", "1024k", "256k"],
                 result["loose"] == 3, result["sized_rows"] == 5,
                 result["counts"] == [4860, 24300, 48600, 145800, 291600, 583200, 230400]]),
            f"the smallest context that actually holds each row is {result['smallest']} "
            f"against labels {result['labels']}. Of the {result['sized_rows']} labels that "
            f"name a size at all, {result['loose']} disagree -- and the 1-hour row is "
            "labelled 1M while fitting 512k",
        ),
        practice.Check(
            "FINDING: the row labelled agentic retrieval is the cheaper of the two",
            all([result["agentic"] == 230_400, result["million"] == 291_600,
                 result["misordered"]]),
            f"a 2-hour video at 32 tokens a frame is {result['agentic']:,} tokens and a "
            f"1-hour video at 81 is {result['million']:,}. The table routes the smaller to "
            "an agent and the larger to a million-token model, so its recommendation column "
            "is not ordered by its own token column",
        ),
        practice.Check(
            "FINDING: the 45-minute case sits between two rows and takes the lower label",
            all([result["neighbours"] == ["256k", "1M / LongVILA"],
                 result["fits"] == "256k"]),
            f"the table's 30- and 60-minute rows are labelled {result['neighbours']}, and the "
            f"true answer for 45 minutes is {result['fits']} -- the lower of the two "
            "neighbours, because the 60-minute label was already 2x loose",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
