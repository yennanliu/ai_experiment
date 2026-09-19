"""Exercise 2 — "ambiguous" means both and neither, and the pipeline pays for both.

    Implement a router function: given prompt text, classify as `understand` or
    `generate`. How do you handle ambiguous prompts like "describe and then
    sketch"?

Reading of the exercise: the lesson already ships `route`, so it is measured
against a labelled set rather than reimplemented -- fourteen prompts including
the five the lesson demos and nine chosen to probe the two failure modes the
keyword design implies. The handling of ambiguity is then read off what
`run_pipeline` actually does with the third verdict.

**ANSWER: 9 of 14, and the five misroutes fall into exactly two classes.** Two
are substring collisions -- "**draw**er" and "with**draw**" both score as
generation -- and three are prompts with no keyword at all, which fall through
to `ambiguous`.

**FINDING: `ambiguous` conflates "both" with "neither".** Of the five prompts it
returns for, **2** score 1-1 (genuinely both) and **3** score 0-0 (neither).
Those are opposite situations -- one wants both encoders, the other wants a
default -- and `route` returns the same string for them.

**FINDING: and `run_pipeline` runs both encoders for all five.** Its `else`
branch encodes with SigLIP *and* with VQ and calls the body twice, so a prompt
with no keywords costs double. The 60% of ambiguous verdicts that mean "neither"
are the ones paying most.

**FINDING: the substring bug is load-bearing.** A word-boundary matcher takes
the router *down*, from **64.3%** to **50.0%**: "paint" inside "painting" and
"what" inside "Somewhat" are two of its nine correct answers. And it rescues
none of the five misroutes -- the two "draw" collisions merely become `ambiguous`
instead of `generate`, because without that substring they have no keyword at
all. The matcher is a bad stemmer, and the fix is the keyword list rather than
the comparison.

**ANSWER: so the way to handle ambiguity is to stop producing it accidentally.**
Split the third verdict in two -- `both` for a 1-1 score and `default` for 0-0 --
route `default` by whether an image is attached rather than by the text, and keep
the dual-encoder path only for the genuine `both`. On this set that is 2 prompts
rather than 5, a **60%** reduction in double-encoded requests.

Structure: `CASES` is the labelled set, `scores` reproduces `route`'s two counts,
`bucket` separates both-from-neither, and `word_route` is the boundary-matching
variant used to size the substring bug.
"""

from __future__ import annotations

import re

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "15-janus-pro-decoupled-encoders"
UNDERSTAND = ("describe", "what", "why", "caption", "explain", "how many")
GENERATE = ("draw", "generate", "sketch", "render", "create", "paint")
CASES = (
    ("Describe what is in this image", "understand"),
    ("Generate a picture of a sunset over the ocean", "generate"),
    ("Sketch a cat and then describe its breed", "ambiguous"),
    ("What is the pose of the person in the image?", "understand"),
    ("Render a cyberpunk cityscape at night", "generate"),
    ("Show me the contents of the drawer", "understand"),
    ("Is this a cat?", "understand"),
    ("A photo of a dog at sunset", "generate"),
    ("Count the cars", "understand"),
    ("Make an oil painting of a harbour", "generate"),
    ("Somewhat blurry - can you read the sign?", "understand"),
    ("Withdraw the overlay and show the original", "understand"),
    ("Caption this and then draw a variant", "ambiguous"),
    ("Explain the chart", "understand"),
)


def scores(prompt):
    lowered = prompt.lower()
    return (sum(word in lowered for word in UNDERSTAND),
            sum(word in lowered for word in GENERATE))


def word_scores(prompt):
    """The same counts with a word-boundary match instead of a substring one."""
    words = set(re.findall(r"[a-z]+", prompt.lower()))
    phrases = prompt.lower()
    return (sum(word in words or (" " in word and word in phrases)
                for word in UNDERSTAND),
            sum(word in words for word in GENERATE))


def decide(understand, generate):
    if generate > understand:
        return "generate"
    return "understand" if understand > generate else "ambiguous"


def word_route(prompt):
    return decide(*word_scores(prompt))


def bucket(prompt):
    understand, generate = scores(prompt)
    return "both" if understand else "neither"


def compare(ref):
    """Every case under the lesson's matcher and under a word-boundary one."""
    return [(prompt, want, ref.route(prompt), word_route(prompt))
            for prompt, want in CASES]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = compare(ref)
    wrong = [row for row in rows if row[1] != row[2]]
    ambiguous = [row[0] for row in rows if row[2] == "ambiguous"]
    buckets = [bucket(prompt) for prompt in ambiguous]
    word_correct = sum(row[1] == row[3] for row in rows)
    return {
        "total": len(rows), "correct": len(rows) - len(wrong),
        "accuracy": round((len(rows) - len(wrong)) / len(rows) * 100, 1),
        "misroutes": [(row[0], row[2]) for row in wrong],
        "ambiguous": len(ambiguous),
        "both": buckets.count("both"), "neither": buckets.count("neither"),
        "neither_share": round(buckets.count("neither") / len(buckets) * 100),
        "rescued_by_substring": [row[0] for row in rows
                                 if row[2] == row[1] != row[3]],
        "still_wrong": [row[0] for row in wrong if row[3] != row[1]],
        "word_correct": word_correct,
        "word_accuracy": round(word_correct / len(rows) * 100, 1),
        "double_encoded": len(ambiguous),
        "genuine_both": buckets.count("both"),
        "saving_pct": round((1 - buckets.count("both") / len(ambiguous)) * 100),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 9 of 14, and the five misroutes fall into exactly two classes",
            all([result["correct"] == 9, result["total"] == 14,
                 result["accuracy"] == 64.3, len(result["misroutes"]) == 5]),
            f"the lesson's route scores {result['correct']} of {result['total']} "
            f"({result['accuracy']}%) on the labelled set. The misroutes are "
            f"{result['misroutes']} -- two substring collisions on 'draw' inside other "
            "words, and three prompts with no keyword at all",
        ),
        practice.Check(
            "FINDING: ambiguous conflates 'both' with 'neither'",
            all([result["ambiguous"] == 5, result["both"] == 2, result["neither"] == 3,
                 result["neither_share"] == 60]),
            f"of the {result['ambiguous']} prompts route calls ambiguous, {result['both']} "
            f"score 1-1 and {result['neither']} score 0-0 -- {result['neither_share']}% of "
            "the verdict. One case wants both encoders and the other wants a default, and "
            "route returns the same string for them",
        ),
        practice.Check(
            "FINDING: run_pipeline runs both encoders for all five",
            all([result["double_encoded"] == result["ambiguous"],
                 result["genuine_both"] == 2, result["saving_pct"] == 60]),
            f"the else branch encodes with SigLIP and with VQ and calls the body twice, so "
            f"all {result['double_encoded']} pay double while only {result['genuine_both']} "
            f"need to. Splitting the verdict would cut double-encoded requests by "
            f"{result['saving_pct']}% on this set",
        ),
        practice.Check(
            "FINDING: the substring bug is load-bearing",
            all([result["word_correct"] == 7, result["word_accuracy"] == 50.0,
                 len(result["rescued_by_substring"]) == 2,
                 len(result["still_wrong"]) == 5]),
            f"a word-boundary matcher takes the router from {result['accuracy']}% to "
            f"{result['word_accuracy']}%, because {result['rescued_by_substring']} are "
            "correct only through 'paint' inside 'painting' and 'what' inside 'Somewhat'. "
            f"And it rescues none of the {len(result['still_wrong'])} misroutes -- the two "
            "'draw' collisions merely become ambiguous instead of generate. The matcher is a "
            "bad stemmer, and the fix is the keyword list, not the comparison",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
