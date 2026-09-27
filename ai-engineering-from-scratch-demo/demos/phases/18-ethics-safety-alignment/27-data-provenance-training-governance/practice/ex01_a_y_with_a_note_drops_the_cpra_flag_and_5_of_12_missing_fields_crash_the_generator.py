"""Exercise 1 — a "Y" with a note drops the CPRA flag, and 5 of 12 missing fields crash the generator.

    Run `code/main.py`. Produce a 12-field summary for a toy dataset and
    identify which fields are under-specified.

Reading of the exercise: the shipped TOY_EXAMPLE is a synthetic dataset with
little to disclose, so a second toy is written the way a real team fills the
form: a forum crawl with abstract placeholders ([FORUM-A]), some answers left
vague. Both go through the reference's `render_markdown` and
`flag_followups`. "Under-specified" is judged against the statute's own item
text in docs/en.md, with five rules: a placeholder answer (TBD, various,
unknown), a count without a number (item 3), a bare "Y" to an either/or item
(5, 6, 12), cleaning described with no stated purpose (item 9), and a collection period
with no ongoing notice (item 10).

**ANSWER: in the forum toy, items 3, 6, 9, 10 and 11 are under-specified;
the shipped toy passes all five rules.** Item 3 says "various", item 6 a bare
"Y" that does not say purchased or licensed, item 9 gives no purpose, item 10
no ongoing-collection notice, and item 11 "TBD". The reference renders all 12
lines of both summaries and never warns, although the lesson's skill file
asks to "flag any missing or placeholder-only fields".

**FINDING: a "Y" with a note drops the CPRA flag.** `flag_followups` tests
items 5 and 12 with `startswith("Y")` but items 6, 7 and 8 with `== "Y"`.
Written in the shipped toy's own style ("N (entirely synthetic ...)"), the
forum toy's "Y (usernames and signatures)" on item 7 raises no CPRA
obligation. An all-"Y (note)" summary raises 2 of the 5 follow-ups; an
all-bare-"Y" one raises all 5.

**FINDING: missing fields show "(missing)" only for 7 of the 12.** Deleting
any of items 5, 6, 7, 8 or 12 makes `render_markdown` raise KeyError, because
`flag_followups` indexes those keys directly. The "(missing)" placeholder is
reachable only for the other 7.

**FINDING: the shipped toy's only follow-up is a false positive.** Its one
flag is "may still trigger obligations on the base model used for
generation", for data drawn from `random.gauss`, which has no base model.
The printed takeaway says items 5 and 7 trigger cascading obligations, and
the toy triggers neither.

Structure: `underspecified()` applies the five rules; `drop_probe()` deletes
each field in turn and records what `render_markdown` does.
"""

from __future__ import annotations

import contextlib
import io
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "27-data-provenance-training-governance"
PLACEHOLDERS = {"", "tbd", "various", "unknown", "n/a"}
WARNINGS = ("under-specified", "placeholder", "warning", "missing")
EITHER_OR = (5, 6, 12)  # "...or fully public domain", "purchased or licensed", "uses or continuously uses"
FORUM_TOY = [
    "public posts crawled from [FORUM-A] and [FORUM-B]",
    "teaches a support assistant a conversational register",
    "various",
    "free-text threads; no labels",
    "Y (posts are copyrighted by their authors)",
    "Y",
    "Y (usernames and signatures)",
    "N",
    "deduplicated; usernames hashed",
    "2023-01 to 2024-06",
    "TBD",
    "N",
]


def as_summary(ref, values):
    return dict(zip(ref.AB_2013_FIELDS, values))


def underspecified(ref, summary):
    """1-based item numbers that fail one of the five rules."""
    bad = []
    for i, field in enumerate(ref.AB_2013_FIELDS, 1):
        v = summary[field].strip()
        rules = [v.lower() in PLACEHOLDERS, i == 3 and not re.search(r"\d", v),
                 i in EITHER_OR and v == "Y", i == 9 and not v.startswith("none") and not re.search(r"\b(for|to)\b", v),
                 i == 10 and "ongoing" not in v]
        if any(rules):
            bad.append(i)
    return bad


def drop_probe(ref):
    """What render_markdown does when each item in turn is absent."""
    out = {}
    for i, field in enumerate(ref.AB_2013_FIELDS, 1):
        summary = {k: v for k, v in ref.TOY_EXAMPLE.items() if k != field}
        try:
            out[i] = "(missing)" if "(missing)" in ref.render_markdown(summary) else "silent"
        except KeyError:
            out[i] = "KeyError"
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    forum = as_summary(ref, FORUM_TOY)
    bare = as_summary(ref, ["Y" if v[0] in "YN" else v for v in FORUM_TOY])
    fixed7 = dict(forum, **{ref.AB_2013_FIELDS[6]: "Y"})
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ref.main()
    skill = (parity.lesson_dir(PHASE, LESSON) / "outputs" / "skill-provenance-check.md").read_text()
    rendered = ref.render_markdown(forum)
    return {
        "forum_bad": underspecified(ref, forum), "toy_bad": underspecified(ref, ref.TOY_EXAMPLE),
        "field_lines": rendered.count("\n- **"), "warns": [w for w in WARNINGS if w in rendered.lower()],
        "skill_asks": "placeholder-only fields" in skill,
        "forum_flags": ref.flag_followups(forum),
        "cpra": ["CPRA" in " ".join(ref.flag_followups(x)) for x in (forum, fixed7)],
        "note_flags": len(ref.flag_followups(as_summary(ref, ["Y (note)"] * 12))),
        "bare_flags": len(ref.flag_followups(bare)),
        "probe": drop_probe(ref),
        "crash": [i for i, v in drop_probe(ref).items() if v == "KeyError"], "toy_flags": ref.flag_followups(ref.TOY_EXAMPLE),
        "gauss": "random.gauss" in ref.TOY_EXAMPLE[ref.AB_2013_FIELDS[0]],
        "takeaway": "Items 5 and 7 trigger cascading obligations" in out.getvalue(),
    }


def verify(result):
    probe, flags = result["probe"], result["toy_flags"]
    return [
        practice.Check(
            "ANSWER: items 3, 6, 9, 10 and 11 are under-specified; the shipped toy passes",
            all([result["forum_bad"] == [3, 6, 9, 10, 11], result["toy_bad"] == [],
                 result["field_lines"] == 12, result["warns"] == [], result["skill_asks"]]),
            f"forum toy fails items {result['forum_bad']}, shipped toy {result['toy_bad']}; the "
            f"reference renders {result['field_lines']} lines, warning words {result['warns']}",
        ),
        practice.Check(
            "FINDING: a 'Y' with a note drops the CPRA flag",
            (result["cpra"], result["note_flags"], result["bare_flags"]) == ([False, True], 2, 5),
            f"forum toy flags {result['forum_flags']}; CPRA raised (as written, item 7 = 'Y'): "
            f"{result['cpra']}; all 'Y (note)' raises {result['note_flags']} of 5, all bare 'Y' "
            f"{result['bare_flags']}",
        ),
        practice.Check(
            "FINDING: missing fields show '(missing)' only for 7 of the 12",
            (result["crash"], list(probe.values()).count("(missing)")) == ([5, 6, 7, 8, 12], 7),
            f"render_markdown with each item removed: {probe}",
        ),
        practice.Check(
            "FINDING: the shipped toy's only follow-up is a false positive",
            all([len(flags) == 1, "base model" in flags[0], result["gauss"], result["takeaway"]]),
            f"TOY_EXAMPLE flags {flags} for data from random.gauss; the takeaway "
            "still says items 5 and 7 trigger cascading obligations",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
