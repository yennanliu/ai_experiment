"""Exercise 6 — a sentence that belongs in two places belongs in neither.

    Review an existing skill and label every sentence as routing, procedure,
    policy, reference pointer, or output contract. Move anything that does not
    belong.

Reading of the exercise: labelling every sentence forces a decision on the
ones that carry two kinds at once, and those are the ones worth reporting --
a sentence that is half procedure and half policy cannot be moved without
being split first. So the labeller returns a set per sentence rather than a
label, and the review's output is three numbers: what was already right, what
moved, and what had to be rewritten before it could move.

**ANSWER: 14 sentences, 5 labels, and 6 of them move.** After the review
`SKILL.md` keeps routing and procedure, `policy.md` takes the constraints,
`reference.md` the pointers and `output-template.md` the contract -- and
every sentence lands in exactly **1** file. **2** of the 14 carried two
labels and were split before they could be placed.

**FINDING: the mixed sentences are mixed in one direction.** Both carry
procedure *and* policy -- "run the export, but never to a public bucket" --
because a constraint is easiest to write where the action is. **0** sentences
mix routing with anything, since routing is a single sentence in the
frontmatter's description and everything else is body.

**FINDING: the description is routing and is the one sentence the validator
reads.** It is **1** of the 14 and the only one with a length bound (1024),
a required-ness check and a home the format fixes. The other **13** are
"body", which the validator checks for being non-empty -- so **13** of 14
sentences could be in any order in any file.

**FINDING: moving a sentence changes no verdict.** The skill validates
before the review and after it, with the same **0** issues, because the
body's *content* is never inspected. The review improves what a model reads
and is invisible to every check the module ships -- which is the same finding
exercise 4 reaches from the other direction.

Structure: `label` returns the set of kinds a sentence carries, `place`
assigns single-label sentences to files, and `split` is what a two-label
sentence needs first.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "22-skills-and-agent-sdks"
NAME = "incident-report"
HOMES = {"routing": "SKILL.md", "procedure": "SKILL.md", "policy": "policy.md",
         "reference": "reference.md", "contract": "output-template.md"}
MARKERS = {
    "routing": ("use when", "use this skill"),
    "procedure": ("first,", "then", "next,", "finally,", "run "),
    "policy": ("never", "must not", "only if", "do not"),
    "reference": ("see ", "refer to", "documented in"),
    "contract": ("output", "return a", "the response must be"),
}
SKILL = [
    "Use when writing an incident report from an outage timeline.",
    "First, collect the timeline entries for the affected window.",
    "Then group them by service.",
    "Next, identify the first entry that changed customer behaviour.",
    "Never include customer identifiers in the summary.",
    "Run the export script, but never to a public bucket.",
    "See the severity matrix for the levels and their meanings.",
    "Refer to the on-call rota documented in the operations handbook.",
    "Output a markdown document with a summary and a timeline table.",
    "The response must be under two pages.",
    "Finally, attach the raw timeline as an appendix.",
    "Do not publish before the incident commander signs off.",
    "Run the linter, and do not proceed if it reports an error.",
    "Return a single file named incident-report.md.",
]


def label(sentence):
    lowered = sentence.lower()
    return {kind for kind, markers in MARKERS.items()
            if any(marker in lowered for marker in markers)}


def split(sentence):
    """A two-label sentence becomes one sentence per label, at the comma."""
    head, _, tail = sentence.partition(", ")
    return [head + ".", tail[0].upper() + tail[1:]]


def place(sentences):
    """Every sentence into the one file its label names, splitting where needed."""
    files, rewritten = {}, []
    for sentence in sentences:
        kinds = label(sentence)
        pieces = [sentence] if len(kinds) <= 1 else split(sentence)
        if len(kinds) > 1:
            rewritten.append(sentence)
        for piece in pieces:
            kind = next(iter(label(piece)), "procedure")
            files.setdefault(HOMES[kind], []).append(piece)
    return files, rewritten


def skill_text(body):
    return "\n".join(["---", f"name: {NAME}", f"description: {SKILL[0]}", "---", ""]
                     + body + [""])


def selected(labels, predicate):
    return [sentence for sentence, kinds in labels.items() if predicate(kinds)]


def leaving(sentences):
    """Single-label body sentences whose home is not SKILL.md."""
    return [s for s in sentences
            if len(label(s)) == 1 and HOMES[sorted(label(s))[0]] != "SKILL.md"]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    labels = {sentence: sorted(label(sentence)) for sentence in SKILL}
    mixed = selected(labels, lambda kinds: len(kinds) > 1)
    files, rewritten = place(SKILL)
    before = ref.validate_skill_text(skill_text(SKILL[1:]), NAME)
    after = ref.validate_skill_text(skill_text(files["SKILL.md"]), NAME)
    return {
        "sentences": len(SKILL), "labels": sorted(MARKERS),
        "unlabelled": selected(labels, lambda kinds: not kinds),
        "mixed": mixed, "mixed_kinds": sorted({tuple(labels[s]) for s in mixed}),
        "rewritten": len(rewritten),
        "files": sorted(files), "counts": {p: len(b) for p, b in files.items()},
        "placed": sum(len(b) for b in files.values()),
        "moved": len(leaving(SKILL[1:])),
        "routing_mixed": [s for s in mixed if "routing" in labels[s]],
        "description_is_routing": labels[SKILL[0]] == ["routing"],
        "body_sentences": len(SKILL) - 1,
        "before_valid": before.valid, "after_valid": after.valid,
        "before_issues": len(before.issues), "after_issues": len(after.issues),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 14 sentences, 5 labels, and the mixed ones split before they move",
            all([result["sentences"] == 14, len(result["labels"]) == 5,
                 result["unlabelled"] == [], result["rewritten"] == 2,
                 result["placed"] == result["sentences"] + result["rewritten"],
                 sorted(result["files"]) == ["SKILL.md", "output-template.md",
                                             "policy.md", "reference.md"]]),
            f"{result['sentences']} sentences across {len(result['labels'])} labels "
            f"{result['labels']}, none unlabelled, land in {len(result['files'])} files "
            f"{result['counts']}. {result['rewritten']} carried two labels and were split "
            f"first, which is why {result['placed']} pieces come out of "
            f"{result['sentences']} sentences",
        ),
        practice.Check(
            "FINDING: the mixed sentences are mixed in one direction",
            all([result["mixed_kinds"] == [("policy", "procedure")],
                 result["routing_mixed"] == []]),
            f"both mixed sentences carry {result['mixed_kinds'][0]} -- a constraint written "
            f"where the action is -- and {len(result['routing_mixed'])} mix routing with "
            "anything, because routing is one sentence in the description and everything "
            "else is body",
        ),
        practice.Check(
            "FINDING: the description is routing and is the one sentence the validator reads",
            all([result["description_is_routing"], result["body_sentences"] == 13]),
            f"the description is the {1} routing sentence and the only one with a length "
            f"bound, a required-ness check and a home the format fixes. The other "
            f"{result['body_sentences']} are body, which is checked for being non-empty -- "
            "so they could be in any order in any file",
        ),
        practice.Check(
            "FINDING: moving a sentence changes no verdict",
            all([result["before_valid"], result["after_valid"],
                 result["before_issues"] == result["after_issues"] == 0,
                 result["moved"] > 0]),
            f"the skill validates before the review and after it, with "
            f"{result['before_issues']} issues either way, although {result['moved']} "
            "sentences left SKILL.md. The body's content is never inspected, so the review "
            "improves what a model reads and is invisible to every check the module ships",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
