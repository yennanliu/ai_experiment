"""Exercise 4 — the clause that disambiguates is the one the budget cuts.

    Create two descriptions that sound nearly identical. Rewrite them so the
    trigger boundaries do not overlap.

Reading of the exercise: "do not overlap" has to be decided by something
other than reading them, so the pair is scored two ways -- term overlap
between the descriptions, and a router run over queries that each belong to
exactly one skill. The rewrite is then judged by the router, and the router
is what turns the last finding up, because the catalog does not publish what
the author wrote.

**ANSWER: the rewrite removes every tie, and it does so by adding words.**
The near-identical pair's trigger clauses share **3** terms and leave **6**
of **6** queries tied; the rewritten triggers share **0** and route **6** of
**6** correctly. The rewritten descriptions are **84** characters longer, not
shorter.

**FINDING: disambiguation is conditions, not brevity.** Both originals say
what the skill produces and neither says when it applies. Adding the
audience, the trigger event and the destination file separates them; deleting
words from either original separates nothing, because the shared terms are
the nouns both skills genuinely own.

**FINDING: the clause that disambiguates is the one the budget cuts.**
`_shorten` truncates at `max_description_chars`, and both rewritten
sentences share their first **59** characters. At a **60**-character budget
the catalog publishes two identical entries and the router is once again
**6** of **6** tied -- the rewrite is intact on disk and gone from the
model's view.

**FINDING: the author is validated against 1024 characters and the model
reads 240.** Discovery rejects a description over 1024; the default catalog
budget publishes 240 with an ellipsis. A description written to the limit
loses **784** characters silently, and nothing in the pipeline reports the
truncation.

Structure: `route()` is the arbiter, `terms()` the tokenizer it shares with
the overlap measure, so "overlap" and "tie" are the same evidence twice.
"""

from __future__ import annotations

import pathlib
import re
import tempfile

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "24-skill-discovery-and-progressive-disclosure"
STOP = frozenset("a an and are be for from in is it of on or the this to use when with".split())
HINGE = "use this when"
BEFORE = {
    "release-notes": f"Summarize what changed in a shipped version; {HINGE} a release "
                     "goes out.",
    "changelog-entry": f"Note what changed in a shipped version; {HINGE} a release goes out.",
}
AFTER = {
    "release-notes": f"Summarize what changed in a shipped version; {HINGE} the audience "
                     "is customers outside the engineering team.",
    "changelog-entry": f"Summarize what changed in a shipped version; {HINGE} each merged "
                       "pull request needs one dated line in CHANGELOG.md.",
}
QUERIES = {
    "release-notes": ["announce the new version to customers",
                      "draft the announcement for customers outside engineering",
                      "readers outside engineering need the announcement"],
    "changelog-entry": ["append a dated line for this merged pull request",
                        "update CHANGELOG.md with today's dated line",
                        "each merged pull request needs one entry"],
}


def terms(text):
    return {word for word in re.findall(r"[a-z0-9.]+", text.lower()) if word not in STOP}


def route(descriptions, query):
    """Highest term overlap wins; a tie at the top is a routing failure."""
    scored = sorted(((len(terms(text) & terms(query)), name)
                     for name, text in descriptions.items()), reverse=True)
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return "tie"
    return scored[0][1]


def score(descriptions):
    """Ties and correct routes over the fixed query set."""
    outcomes = [(expected, route(descriptions, query))
                for expected, queries in QUERIES.items() for query in queries]
    return {"ties": sum(got == "tie" for _, got in outcomes),
            "correct": sum(want == got for want, got in outcomes), "total": len(outcomes)}


def shared(descriptions, trigger=False):
    texts = [text.split(HINGE)[-1] if trigger else text for text in descriptions.values()]
    return len(terms(texts[0]) & terms(texts[1]))


def published(ref, base, descriptions, limit):
    """What the catalog actually shows the model at this budget."""
    for name, text in descriptions.items():
        ref._write_skill(base / str(limit), name, text, f"# {name}\n")
    candidates = ref.discover_scope(ref.Scope("user", base / str(limit)))
    catalog = ref.build_catalog(candidates, ("user",),
                                ref.CatalogBudget(max_description_chars=limit))
    return {entry.name: entry.description for entry in catalog.entries}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    with tempfile.TemporaryDirectory() as temp:
        base = pathlib.Path(temp)
        wide = published(ref, base, AFTER, 240)
        narrow = published(ref, base, AFTER, 60)
        long_text = "Report evidence. " + "x" * (1_024 - 17)
        ref._write_skill(base / "long", "evidence-report", long_text, "# Evidence\n")
        candidates = ref.discover_scope(ref.Scope("user", base / "long"))
        catalog = ref.build_catalog(candidates, ("user",), ref.CatalogBudget())
        entry = catalog.entries[0]
    return {
        "before": score(BEFORE), "after": score(AFTER),
        "before_shared": shared(BEFORE), "after_shared": shared(AFTER),
        "before_trigger": shared(BEFORE, trigger=True),
        "after_trigger": shared(AFTER, trigger=True),
        "before_chars": sum(map(len, BEFORE.values())),
        "after_chars": sum(map(len, AFTER.values())),
        "wide": score(wide), "narrow": score(narrow),
        "narrow_lengths": sorted({len(text) for text in narrow.values()}),
        "authored": len(long_text), "published": len(entry.description),
        "dropped": len(long_text) - len(entry.description),
        "elided": entry.description.endswith("…"),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the rewrite removes every tie, and it does so by adding words",
            all([result["before"] == {"ties": 6, "correct": 0, "total": 6},
                 result["after"] == {"ties": 0, "correct": 6, "total": 6},
                 result["before_trigger"] == 3, result["after_trigger"] == 0,
                 result["after_chars"] > result["before_chars"]]),
            f"the near-identical triggers share {result['before_trigger']} terms and leave "
            f"{result['before']['ties']} of {result['before']['total']} queries tied; the "
            f"rewritten triggers share {result['after_trigger']} and route "
            f"{result['after']['correct']} of {result['after']['total']} correctly, at "
            f"{result['after_chars'] - result['before_chars']} characters more, not fewer",
        ),
        practice.Check(
            "FINDING: disambiguation is conditions, not brevity",
            all([result["before"]["correct"] == 0, result["after"]["correct"] == 6,
                 result["after_chars"] - result["before_chars"] == 84,
                 result["after_shared"] == 5]),
            f"both originals say what the skill produces and neither says when it applies, "
            f"and the rewrite leaves {result['after_shared']} shared terms in the part that "
            "says what. Adding the audience, the destination file and the unit of work "
            "separates them; deleting words separates nothing, because the shared terms are "
            "the nouns both skills genuinely own",
        ),
        practice.Check(
            "FINDING: the clause that disambiguates is the one the budget cuts",
            all([result["wide"]["ties"] == 0, result["wide"]["correct"] == 6,
                 result["narrow"]["ties"] == 6, result["narrow_lengths"] == [59]]),
            f"at a 240-character budget the catalog routes {result['wide']['correct']} of 6; "
            f"at 60 both entries truncate to the same {result['narrow_lengths'][0]} "
            f"characters and the router is {result['narrow']['ties']} of 6 tied, because "
            "_shorten cuts from the end and the conditions are at the end. The rewrite is "
            "intact on disk and gone from the model's view",
        ),
        practice.Check(
            "FINDING: the author is validated against 1024 characters and the model reads 240",
            all([result["authored"] == 1_024, result["published"] == 240,
                 result["dropped"] == 784, result["elided"]]),
            f"discovery rejects a description over {result['authored']} and the default "
            f"budget publishes {result['published']}, so {result['dropped']} characters go "
            "missing behind an ellipsis. Nothing in the catalog reports the truncation, so "
            "the text the author validated is not the text the model reads",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
