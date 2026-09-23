"""Exercise 1 — the sanitizer reports stripping and the attack lands.

    Run `code/main.py`. Identify which attack the sanitizer catches but the
    read/write boundary does not, and which attack only the read/write
    boundary catches.

Reading of the exercise: two questions, so both are answered by running the
full 3-by-4 grid rather than by reading the headline. One of them turns out to
have no answer, which is worth establishing rather than working around.

**ANSWER: none, and the URL-fragment injection.** The read/write boundary
refuses the write on **both** attacks, so the set "caught by the sanitizer and
not by the boundary" is empty. The fragment payload lives in the URL and the
sanitizer only inspects HTML, so it is the one the boundary catches alone.

**FINDING: the two defenses disagree about what "caught" means.** On the
visible-text injection the sanitizer leaves the agent posting **'ok'** to
`/api/post` -- the user's actual request, completed -- while the boundary
returns **no write at all**. One defense repairs the trajectory and the other
halts it; the grid prints both as a success and the difference is the entire
operational question.

**FINDING: one verb defeats the sanitizer, and it still reports a catch.**
Rule **2** of 3 is `post .* to /api/exfil`, so rewriting the payload's verb
from "Post" to "Send", "Submit", "Deliver" or "Forward" leaves `/api/exfil`
in the context and the agent follows it -- **4** of **5** verbs land the
attack. In all five the sanitizer still reports that it stripped rules,
because rules 1 and 3 match the surrounding prose either way.

**FINDING: the agent has one trigger, not a behaviour.** `run_agent` follows
an injection if and only if the literal substring `exfil` appears in the
context. No other endpoint is ever followed, and the page's own payload text
carries the unformatted placeholder `{sensitive}` -- the body that actually
gets posted comes from the module constant. The simulator demonstrates a
defense grid against an attacker it has hard-coded.

Structure: `grid()` runs every page against every defense; `reverbed()`
re-runs the visible-text payload with its verb swapped.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "11-browser-agents"

VERBS = ("Post", "Send", "Submit", "Deliver", "Forward")


def grid(ref):
    """{(case, defense): endpoint or None} over every page and defense."""
    return {(name, defense): ref.run_agent(page, defense).posted_to
            for name, page in ref.CASES for defense in ref.DEFENSES}


def caught(cells, defense):
    """Cases where this defense stopped the write reaching /api/exfil."""
    return {name for (name, used), endpoint in cells.items()
            if used == defense and endpoint != "/api/exfil"} - {"benign page"}


def reverbed(ref, verb):
    """The visible-text payload with its verb swapped, under the sanitizer."""
    original = ref.PAGE_VISIBLE_INJECTION
    page = {"url": original["url"],
            "html": original["html"].replace("Post '", f"{verb} '")}
    result = ref.run_agent(page, "sanitizer")
    return result.posted_to, any("stripped" in note for note in result.notes)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    cells = grid(ref)
    swapped = {verb: reverbed(ref, verb) for verb in VERBS}
    agent = inspect.getsource(ref.run_agent)
    return {
        "cases": len(ref.CASES), "defenses": len(ref.DEFENSES),
        "sanitizer_only": sorted(caught(cells, "sanitizer") - caught(cells, "rw_boundary")),
        "boundary_only": sorted(caught(cells, "rw_boundary") - caught(cells, "sanitizer")),
        "sanitized_write": cells[("visible-text injection", "sanitizer")],
        "boundary_write": cells[("visible-text injection", "rw_boundary")],
        "rules": len(ref.SANITIZER_RULES),
        "verb_rule": ref.SANITIZER_RULES[1],
        "landed": sorted(verb for verb, (endpoint, _r) in swapped.items()
                         if endpoint == "/api/exfil"),
        "reported_anyway": sum(1 for _e, reported in swapped.values() if reported),
        "verbs": len(VERBS),
        "trigger": agent.count('"exfil" in context.lower()'),
        "placeholder": "{sensitive}" in ref.PAGE_VISIBLE_INJECTION["html"],
        "body_from_constant": "target_body = SENSITIVE" in agent,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: none, and the URL-fragment injection",
            all([result["sanitizer_only"] == [],
                 result["boundary_only"] == ["URL-fragment injection"],
                 result["cases"] == 3, result["defenses"] == 4]),
            f"over the {result['cases']}-by-{result['defenses']} grid the boundary "
            f"stops both attacks, so the sanitizer-only set is "
            f"{result['sanitizer_only']}; the boundary-only set is "
            f"{result['boundary_only']}, whose payload lives in the URL",
        ),
        practice.Check(
            "FINDING: the two defenses disagree about what caught means",
            all([result["sanitized_write"] == "/api/post",
                 result["boundary_write"] is None]),
            f"on the visible-text injection the sanitizer leaves the agent posting to "
            f"{result['sanitized_write']} -- the user's real request, completed -- while "
            "the boundary returns no write at all",
        ),
        practice.Check(
            "FINDING: one verb defeats the sanitizer, and it still reports a catch",
            all([result["landed"] == ["Deliver", "Forward", "Send", "Submit"],
                 result["reported_anyway"] == result["verbs"] == 5,
                 result["rules"] == 3]),
            f"rule 2 of {result['rules']} is {result['verb_rule']!r}, so "
            f"{len(result['landed'])} of {result['verbs']} verbs land the attack -- "
            f"{result['landed']} -- and all {result['reported_anyway']} still report "
            "stripped rules",
        ),
        practice.Check(
            "FINDING: the agent has one trigger, not a behaviour",
            all([result["trigger"] == 1, result["placeholder"],
                 result["body_from_constant"]]),
            f"run_agent follows an injection on {result['trigger']} literal substring "
            "test, the payload text carries an unformatted {sensitive} placeholder, and "
            "the posted body comes from the module constant",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
