"""Exercise 1 — main() measures dialogues, not turns.

    **Easy.** Build the rule-based state tracker in `code/main.py` for 3 slots
    (cuisine, area, price). Test on 10 hand-crafted dialogues. Measure JGA.

Reading of the exercise: the tracker ships, so "build it" is "run it on ten
dialogues I wrote". `DIALOGUES` is that invented set -- ten restaurant bookings,
32 turns, hand-labelled turn by turn with the slots each turn changes, written to
cover the failure modes the lesson's own Pitfalls section names. Over the three
slots the exercise asks for it scores **29 of 32 turns (0.9062)** and **8 of 10
dialogues**; `main()`'s own three dialogues score **3 of 3**.

The exercise names three slots; `SLOT_EXTRACTORS` ships four. Dropping `people`
can only raise an all-or-nothing metric, and it does: **28 of 32 turns at four
slots against 29 at three**, the same 8 of 10 dialogues either way.

The doc defines JGA as "the fraction of *turns* where every slot is correct" and
Step 4 zips per-turn state lists. `main()` does something else -- one comparison
per dialogue, against the final state only. Its gold carries **3 labels for 11
turns**, so the metric the doc defines is not computable from the shipped data.

That substitution is what makes the exercise's number unreadable. At `main()`'s
n=3 the metric moves in steps of 33.3 points; at the exercise's n=10 it moves in
steps of 10. The doc says to beat MultiWOZ's ~83%, and no ten-dialogue evaluation
can express 83% -- the measured **0.80 is one dialogue below 0.90**, so the whole
comparison turns on whether a single dialogue flips.

The second of the doc's three update invariants is never executed by the lesson's
demo. The clear-on-negation branch fires **0 times across `main()`'s 11 turns** --
in "Never mind the cuisine, any food is fine" the extractor already returns `any`
and `continue` skips the branch -- against 2 times across my 32.

Structure: in `DIALOGUES` -- invented, and labelled by hand -- `>` opens a
dialogue, `~` splits turns packed onto one line and `|` separates an utterance
from the slot delta labelled on it. `parse` expands that into (utterance,
gold-state) pairs; `track` runs `update_state` over one dialogue; `jga` scores a slot subset
at both granularities; `lesson_cases` reads `main()`'s own dialogues out of its
source; `negation_fires` lists the turns on which the clearing branch runs.
"""

from __future__ import annotations

import ast
import inspect

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "29-dialogue-state-tracking"

DIALOGUES = """>We would like an Indian place in the west.|c=indian,a=west~Make it cheap.|p=cheap
Three people.|n=3
>Pizza somewhere in the south, please.|c=italian,a=south~Something moderate.|p=moderate
For two people.|n=2
>A fancy Thai restaurant.|c=thai,p=expensive~In the east.|a=east
Actually, no wait, make it the north.|a=north~Five guests.|n=5
>Dim sum for eight diners.|c=chinese,n=8~Budget friendly.|p=cheap~Centre of town.|a=center
>Curry house in the north, moderate price.|c=indian,a=north,p=moderate
On second thought, make it expensive.|p=expensive~Four guests.|n=4
>An expensive place in the south for six people.|p=expensive,a=south,n=6
Never mind the price, it doesn't matter.|p=~Pad thai.|c=thai
>Chinese food in the east.|c=chinese,a=east~A table for two.|n=2~Nothing too expensive.|p=cheap
>Somewhere in the north, moderate price.|a=north,p=moderate
Actually never mind the north, anywhere is fine.|a=~Thai, for four people.|c=thai,n=4
>Cheap curry in the centre.|p=cheap,c=indian,a=center~Sounds good.|~Yes, that one.|
For seven guests.|n=7
>An Italian place, cheap, in the east, for three people.|c=italian,p=cheap,a=east,n=3
Change that to Chinese.|c=chinese~Forget about the area.|a="""
KEYS = {"c": "cuisine", "a": "area", "p": "price", "n": "people"}
SLOTS, THREE = tuple(KEYS.values()), ("cuisine", "area", "price")
EMPTY = dict.fromkeys(SLOTS)


def parse(text):
    """Dialogues as (utterance, gold state) pairs, accumulating the labelled deltas."""
    out = []
    for line in text.splitlines():
        if line.startswith(">"):
            out.append([])
            state, line = dict(EMPTY), line[1:]
        for turn in line.split("~"):
            utterance, _, delta = turn.partition("|")
            state = {**state, **{KEYS[k]: None if not v else int(v) if v.isdigit() else v
                                 for k, v in [p.split("=") for p in delta.split(",") if p]}}
            out[-1].append((utterance, state))
    return out


def track(ref, dialogue):
    """The lesson's incremental tracker: one predicted state per turn, from turns or pairs."""
    states, state = [], dict(EMPTY)
    for turn in dialogue:
        state = ref.update_state(state, turn[0] if isinstance(turn, tuple) else turn)
        states.append(state)
    return states


def jga(ref, data, slots):
    """Turn-level hits, turn count and dialogue-final hits over a slot subset."""
    scored = [(track(ref, d), d) for d in data]
    pairs = [(p, g) for pred, d in scored for p, (_, g) in zip(pred, d)]
    hits = sum(all(p[s] == g[s] for s in slots) for p, g in pairs)
    final = sum(all(pred[-1][s] == d[-1][1][s] for s in slots) for pred, d in scored)
    return {"hits": hits, "turns": len(pairs), "final": final, "n": len(data)}


def lesson_cases(ref):
    """`main()`'s own three dialogues, read out of its source rather than retyped."""
    tree = ast.parse(inspect.getsource(ref.main))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", "") == "dialogues")
    return ast.literal_eval(node.value)


def negation_fires(ref, utterances):
    """(utterance, slot) pairs on which update_state's clear-on-negation branch runs."""
    return [(u, s) for u in utterances for s, extract in ref.SLOT_EXTRACTORS.items()
            if extract(u) is None and ref.is_negation(u, s)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    data, cases = parse(DIALOGUES), lesson_cases(ref)
    said = [u for d in data for u, _ in d]
    upstream = [u for c in cases for u in c["turns"]]
    return {
        "three": jga(ref, data, THREE), "four": jga(ref, data, SLOTS),
        "lesson_ok": sum(track(ref, c["turns"])[-1] == c["gold"] for c in cases),
        "lesson_n": len(cases), "lesson_turns": len(upstream),
        "fires": negation_fires(ref, said), "up_fires": negation_fires(ref, upstream),
        "slots": list(ref.SLOT_EXTRACTORS), "dead": "is_correction" not in
        inspect.getsource(ref.update_state), "cues": [u for u in said if ref.is_correction(u)],
    }


def verify(result):
    three, four, n, up = result["three"], result["four"], result["three"]["n"], result["lesson_n"]
    return [
        practice.Check(
            "ANSWER: 29 of 32 turns and 8 of 10 dialogues on ten hand-crafted dialogues",
            three["hits"] == 29 and three["final"] == 8 and three["turns"] == 32,
            f"over the three slots the exercise names the shipped tracker scores "
            f"{three['hits']}/{three['turns']} ({three['hits'] / three['turns']:.4f}) turns and "
            f"{three['final']}/{n} final states, against {result['lesson_ok']}/{up} on the "
            "lesson's own three",
        ),
        practice.Check(
            "MECHANISM: the exercise names three slots, the code ships four",
            len(result["slots"]) == 4 and three["hits"] > four["hits"],
            f"`SLOT_EXTRACTORS` holds {result['slots']}; dropping `people` can only raise an "
            f"all-or-nothing metric and does -- {four['hits']}/{four['turns']} turns at four "
            f"slots against {three['hits']} at three, the same {three['final']}/{n} dialogues",
        ),
        practice.Check(
            "FINDING: the shipped metric is not the one the doc defines",
            result["lesson_turns"] > up,
            f"the doc calls JGA the fraction of *turns* where every slot is correct; `main()` "
            f"compares once per dialogue against the final state, on gold carrying {up} labels "
            f"for {result['lesson_turns']} turns. Per-turn JGA is not computable from it",
        ),
        practice.Check(
            "FINDING: at ten dialogues the doc's own 83% target is not on the grid",
            three["final"] / n < 0.83 < (three["final"] + 1) / n,
            f"dialogue-final JGA over {n} dialogues moves in steps of {1 / n:.2f}, straddling the "
            f"~0.83 the doc says to beat: {three['final'] / n:.2f} measured, "
            f"{(three['final'] + 1) / n:.2f} next. At n={up} the step is 33.3 points",
        ),
        practice.Check(
            "FINDING: the clear-on-negation invariant never runs in the lesson's own demo",
            not result["up_fires"] and len(result["fires"]) > 0,
            f"it fires {len(result['up_fires'])} times across `main()`'s "
            f"{result['lesson_turns']} turns -- in 'Never mind the cuisine, any food is fine' "
            f"`extract_cuisine` returns 'any' and `continue` skips it -- against "
            f"{len(result['fires'])} here: {result['fires']}",
        ),
        practice.Check(
            "CONTROL: `is_correction` is defined and never called",
            result["dead"] and len(result["cues"]) > 0,
            f"`update_state`'s source never mentions it, though {len(result['cues'])} of "
            f"{four['turns']} turns match a cue: {result['cues']}. Corrections survive only "
            "because extraction overwrites unconditionally, so the doc's 'overwrite the "
            "last-updated slot' rule has nothing to hook into",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
