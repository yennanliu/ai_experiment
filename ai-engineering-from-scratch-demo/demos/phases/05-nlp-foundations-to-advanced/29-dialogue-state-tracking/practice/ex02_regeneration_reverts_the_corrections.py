"""Exercise 2 — regenerating from history reverts the corrections.

    **Medium.** Same dataset with Instructor + Pydantic + a small LLM. Compare
    JGA. Inspect the hardest turns.

Reading of the exercise: `instructor`, `pydantic`, `openai`, `transformers` and
`torch` are all absent -- `importlib.util.find_spec` returns `None` for every one
-- so the LLM arm cannot be built at all. What can be built is the thing the doc
recommends in its place: "always let the LLM regenerate the whole state from
history rather than incrementally updating -- this naturally handles
corrections". Swap the LLM for the lesson's own extractors and that pattern is
exactly `update_state(empty, " ".join(history))`, which is measurable. `DIALOGUES`
is exercise 1's invented ten-dialogue set, unchanged.

Regeneration is **much worse, not better: 20 of 32 turns and 4 of 10 dialogues
against the incremental loop's 28 of 32 and 8 of 10**. The doc's claim about
corrections is backwards for this extractor.

The reason is that the extractors return the first *canonical key* that appears
anywhere in the string, not the last value the user said. `PRICE_WORDS` is a dict
ordered `cheap, moderate, expensive`, so on the concatenated history of "Curry
house in the north, moderate price. / On second thought, make it expensive."
`extract_price` returns **moderate** -- it scans its own key order, and the user's
second thought is behind the first one in that order. Every correction and every
negation in the set is reverted the same way.

The arm is also not reproducible. `AREA_WORDS` is a `set`, and `extract_area`
returns the first member that matches, so on a history mentioning two areas the
answer depends on the set's iteration order -- which is a function of
`PYTHONHASHSEED`. Pinning the container to ascending order scores **20 of 32**;
pinning it to descending order scores **22 of 32**. The incremental loop never
sees two areas in one utterance and is unaffected.

And the pipeline the exercise asks you to reproduce does not exist in the lesson.
The doc's own code blocks call `is_negated`, `NEGATION_CLEARS`,
`joint_goal_accuracy` and `render`; **all four are missing** from `code/main.py`.
Its `RestaurantState` schema declares **5 slots against `SLOT_EXTRACTORS`' 4** --
it adds `day`, which no extractor fills -- so the two arms' JGAs would not even be
over the same slot set.

Structure: `parse` expands exercise 1's labelled fixture (`>` opens a dialogue,
`~` splits packed turns, `|` separates utterance from delta); `incremental` is the
lesson's own loop; `regenerated` is the doc's whole-history pattern; `score`
reports both granularities; `solve` re-imports the lesson twice so
`AREA_WORDS` can be pinned two ways.
"""

from __future__ import annotations

import importlib.util
import re

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "29-dialogue-state-tracking"
LIBS = ("instructor", "pydantic", "openai", "transformers", "torch")
NAMED = ("is_negated", "NEGATION_CLEARS", "joint_goal_accuracy", "render")
ABSENT = [m for m in LIBS if importlib.util.find_spec(m) is None]
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
SLOTS, EMPTY = tuple(KEYS.values()), dict.fromkeys(KEYS.values())


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


def incremental(ref, utterances):
    """The lesson's own loop: `update_state` applied turn by turn."""
    out = [dict(EMPTY)]
    for utterance in utterances:
        out.append(ref.update_state(out[-1], utterance))
    return out[1:]


def regenerated(ref, utterances):
    """The doc's production pattern: rebuild the whole state from the history each turn."""
    return [ref.update_state(dict(EMPTY), " ".join(utterances[:i + 1]))
            for i in range(len(utterances))]


def score(preds, data):
    """Turn-level hits, dialogue-final hits and the turn count."""
    return {"hits": sum(p == g for pred, d in zip(preds, data) for p, (_, g) in zip(pred, d)),
            "final": sum(pred[-1] == d[-1][1] for pred, d in zip(preds, data)),
            "turns": sum(len(d) for d in data)}


def cued(ref, dialogue):
    """Does this dialogue take something back -- a correction cue or a negation cue."""
    return any(ref.is_correction(u) or any(c in u.lower() for c in ref.NEGATION_CUES)
               for u, _ in dialogue)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    data = parse(DIALOGUES)
    said = [[u for u, _ in d] for d in data]
    asc = parity.load_reference(PHASE, LESSON, "main")
    asc.AREA_WORDS = tuple(sorted(ref.AREA_WORDS))
    inc, reg = [incremental(ref, s) for s in said], [regenerated(asc, s) for s in said]
    asc.AREA_WORDS = asc.AREA_WORDS[::-1]
    flipped = score([regenerated(asc, s) for s in said], data)
    fixed = ref.update_state(dict(EMPTY), " ".join(said[4]))
    schema = parity.doc_text(PHASE, LESSON).split("class RestaurantState")[1].split("```")[0]
    return {
        "inc": score(inc, data), "reg": score(reg, data), "prices": list(ref.PRICE_WORDS),
        "desc": flipped,
        "absent": ABSENT, "missing": sorted(set(NAMED) - set(dir(ref))),
        "slots": list(ref.SLOT_EXTRACTORS),
        "schema": re.findall(r"^    (\w+): Optional", schema, re.M),
        "reverted": (fixed["price"], data[4][-1][1]["price"]),
        "differ": [i for i, (a, b) in enumerate(zip(inc, reg)) if a[-1] != b[-1]],
        "cued": [i for i, d in enumerate(data) if cued(ref, d)],
        "is_set": type(ref.AREA_WORDS) is set,
    }


def verify(result):
    inc, reg, n = result["inc"], result["reg"], result["inc"]["turns"]
    return [
        practice.Check(
            "ANSWER: the LLM arm cannot be built; the doc's own substitute scores 20 of 32",
            len(result["absent"]) == len(LIBS) and reg["hits"] < inc["hits"],
            f"{result['absent']} are all absent. The doc's own substitute -- rebuild the state "
            f"from the whole history each turn -- scores {reg['hits']}/{n} turns and "
            f"{reg['final']}/10 dialogues, against {inc['hits']}/{n} and {inc['final']}/10",
        ),
        practice.Check(
            "MECHANISM: the extractors return the first canonical key, not the last mention",
            result["reverted"][0] != result["reverted"][1],
            f"`PRICE_WORDS` is ordered {result['prices']}, so on the history that says 'moderate "
            f"price' then 'on second thought, make it expensive' `extract_price` returns "
            f"{result['reverted'][0]!r} against a gold of {result['reverted'][1]!r}",
        ),
        practice.Check(
            "FINDING: every dialogue the two arms disagree on holds a correction or a negation",
            set(result["differ"]) <= set(result["cued"]) and result["differ"],
            f"the arms end on different states for dialogues {result['differ']}, all inside the "
            f"cue-carrying set {result['cued']} -- the ones it gets right took nothing back",
        ),
        practice.Check(
            "FINDING: the regeneration arm's score is not reproducible across processes",
            result["is_set"] and reg["hits"] != result["desc"]["hits"],
            f"`AREA_WORDS` is a `set` and `extract_area` returns its first matching member, so a "
            f"history naming two areas resolves by iteration order: pinned ascending the arm "
            f"scores {reg['hits']}/{n}, pinned descending {result['desc']['hits']}/{n}",
        ),
        practice.Check(
            "CONTROL: the pipeline the doc writes down is not the one it ships",
            set(result["missing"]) == set(NAMED),
            f"the doc's Step 2, 3 and 4 blocks call {result['missing']} -- `hasattr` is False for "
            "all four, so reproducing its own snippets starts from an AttributeError",
        ),
        practice.Check(
            "CONTROL: the two arms would not even be scored over the same slots",
            len(result["schema"]) > len(result["slots"]),
            f"`RestaurantState` declares {result['schema']} while `SLOT_EXTRACTORS` fills "
            f"{result['slots']}. It adds `day`, which no extractor can produce, and JGA is "
            "all-or-nothing over whatever slot set it gets, so 'compare JGA' compares two metrics",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
