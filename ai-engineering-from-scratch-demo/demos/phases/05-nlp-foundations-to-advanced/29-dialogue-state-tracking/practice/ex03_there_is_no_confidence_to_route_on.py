"""Exercise 3 — there is no confidence to route on.

    **Hard.** Implement both and route: rule-based primary, LLM fallback when
    rule-based emits <2 slots with confidence. Measure the combined JGA and
    inference cost per turn.

Reading of the exercise: the LLM arm is unbuildable here (exercise 2: instructor,
pydantic, openai, transformers and torch are all absent), so the fallback is
modelled as an oracle that writes the gold state -- an upper bound on any model,
which is the honest way to price a router you cannot run. The dataset is exercise
1's invented ten dialogues, 32 turns.

Nothing in `code/main.py` emits a confidence. `extract_cuisine` returns a string
or `None`, `update_state` returns a plain `dict` of values, and asking for a score
raises **TypeError: extract_cuisine() got an unexpected keyword argument
'confidence'**. The only quantity the trigger can read is the count of non-`None`
slots, so "emits <2 slots with confidence" collapses to "emits <2 slots".

That sentence still has two readings, and they do not merely differ -- they have
no overlap. Counting the slots *this turn's extractors* produce, the fallback
fires on **21 of 32 turns (0.6562)** and an oracle lifts turn-JGA from **28/32 to
32/32**. Counting the slots *the state* holds after the turn, it fires on **0 of
32**: every dialogue's opening utterance already fills two slots, so the fallback
is never reached and the "combined JGA" is the rule-only 28/32, exactly.

Under the reading that does fire, the trigger is nearly uninformative. **18 of the
21 escalated turns were already correct**, a trigger precision of 3/21 = 0.1429,
because the slot count measures how much the user said, not whether the extractor
understood it. Its lowest-confidence bucket makes the point: of the **5 turns that
emit nothing at all, 4 are correct** -- "Sounds good.", "Yes, that one." and the
two negations, where emitting nothing is the right answer.

So the whole measured gain belongs to the fallback, and the combined JGA is a
measurement of the oracle. One of the four repairs even lands on a turn the router
never escalated: a repaired state carries forward, so the next turn inherits it.

On cost, only one arm is measurable. `update_state` runs in **microseconds per
turn**; the routed cost per turn is that plus 0.6562 x the LLM call, so the
rule-based primary removes at most 34% of the model traffic under one reading and
100% of it -- along with all of the benefit -- under the other.

Structure: `parse` expands exercise 1's labelled fixture; `emitted` counts the
slots one turn's extractors fill; `route` runs the policy for either reading of
the threshold, with or without the oracle fallback, and also tallies the
zero-emission bucket; `cost` times the rule arm.
"""

from __future__ import annotations

import timeit

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
EMPTY = dict.fromkeys(KEYS.values())


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


def emitted(ref, utterance):
    """Slots this turn's extractors fill -- the only quantity standing in for confidence."""
    return sum(extract(utterance) is not None for extract in ref.SLOT_EXTRACTORS.values())


def route(ref, data, reading, oracle):
    """Escalations, hits, escalations onto right turns, unseen errors, the emit-nothing bucket."""
    tally = dict.fromkeys(("fired", "hits", "spare", "missed", "zero", "zero_wrong"), 0)
    for dialogue in data:
        state = dict(EMPTY)
        for utterance, gold in dialogue:
            state = ref.update_state(state, utterance)
            fills = emitted(ref, utterance)
            count = fills if reading == "turn" else sum(v is not None for v in state.values())
            tally["zero"] += fills == 0
            tally["zero_wrong"] += fills == 0 and state != gold
            if count < 2:
                tally["fired"] += 1
                tally["spare"] += state == gold
                state = dict(gold) if oracle else state
            else:
                tally["missed"] += state != gold
            tally["hits"] += state == gold
    return tally


def cost(ref, said, repeats=300):
    """Seconds per turn for the rule arm, best of three timed passes."""
    passes = timeit.repeat(lambda: [ref.update_state(EMPTY, u) for u in said],
                           repeat=3, number=repeats)
    return min(passes) / (repeats * len(said))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    data = parse(DIALOGUES)
    said = [u for d in data for u, _ in d]
    try:
        ref.extract_cuisine("cheap italian food", confidence=True)
        raised = None
    except TypeError as exc:
        raised = f"{type(exc).__name__}: {exc}"
    return {
        "turn": route(ref, data, "turn", False), "turn_or": route(ref, data, "turn", True),
        "state": route(ref, data, "state", False), "state_or": route(ref, data, "state", True),
        "n": len(said), "seconds": cost(ref, said), "raised": raised,
        "returns": type(ref.update_state(dict(EMPTY), said[0])).__name__,
        "scored": [n for n in dir(ref) if "conf" in n.lower() or "score" in n.lower()],
    }


def verify(result):
    turn, state, n, micro = result["turn"], result["state"], result["n"], result["seconds"] * 1e6
    return [
        practice.Check(
            "ANSWER: no confidence exists, so the trigger is a slot count; 32/32 with an oracle",
            result["raised"] is not None and not result["scored"],
            f"asking for one raises {result['raised']!r}; `update_state` returns a plain "
            f"{result['returns']} and nothing in the module is named for a score "
            f"({result['scored']}). Counting emitted slots instead, the fallback fires "
            f"{turn['fired']}/{n} times and an oracle lifts turn-JGA {turn['hits']}/{n} to "
            f"{result['turn_or']['hits']}/{n}, at {micro:.2f} us/turn for the rule arm",
        ),
        practice.Check(
            "MECHANISM: the threshold's two readings have no overlap",
            state["fired"] == 0 and turn["fired"] > 0,
            f"slots emitted by the turn -- {turn['fired']}/{n} escalations "
            f"({turn['fired'] / n:.4f}); slots filled in the state -- {state['fired']}/{n}, "
            "because every dialogue's opening utterance already fills two",
        ),
        practice.Check(
            "FINDING: under the state reading the combined JGA is the rule-only JGA, exactly",
            result["state_or"]["hits"] == state["hits"] == turn["hits"],
            f"{result['state_or']['hits']}/{n} with the oracle wired in and {state['hits']}/{n} "
            "without: the cheap reading of the exercise's own sentence buys precisely nothing",
        ),
        practice.Check(
            "FINDING: the trigger that does fire is nearly uninformative",
            turn["spare"] > turn["fired"] - turn["spare"],
            f"{turn['spare']} of the {turn['fired']} escalated turns were already correct -- a "
            f"trigger precision of {(turn['fired'] - turn['spare']) / turn['fired']:.4f}. Slot "
            "count measures how much the user said, not whether the extractor understood it",
        ),
        practice.Check(
            "CONTROL: the lowest-confidence bucket is the one that is mostly right",
            turn["zero"] - turn["zero_wrong"] > turn["zero_wrong"],
            f"of the {turn['zero']} turns that emit nothing at all, {turn['zero_wrong']} is "
            "wrong: 'Sounds good.', 'Yes, that one.' and the two negations are turns where "
            "emitting nothing is right -- and the trigger ranks them least confident",
        ),
        practice.Check(
            "FINDING: the combined JGA measures the oracle, and only one arm has a price",
            result["turn_or"]["hits"] == n and turn["missed"] > 0,
            f"all {n} turns are right once the fallback is perfect, {turn['missed']} of them on "
            f"a turn the router never escalated -- a repaired state carries forward. Cost is "
            f"{micro:.2f} us plus {turn['fired'] / n:.4f} of an LLM call, so the rule primary "
            f"sheds at most {100 - 100 * turn['fired'] / n:.1f}% of the model traffic",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
