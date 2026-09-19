"""Exercise 5 — the second round changes no ranking, so the tax buys nothing.

    Agentic multi-hop RAG has a latency tax per round-trip. At what query
    difficulty does the accuracy gain justify the latency?

Reading of the exercise: the break-even is derived and then the lesson's own
`agentic_loop` is run to measure its gain, because a formula for when a round
pays off is not useful until one round has been priced -- and the round the
lesson ships turns out to change every score and no answer.

**ANSWER: when the extra round's recall gain exceeds the latency cost divided by
the value of a correct answer** -- and the lesson's own round gains **0.00**.
Round 1 ranks `['r1', 'r4', 'r3']`; round 2 ranks `['r1', 'r4', 'r3']`. The top
score moves from **0.686** to **0.734** and the ordering does not move at all.

**FINDING: the trigger fires on a number that the reformulation is guaranteed to
raise.** Round 2 reweights toward the image retriever, whose scores are 0 or 1 --
the widest span of the three (Exercise 2) -- so shifting weight into it raises the
top fused score by construction. The loop then reports a higher "confidence" and
stops. It is a thermostat wired to its own heater.

**FINDING: and it stops below its own floor without saying so.** At a floor of
0.99 the loop runs its one reformulation, reaches **0.734**, and returns -- there
is no third round and no signal that the threshold was never met. A caller
reading only the answer cannot tell a satisfied loop from an exhausted one.

**ANSWER: so the difficulty threshold is a property of the retrieval, not of the
query.** A round pays when the first-round candidate set *misses* gold that a
reformulation can reach. Exercise 4's measurement says that cannot happen here:
fused recall@2 is already **1.00**, so no reformulation has anything to add and
every extra round is pure tax. The condition to test before building the loop is
first-round recall -- and if it is 1.0, difficulty is not the variable.

Structure: `rounds` reproduces the lesson's own two-round reweighting, `ordering`
compares the rankings, and `gain` is the recall difference the tax has to buy.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "24-multimodal-rag-cross-modal"
QUERY = "find me a quiet vegan brunch with natural light"
GOLD = frozenset({"r1", "r4"})
ROUND_ONE = (0.3, 0.4, 0.3)
ROUND_TWO = (0.3, 0.5, 0.2)
SUFFIX = " bright windows low noise"
HIGH_FLOOR = 0.99


def fuse_round(ref, query, weights):
    rows = [ref.text_retrieve(QUERY), ref.image_retrieve(query), ref.audio_retrieve(query)]
    return ref.fuse(rows, list(weights))


def ordering(ref, scores, k=3):
    return [doc for doc, _ in ref.top_k(scores, k)]


def recall_at(ref, scores, k=2, gold=GOLD):
    picked = {doc for doc, _ in ref.top_k(scores, k)}
    return round(len(picked & gold) / len(gold), 2)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    first = fuse_round(ref, QUERY, ROUND_ONE)
    second = fuse_round(ref, QUERY + SUFFIX, ROUND_TWO)
    trace = ref.agentic_loop(QUERY, HIGH_FLOOR)
    spans = {name: round(max(fn(QUERY).values()) - min(fn(QUERY).values()), 4)
             for name, fn in (("text", ref.text_retrieve), ("image", ref.image_retrieve),
                              ("audio", ref.audio_retrieve))}
    return {
        "first_order": ordering(ref, first), "second_order": ordering(ref, second),
        "same_order": ordering(ref, first) == ordering(ref, second),
        "first_top": round(max(first.values()), 4),
        "second_top": round(max(second.values()), 4),
        "score_gain": round(max(second.values()) - max(first.values()), 4),
        "first_recall": recall_at(ref, first), "second_recall": recall_at(ref, second),
        "recall_gain": round(recall_at(ref, second) - recall_at(ref, first), 2),
        "spans": spans, "widest": max(spans, key=spans.get),
        "image_weight_change": round(ROUND_TWO[1] - ROUND_ONE[1], 1),
        "rounds_run": trace.count("round "),
        "stops_below_floor": max(second.values()) < HIGH_FLOOR,
        "floor": HIGH_FLOOR,
        "signals_failure": "threshold" in trace.lower(),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: when the recall gain beats the tax -- and this round gains 0.00",
            all([result["same_order"], result["first_order"] == ["r1", "r4", "r3"],
                 result["recall_gain"] == 0.0, result["first_recall"] == 1.0,
                 result["score_gain"] == 0.0475]),
            f"round 1 ranks {result['first_order']} and round 2 ranks "
            f"{result['second_order']} -- identical. The top score moves "
            f"{result['first_top']} -> {result['second_top']} "
            f"({result['score_gain']:+}) and the recall does not move at all "
            f"({result['recall_gain']:+})",
        ),
        practice.Check(
            "FINDING: the trigger fires on a number the reformulation must raise",
            all([result["widest"] == "image", result["spans"]["image"] == 1.0,
                 result["image_weight_change"] == 0.1, result["score_gain"] > 0]),
            f"round 2 shifts {result['image_weight_change']} of weight into the image "
            f"retriever, whose span is {result['spans']['image']} against "
            f"{ {k: v for k, v in result['spans'].items() if k != 'image'} } -- the widest of "
            "the three. Moving weight into it raises the top fused score by construction, so "
            "the loop reports a higher confidence and stops. A thermostat wired to its own "
            "heater",
        ),
        practice.Check(
            "FINDING: and it stops below its own floor without saying so",
            all([result["rounds_run"] == 2, result["stops_below_floor"],
                 not result["signals_failure"]]),
            f"at a floor of {result['floor']} the loop runs {result['rounds_run']} rounds, "
            f"reaches {result['second_top']} and returns. There is no third round and no "
            "signal that the threshold was never met, so a caller reading the answer cannot "
            "tell a satisfied loop from an exhausted one",
        ),
        practice.Check(
            "ANSWER: the threshold is a property of the retrieval, not the query",
            all([result["first_recall"] == 1.0, result["recall_gain"] == 0.0]),
            f"a round pays when the first-round candidates miss gold that a reformulation can "
            f"reach. Exercise 4 measures fused recall@2 at {result['first_recall']} here, so "
            "no reformulation has anything to add and every extra round is pure tax. The "
            "condition to test before building the loop is first-round recall -- and if it is "
            "1.0, difficulty is not the variable",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
