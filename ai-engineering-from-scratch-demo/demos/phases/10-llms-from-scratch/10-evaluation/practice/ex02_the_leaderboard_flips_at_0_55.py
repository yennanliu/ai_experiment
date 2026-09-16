"""Exercise 2 — the leaderboard flips at weight 0.55, and never on the lesson's own two models.

    Extend the ELO tracker to support multiple judge functions (exact match, F1,
    LLM-as-judge) and weight them. Compare how the leaderboard changes when you
    weight exact match heavily versus F1 heavily.

Reading of the exercise: the three judges are the lesson's own `exact_match`,
`token_f1` and `llm_judge_simulated`, and the weighted score decides each
pairwise match fed to `ELOTracker.record_match`. The comparison is run on the
lesson's own two demo models first, and then on a third that the metrics
disagree about, because two models the metrics agree on cannot show a
leaderboard changing.

**FINDING: on the lesson's own two models no weighting changes anything.**
`demo_model_good` scores 1.000 on all three judges and `demo_model_bad` scores
0.000, 0.291 and 0.630 -- good dominates on every metric, so every weighting
produces the same order. The comparison the exercise asks for has no degrees of
freedom on the data it ships.

**ANSWER: add a model the metrics disagree about and the leaderboard flips at
weight 0.55.** A `terse` model that answers one question exactly and says "no"
to the rest scores **0.143 exact / 0.143 F1**, against bad's **0.000 exact /
0.291 F1**. Below 0.55 the order is `good, bad, terse`; at 0.55 and above it is
`good, terse, bad`. The crossover is where `w * 0.143 + (1-w) * 0.143` overtakes
`w * 0 + (1-w) * 0.291`.

**MECHANISM: the two judges disagree about what a wrong answer is worth.**
`exact_match` gives a verbose-but-correct answer 0. `token_f1` gives it partial
credit for the words it shares. So a model that is never exactly right but
always nearly right beats a model that is occasionally exactly right, under F1,
and loses under exact match. Weighting them is choosing which of those two
failures to call worse.

**FINDING: ELO adds nothing here except path dependence.** The judges are
deterministic, so every match between two models has the same outcome every
time and the ELO ranking is the ranking by mean score. Running the same three
matches in a different order moves the ratings -- 1532.0 against 1530.5 for the
same player -- without changing the order, so what ELO contributes to this
leaderboard is a number that depends on scheduling.

Structure: `weighted` combines the three judges at one weight; `leaderboard`
runs the round robin through the lesson's own `ELOTracker`.
"""

from __future__ import annotations

import statistics

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "10-evaluation"
ROUNDS = 30
WEIGHTS = tuple(i / 20 for i in range(21))
QA = (
    ("What is the capital of France?", "Paris"),
    ("What is 2 + 2?", "4"),
    ("Who wrote Hamlet?", "William Shakespeare"),
    ("What language is PyTorch written in?", "Python and C++"),
    ("What is the boiling point of water?", "100 degrees Celsius"),
    ("Name the largest planet.", "Jupiter"),
    ("What year did World War 2 end?", "1945"),
)


def terse(prompt):
    """Exactly right once, tersely wrong otherwise -- the metrics disagree about it."""
    return {"Name the largest planet.": "Jupiter"}.get(prompt, "no")


def judge_means(ref, model_fn):
    """Mean score of one model under each of the lesson's three judges."""
    judges = {"exact": ref.exact_match, "f1": ref.token_f1,
              "judge": ref.llm_judge_simulated}
    suite = ref.EvalSuite("demo", [ref.EvalCase(p, e) for p, e in QA], judges)
    rows = suite.run(model_fn)
    return {name: statistics.fmean(row["scores"][name] for row in rows) for name in judges}


def weighted(scores, weight):
    return weight * scores["exact"] + (1 - weight) * scores["f1"]


def leaderboard(ref, scores, weight):
    """A round robin scored at `weight`, run through the lesson's own ELOTracker."""
    tracker = ref.ELOTracker()
    names = list(scores)
    for _ in range(ROUNDS):
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                left, right = weighted(scores[a], weight), weighted(scores[b], weight)
                tracker.record_match(a, b, "a" if left > right else
                                     ("b" if right > left else "tie"))
    return [name for name, _ in tracker.leaderboard()]


def order_dependence(ref):
    """The same three matches in two orders: ratings differ, order does not."""
    out = []
    for schedule in ([("a", "b"), ("b", "c"), ("a", "c")],
                     [("a", "c"), ("b", "c"), ("a", "b")]):
        tracker = ref.ELOTracker()
        for left, right in schedule:
            tracker.record_match(left, right, "a")
        out.append(tracker.leaderboard())
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    shipped = {"good": ref.demo_model_good, "bad": ref.demo_model_bad}
    scores = {name: judge_means(ref, fn) for name, fn in shipped.items()}
    extended = dict(scores, terse=judge_means(ref, terse))
    boards = {w: leaderboard(ref, extended, w) for w in WEIGHTS}
    first, second = order_dependence(ref)
    flips = [w for w, board in boards.items() if board != boards[0.0]]
    return {
        "shipped": scores,
        "terse": extended["terse"],
        "shipped_boards": {w: leaderboard(ref, scores, w) for w in (0.0, 0.5, 1.0)},
        "boards": (boards[0.0], boards[1.0]),
        "flip": min(flips) if flips else None,
        "orders": ([n for n, _ in first], [n for n, _ in second]),
        "ratings": (dict(first)["a"], dict(second)["a"]),
    }


def verify(result):
    good, bad, terse_scores = result["shipped"]["good"], result["shipped"]["bad"], result["terse"]
    low, high = result["boards"]
    first_order, second_order = result["orders"]
    left, right = result["ratings"]
    return [
        practice.Check(
            "FINDING: on the lesson's own two models no weighting changes anything",
            len({tuple(b) for b in result["shipped_boards"].values()}) == 1,
            f"demo_model_good scores {good['exact']:.3f}, {good['f1']:.3f} and "
            f"{good['judge']:.3f} on the three judges and demo_model_bad scores "
            f"{bad['exact']:.3f}, {bad['f1']:.3f} and {bad['judge']:.3f}. Good dominates on "
            "every metric, so every weighting gives the same order and the comparison the "
            "exercise asks for has no degrees of freedom on the data it ships",
        ),
        practice.Check(
            f"ANSWER: with a third model the leaderboard flips at weight {result['flip']}",
            low != high and result["flip"] is not None,
            f"a terse model that answers one question exactly and says 'no' to the rest scores "
            f"{terse_scores['exact']:.3f} exact and {terse_scores['f1']:.3f} F1, against bad's "
            f"{bad['exact']:.3f} and {bad['f1']:.3f}. Weighting F1 fully gives {low} and "
            f"weighting exact match fully gives {high}, with the crossover at "
            f"{result['flip']}",
        ),
        practice.Check(
            "MECHANISM: the two judges disagree about what a wrong answer is worth",
            bad["exact"] < terse_scores["exact"] and bad["f1"] > terse_scores["f1"],
            "exact_match gives a verbose-but-correct answer 0; token_f1 gives it partial credit "
            "for the words it shares. So a model that is never exactly right but always nearly "
            "right beats a model that is occasionally exactly right under F1 and loses under "
            "exact match. Weighting them is choosing which of those two failures to call worse",
        ),
        practice.Check(
            "FINDING: ELO adds nothing here except path dependence",
            first_order == second_order and left != right,
            f"the judges are deterministic, so every match has the same outcome every time and "
            f"the ELO ranking is the ranking by mean score. The same three matches in two "
            f"orders give the same leaderboard {first_order} and different ratings for the same "
            f"player -- {left:.1f} against {right:.1f}. What ELO contributes here is a number "
            "that depends on scheduling",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
