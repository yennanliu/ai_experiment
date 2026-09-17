"""Exercise 2 — the leaderboard flips at weight 0.55, and never on the lesson's own two models.

    Extend the ELO tracker to support multiple judge functions (exact match, F1,
    LLM-as-judge) and weight them. Compare how the leaderboard changes when you
    weight exact match heavily versus F1 heavily.

Reading of the exercise: the three judges are the lesson's own `exact_match`,
`token_f1` and `llm_judge_simulated`, and the weight is a distribution over all
three -- validated as one -- whose weighted score decides each pairwise match fed
to `ELOTracker.record_match`. The comparison is run on the lesson's own two demo
models first, and then on a third that the judges disagree about, because two
models they agree on cannot show a leaderboard changing.

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

**FINDING: give the judge a third of the weight and the flip becomes
unreachable.** `llm_judge_simulated` agrees with F1 about these two models --
bad **0.630** against terse **0.240** -- so weighting it in pushes the crossover
up: 0.55 at judge weight 0, 0.60 at 0.1, and at an equal third there is no
crossover at all, because only 0.67 of the weight is left for exact match. Which
of the three judges carries the weight decides whether the exercise's comparison
has an answer.

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

Structure: `parts` builds and validates a distribution over the three judges,
`weighted` combines them, `leaderboard` runs the round robin through the
lesson's own `ELOTracker`, and `flip_point` finds where the board changes.
"""

from __future__ import annotations

import statistics

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "10-evaluation"
ROUNDS = 30
GRID = tuple(i / 20 for i in range(21))
JUDGE_SHARES = (0.0, 0.1, 1 / 3)
QA = (("What is the capital of France?", "Paris"), ("What is 2 + 2?", "4"),
      ("Who wrote Hamlet?", "William Shakespeare"),
      ("What language is PyTorch written in?", "Python and C++"),
      ("What is the boiling point of water?", "100 degrees Celsius"),
      ("Name the largest planet.", "Jupiter"), ("What year did World War 2 end?", "1945"))


def terse(prompt):
    """Exactly right once, tersely wrong otherwise -- the judges disagree about it."""
    return "Jupiter" if prompt == "Name the largest planet." else "no"


def judge_means(ref, model_fn):
    """Mean score of one model under each of the lesson's three judges."""
    judges = {"exact": ref.exact_match, "f1": ref.token_f1, "judge": ref.llm_judge_simulated}
    rows = ref.EvalSuite("demo", [ref.EvalCase(p, e) for p, e in QA], judges).run(model_fn)
    return {name: statistics.fmean(row["scores"][name] for row in rows) for name in judges}


def parts(exact_share, judge_share):
    """Three weights summing to 1 -- whatever the other two leave goes to F1."""
    shares = {"exact": exact_share, "judge": judge_share,
              "f1": 1 - exact_share - judge_share}
    if min(shares.values()) < -1e-9:
        raise ValueError(f"weights must be a distribution over the judges: {shares}")
    return shares


def weighted(scores, shares):
    return sum(share * scores[name] for name, share in shares.items())


def leaderboard(ref, scores, shares):
    """A round robin scored at `shares`, run through the lesson's own ELOTracker."""
    tracker, names = ref.ELOTracker(), list(scores)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    for _ in range(ROUNDS):
        for a, b in pairs:
            left, right = weighted(scores[a], shares), weighted(scores[b], shares)
            tracker.record_match(a, b, "a" if left > right
                                 else ("b" if right > left else "tie"))
    return [name for name, _ in tracker.leaderboard()]


def flip_point(ref, scores, judge_share):
    """The smallest exact-match weight at which the board changes, and the board either side."""
    base = leaderboard(ref, scores, parts(0.0, judge_share))
    for share in (s for s in GRID if s + judge_share <= 1 + 1e-9):
        board = leaderboard(ref, scores, parts(share, judge_share))
        if board != base:
            return share, base, board
    return None, base, base


def order_dependence(ref):
    """The same three matches in two orders: ratings differ, order does not."""
    boards = []
    for matches in ([("a", "b"), ("b", "c"), ("a", "c")],
                    [("a", "c"), ("b", "c"), ("a", "b")]):
        tracker = ref.ELOTracker()
        for left, right in matches:
            tracker.record_match(left, right, "a")
        boards.append(tracker.leaderboard())
    return boards


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    shipped = {"good": ref.demo_model_good, "bad": ref.demo_model_bad}
    scores = {name: judge_means(ref, fn) for name, fn in shipped.items()}
    extended = dict(scores, terse=judge_means(ref, terse))
    first, second = order_dependence(ref)
    return {"shipped": scores, "terse": extended["terse"],
            "shipped_boards": [leaderboard(ref, scores, parts(e, j))
                               for e, j in ((0.0, 0.0), (0.5, 0.0), (0.0, 1 / 3), (1.0, 0.0))],
            "sweeps": {s: flip_point(ref, extended, s) for s in JUDGE_SHARES},
            "orders": ([n for n, _ in first], [n for n, _ in second]),
            "ratings": (dict(first)["a"], dict(second)["a"])}


def verify(result):
    good, bad, terse_scores = result["shipped"]["good"], result["shipped"]["bad"], result["terse"]
    first_order, second_order = result["orders"]
    left, right = result["ratings"]
    flat, tilted, thirds = (result["sweeps"][share] for share in JUDGE_SHARES)
    return [
        practice.Check(
            "FINDING: on the lesson's own two models no weighting changes anything",
            len({tuple(b) for b in result["shipped_boards"]}) == 1,
            f"good scores {good['exact']:.3f}/{good['f1']:.3f}/{good['judge']:.3f} on "
            f"exact/F1/judge and bad {bad['exact']:.3f}/{bad['f1']:.3f}/{bad['judge']:.3f}. It dominates on every metric, so every distribution over the "
            "three gives the same order and the comparison has no degrees of freedom on the "
            "data the lesson ships",
        ),
        practice.Check(
            f"ANSWER: with a third model the board flips at exact-match weight {flat[0]}",
            flat[0] is not None and flat[1] != flat[2],
            f"a terse model answering one question exactly and 'no' to the rest scores "
            f"{terse_scores['exact']:.3f}/{terse_scores['f1']:.3f}/{terse_scores['judge']:.3f} "
            f"against bad's {bad['exact']:.3f}/{bad['f1']:.3f}/{bad['judge']:.3f}. With the judge "
            f"weighted 0, F1 alone gives {flat[1]} and the board becomes {flat[2]} at {flat[0]}",
        ),
        practice.Check(
            "FINDING: give the judge a third of the weight and the flip becomes unreachable",
            tilted[0] > flat[0] and thirds[0] is None,
            f"the judge agrees with F1 here -- bad {bad['judge']:.3f} against terse "
            f"{terse_scores['judge']:.3f} -- so weighting it in pushes the crossover up: "
            f"{flat[0]} at judge weight 0, {tilted[0]} at {JUDGE_SHARES[1]}, and none at all at "
            f"an equal third, since only {1 - JUDGE_SHARES[2]:.2f} is left for exact match and "
            f"the board stays {thirds[1]}. Which judge carries the weight decides whether the "
            "comparison has an answer",
        ),
        practice.Check(
            "MECHANISM: exact match is alone against the other two judges",
            bad["exact"] < terse_scores["exact"] and bad["f1"] > terse_scores["f1"]
            and bad["judge"] > terse_scores["judge"],
            "exact_match gives a verbose-but-correct answer 0; token_f1 gives partial credit for "
            "shared words and llm_judge_simulated rewards length. A model never exactly right but "
            "always nearly right beats one occasionally exactly right under two judges and loses "
            "under the third -- weighting them is choosing which failure to call worse",
        ),
        practice.Check(
            "FINDING: ELO adds nothing here except path dependence",
            first_order == second_order and left != right,
            f"the judges are deterministic, so every match has the same outcome every time and "
            f"the ELO ranking is the ranking by mean score. The same three matches in two orders "
            f"give the leaderboard {first_order} either way and ratings of {left:.1f} against "
            f"{right:.1f} for one player -- what ELO adds is a scheduling-dependent number",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
