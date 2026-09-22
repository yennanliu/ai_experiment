"""Exercise 1 — the shipped debaters agree before the first round.

    Implement a "forced disagreement" rule: in round 1, every debater must
    produce a distinct proposal. Measure effect on convergence speed.

Reading of the exercise: measuring the effect needs a baseline that debates,
and the shipped one does not. `_make_debater` reads `corrections` before
`bias`, and beta and gamma carry a correction for every question, so all
three produce the same answer at round 0 on all 3 questions. Forced
disagreement is therefore not a tweak to this demo -- it is the thing that
makes it a demo of debate.

**ANSWER: convergence goes from round 1 on everything to round 3 on one
question and never on the other two.** With the shipped debaters
`converged_round` is **1** for **3** of **3** questions under both
topologies, because nothing ever disagrees. Forcing distinct round-1
proposals gives `capital_of_portugal` at round **3** and **-1** for the
other two -- with three debaters holding three distinct answers, every
debater's two peers disagree, `Counter.most_common` breaks the tie by
insertion order, and the answers rotate instead of settling. The majority
answer is still correct on **3** of **3**, so forced disagreement cost
convergence and bought nothing here.

**FINDING: the baseline never changes an answer.** Across **3** questions and
**2** topologies, the number of debater answers that differ from their round-0
proposal is **0**. The demo's **18** full-mesh critique ops and **12** star
ops buy exactly the same output as **0** ops would.

**FINDING: the update rule refuses answers that match a debater's own bias.**
`drift` adopts the peer majority only `if common != current and common !=
bias`. beta holds `"Lisbon"` by correction and `"Madrid"` by bias: peers
saying `"Madrid"` unanimously leave it on `"Lisbon"`, while peers saying
`"Porto"` move it to `"Porto"`. It accepts any consensus except its own bias
-- and that accident is the only reason `capital_of_portugal` converges
above, since the clause is what stops the rotation.

**FINDING: convergence does not stop the spending.** `run_debate` loops
`rounds` times whatever happens, so of the **18** full-mesh ops **12** are
spent after consensus, and under forced disagreement **6** of **18** are.
Recording `converged_round` while continuing to pay is the shape of a
benchmark that reports the cost of its slowest configuration.

Structure: `forced()` wraps the lesson's debaters to make round 1 distinct;
`trace_rounds()` records every answer so changes can be counted.
"""

from __future__ import annotations

from collections import Counter

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "25-multi-agent-debate"
QUESTIONS = {"capital_of_portugal": "Lisbon", "is_2_plus_2_equal_4": "yes",
             "chess_legal_e4": "legal"}
BIASES = {"alpha": "Lisbon", "beta": "Madrid", "gamma": "Porto"}
CORRECTIONS = {
    "alpha": {"is_2_plus_2_equal_4": "yes", "chess_legal_e4": "legal"},
    "beta": {"capital_of_portugal": "Lisbon", "is_2_plus_2_equal_4": "yes",
             "chess_legal_e4": "legal"},
    "gamma": {"capital_of_portugal": "Lisbon", "is_2_plus_2_equal_4": "yes",
              "chess_legal_e4": "legal"},
}
DISTINCT = {"capital_of_portugal": ("Lisbon", "Madrid", "Porto"),
            "is_2_plus_2_equal_4": ("yes", "no", "maybe"),
            "chess_legal_e4": ("legal", "illegal", "unclear")}


def build(ref):
    return [ref._make_debater(n, bias=BIASES[n], corrections=CORRECTIONS[n])
            for n in BIASES]


def forced(ref, debaters, question):
    """Round 1 proposals are pinned distinct; later rounds use the shipped rule."""
    pinned = dict(zip([d.name for d in debaters], DISTINCT[question]))

    def pin(debater):
        def drift(q, peers, inner=debater.drift, name=debater.name):
            return pinned[name] if not peers else inner(q, peers)
        return ref.Debater(name=debater.name, drift=drift)
    return [pin(debater) for debater in debaters]


def trace_rounds(ref, debaters, question, rounds=3, topology="full_mesh"):
    """Every round's answers, so changes and consensus can both be counted."""
    prior = {d.name: d.drift(question, []) for d in debaters}
    history, ops, converged = [dict(prior)], 0, -1
    for index in range(rounds):
        if topology == "full_mesh":
            new, cost = ref.full_mesh_round(debaters, question, prior)
        else:
            new, cost = ref.sparse_star_round(debaters[0], debaters[1:],
                                              question, prior)
        ops += cost
        if len(set(new.values())) == 1 and converged == -1:
            converged = index + 1
        history.append(dict(new))
        prior = new
    return {"history": history, "ops": ops, "converged": converged,
            "answer": Counter(prior.values()).most_common(1)[0][0],
            "per_round": ops // rounds}


def changes(history):
    return sum(history[i][name] != history[i - 1][name]
               for i in range(1, len(history)) for name in history[0])


def bias_refusal(ref, debaters):
    """beta holds 'Lisbon' by correction and 'Madrid' by bias. Offer it each."""
    beta, q = next(d for d in debaters if d.name == "beta"), "capital_of_portugal"
    return {"offered_bias": beta.drift(q, ["Madrid", "Madrid"]),
            "offered_other": beta.drift(q, ["Porto", "Porto"]),
            "bias": BIASES["beta"], "current": CORRECTIONS["beta"][q]}


def summarise(runs, label):
    return {f"{label}_converged": sorted({r["converged"] for r in runs.values()}),
            f"{label}_changes": sum(changes(r["history"]) for r in runs.values()),
            f"{label}_correct": sum(runs[q]["answer"] == truth
                                    for q, truth in QUESTIONS.items())}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    base = build(ref)
    plain = {q: trace_rounds(ref, base, q) for q in QUESTIONS}
    star = {q: trace_rounds(ref, base, q, topology="star") for q in QUESTIONS}
    pushed = {q: trace_rounds(ref, forced(ref, base, q), q) for q in QUESTIONS}
    first = plain["capital_of_portugal"]
    return {
        "questions": len(QUESTIONS), "refusal": bias_refusal(ref, base),
        "plain_ops": first["ops"], "star_ops": star["capital_of_portugal"]["ops"],
        "forced_per_question": {q: pushed[q]["converged"] for q in QUESTIONS},
        "wasted_plain": first["ops"] - first["per_round"],
        "wasted_forced": pushed["capital_of_portugal"]["ops"]
        - 2 * pushed["capital_of_portugal"]["per_round"],
        **summarise(plain, "plain"), **summarise(star, "star"),
        **summarise(pushed, "forced"),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: round 1 on everything becomes round 3 on one and never on two",
            all([result["plain_converged"] == [1], result["star_converged"] == [1],
                 result["forced_converged"] == [-1, 3], result["forced_correct"] == 3,
                 result["plain_correct"] == 3, result["questions"] == 3,
                 result["forced_per_question"]["capital_of_portugal"] == 3]),
            f"the shipped debaters converge at round {result['plain_converged']} on all "
            f"{result['questions']} questions under both topologies; forced disagreement "
            f"gives {result['forced_per_question']} -- the answers rotate on a three-way "
            f"tie -- with the majority still correct {result['forced_correct']}/3",
        ),
        practice.Check(
            "FINDING: the baseline never changes an answer",
            all([result["plain_changes"] == 0, result["plain_ops"] == 18,
                 result["star_ops"] == 12, result["forced_changes"] > 0]),
            f"across {result['questions']} questions and two topologies, "
            f"{result['plain_changes']} debater answers differ from their round-0 "
            f"proposal. The {result['plain_ops']} full-mesh and {result['star_ops']} star "
            f"ops buy the same output as zero would; forced disagreement produces "
            f"{result['forced_changes']} changes",
        ),
        practice.Check(
            "FINDING: the update rule refuses answers matching a debater's own bias",
            all([result["refusal"]["offered_bias"] == "Lisbon",
                 result["refusal"]["offered_other"] == "Porto",
                 result["refusal"]["bias"] == "Madrid"]),
            f"beta holds {result['refusal']['current']!r} by correction and "
            f"{result['refusal']['bias']!r} by bias. Peers unanimously saying "
            f"{result['refusal']['bias']!r} leave it at "
            f"{result['refusal']['offered_bias']!r}; peers saying 'Porto' move it to "
            f"{result['refusal']['offered_other']!r}. It accepts any consensus but its own "
            "bias",
        ),
        practice.Check(
            "FINDING: convergence does not stop the spending",
            all([result["wasted_plain"] == 12, result["wasted_forced"] == 6,
                 result["plain_ops"] == 18]),
            f"run_debate loops its full round count whatever happens, so "
            f"{result['wasted_plain']} of {result['plain_ops']} ops are spent after "
            f"consensus in the baseline and {result['wasted_forced']} under forced "
            "disagreement. converged_round is recorded and never acted on",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
