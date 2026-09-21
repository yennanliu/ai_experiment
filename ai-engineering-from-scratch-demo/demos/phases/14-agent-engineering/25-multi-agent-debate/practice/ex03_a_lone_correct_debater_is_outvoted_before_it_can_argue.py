"""Exercise 3 — a lone correct debater is outvoted before it can argue.

    Swap one "agent" for a different scripted LLM with different opinions.
    Does heterogeneity improve accuracy?

Reading of the exercise: the lesson's own claim is that cross-model
combinations beat single-model debates, so the swap is worth making properly
-- a debater whose opinions come from a different source, right where the
others are wrong and wrong where they are right. Whether that improves
accuracy is then a property of the *update rule*, not of the expert, and the
shipped rule decides it before any argument happens.

**ANSWER: heterogeneity does not improve accuracy here, because the update
rule decides the question before any argument happens.** Over **8** questions where the swapped
expert is right and the other two agree on a wrong answer, the expert adopts
the majority in round **1** on **8** of **8**, and the debate's answer is
wrong **8** of **8**. A homogeneous panel scores the same **0/8**: the swap
changes who is right and nothing else.

**FINDING: `drift` is majority-following with no notion of evidence.** A
debater compares `Counter(peer_answers).most_common(1)[0][0]` against its own
answer and adopts the peer answer whenever it differs -- there is no
confidence, no justification and no tie-break other than list order, so **2**
peers always outrank **1** debater regardless of which is correct.

**FINDING: the swapped expert is memoryless, so it cannot even hold out.**
`drift` recomputes `current = corrections.get(question, bias)` on every call
and never reads its own previous answer, so an expert that was argued out of
a correct answer in round 1 re-proposes it in round 2 from scratch and is
argued out again. Across **3** rounds its answer changes **4** times and
ends where the majority is.

**FINDING: adding a third correct agent makes the debate wrong again.**
Sweeping 1 to 4 experts against 2 incumbents gives, as (debate, plain vote),
`{1: (0, 0), 2: (8, 0), 3: (0, 8), 4: (8, 8)}`. The plain vote is monotone
and flips exactly when the experts outnumber the incumbents. The debate goes
**0, 8, 0, 8** -- correct on a **2-2** tie and wrong on a **3-2** expert
majority -- because every update is decided by `most_common` breaking a tie
on list order. Accuracy that is not monotone in the number of correct agents
is not an aggregation rule; it is a lottery with a seed.

Structure: `expert()` is the swapped debater; `run()` drives the lesson's own
rounds; `no_imitation()` is the alternative update rule.
"""

from __future__ import annotations

from collections import Counter

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "25-multi-agent-debate"
# Eight questions the swapped expert knows and the incumbents get wrong.
EXPERT_QUESTIONS = {
    "half_life_of_c14": ("5730 years", "5000 years"),
    "author_of_dune": ("Herbert", "Asimov"),
    "boiling_point_at_4km": ("86 C", "100 C"),
    "tcp_handshake_steps": ("3", "2"),
    "first_moon_landing": ("1969", "1968"),
    "sorting_lower_bound": ("n log n", "n"),
    "capital_of_australia": ("Canberra", "Sydney"),
    "atomic_number_of_lead": ("82", "80"),
}


def incumbent(ref, name):
    """A debater that is confidently wrong on every expert question."""
    return ref._make_debater(
        name, bias="unknown",
        corrections={q: wrong for q, (_, wrong) in EXPERT_QUESTIONS.items()})


def expert(ref):
    """The swapped agent: different source, right where the others are wrong."""
    return ref._make_debater(
        "delta", bias="unknown",
        corrections={q: right for q, (right, _) in EXPERT_QUESTIONS.items()})


def run(ref, debaters, question, rounds=3):
    prior = {d.name: d.drift(question, []) for d in debaters}
    history = [dict(prior)]
    for _ in range(rounds):
        prior, _ = ref.full_mesh_round(debaters, question, prior)
        history.append(dict(prior))
    return Counter(prior.values()).most_common(1)[0][0], history


def no_imitation(ref, debaters, question):
    """Everyone keeps their own answer; the vote happens once, at the end."""
    answers = [d.drift(question, []) for d in debaters]
    return Counter(answers).most_common(1)[0][0]


def panel(ref, experts):
    """Two incumbents plus `experts` copies of the swapped agent."""
    rows = [incumbent(ref, "alpha"), incumbent(ref, "beta")]
    return rows + [ref._make_debater(
        f"delta{i}", bias="unknown",
        corrections={q: right for q, (right, _) in EXPERT_QUESTIONS.items()})
        for i in range(experts)]


def score(ref, debaters, rule=run):
    hits = 0
    for question, (right, _) in EXPERT_QUESTIONS.items():
        answer = rule(ref, debaters, question)
        hits += (answer[0] if isinstance(answer, tuple) else answer) == right
    return hits


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    mixed = [incumbent(ref, "alpha"), incumbent(ref, "beta"), expert(ref)]
    same = [incumbent(ref, name) for name in ("alpha", "beta", "gamma")]
    flips, changes = 0, 0
    for question, (right, _) in EXPERT_QUESTIONS.items():
        _, history = run(ref, mixed, question)
        flips += history[1]["delta"] != right
        changes += sum(history[i]["delta"] != history[i - 1]["delta"]
                       for i in range(1, len(history)))
    sample = run(ref, mixed, "author_of_dune")[1]
    delta = mixed[-1]
    run(ref, mixed, "author_of_dune")
    sweep = {n: (score(ref, panel(ref, n)),
                 score(ref, panel(ref, n), no_imitation))
             for n in (1, 2, 3, 4)}
    return {
        "sweep": sweep, "stored_after": delta.drift("author_of_dune", []),
        "questions": len(EXPERT_QUESTIONS),
        "mixed": score(ref, mixed), "homogeneous": score(ref, same),
        "round1_flips": flips, "delta_changes": changes,
        "delta_path": [row["delta"] for row in sample],
        "mixed_no_imitation": score(ref, mixed, no_imitation),
        "peers_seen": 2,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: heterogeneity does not improve accuracy here",
            all([result["questions"] == 8, result["mixed"] == 0,
                 result["homogeneous"] == 0, result["round1_flips"] == 8]),
            f"over {result['questions']} questions where the swapped expert is right and "
            f"the incumbents agree on a wrong answer, the expert adopts the majority in "
            f"round 1 on {result['round1_flips']}/{result['questions']}, and the debate "
            f"scores {result['mixed']} against the homogeneous panel's "
            f"{result['homogeneous']}",
        ),
        practice.Check(
            "FINDING: drift is majority-following with no notion of evidence",
            all([result["peers_seen"] == 2, result["round1_flips"] == 8,
                 result["mixed"] == 0]),
            f"drift compares the peer majority against its own answer and adopts it "
            f"whenever they differ -- no confidence, no justification, no tie-break but "
            f"list order. {result['peers_seen']} peers outrank 1 debater "
            f"{result['round1_flips']}/{result['questions']} times regardless of who is "
            "correct",
        ),
        practice.Check(
            "FINDING: the swapped expert is memoryless, so it cannot hold out",
            all([result["delta_path"] == ["Herbert", "Asimov", "Asimov", "Asimov"],
                 result["stored_after"] == "Herbert"]),
            f"delta's answers over the debate are {result['delta_path']}, but asked again "
            f"with no peers it still says {result['stored_after']!r}: drift recomputes "
            "current from corrections every call and never reads its own last answer. The "
            "debate changed what delta said and nothing about what delta holds",
        ),
        practice.Check(
            "FINDING: adding a third correct agent makes the debate wrong again",
            all([result["sweep"] == {1: (0, 0), 2: (8, 0), 3: (0, 8), 4: (8, 8)}]),
            f"sweeping 1 to 4 experts against 2 incumbents gives (debate, plain vote) "
            f"{result['sweep']}. The plain vote is monotone and flips once the experts "
            "outnumber the incumbents; the debate goes 0, 8, 0, 8 -- right on a 2-2 tie "
            "and wrong on a 3-2 expert majority, because every update is a tie-break by "
            "list order",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
