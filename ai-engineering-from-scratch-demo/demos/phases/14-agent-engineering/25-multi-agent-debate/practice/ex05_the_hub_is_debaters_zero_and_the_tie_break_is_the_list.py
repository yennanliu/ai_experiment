"""Exercise 5 — the hub is debaters[0] and the tie-break is the list.

    Read the Society of Minds paper. Port your toy to N=5, R=3. What breaks?
    What gets better?

Reading of the exercise: the paper's own experiments used N=3, R=2 for cost,
and the lesson says accuracy improves with more agents and more rounds. The
port to N=5, R=3 is four lines. What it exposes is that two of the shipped
design decisions are only invisible at N=3 -- an even number of peers, and a
hub chosen by list index -- and both of them decide answers.

**ANSWER: N=5, R=3 costs 5.0x the paper's N=3, R=2 and returns the same 8 of
8.** Widening the panel is a parameter change with no code change: **60**
mesh ops and **24** star ops against **12** at N=3, R=2. With a correct
majority in both panels the answers are identical, **8/8** either way, so on
this toy more agents buy nothing. That is not a refutation of the paper --
its gains come from models that genuinely propose differently on hard
problems, and these debaters produce one distinct proposal between them.

**FINDING: every debater now reads an even number of peers, and ties are
decided by list order.** At N=5 each debater sees **4** peers, so a **2-2**
split among them is resolved by `Counter.most_common`, which returns the
answer belonging to the earliest debater in the list. Reversing the panel
changes the final answer on **8** of **8** questions with nobody changing
their mind -- at N=3 the same reversal changes **0**, because 2 peers can
only tie when they agree. Going from N=3 to N=5 turns list order from an
irrelevance into the decider.

**FINDING: the hub cannot be rotated.** `run_debate` computes
`hub = debaters[0]` and `spokes = debaters[1:]`, so the star's hub is
whichever debater was listed first. Rotating the hub through all **5**
positions produces **2** distinct answers over the same panel and question,
and the lesson's own mitigation -- "rotate or use multiple hubs" -- has no
parameter to express it.

**FINDING: a bad hub costs more at N=5 than at N=3.** With a wrong hub, every
spoke reads only that hub, so **4** of **5** debaters are corrupted in one
round against **2** of **3**: the star answers wrong on **8** of **8**
questions at N=5 and **8** of **8** at N=3, but the mesh survives at N=3 and
not at N=5. Sparsity concentrates the blast radius exactly where the lesson
recommends sparsity for cost.

Structure: `panel()` builds N debaters; `sweep_hub()` rotates the hub the
shipped signature cannot.
"""

from __future__ import annotations

from collections import Counter

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "25-multi-agent-debate"
# Truth is a 3-2 minority at N=5 and a 2-1 minority at N=3.
QUESTIONS = {
    "half_life_of_c14": ("5730 years", "5000 years"),
    "author_of_dune": ("Herbert", "Asimov"),
    "boiling_point_at_4km": ("86 C", "100 C"),
    "tcp_handshake_steps": ("3", "2"),
    "first_moon_landing": ("1969", "1968"),
    "sorting_lower_bound": ("n log n", "n"),
    "capital_of_australia": ("Canberra", "Sydney"),
    "atomic_number_of_lead": ("82", "80"),
}


def debater(ref, name, knows_truth):
    index = 0 if knows_truth else 1
    return ref._make_debater(
        name, bias="unknown",
        corrections={q: pair[index] for q, pair in QUESTIONS.items()})


def panel(ref, agents, correct):
    """`correct` debaters that know the answer, the rest confidently wrong."""
    return [debater(ref, f"d{i}", i < correct) for i in range(agents)]


def ops(topology, agents, rounds):
    per_round = agents * (agents - 1) if topology == "full_mesh" else 2 * (agents - 1)
    return per_round * rounds


def answer(ref, debaters, question, rounds, topology="full_mesh"):
    return ref.run_debate(debaters, question, rounds=rounds, topology=topology)[0]


def score(ref, debaters, rounds, topology="full_mesh"):
    return sum(answer(ref, debaters, q, rounds, topology) == right
               for q, (right, _) in QUESTIONS.items())


def sweep_hub(ref, debaters, question, rounds=3):
    """Rotate the hub by rotating the list, since run_debate takes no hub."""
    seen = []
    for start in range(len(debaters)):
        rotated = debaters[start:] + debaters[:start]
        seen.append(answer(ref, rotated, question, rounds, "sparse_star"))
    return seen


def reversal_flips(ref, agents, correct, rounds):
    debaters = panel(ref, agents, correct)
    return sum(answer(ref, debaters, q, rounds)
               != answer(ref, debaters[::-1], q, rounds) for q in QUESTIONS)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    wide, narrow = panel(ref, 5, 3), panel(ref, 3, 2)
    bad_hub = [debater(ref, "d0", False)] + [debater(ref, f"d{i}", True)
                                             for i in range(1, 5)]
    return {
        "wide_score": score(ref, wide, 3), "narrow_score": score(ref, narrow, 2),
        "questions": len(QUESTIONS),
        "mesh_ops": ops("full_mesh", 5, 3), "star_ops": ops("star", 5, 3),
        "narrow_ops": ops("full_mesh", 3, 2),
        "cost_ratio": round(ops("full_mesh", 5, 3) / ops("full_mesh", 3, 2), 2),
        "peers_wide": 4, "peers_narrow": 2,
        "flips_wide": reversal_flips(ref, 5, 3, 3),
        "flips_narrow": reversal_flips(ref, 3, 2, 2),
        "hub_answers": sorted(set(sweep_hub(ref, wide, "author_of_dune"))),
        "hub_positions": 5,
        "bad_hub_star": score(ref, bad_hub, 3, "sparse_star"),
        "bad_hub_mesh": score(ref, bad_hub, 3),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: N=5 R=3 costs 5.0x the paper's N=3 R=2 and returns the same 8 of 8",
            all([result["wide_score"] == 8, result["narrow_score"] == 8,
                 result["mesh_ops"] == 60, result["star_ops"] == 24,
                 result["narrow_ops"] == 12, result["cost_ratio"] == 5.0,
                 result["questions"] == 8]),
            f"widening to N=5, R=3 is a parameter change: {result['mesh_ops']} mesh ops "
            f"and {result['star_ops']} star ops against the paper's N=3, R=2 at "
            f"{result['narrow_ops']}. Both panels answer "
            f"{result['wide_score']}/{result['questions']}, so the extra "
            f"{result['cost_ratio']}x buys nothing on debaters that all propose the same "
            "thing",
        ),
        practice.Check(
            "FINDING: an even peer count makes ties decidable by list order",
            all([result["peers_wide"] == 4, result["peers_narrow"] == 2,
                 result["flips_wide"] == 8, result["flips_narrow"] == 0]),
            f"at N=5 each debater reads {result['peers_wide']} peers, so a 2-2 split among "
            f"them is resolved by most_common taking the earliest debater. Reversing the "
            f"panel changes {result['flips_wide']}/{result['questions']} answers; at N=3 "
            f"it changes {result['flips_narrow']}, because 2 peers can only tie by "
            "agreeing",
        ),
        practice.Check(
            "FINDING: the hub cannot be rotated",
            all([len(result["hub_answers"]) == 2, result["hub_positions"] == 5]),
            f"run_debate computes hub = debaters[0], so the star's hub is whoever was "
            f"listed first. Rotating through all {result['hub_positions']} positions "
            f"produces {result['hub_answers']} on one question -- the lesson's own "
            "mitigation, rotate or use multiple hubs, has no parameter to express it",
        ),
        practice.Check(
            "FINDING: a bad hub is worse at N=5 than the same panel in full mesh",
            all([result["bad_hub_star"] == 0, result["bad_hub_mesh"] == 8]),
            f"with four correct spokes and one wrong hub, the star scores "
            f"{result['bad_hub_star']}/{result['questions']} because every spoke reads "
            f"only the hub, while the same panel in full mesh scores "
            f"{result['bad_hub_mesh']}. Sparsity concentrates the blast radius exactly "
            "where it is recommended for cost",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
