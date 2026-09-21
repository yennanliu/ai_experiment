"""Exercise 4 — the lesson counts the hub's reads and the code pays for them.

    Measure token cost for full mesh vs sparse on your 3 questions. Plot cost
    vs accuracy.

Reading of the exercise: "plot" is a table when the x-axis has two points, so
what matters is which cost is being plotted. `run_debate` returns a critique-op
count, and the lesson's prose gives its own arithmetic for N=5, R=3 -- **60**
ops for full mesh and **12** for a star. One of those two numbers does not
match the code, and finding out which is the measurement.

**ANSWER: 18 ops against 12 on the shipped 3 questions, both 3/3 correct.**
Full mesh costs **6** ops per round and the star **4**, so over 3 rounds the
table is `(18, 3/3)` against `(12, 3/3)`: the star is **33.3%** cheaper at
identical accuracy. That is the lesson's claim reproduced -- and it is
reproduced on questions where nobody disagrees, so it is a statement about
arithmetic, not about debate.

**FINDING: the star's published cost is half what the code charges.** The
lesson computes N=5, R=3 as "spokes read only the hub = 12 critique ops",
counting **4** spoke reads per round and ignoring the hub's **4**.
`sparse_star_round` charges both: **8** per round, **24** over three. The
full-mesh number matches exactly at **60**, so the star's advantage is
**2.5x** in the code and **5.0x** in the prose.

**FINDING: ops are not tokens, and here they understate the star.** Weighting
each read by the length of what is read -- the mesh's reads are all
full-length peer answers, the star's include short hub summaries -- gives a
character ratio of **3.1x** against the op ratio's **2.5x** at N=5. Neither
number is wrong; they answer different questions, and only one of them is
what a provider bills. Which way the correction runs depends entirely on the
relative answer lengths, so the proxy has to be checked per workload.

**FINDING: accuracy is flat, so the plot is a vertical line.** Across **3**
questions and **2** topologies every run returns the correct answer, giving
**1** distinct accuracy value over **6** configurations. A cost-accuracy plot
needs a question the panel can get wrong, and the shipped panel converges on
the truth before round 1.

Structure: `ops_for()` is the closed-form cost of each topology; `table()`
runs the lesson's own debate to pair cost with accuracy.
"""

from __future__ import annotations

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
DOC = {"mesh_n5_r3": 60, "star_n5_r3": 12}
HUB_ANSWER, SPOKE_ANSWER = 24, 40


def build(ref):
    return [ref._make_debater(name, bias=BIASES[name], corrections=CORRECTIONS[name])
            for name in BIASES]


def ops_for(topology, agents, rounds):
    """What the shipped round functions charge, in critique ops."""
    per_round = agents * (agents - 1) if topology == "full_mesh" else 2 * (agents - 1)
    return per_round * rounds


def doc_star_ops(agents, rounds):
    """The lesson's own arithmetic: spokes read the hub, the hub reads nobody."""
    return (agents - 1) * rounds


def characters(topology, agents, rounds):
    """The same traffic weighted by the length of what is read."""
    if topology == "full_mesh":
        return agents * (agents - 1) * SPOKE_ANSWER * rounds
    return ((agents - 1) * SPOKE_ANSWER + (agents - 1) * HUB_ANSWER) * rounds


def table(ref, debaters, rounds=3):
    rows = {}
    for topology in ("full_mesh", "sparse_star"):
        correct, ops = 0, 0
        for question, truth in QUESTIONS.items():
            answer, _, cost = ref.run_debate(debaters, question, rounds=rounds,
                                             topology=topology)
            correct += answer == truth
            ops = cost
        rows[topology] = {"ops": ops, "correct": correct,
                          "per_round": ops // rounds}
    return rows


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = table(ref, build(ref))
    mesh, star = rows["full_mesh"], rows["sparse_star"]
    return {
        "rows": rows, "questions": len(QUESTIONS),
        "saving": round(100 * (mesh["ops"] - star["ops"]) / mesh["ops"], 1),
        "mesh_n5": ops_for("full_mesh", 5, 3), "star_n5": ops_for("star", 5, 3),
        "doc": DOC, "doc_star": doc_star_ops(5, 3),
        "code_ratio": round(ops_for("full_mesh", 5, 3) / ops_for("star", 5, 3), 1),
        "doc_ratio": round(DOC["mesh_n5_r3"] / DOC["star_n5_r3"], 1),
        "char_ratio": round(characters("full_mesh", 5, 3)
                            / characters("star", 5, 3), 1),
        "accuracies": sorted({row["correct"] for row in rows.values()}),
        "configurations": len(rows) * len(QUESTIONS),
    }


def verify(result):
    mesh, star = result["rows"]["full_mesh"], result["rows"]["sparse_star"]
    return [
        practice.Check(
            "ANSWER: 18 ops against 12 on the three questions, both 3/3 correct",
            all([mesh["ops"] == 18, star["ops"] == 12, mesh["per_round"] == 6,
                 star["per_round"] == 4, mesh["correct"] == 3,
                 star["correct"] == 3, result["saving"] == 33.3]),
            f"full mesh costs {mesh['per_round']} ops per round and the star "
            f"{star['per_round']}, so the table is ({mesh['ops']}, "
            f"{mesh['correct']}/3) against ({star['ops']}, {star['correct']}/3) -- "
            f"{result['saving']}% cheaper at identical accuracy, on questions where "
            "nobody disagrees",
        ),
        practice.Check(
            "FINDING: the star's published cost is half what the code charges",
            all([result["mesh_n5"] == 60, result["doc"]["mesh_n5_r3"] == 60,
                 result["star_n5"] == 24, result["doc"]["star_n5_r3"] == 12,
                 result["doc_star"] == 12, result["code_ratio"] == 2.5,
                 result["doc_ratio"] == 5.0]),
            f"at N=5, R=3 the lesson gives {result['doc']} and sparse_star_round charges "
            f"{result['star_n5']}, because it counts the hub's reads as well as the "
            f"spokes'. The mesh number matches exactly, so the star's advantage is "
            f"{result['code_ratio']}x in the code and {result['doc_ratio']}x in the prose",
        ),
        practice.Check(
            "FINDING: ops are not tokens, and the two rank differently",
            all([result["char_ratio"] == 3.1, result["code_ratio"] == 2.5,
                 result["char_ratio"] > result["code_ratio"]]),
            f"weighting each read by the length of what is read, the mesh's reads are all "
            f"full-length peer answers while the star's include short hub ones: the op "
            f"ratio is {result['code_ratio']}x and the character ratio "
            f"{result['char_ratio']}x. Counting messages understates the star here, and "
            "which way the correction goes depends on the answer lengths",
        ),
        practice.Check(
            "FINDING: accuracy is flat, so the plot is a vertical line",
            all([result["accuracies"] == [3], result["configurations"] == 6,
                 result["questions"] == 3]),
            f"across {result['configurations']} configurations every run returns the "
            f"correct answer, giving {len(result['accuracies'])} distinct accuracy value. "
            "A cost-accuracy plot needs a question the panel can get wrong, and this one "
            "converges on the truth before round 1",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
