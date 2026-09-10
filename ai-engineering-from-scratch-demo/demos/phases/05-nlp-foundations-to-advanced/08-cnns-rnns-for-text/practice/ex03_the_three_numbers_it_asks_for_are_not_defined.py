"""Exercise 3 — the three numbers it asks for are not defined.

    **Hard.** Build a BiLSTM-CRF NER tagger (combine lesson 06 and this one).
    Train on CoNLL-2003. Compare to the CRF-alone baseline from lesson 06 and to
    a BERT fine-tune. Report training time, memory, and F1.

Reading of the exercise: torch, tensorflow and jax are all absent and CoNLL-2003
is not downloadable, so none of the three systems can be trained here and none
of the three numbers is reported as though it had been. What is available is
that two of the three quantities are determined by hyperparameters the exercise
does not state, and the arithmetic settles them without a GPU.

Memory first. A BiLSTM stores 2 n H activations, linear in sequence length; a
transformer stack stores L H_heads n^2 attention scores, quadratic. At the sizes
below the crossover is at n = 8 tokens -- shorter than a CoNLL sentence -- and by
n = 512 the attention side is 72 times the recurrent side. Meanwhile the CRF's
lattice is n |tags|^2, which at 9 tags is 81 numbers per token and never
competes for the answer. Reporting one memory figure for a "comparison" is
reporting a choice of n.

Parameters go the other way, and are dominated by a third thing entirely. The
BiLSTM-CRF's recurrent weights are 2,099,200 and its transition matrix is 81,
against an embedding table of 23,040,000 -- so 92% of the model the exercise
calls a BiLSTM-CRF is a vocabulary, and shrinking the vocabulary changes the
headline more than replacing the encoder does. The transformer stack is 42
million on top of the same embedding table, for 2.61 times the total.

F1 is the third number, and lesson 06 already measured what moves it most on
this task: splitting the same corpus by sentence rather than by entity was worth
0.9048 F1, with the model, the features and the data held fixed. No architecture
change in this exercise is in that range.

Structure: `activations` and `parameters` are closed-form counts at the sizes in
`CONFIG`; `crossover` finds the smallest n at which one exceeds another. Nothing
here is timed, because a wall-clock number from this machine would not be the
one the exercise is asking for.
"""

from __future__ import annotations

import importlib.util

from harness import practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "08-cnns-rnns-for-text"

CONFIG = {"hidden": 256, "model_dim": 768, "heads": 12, "layers": 6,
          "vocab": 30000, "tags": 9}
LENGTHS = (8, 32, 128, 512)
FRAMEWORKS = ("torch", "tensorflow", "jax", "transformers")
LESSON_06_LEAK = 0.9048


def activations(length: int) -> dict:
    """Stored per sequence: recurrent states, attention scores, CRF lattice."""
    return {"bilstm": 2 * length * CONFIG["hidden"],
            "attention": CONFIG["layers"] * CONFIG["heads"] * length * length,
            "crf": length * CONFIG["tags"] ** 2}


def parameters() -> dict:
    dim, hidden = CONFIG["model_dim"], CONFIG["hidden"]
    return {"embedding": CONFIG["vocab"] * dim,
            "bilstm": 4 * 2 * (hidden * hidden + dim * hidden + hidden),
            "transformer": CONFIG["layers"] * (4 * dim * dim + 2 * dim * 4 * dim),
            "transitions": CONFIG["tags"] ** 2}


def crossover(left: str, right: str, limit: int = 4096) -> int | None:
    """The shortest sequence at which `left` stores more than `right`."""
    return next((n for n in range(1, limit) if activations(n)[left] > activations(n)[right]), None)


def solve():
    counts = parameters()
    grid = {n: activations(n) for n in LENGTHS}
    bilstm_crf = counts["embedding"] + counts["bilstm"] + counts["transitions"]
    transformer = counts["embedding"] + counts["transformer"]
    return {
        "missing": [m for m in FRAMEWORKS if importlib.util.find_spec(m) is None],
        "grid": grid, "parameters": counts, "config": CONFIG,
        "crossover": {"attention over bilstm": crossover("attention", "bilstm"),
                      "attention over crf": crossover("attention", "crf"),
                      "bilstm over crf": crossover("bilstm", "crf")},
        "ratio": {n: round(grid[n]["attention"] / grid[n]["bilstm"], 2) for n in LENGTHS},
        "totals": {"bilstm_crf": bilstm_crf, "transformer": transformer,
                   "ratio": round(transformer / bilstm_crf, 2)},
        "embedding_share": round(counts["embedding"] / bilstm_crf, 4),
        "recurrent_share": round(counts["bilstm"] / bilstm_crf, 4),
        "leak": LESSON_06_LEAK,
    }


def verify(result):
    grid, counts, totals = result["grid"], result["parameters"], result["totals"]
    cross, ratio = result["crossover"], result["ratio"]
    longest = LENGTHS[-1]
    return [
        practice.Check(
            "ANSWER: none of the three systems is trainable here, and two of the three numbers are arithmetic",
            result["missing"] == list(FRAMEWORKS),
            f"{result['missing']} are all absent and CoNLL-2003 is not downloadable, so no training "
            f"time and no measured F1 are reported. Memory and parameter count do not need a GPU: "
            f"at {CONFIG['hidden']}-unit recurrence and a {CONFIG['layers']}-layer, "
            f"{CONFIG['heads']}-head stack, the stored activations per sequence are {grid}"),
        practice.Check(
            "MECHANISM: the memory comparison crosses over at 8 tokens, shorter than a sentence",
            cross["attention over bilstm"] <= 8,
            f"a BiLSTM stores 2nH activations and a transformer stores L*heads*n^2 attention "
            f"scores, so the ranking inverts at n = {cross['attention over bilstm']}. Every "
            f"CoNLL-2003 sentence is longer than that, and by n = {longest} the attention side is "
            f"{ratio[longest]}x the recurrent side. 'Report memory' is answered by choosing n"),
        practice.Check(
            "FINDING: the CRF lattice never competes for the answer",
            grid[longest]["crf"] < grid[longest]["bilstm"] < grid[longest]["attention"],
            f"at {CONFIG['tags']} tags the lattice is n*{CONFIG['tags'] ** 2} numbers -- "
            f"{grid[longest]['crf']:,} at n={longest}, against {grid[longest]['bilstm']:,} for the "
            f"recurrence and {grid[longest]['attention']:,} for attention. The CRF is the part the "
            f"exercise treats as the baseline and the part that costs nothing"),
        practice.Check(
            "FINDING: 92% of the 'BiLSTM-CRF' is a vocabulary, not a BiLSTM",
            result["embedding_share"] > 0.9 and result["recurrent_share"] < 0.1,
            f"the recurrent weights are {counts['bilstm']:,} and the transition matrix is "
            f"{counts['transitions']}, against an embedding table of {counts['embedding']:,} -- "
            f"{result['embedding_share']:.1%} of the {totals['bilstm_crf']:,} total. Halving the "
            f"vocabulary moves the reported memory more than deleting the encoder does"),
        practice.Check(
            "MECHANISM: the two parameter totals differ by less than the names suggest",
            1.5 < totals["ratio"] < 4.0,
            f"the BiLSTM-CRF totals {totals['bilstm_crf']:,} parameters and a "
            f"{CONFIG['layers']}-layer transformer over the same embedding table totals "
            f"{totals['transformer']:,} -- {totals['ratio']}x, not an order of magnitude, because "
            f"both carry the same {counts['embedding']:,}-parameter table underneath"),
        practice.Check(
            "CONTROL: no architecture in this exercise moves F1 as much as lesson 06's split did",
            result["leak"] > 0.9,
            f"lesson 06 measured the same model on the same corpus scoring 1.0000 entity F1 under a "
            f"by-sentence split and 0.0952 under a by-entity one -- {result['leak']} of F1 from an "
            f"evaluation choice, with model, features and data held fixed. 'Report F1' for three "
            f"architectures is a smaller question than the one the split answers"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
