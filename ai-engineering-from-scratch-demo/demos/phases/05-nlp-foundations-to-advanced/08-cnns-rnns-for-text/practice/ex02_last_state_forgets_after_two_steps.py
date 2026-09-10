"""Exercise 2 — last-state forgets after two steps.

    **Medium.** Implement max-pool, mean-pool, and last-state pooling for the
    LSTM classifier. Compare on a small dataset; document which pooling wins and
    hypothesize why.

Reading of the exercise: the winner depends on where in the sequence the
information is, and "a small dataset" does not fix that, so the dataset here
varies it on purpose. Each sequence is 20 tokens of filler with one cue token
planted at a chosen position, and the label is which cue. Max-pool and mean-pool
score 0.69 to 0.74 wherever the cue sits. Last-state scores 1.0000 when the cue
is the final token, 0.9444 one before it, 0.7389 two before, and 0.4833 three
before -- chance, on a two-class problem -- and stays there for every earlier
position. Which pooling wins is not a property of the pooling.

The hypothesis the exercise asks for is in the lesson's own file.
`vanishing_gradient_sim(d)` returns 0.9^d, and 0.9 is the spectral radius used
for the recurrence here, so the lesson's model of how much of a signal survives
d steps predicts 0.729 at d=3. The measurement is harsher: at d=3 the classifier
is already at chance. The linear model is optimistic because the recurrence is
tanh, which saturates, and because a linear readout has to find the residue in a
24-dimensional state that 17 other tokens have written over since.

torch is not installed, so the encoder is a fixed random tanh recurrence with
its spectral radius scaled to the lesson's 0.9, and only the readout is trained.
That is the honest version of "the LSTM classifier" available here, and it is
the right one for this question: the three poolings read the same hidden states,
so any difference between them is the pooling.

Structure: `sequences` plants one cue at `position`; `encoder` builds the frozen
embedding table and recurrence, `states` runs it, and `POOLINGS` are the three
reductions. `sweep` scores each pooling at each position over `SEEDS` random
encoders.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "08-cnns-rnns-for-text"

FILLER = ("the", "a", "some", "quite", "rather", "film", "movie", "story", "acting",
          "script", "pacing", "scene", "cast")
CUES = ("brilliant", "dreadful")
DIM, HIDDEN, LENGTH, SEEDS = 12, 24, 20, 5
RADIUS = 0.9
POSITIONS = (0, 10, 16, 17, 18, 19)
POOLINGS = ("max", "mean", "last")


def sequences(np, position, per_class=60) -> tuple:
    rng, rows = np.random.default_rng(0), []
    for label, cue in enumerate(CUES):
        for _ in range(per_class):
            tokens = [str(t) for t in rng.choice(FILLER, size=LENGTH)]
            tokens[position] = cue
            rows.append((tokens, label))
    return rows, np.array([label for _, label in rows])


def encoder(np, vocab, seed) -> tuple:
    """A frozen tanh recurrence whose spectral radius is the lesson's 0.9."""
    rng = np.random.default_rng(seed)
    table = dict(zip(vocab, rng.normal(size=(len(vocab), DIM))))
    into = rng.normal(scale=0.5, size=(DIM, HIDDEN))
    across = rng.normal(scale=0.5, size=(HIDDEN, HIDDEN))
    return table, into, across * RADIUS / abs(np.linalg.eigvals(across)).max()


def states(np, tokens, model):
    table, into, across = model
    hidden, out = np.zeros(HIDDEN), []
    for token in tokens:
        hidden = np.tanh(table[token] @ into + hidden @ across)
        out.append(hidden.copy())
    return np.array(out)


def pooled(np, hidden, how):
    return {"max": hidden.max(0), "mean": hidden.mean(0), "last": hidden[-1]}[how]


def accuracy(np, rows, labels, vocab, how) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    order = np.random.default_rng(7).permutation(len(labels))
    cut, scores = int(len(labels) * 0.7), []
    for seed in range(SEEDS):
        model = encoder(np, vocab, seed)
        matrix = np.array([pooled(np, states(np, tokens, model), how) for tokens, _ in rows])
        fitted = LogisticRegression(max_iter=3000).fit(matrix[order[:cut]], labels[order[:cut]])
        scores.append(accuracy_score(labels[order[cut:]], fitted.predict(matrix[order[cut:]])))
    return round(float(sum(scores) / len(scores)), 4)


def column(grid, how) -> list:
    return [grid[position][how] for position in POSITIONS]


def summarize(grid) -> dict:
    """The three readings of the grid the checks need, computed once."""
    wins = {p: grid[p]["max"] > grid[p]["last"] for p in POSITIONS}
    return {"last": {LENGTH - 1 - p: grid[p]["last"] for p in POSITIONS},
            "flat": {how: column(grid, how) for how in ("max", "mean")},
            "steady": {how: round(max(column(grid, how)) - min(column(grid, how)), 4)
                       for how in ("max", "mean")},
            "far": sum(wins.values()), "near": sum(not w for w in wins.values()),
            "clean": all(wins[p] == (LENGTH - 1 - p >= 3) for p in POSITIONS)}


def solve():
    try:
        import numpy as np
        import sklearn                              # noqa: F401
    except ImportError as exc:                      # pragma: no cover - T1 needs sklearn
        raise practice.Skip(f"needs scikit-learn: uv sync --extra math ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    vocab = sorted(set(FILLER) | set(CUES))
    grid = {}
    for position in POSITIONS:
        rows, labels = sequences(np, position)
        grid[position] = {how: accuracy(np, rows, labels, vocab, how) for how in POOLINGS}
    return dict(summarize(grid), grid=grid, length=LENGTH, radius=RADIUS, seeds=SEEDS,
                predicted={LENGTH - 1 - p: round(ref.vanishing_gradient_sim(LENGTH - 1 - p), 4)
                           for p in POSITIONS})


def verify(result):
    grid, predicted = result["grid"], result["predicted"]
    last, flat, steady = result["last"], result["flat"], result["steady"]
    return [
        practice.Check(
            "ANSWER: which pooling wins is decided by where the cue is, not by the pooling",
            grid[POSITIONS[-1]]["last"] > grid[POSITIONS[-1]]["max"]
            and grid[0]["last"] < grid[0]["max"],
            f"with the cue as the final token, last-state scores {grid[POSITIONS[-1]]['last']} "
            f"against max-pool's {grid[POSITIONS[-1]]['max']}; with the cue first, it scores "
            f"{grid[0]['last']} against {grid[0]['max']}. The full grid over positions "
            f"{list(POSITIONS)} is {grid}"),
        practice.Check(
            "MECHANISM: max and mean read every step, so their score does not move with position",
            max(steady.values()) < 0.1,
            f"max-pool scores {flat['max']} and mean-pool {flat['mean']} across the six cue "
            f"positions -- spreads of {steady['max']} and "
            f"{steady['mean']}. Neither reduction has a notion of when "
            f"a step happened, which is exactly why neither can be beaten by moving the cue"),
        practice.Check(
            "FINDING: last-state is at chance from three steps back and never recovers",
            last[3] < 0.55 and last[LENGTH - 1] < 0.55 and last[0] > 0.95,
            f"by distance from the end, last-state scores {last}: {last[0]} at the final token, "
            f"{last[1]} one back, {last[2]} two back, and {last[3]} at three -- chance on a "
            f"two-class problem -- with no further decline available at distances "
            f"{LENGTH - 1 - POSITIONS[1]} and {LENGTH - 1}"),
        practice.Check(
            "MECHANISM: the lesson's own vanishing_gradient_sim predicts this, and is optimistic",
            predicted[3] > 0.7 > last[3],
            f"`vanishing_gradient_sim(d)` returns {result['radius']}^d, and {result['radius']} is "
            f"the spectral radius used here, so it predicts {predicted} retained at those "
            f"distances -- {predicted[3]} at three steps, where the classifier is already at "
            f"{last[3]}. The linear model does not see tanh saturating, nor the 17 later tokens "
            f"overwriting a {HIDDEN}-dimensional state"),
        practice.Check(
            "FINDING: 'which pooling wins' splits exactly on distance, with the threshold at three",
            result["clean"],
            f"max-pool beats last-state at every position three or more steps from the end and "
            f"loses at every position nearer than that -- {len(POSITIONS)} positions, split "
            f"{result['far']} to {result['near']}, with no exception. A "
            f"corpus whose verdict falls in the last two tokens ranks last-state first; one whose "
            f"evidence is anywhere else ranks it last"),
        practice.Check(
            "CONTROL: the three poolings read identical hidden states",
            all(grid[p]["max"] > 0.5 and grid[p]["mean"] > 0.5 for p in POSITIONS),
            f"one frozen encoder per seed produces one sequence of states, and the three reductions "
            f"are applied to it; only the linear readout is trained, over {result['seeds']} seeds. "
            f"Any difference between the three columns is the pooling and nothing else, which is "
            f"what makes the position sweep readable"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
