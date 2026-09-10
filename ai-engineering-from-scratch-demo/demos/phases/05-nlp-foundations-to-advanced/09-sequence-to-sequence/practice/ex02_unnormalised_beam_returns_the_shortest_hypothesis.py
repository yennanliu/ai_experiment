"""Exercise 2 — unnormalised beam returns the shortest hypothesis.

    **Medium.** Add beam search decoding with beam width 3. Measure BLEU on a
    small parallel corpus against greedy. Document where beam search wins
    (usually last tokens) and where it makes no difference.

Reading of the exercise: with the score the exercise implies -- the sum of token
log-probabilities -- beam-3 scores BLEU 0.0253 against greedy's 0.2507. It does
not win somewhere and tie elsewhere; it loses everywhere, and for a reason that
has nothing to do with the search. Every extra token adds a negative number to
the score, so among complete hypotheses the highest-scoring one is the shortest
that the model will end, and beam search finds it exactly because it is a better
search. All five outputs come back one token long against references of 3 to 7.
Greedy never sees that hypothesis because greedy never compares a finished
sequence with an unfinished one.

Divide by length and the exercise's claim comes back, sharply. Length-normalised
beam-3 scores 0.2982, above greedy, and the output differs from greedy's at
exactly one position on every sentence: the last one. The decoder here is built
around a trap -- the token `x` has the higher emission at every step and a worse
continuation than `y` -- so greedy takes `x` throughout and normalised beam takes
`x` until the final step, where the continuation no longer matters. "Usually
last tokens" is right, and it is right about the normalised variant only.

One more thing the phrasing hides: beam width 1 is not greedy. Beam-1 returns a
one-token output at every length where greedy returns n, because beam-1 still
compares completed hypotheses against live ones at each step and greedy does
not. The two differ in the stopping rule, not the width.

Structure: `EMISSION` and `TRANSITION` define the trap, `step` returns one
token's log-probability, and `greedy` and `beam` are the two decoders --
`beam(k, norm=True)` divides the final score by length. `bleu` is a smoothed
4-gram BLEU with the standard brevity penalty, written here because sacrebleu is
not installed.
"""

from __future__ import annotations

import collections
import math

from harness import practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "09-sequence-to-sequence"

TOKENS, END = ("v", "w", "x", "y", "z"), "</s>"
EMISSION = {"y": 0.40, "x": 0.45, "v": 0.05, "w": 0.05, "z": 0.05}
CONTINUE = {"y": 0.05, "x": 0.05, "v": 0.05, "w": 0.05, "z": 0.05, END: 0.75}
DEAD_END = {"y": 0.10, "x": 0.10, "v": 0.10, "w": 0.10, "z": 0.10, END: 0.50}
START = dict.fromkeys(TOKENS, 1.0) | {END: 0.02}
LENGTHS, WIDTH, MAXLEN = (3, 4, 5, 6, 7), 3, 10


def step(previous: str, token: str, index: int, target: int) -> float:
    emission = (0.02 if index < target else 0.9) if token == END else EMISSION[token]
    table = START if previous == "<s>" else (DEAD_END if previous == "x" else CONTINUE)
    return math.log(max(emission * table[token], 1e-12))


def greedy(target: int) -> list:
    sequence, previous = [], "<s>"
    for index in range(MAXLEN):
        best = max((*TOKENS, END), key=lambda t: step(previous, t, index, target))
        if best == END:
            break
        sequence.append(best)
        previous = best
    return sequence


def beam(target: int, width: int, norm: bool = False) -> list:
    live, done = [([], "<s>", 0.0)], []
    for index in range(MAXLEN):
        candidates = []
        for sequence, previous, total in live:
            for token in (*TOKENS, END):
                score = total + step(previous, token, index, target)
                if token == END:
                    done.append((sequence, score))
                else:
                    candidates.append((sequence + [token], token, score))
        live = sorted(candidates, key=lambda row: -row[2])[:width]
    done += [(sequence, score) for sequence, _, score in live]
    rank = (lambda row: row[1] / max(1, len(row[0]))) if norm else (lambda row: row[1])
    return max(done, key=rank)[0]


def bleu(hypothesis, reference, order=4) -> float:
    if not hypothesis:
        return 0.0
    precisions = []
    for size in range(1, order + 1):
        got = collections.Counter(tuple(hypothesis[i:i + size])
                                  for i in range(len(hypothesis) - size + 1))
        want = collections.Counter(tuple(reference[i:i + size])
                                   for i in range(len(reference) - size + 1))
        matched = sum(min(count, want[gram]) for gram, count in got.items())
        precisions.append((matched + 1) / (max(1, sum(got.values())) + 1))
    penalty = 1.0 if len(hypothesis) >= len(reference) else math.exp(
        1 - len(reference) / max(1, len(hypothesis)))
    return penalty * math.exp(sum(map(math.log, precisions)) / order)


DECODERS = {"greedy": lambda n: greedy(n), "beam-3": lambda n: beam(n, WIDTH),
            "beam-3 normalised": lambda n: beam(n, WIDTH, norm=True),
            "beam-1": lambda n: beam(n, 1)}


def scored(arms) -> dict:
    references = {n: ["y"] * n for n in LENGTHS}
    return {name: round(sum(bleu(out[n], references[n]) for n in LENGTHS) / len(LENGTHS), 4)
            for name, out in arms.items()}


def solve():
    arms = {name: {n: run(n) for n in LENGTHS} for name, run in DECODERS.items()}
    diverge = [[i for i, (a, b) in enumerate(zip(arms["greedy"][n], arms["beam-3 normalised"][n]))
                if a != b] for n in LENGTHS]
    return {
        "bleu": scored(arms), "reference_lengths": list(LENGTHS), "diverge": diverge,
        "lengths": {name: [len(out[n]) for n in LENGTHS] for name, out in arms.items()},
        "last_index": [len(arms["greedy"][n]) - 1 for n in LENGTHS],
        "sample": {"greedy": arms["greedy"][5], "normalised": arms["beam-3 normalised"][5]},
    }


def verify(result):
    scores, lengths, diverge = result["bleu"], result["lengths"], result["diverge"]
    return [
        practice.Check(
            "ANSWER: unnormalised beam-3 loses to greedy everywhere -- BLEU 0.0253 against 0.2507",
            scores["beam-3"] < scores["greedy"] < scores["beam-3 normalised"],
            f"averaged over reference lengths {result['reference_lengths']}: greedy "
            f"{scores['greedy']}, beam-{WIDTH} {scores['beam-3']}, beam-{WIDTH} with length "
            f"normalisation {scores['beam-3 normalised']}. The exercise expects beam to win "
            f"somewhere and tie elsewhere; unnormalised it loses on every sentence"),
        practice.Check(
            "MECHANISM: every extra token subtracts, so the argmax is the shortest complete hypothesis",
            lengths["beam-3"] == [1] * len(LENGTHS),
            f"the score is a sum of log-probabilities, each negative, so a longer hypothesis cannot "
            f"outscore its own prefix unless a step is free. Beam-{WIDTH} returns "
            f"{lengths['beam-3']} tokens against references of {result['reference_lengths']}. That "
            f"is not a search failure -- it is the search succeeding at the stated objective"),
        practice.Check(
            "FINDING: divide by length and the exercise's claim is exactly right",
            scores["beam-3 normalised"] > scores["greedy"]
            and all(d == [last] for d, last in zip(diverge, result["last_index"])),
            f"length-normalised beam-{WIDTH} beats greedy, {scores['beam-3 normalised']} against "
            f"{scores['greedy']}, and differs from it at exactly one position on every sentence: "
            f"{diverge}, against final indices {result['last_index']}. At length 5 that is "
            f"{result['sample']['greedy']} versus {result['sample']['normalised']}"),
        practice.Check(
            "MECHANISM: it is the last token because that is where the continuation stops mattering",
            result["sample"]["greedy"][-1] == "x" and result["sample"]["normalised"][-1] == "y",
            f"'x' has the higher emission at every step ({EMISSION['x']} against "
            f"{EMISSION['y']}) and the worse continuation, so greedy takes it throughout. Beam "
            f"takes it too until the final step, where nothing follows and the emission is the whole "
            f"score. 'Usually last tokens' is right, about the normalised variant"),
        practice.Check(
            "FINDING: beam width 1 is not greedy decoding",
            lengths["beam-1"] != lengths["greedy"],
            f"beam-1 returns {lengths['beam-1']} tokens where greedy returns {lengths['greedy']}. "
            f"Both keep one hypothesis, but beam-1 compares completed hypotheses against live ones "
            f"at every step and greedy only asks which token is best next. The two differ in the "
            f"stopping rule, not the width"),
        practice.Check(
            "CONTROL: the brevity penalty is what makes the short outputs score badly, not the n-grams",
            scores["beam-1"] < scores["greedy"],
            f"a one-token output can still match a unigram, so its n-gram precisions are not zero; "
            f"BLEU's brevity penalty is what takes beam-1 to {scores['beam-1']} and beam-{WIDTH} to "
            f"{scores['beam-3']}. A metric without it would rank the shortest hypothesis first, "
            f"which is the same ranking the decoder already applied"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
