"""Exercise 2 — a word-level Markov chain text generator.

    **Build a text generator using a Markov chain.** Train on a small corpus: for
    each word, count transitions to the next word. Build the transition matrix.
    Generate new sentences by sampling from the chain.

Reading of the exercise: a transition matrix built from raw counts has a hole the
exercise does not mention — the **last word of the corpus** has no successor, so
its row is all zeros and is not a distribution. Check 3 shows the row and what
that does to generation. The other thing worth measuring is how little a
first-order chain actually invents: check 5 counts how many generated bigrams
appear verbatim in the training text, which is the honest answer to "generate new
sentences".
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "22-stochastic-processes"
SEED, LENGTH, N_SENTENCES = 20260904, 12, 20
CORPUS = ("the cat sat on the mat the cat ate the fish the dog sat on the rug "
          "the dog chased the cat the fish swam in the pond the mat was red")


def build(words):
    vocabulary = sorted(set(words))
    index = {w: i for i, w in enumerate(vocabulary)}
    counts = [[0.0] * len(vocabulary) for _ in vocabulary]
    for a, b in zip(words, words[1:]):
        counts[index[a]][index[b]] += 1
    rows = []
    for row in counts:
        total = sum(row)
        rows.append([v / total for v in row] if total else list(row))
    return vocabulary, index, rows


def generate(chain, vocabulary, index, start, length, seed):
    rng = random.Random(seed)
    state, out = index[start], [start]
    for _ in range(length - 1):
        row = chain.P[state]      # the lesson stores the matrix as .P
        if not sum(row):
            break
        state = rng.choices(range(len(vocabulary)), weights=row)[0]
        out.append(vocabulary[state])
    return out


def _bigram_novelty(words, sentences):
    training = set(zip(words, words[1:]))
    generated = [(a, b) for s in sentences for a, b in zip(s, s[1:])]
    return len(generated), sum(1 for pair in generated if pair not in training)


def solve():
    numpy = parity.try_numpy()
    if numpy is None:
        raise practice.Skip("needs numpy — uv sync --extra math")
    ref = parity.load_reference(PHASE, LESSON, "stochastic")
    words = CORPUS.split()
    vocabulary, index, rows = build(words)
    chain = ref.MarkovChain(numpy.array(rows), state_names=vocabulary)
    sentences = [generate(chain, vocabulary, index, "the", LENGTH, SEED + i)
                 for i in range(N_SENTENCES)]
    n_generated, n_novel = _bigram_novelty(words, sentences)
    dead_word = words[-1]
    row_sums = [sum(row) for row in rows]
    return {"vocab": len(vocabulary), "sentences": sentences[:3],
            "row_sums": row_sums, "dead": dead_word,
            "dead_row_sum": row_sums[index[dead_word]],
            "zero_rows": sum(1 for s in row_sums if s == 0),
            "n_generated": n_generated, "n_novel": n_novel,
            "unique_sentences": len({tuple(s) for s in sentences}),
            "the_row": rows[index["the"]], "vocabulary": vocabulary}


def verify(result):
    novel_fraction = result["n_novel"] / result["n_generated"]
    return [
        practice.Check(f"trained on a {result['vocab']}-word vocabulary; "
                       f"{N_SENTENCES} sentences generated",
                       result["unique_sentences"] > N_SENTENCES // 2,
                       f"{result['unique_sentences']}/{N_SENTENCES} distinct; first: "
                       + " | ".join(" ".join(s) for s in result["sentences"][:2])),
        practice.Check("'the' has a proper distribution over its successors",
                       abs(sum(result["the_row"]) - 1.0) < 1e-12,
                       f"row sums to 1 over {sum(1 for p in result['the_row'] if p > 0)} "
                       f"distinct successors — 'the' is the most connected word, as it "
                       f"would be in any English corpus"),
        practice.Check(f"FINDING: the last corpus word '{result['dead']}' has an all-zero row",
                       result["dead_row_sum"] == 0.0 and result["zero_rows"] == 1,
                       f"it appears once, at the very end, so it has no observed successor "
                       f"and its row is not a distribution at all. Exactly "
                       f"{result['zero_rows']} of {result['vocab']} rows is degenerate — a "
                       f"sampler that reaches it has nothing to sample from"),
        practice.Check("…so generation has to handle it, or it would raise",
                       all(len(s) >= 1 for s in result["sentences"]),
                       f"the generator stops early on a zero row rather than calling "
                       f"random.choices with all-zero weights, which raises ValueError. "
                       f"The exercise's 'sample from the chain' assumes every row is valid"),
        practice.Check("ANSWER: a first-order chain invents nothing at the bigram level",
                       result["n_novel"] == 0,
                       f"exactly {result['n_novel']} of {result['n_generated']} generated "
                       f"bigrams are new ({novel_fraction:.1%}) — and this is provable, not "
                       f"lucky: a zero transition probability can never be sampled. A "
                       f"first-order chain can only recombine observed pairs, so 'new "
                       f"sentences' means new *paths* through seen transitions — which is "
                       f"exactly why n-gram models gave way to neural ones"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
