"""Exercise 4 — the only paraphrase detector the lesson ships scores a word shuffle as 1.000.

    Implement contamination detection: given a set of eval questions and a
    training corpus, check what percentage of eval questions (or close
    paraphrases) appear in the training data. This is how researchers audit
    benchmark validity.

Reading of the exercise: "or close paraphrases" needs a similarity function, and
the lesson ships exactly one that could serve -- `token_f1` -- so the detector is
built on it and then tested on the cases it has to get right. Exact
contamination is checked with `exact_match`, which is the other shipped scorer.
The corpus is planted: some questions verbatim, some reworded, some absent.

**ANSWER: 40% exact and 70% at a 0.6 paraphrase threshold**, on a corpus where
40% of the questions were planted verbatim and another 30% reworded. The
detector recovers the plant, which is what makes the next finding a problem
rather than a curiosity.

**FINDING: `token_f1` scores a word shuffle as a perfect match.**
`"mat the on sat cat the"` against `"the cat sat on the mat"` is **1.000** --
the function takes `set(...)` of both sides, so word order and repetition are
both discarded. Any permutation of a question is indistinguishable from the
question, and a contamination audit built on it cannot tell a quotation from an
anagram.

**FINDING: repetition alone buys a third of the score.** `"the the the the the
the"` against the same sentence scores **0.333**, because the six repeats
collapse to one token that the target happens to contain. A corpus of filler
words registers as partially contaminated with everything.

**FINDING: the threshold is doing the work, and there is no principled
value.** Sweeping it from 0.3 to 0.9 takes the reported contamination rate
across a wide range on the same corpus -- the number the exercise asks for is a
function of a constant the exercise does not name. "How researchers audit
benchmark validity" is a sentence about a knob.

Structure: `contaminated` is the detector; `CORPUS` plants verbatim copies,
rewordings and absences so the detector's own recall can be checked.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "10-evaluation"
THRESHOLD = 0.6
SWEEP = (0.3, 0.5, 0.6, 0.7, 0.9)
QUESTIONS = (
    "what is the capital of france",
    "who wrote hamlet",
    "what is the boiling point of water",
    "name the largest planet",
    "what year did world war two end",
    "how many continents are there",
    "what is the speed of light",
    "who painted the mona lisa",
    "what is the largest ocean",
    "what is the chemical symbol for gold",
)
VERBATIM, REWORDED = 4, 3
PROBE = "the cat sat on the mat"


def corpus():
    """A training corpus with 4 questions verbatim, 3 reworded and 3 absent."""
    documents = list(QUESTIONS[:VERBATIM])
    documents += ["tell me " + q for q in QUESTIONS[VERBATIM:VERBATIM + REWORDED]]
    return documents + ["an unrelated document about gardening and weather"]


def contaminated(ref, question, documents, threshold):
    """True if any document matches the question at or above `threshold` on token_f1."""
    return any(ref.token_f1(document, question) >= threshold for document in documents)


def exact_hits(ref, documents):
    return sum(any(ref.exact_match(document, question) == 1.0 for document in documents)
               for question in QUESTIONS)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    documents = corpus()
    rates = {t: sum(contaminated(ref, q, documents, t) for q in QUESTIONS) / len(QUESTIONS)
             for t in SWEEP}
    return {
        "questions": len(QUESTIONS),
        "planted": (VERBATIM, REWORDED),
        "exact": exact_hits(ref, documents) / len(QUESTIONS),
        "rates": rates,
        "shuffle": ref.token_f1("mat the on sat cat the", PROBE),
        "identical": ref.token_f1(PROBE, PROBE),
        "repeats": ref.token_f1("the the the the the the", PROBE),
        "unrelated": ref.token_f1("a dog ran through a field", PROBE),
    }


def verify(result):
    rates, questions = result["rates"], result["questions"]
    verbatim, reworded = result["planted"]
    return [
        practice.Check(
            f"ANSWER: {result['exact']:.0%} exact and {rates[THRESHOLD]:.0%} at threshold {THRESHOLD}",
            result["exact"] == verbatim / questions
            and rates[THRESHOLD] == (verbatim + reworded) / questions,
            f"of {questions} eval questions, {verbatim} were planted verbatim and {reworded} "
            f"reworded. exact_match recovers {result['exact']:.0%} and the token_f1 detector at "
            f"{THRESHOLD} recovers {rates[THRESHOLD]:.0%} -- exactly the plant, which is what "
            "makes the next finding a problem rather than a curiosity",
        ),
        practice.Check(
            "FINDING: token_f1 scores a word shuffle as a perfect match",
            result["shuffle"] == result["identical"] == 1.0,
            f"'mat the on sat cat the' against {PROBE!r} scores {result['shuffle']:.3f}, the same "
            f"{result['identical']:.3f} the sentence scores against itself. token_f1 takes "
            "set(...) of both sides, so word order and repetition are both discarded -- any "
            "permutation of a question is indistinguishable from the question, and an audit "
            "built on it cannot tell a quotation from an anagram",
        ),
        practice.Check(
            "FINDING: repetition alone buys a third of the score",
            0.3 < result["repeats"] < 0.4 and result["unrelated"] == 0.0,
            f"'the the the the the the' against {PROBE!r} scores {result['repeats']:.3f} while an "
            f"unrelated sentence scores {result['unrelated']:.3f}. The six repeats collapse to "
            "one token the target happens to contain, so a corpus of filler words registers as "
            "partially contaminated with everything that uses the same stop words",
        ),
        practice.Check(
            "FINDING: the threshold is doing the work and the exercise does not name one",
            max(rates.values()) > min(rates.values()),
            "sweeping the threshold gives "
            + ", ".join(f"{t} -> {rate:.0%}" for t, rate in rates.items())
            + " on one fixed corpus. The percentage the exercise asks for is a function of a "
            "constant it never states, so 'how researchers audit benchmark validity' is, at "
            "this level of specification, a sentence about a knob",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
