"""Exercise 2 — the direction of the result is a preprocessing choice.

    **Medium.** Implement perplexity for your KN model on a held-out Shakespeare
    split. Compare against Laplace. You should see KN lower perplexity by 30-50%.

Reading of the exercise: on a held-out split of real English -- this phase's own
lesson documents, 725 sentences, since no Shakespeare corpus ships -- Kneser-Ney
scores perplexity 6763 against Laplace's 821. Not 30-50% lower: **eight times
higher**. Close the vocabulary and it reverses to 125 against 590, a 78.7%
reduction, past the top of the predicted band. The exercise's claim and its
opposite are both available from the same two functions on the same corpus, and
which one you get is decided by how out-of-vocabulary tokens are handled -- a
step it does not mention.

The cause is in `kneser_ney_prob`. Its continuation term is
`len(unigram_contexts[w]) / total_unique_bigrams`, and a word never seen in
training has no contexts, so that term is zero; with a zero bigram count the
whole probability is zero. 18.1% of held-out tokens here are out of vocabulary
and 14.2% of positions receive probability zero, which `perplexity` clamps to
1e-12 -- a penalty of 27.6 nats each. Laplace's +1 gives the same token
1/(count + V) instead. KN has no unknown-word floor and Laplace is nothing but
floor.

There is a second, smaller defect on the Laplace side, and it runs the other way.
`laplace_prob` divides by `unigrams[prev] + vocab_size`, but the quantity it
needs is the number of times `prev` appears *as a context*. Those differ for any
token that ends a sentence: on a three-sentence corpus `</s>` occurs 3 times and
is a context 0 times, so its distribution sums to 0.8235 instead of 1.
Kneser-Ney sums to exactly 1.0 for every context tested.

Structure: `corpus` reads this phase's English lesson documents through the
harness and splits them into sentences; `models` builds both probability
functions over a training split. `mass` sums a distribution over the whole
vocabulary, and `arms` scores perplexity on the open, closed and
OOV-substituted versions of the same test set.
"""

from __future__ import annotations

import collections
import re

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "16-text-generation-pre-transformer"

SPLIT, MIN_LEN, MAX_LEN, DOCS = 0.8, 5, 30, 8
FENCE = re.compile(r"```.*?```", re.S)
KEEP = re.compile(r"[^A-Za-z .,']")


def corpus(ref) -> list:
    """This phase's own lesson documents, read through the harness and split into sentences."""
    root = parity.find_reference_root() / "phases" / PHASE
    names = sorted(d.name for d in root.iterdir() if (d / "docs" / "en.md").is_file())[:DOCS]
    out = []
    for name in names:
        body = FENCE.sub("", parity.doc_text(PHASE, name, "en"))
        for chunk in re.split(r"(?<=[.!?])\s+", body):
            tokens = ref.tokenize(KEEP.sub(" ", chunk))
            if MIN_LEN <= len(tokens) <= MAX_LEN:
                out.append(tokens)
    return out


def models(ref, train) -> tuple:
    bigrams, unigrams, contexts = ref.train_bigrams(train)
    totals, follow = collections.Counter(), collections.defaultdict(set)
    for (prev, word), count in bigrams.items():
        totals[prev] += count
        follow[prev].add(word)
    unique = sum(len(seen) for seen in contexts.values())
    kn = lambda p, w: ref.kneser_ney_prob(bigrams, contexts, totals, follow, unique, p, w)  # noqa: E731
    laplace = lambda p, w: ref.laplace_prob(bigrams, unigrams, len(unigrams), p, w)         # noqa: E731
    return kn, laplace, unigrams, totals


def mass(prob, vocab, prev) -> float:
    return round(sum(prob(prev, w) for w in vocab), 6)


def arms(ref, kn, laplace, test, unigrams) -> dict:
    known = [s for s in test if all(w in unigrams for w in s)]
    filler = max(unigrams, key=lambda w: unigrams[w] if w not in ("<s>", "</s>") else 0)
    mapped = [[w if w in unigrams else filler for w in s] for s in test]
    return {name: {"laplace": round(ref.perplexity(laplace, rows), 2),
                   "kn": round(ref.perplexity(kn, rows), 2), "sentences": len(rows)}
            for name, rows in (("open", test), ("closed", known), ("mapped", mapped))}


def normalisation(ref) -> dict:
    """Does each model's distribution sum to one, on a corpus small enough to enumerate?"""
    toy = [ref.tokenize(s) for s in ("the cat sat on the mat .", "the dog sat by the window .",
                                     "a cat chased the mouse .")]
    kn, laplace, unigrams, totals = models(ref, toy)
    return {"mass": {name: {prev: mass(prob, list(unigrams), prev) for prev in ("the", "</s>")}
                     for name, prob in (("kn", kn), ("laplace", laplace))},
            "counts": {prev: (unigrams[prev], totals[prev]) for prev in ("the", "</s>")}}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    sentences = corpus(ref)
    cut = int(len(sentences) * SPLIT)
    train, test = sentences[:cut], sentences[cut:]
    kn, laplace, unigrams, totals = models(ref, train)
    scored = arms(ref, kn, laplace, test, unigrams)
    positions = [(prev, word) for s in test for prev, word in zip(["<s>"] + s, s + ["</s>"])]
    return dict(
        normalisation(ref), arms=scored, vocab=len(unigrams), train=len(train), test=len(test),
        zeros=sum(kn(p, w) <= 0 for p, w in positions), positions=len(positions),
        oov=sum(w not in unigrams for _, w in positions),
        reduction={name: round((row["laplace"] - row["kn"]) / row["laplace"], 4)
                   for name, row in scored.items()})


def verify(result):
    arms_, reduction, mass_ = result["arms"], result["reduction"], result["mass"]
    return [
        practice.Check(
            "ANSWER: on the open vocabulary Kneser-Ney is eight times worse, not 30-50% better",
            reduction["open"] < -1.0,
            f"no Shakespeare corpus ships, so the corpus is {result['train'] + result['test']} "
            f"sentences from this phase's own lesson documents over a {result['vocab']}-word "
            f"vocabulary. Held out: Laplace {arms_['open']['laplace']}, Kneser-Ney "
            f"{arms_['open']['kn']} -- a change of {reduction['open']:+.1%} where the exercise "
            f"predicts 30-50% in the other direction"),
        practice.Check(
            "FINDING: close the vocabulary and it reverses past the top of the predicted band",
            reduction["closed"] > 0.5 and reduction["mapped"] > 0.5,
            f"restricted to the {arms_['closed']['sentences']} test sentences whose every token was "
            f"seen in training, Laplace scores {arms_['closed']['laplace']} and Kneser-Ney "
            f"{arms_['closed']['kn']} -- {reduction['closed']:+.1%}. Replacing unseen tokens with a "
            f"frequent one instead gives {reduction['mapped']:+.1%}. Both the claim and its "
            f"opposite come from the same two functions on the same corpus"),
        practice.Check(
            "MECHANISM: KN has no unknown-word floor, and Laplace is nothing but floor",
            result["zeros"] > 0 and result["oov"] > result["zeros"],
            f"the continuation term is len(contexts[w]) / total_unique_bigrams, which is zero for a "
            f"word never seen in training, and with a zero bigram count the whole probability is "
            f"zero. {result['oov']} of {result['positions']} held-out tokens are out of vocabulary "
            f"({result['oov'] / result['positions']:.1%}) and {result['zeros']} positions "
            f"({result['zeros'] / result['positions']:.1%}) receive probability zero, clamped to "
            f"1e-12 by `perplexity`"),
        practice.Check(
            "MECHANISM: so the whole result is an out-of-vocabulary policy the exercise never names",
            reduction["open"] < 0 < reduction["closed"],
            f"the three arms differ only in what happens to unseen tokens and they read "
            f"{reduction}. Nothing about the smoothing changed between them. A perplexity "
            f"comparison on an open vocabulary is a comparison of unknown-word handling wearing a "
            f"smoothing label"),
        practice.Check(
            "FINDING: Laplace has its own defect, and it runs the other way",
            mass_["laplace"]["</s>"] < 0.9 < mass_["kn"]["</s>"],
            f"`laplace_prob` divides by unigrams[prev] + vocab_size where it needs the count of "
            f"prev as a *context*. On a three-sentence corpus `</s>` occurs "
            f"{result['counts']['</s>'][0]} times and is a context "
            f"{result['counts']['</s>'][1]} times, so its distribution sums to "
            f"{mass_['laplace']['</s>']} rather than 1"),
        practice.Check(
            "CONTROL: Kneser-Ney sums to exactly one for every context tested",
            set(mass_["kn"].values()) == {1.0},
            f"the same check gives {mass_['kn']} for KN and {mass_['laplace']} for Laplace. The "
            f"model that loses the perplexity comparison by a factor of eight is the one that is "
            f"correctly normalised, and the defect that decides the comparison is in neither of "
            f"these two numbers"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
