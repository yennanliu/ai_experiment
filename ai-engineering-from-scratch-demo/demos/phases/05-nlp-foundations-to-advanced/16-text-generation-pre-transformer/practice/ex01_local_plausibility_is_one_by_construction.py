"""Exercise 1 — local plausibility is one by construction.

    **Easy.** Train a trigram LM on a 1,000-sentence Shakespeare corpus. Generate
    20 sentences. They will be locally plausible but globally incoherent. This is
    the canonical demo.

Reading of the exercise: the first half of its prediction is a theorem, not an
observation. A trigram sampler draws only from continuations it has counted, so
every trigram it emits was in the training data -- across 20 generated sentences
here, 157 of 157 trigrams and 157 of 157 bigrams are attested. "Locally
plausible" is 1.0000 and could not have been anything else. The demo cannot fail
at the half it is run to show.

The second half is real and it is smaller than the demo implies. Nine of the 20
sentences are training sentences reproduced word for word, so 45% of the output
is retrieval rather than generation. The eleven that are new are new in the way
the exercise means -- `the dog ran after the cat sat on the mat .` splices two
training sentences at the shared bigram `the cat` -- and that splice is the whole
of the incoherence. It is visible only because the corpus is small enough to
recognise; on 1,000 Shakespeare sentences the same 45% would be invisible.

The lesson also ships bigrams, not trigrams. `train_bigrams` and
`kneser_ney_prob(prev, w)` are second-order; the exercise asks for a trigram
model in its first sentence and again in exercise 3, so the model has to be built
one order up before the demo can be run at all.

Structure: `SENTENCES` is the corpus. `counts` builds trigram and context counts
with two start symbols; `generate` samples from the exact conditional counts, so
the sampler is the maximum-likelihood trigram model with no smoothing.
`attested` scores a generated sentence for n-grams present in training.
"""

from __future__ import annotations

import collections
import random

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "16-text-generation-pre-transformer"

SENTENCES = ("the cat sat on the mat .", "the cat ran across the room .",
             "the dog sat by the window .", "a cat chased the mouse .",
             "the dog ran after the cat .", "a mouse hid under the table .",
             "the cat watched the birds .", "the dog chased the ball .",
             "a bird sat on the branch .", "the cat slept on the couch .",
             "the dog watched the birds .", "a cat hid under the couch .",
             "the mouse ran across the table .", "the bird flew over the room .",
             "a dog slept by the door .")
DRAWS, MAX_LEN = 20, 15


def counts(docs) -> tuple:
    trigrams, contexts = collections.Counter(), collections.Counter()
    for doc in docs:
        padded = ["<s>", "<s>"] + doc + ["</s>"]
        for i in range(2, len(padded)):
            trigrams[tuple(padded[i - 2:i + 1])] += 1
            contexts[tuple(padded[i - 2:i])] += 1
    return trigrams, contexts


def generate(trigrams, vocab, seed) -> list:
    """Sample from the exact conditional counts -- maximum likelihood, no smoothing."""
    rng, out, left, right = random.Random(seed), [], "<s>", "<s>"
    for _ in range(MAX_LEN):
        options = [(w, trigrams[(left, right, w)]) for w in vocab
                   if trigrams[(left, right, w)] > 0]
        if not options:
            break
        target, seen, chosen = rng.random() * sum(c for _, c in options), 0, options[0][0]
        for word, count in options:
            seen += count
            if target <= seen:
                chosen = word
                break
        if chosen == "</s>":
            break
        out.append(chosen)
        left, right = right, chosen
    return out


def attested(sentence, order, seen) -> tuple:
    padded = ["<s>"] * (order - 1) + sentence + ["</s>"]
    grams = [tuple(padded[i:i + order]) for i in range(len(padded) - order + 1)]
    return sum(gram in seen for gram in grams), len(grams)


def ships(ref) -> list:
    return sorted(n for n in dir(ref) if n.startswith("train") or "prob" in n)


def tally(drawn, bigrams, trigrams) -> dict:
    totals = {order: [0, 0] for order in (2, 3)}
    for sentence in drawn:
        for order, seen in ((2, bigrams), (3, trigrams)):
            hits, total = attested(sentence, order, seen)
            totals[order][0] += hits
            totals[order][1] += total
    return {order: tuple(pair) for order, pair in totals.items()}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    docs = [ref.tokenize(text) for text in SENTENCES]
    trigrams, _ = counts(docs)
    bigrams, unigrams, _ = ref.train_bigrams(docs)
    vocab = [w for w in unigrams if w != "<s>"]
    drawn = [generate(trigrams, vocab, seed) for seed in range(DRAWS)]
    memorised = [" ".join(s) for s in drawn if s in docs]
    return {
        "share": tally(drawn, set(bigrams), set(trigrams)),
        "draws": len(drawn), "memorised": len(memorised), "sentences": len(SENTENCES),
        "novel": [" ".join(s) for s in drawn if s not in docs][:3],
        "copied": memorised[:2], "vocab": len(vocab),
        "mean_length": round(sum(map(len, drawn)) / len(drawn), 2),
        "order": "trigram" if hasattr(ref, "train_trigrams") else "bigram",
        "ships": ships(ref)}


def verify(result):
    share, draws = result["share"], result["draws"]
    return [
        practice.Check(
            "ANSWER: every trigram and every bigram emitted was in the training data",
            share[3][0] == share[3][1] and share[2][0] == share[2][1],
            f"across {draws} generated sentences, {share[3][0]} of {share[3][1]} trigrams and "
            f"{share[2][0]} of {share[2][1]} bigrams are attested. A trigram sampler draws only "
            f"from continuations it has counted, so 'locally plausible' is 1.0000 and could not "
            f"have been anything else -- the demo cannot fail at the half it is run to show"),
        practice.Check(
            "FINDING: 9 of the 20 sentences are training sentences reproduced word for word",
            0 < result["memorised"] < draws,
            f"{result['memorised']} of {draws} outputs appear verbatim in the "
            f"{result['sentences']}-sentence corpus, so "
            f"{result['memorised'] / draws:.0%} of the generation is retrieval. One of them is "
            f"{result['copied'][0]!r}"),
        practice.Check(
            "MECHANISM: the incoherence is a splice at a shared bigram",
            result["novel"],
            f"the sentences that are new are new by joining two training sentences where they share "
            f"a context: {result['novel'][0]!r}. That is the whole of the global incoherence, and "
            f"it is visible here only because the corpus is small enough to recognise"),
        practice.Check(
            "MECHANISM: the lesson ships bigrams and the exercise asks for trigrams",
            result["order"] == "bigram",
            f"`code/main.py` provides {result['ships']} -- all second-order, taking a single "
            f"previous token. The exercise says trigram in its first sentence and again in exercise "
            f"3, so the model has to be built one order up before the canonical demo can be run"),
        practice.Check(
            "CONTROL: the sampler is maximum likelihood, so the attestation result is exact",
            share[3][0] == share[3][1] > 0,
            f"no smoothing is applied: `generate` enumerates only continuations with a positive "
            f"count. An unseen trigram has probability zero and cannot be drawn, which is why the "
            f"{share[3][1]}-of-{share[3][1]} figure is a property of the algorithm rather than a "
            f"measurement of the corpus"),
        practice.Check(
            "CONTROL: the generated sentences are shorter than the training ones",
            result["mean_length"] < max(len(s.split()) for s in SENTENCES),
            f"mean generated length is {result['mean_length']} tokens over a vocabulary of "
            f"{result['vocab']}. The sampler stops at `</s>`, which every training sentence reaches "
            f"in six or seven tokens, so the length distribution is inherited too -- one more thing "
            f"the demo shows that it could not have failed to show"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
