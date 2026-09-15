"""Exercise 3 — the quality filter ranks the duplicates best and the improvement is the vocabulary.

    **Hard:** Build a perplexity-based quality filter. Train a small bigram
    language model on Wikipedia text, score each document by perplexity, and
    remove the bottom 20%. Compare model output quality when training on
    filtered vs unfiltered data.

Reading of the exercise: "Wikipedia text" is read as *encyclopedic prose*, and
the corpus used is the lesson's own `generate_sample_corpus` after `clean_text`
and `quality_filter` -- the 13 documents `run_pipeline` actually carries, which
is the corpus the filter would sit in. "Compare model output quality" is read as
held-out perplexity, the only quality a bigram model has, and it is measured
twice: once the way it is naturally computed and once correctly.

**ANSWER: the bottom 20% is three ordinary documents.** Natural language
processing (145.9), Generative adversarial networks (152.9), Computer vision
(159.4), against a corpus that runs 92.5 to 159.4 -- a factor of 1.72, end to
end, with no gap anywhere to cut at.

**FINDING: the filter ranks the duplicates best.** The two lowest-perplexity
documents are `base_docs[1]` and `near_dup_2` at **92.5 each** -- the exact
duplicate pair Exercise 2 removes -- and the next two are the near-duplicate
pair at 110.1 and 115.8. A document repeated in the training set is maximally
predictable by a model trained on it, so the four documents a deduplicator
targets are the four this filter protects. It is a redundancy detector.

**FINDING: the one genuinely damaged document survives.** `clean_text` strips
`</h1><p>` without leaving a separator, so the HTML document begins
`TitleMachine`. It ranks **9th of 13**, mid-pack, and the bottom-20% cut does
not reach it.

**FINDING: the filtered/unfiltered comparison measures the vocabulary.** Each
model scored under its own vocabulary, filtering looks like a **20.4%**
perplexity win on held-out text. But add-alpha smoothing divides by `alpha * V`,
and dropping three documents drops the vocabulary from 327 types to 258 -- 21%
-- which raises every probability for free. Score both models under the shared
vocabulary, the only comparison that means anything, and the win is **-0.3%**.
The entire improvement was the denominator.

Structure: `bigram` and `perplexity` are the model; `V` is passed in explicitly
so the two arms can be scored under one vocabulary or under their own.
"""

from __future__ import annotations

import collections
import math

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "03-data-pipelines"
ALPHA, CUT = 1.0, 0.2
HELD_OUT = (
    "Supervised learning uses labelled examples to fit a model that predicts a target value.",
    "A language model assigns a probability to every sequence of tokens in its vocabulary.",
    "Gradient descent updates the parameters of a model in the direction that reduces the loss.",
)


def bigram(documents):
    """Unigram and bigram counts over whitespace tokens, each document opened with <s>."""
    unigrams, bigrams = collections.Counter(), collections.Counter()
    for document in documents:
        words = ["<s>"] + document.lower().split()
        unigrams.update(words)
        bigrams.update(zip(words, words[1:]))
    return unigrams, bigrams


def perplexity(model, document, vocabulary):
    """Add-alpha bigram perplexity, with the vocabulary size passed in rather than assumed."""
    unigrams, bigrams = model
    words = ["<s>"] + document.lower().split()
    total = sum(math.log((bigrams[pair] + ALPHA) / (unigrams[pair[0]] + ALPHA * vocabulary))
                for pair in zip(words, words[1:]))
    return math.exp(-total / max(len(words) - 1, 1))


def held_out(model, vocabulary):
    return sum(perplexity(model, text, vocabulary) for text in HELD_OUT) / len(HELD_OUT)


def corpus(ref):
    """The documents run_pipeline carries into deduplication, cleaned and quality-filtered."""
    cleaned = [ref.clean_text(doc) for doc in ref.generate_sample_corpus()]
    return [doc for doc in cleaned if ref.quality_filter(doc)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    documents = corpus(ref)
    full = bigram(documents)
    own = len(full[0]) + 1
    scored = sorted((perplexity(full, doc, own), i) for i, doc in enumerate(documents))
    dropped = sorted(i for _, i in scored[-round(CUT * len(documents)):])
    kept = bigram([doc for i, doc in enumerate(documents) if i not in dropped])
    shared = len(set(full[0]) | set(kept[0])) + 1
    return {
        "scored": [(round(p, 1), i, documents[i][:30]) for p, i in scored],
        "dropped": dropped,
        "vocab": (own, len(kept[0]) + 1, shared),
        "own_v": (held_out(full, own), held_out(kept, len(kept[0]) + 1)),
        "shared_v": (held_out(full, shared), held_out(kept, shared)),
        "welded": [doc.startswith("TitleMachine") for doc in documents].index(True),
        "rank": {i: r for r, (_, i) in enumerate(scored, 1)},
    }


def verify(result):
    scored, dropped, rank = result["scored"], result["dropped"], result["rank"]
    own, filtered_v, shared = result["vocab"]
    naive = 100 * (1 - result["own_v"][1] / result["own_v"][0])
    honest = 100 * (1 - result["shared_v"][1] / result["shared_v"][0])
    welded = result["welded"]
    return [
        practice.Check(
            "ANSWER: the bottom 20% is three ordinary encyclopedic documents, and there is no gap",
            len(dropped) == round(CUT * len(scored)) and scored[-1][0] / scored[0][0] < 2,
            f"the cut removes documents {dropped} at perplexity "
            + ", ".join(f"{p}" for p, i, _ in scored if i in dropped)
            + f", against a corpus that runs {scored[0][0]} to {scored[-1][0]} -- a factor of "
            f"{scored[-1][0] / scored[0][0]:.2f} end to end. The three are "
            + "; ".join(repr(t) for _, i, t in scored if i in dropped)
            + " -- ordinary prose, and the distribution has no gap anywhere to cut at",
        ),
        practice.Check(
            "FINDING: the two best-scoring documents are the exact-duplicate pair",
            scored[0][0] == scored[1][0] and scored[0][0] < 0.7 * scored[-1][0],
            f"documents {scored[0][1]} and {scored[1][1]} tie at {scored[0][0]}, the lowest in "
            f"the corpus -- they are base_docs[1] and near_dup_2, byte-identical, the pair "
            f"Exercise 2 removes -- and the next two at {scored[2][0]} and {scored[3][0]} are "
            "the near-duplicate pair. A document repeated in the training set is maximally "
            "predictable by a model trained on it, so the four documents a deduplicator targets "
            "are the four this filter protects. It scores redundancy and calls it quality",
        ),
        practice.Check(
            "FINDING: the one genuinely damaged document is not the one removed",
            welded >= 0 and welded not in dropped and rank[welded] > len(scored) // 2,
            f"clean_text strips '</h1><p>' without leaving a separator, so the HTML document "
            f"reaches the filter beginning 'TitleMachine'. It ranks {rank[welded]} of "
            f"{len(scored)} -- mid-pack -- and the bottom-20% cut does not reach it. The filter "
            "is measuring how well the rest of the corpus predicts a document, and a single "
            "welded word costs almost nothing on that scale",
        ),
        practice.Check(
            "FINDING: the filtered/unfiltered comparison measures the vocabulary, not the data",
            naive > 15 and abs(honest) < 2,
            f"scored under each model's own vocabulary, filtering looks like a {naive:.1f}% "
            f"perplexity win on held-out text ({result['own_v'][0]:.1f} -> "
            f"{result['own_v'][1]:.1f}). But add-alpha divides by alpha*V, and dropping three "
            f"documents takes the vocabulary from {own} types to {filtered_v}, "
            f"{100 * (1 - filtered_v / own):.0f}% smaller, which raises every probability for "
            f"free. Score both under the shared vocabulary of {shared} -- the only comparison "
            f"that means anything -- and the win is {honest:+.1f}%. The improvement was the "
            "denominator, and the exercise's own metric cannot see the difference",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
