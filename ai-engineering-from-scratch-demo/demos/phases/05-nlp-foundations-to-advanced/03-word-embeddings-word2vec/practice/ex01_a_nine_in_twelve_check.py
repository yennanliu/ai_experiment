"""Exercise 1 — a nine-in-twelve check.

    **Easy.** Run the training loop on a tiny corpus (20 sentences about cats and
    dogs). After 200 epochs, verify `nearest(vocab, W, W[vocab["cat"]])` returns
    `dog` in its top 3. If not, increase epochs or vocabulary.

Reading of the exercise: the corpus is 20 sentences written below, because the
lesson's own `main()` uses ten repeated twenty times and the exercise asks for
twenty. At the epoch count the exercise names, its check passes on 9 of 12
seeds -- it is a coin, stated as a fact, and the "if not" clause is the only
acknowledgement that it might not hold. Two of the three failures are not the
model's. The call in the exercise omits `exclude`, which the lesson's own
`main()` passes on the identical call one file away, so `cat` itself takes the
first of the three slots at similarity 1.0 and only two are left to win. Scored
with the word excluded, the same 12 runs give 11.

Doubling the epochs -- the exercise's own remedy -- takes the as-written score
from 9 to 11 of 12 and leaves the self-excluded score at 11. That is the shape
of the thing: more training buys back the wasted slot and does not move the
neighbourhood. What does not move is which word wins it. Across the 12 seeds
`mouse` is `cat`'s nearest neighbour 7 times at 200 epochs and 6 at 400, `dog`
4 and 6 -- and that is the objective doing its job, not failing. Skip-gram
raises the probability of the words that *co-occur* with `cat`, and `mouse`
co-occurs with it. `dog` never appears next to `cat` except in one sentence;
what it shares is the slots `cat` fills. The exercise asks a co-occurrence
model for a substitutability judgement.

Structure: `SENTENCES` is the corpus, `run` is one training run reduced to the
two rankings the exercise's line and the lesson's line produce, and `sweep`
collects both over `SEEDS` seeds at one epoch count. `sampler` measures the
negative sampler without training anything.
"""

from __future__ import annotations

import collections

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "03-word-embeddings-word2vec"

SENTENCES = ("the cat sat on the mat", "the dog sat on the rug", "a cat chased a mouse",
             "a dog chased a cat", "the kitten slept on the mat", "the puppy slept on the rug",
             "cats and dogs are pets", "kittens and puppies are young", "cats chase mice",
             "dogs chase squirrels", "the cat drank the milk", "the dog drank the water",
             "a kitten found a ball", "a puppy found a bone", "cats climb tall trees",
             "dogs dig deep holes", "the cat purred softly", "the dog barked loudly",
             "pets need food and care", "cats and dogs need water")
WORD, TARGET, TOPK, SEEDS, EPOCHS = "cat", "dog", 3, 12, (200, 400)
CONFIG = {"dim": 16, "window": 2, "k_neg": 5, "lr": 0.05}

rate = lambda flags: sum(flags)                                                  # noqa: E731


def run(ref, docs, epochs, seed) -> tuple:
    """One run, reduced to the exercise's ranking and the lesson's own."""
    vocab, weights = ref.train(docs, epochs=epochs, seed=seed, **CONFIG)
    vector, index = weights[vocab[WORD]], vocab[WORD]
    written = [w for w, _ in ref.nearest(vocab, weights, vector, topk=TOPK)]
    excluded = [w for w, _ in ref.nearest(vocab, weights, vector, topk=TOPK, exclude={index})]
    return written, excluded


def sweep(ref, docs, epochs) -> dict:
    runs = [run(ref, docs, epochs, seed) for seed in range(SEEDS)]
    return {"written": rate([TARGET in w for w, _ in runs]),
            "excluded": rate([TARGET in e for _, e in runs]),
            "nearest": collections.Counter(e[0] for _, e in runs),
            "self_first": rate([w[0] == WORD for w, _ in runs])}


def sampler(ref, docs, draws=20000) -> dict:
    """The negative sampler, measured without training: uniform, and never short."""
    import numpy as np
    vocab = ref.build_vocab(docs)
    counts = collections.Counter(token for doc in docs for token in doc)
    rng = np.random.default_rng(0)
    short = 0
    for _ in range(draws):
        center, context = rng.integers(0, len(vocab), size=2)
        candidates = rng.integers(0, len(vocab), size=CONFIG["k_neg"] * 2)
        short += len([n for n in candidates if n != context and n != center]) < CONFIG["k_neg"]
    return {"short": short, "draws": draws, "vocab": len(vocab),
            "top": counts.most_common(1)[0], "hapax": sum(c == 1 for c in counts.values())}


def solve():
    try:
        import numpy                                # noqa: F401
    except ImportError as exc:                      # pragma: no cover - T1 needs numpy
        raise practice.Skip(f"needs numpy: uv sync --extra math ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    docs = [ref.tokenize(sentence) for sentence in SENTENCES]
    arms = {epochs: sweep(ref, docs, epochs) for epochs in EPOCHS}
    return {"arms": arms, "seeds": SEEDS, "sentences": len(SENTENCES),
            "pairs": len(ref.skipgram_pairs(docs, window=CONFIG["window"])),
            "sampler": sampler(ref, docs)}


def verify(result):
    base, more = result["arms"][EPOCHS[0]], result["arms"][EPOCHS[1]]
    seeds, samp = result["seeds"], result["sampler"]
    return [
        practice.Check(
            "ANSWER: at the exercise's own 200 epochs the check passes on 9 of 12 seeds",
            base["written"] == 9 and base["written"] < seeds,
            f"{result['sentences']} sentences, {samp['vocab']} word types, "
            f"{result['pairs']} skip-gram pairs. `nearest(vocab, W, W[vocab['cat']])` returns 'dog' "
            f"in its top {TOPK} on {base['written']}/{seeds} seeds at {EPOCHS[0]} epochs. The "
            f"exercise states this as a fact and its 'if not' clause is the only hint it is a coin"),
        practice.Check(
            "MECHANISM: the exercise's own call wastes a slot the lesson's call does not",
            base["self_first"] == seeds and base["excluded"] > base["written"],
            f"the line in the exercise omits `exclude`, which the lesson's `main()` passes on the "
            f"same call. So 'cat' itself takes slot 1 at similarity 1.0 on all {seeds} runs and only "
            f"2 of the {TOPK} are left to win. Scored with the word excluded the same runs give "
            f"{base['excluded']}/{seeds}, recovering {base['excluded'] - base['written']} of the "
            f"{seeds - base['written']} failures"),
        practice.Check(
            "FINDING: more epochs buys back the wasted slot and does not move the neighbourhood",
            more["written"] > base["written"] and more["excluded"] == base["excluded"],
            f"doubling to {EPOCHS[1]} epochs -- the exercise's own remedy -- takes the as-written "
            f"score from {base['written']} to {more['written']} of {seeds}, and leaves the "
            f"self-excluded score at {more['excluded']}. The run that fails on its merits at "
            f"{EPOCHS[0]} epochs still fails at {EPOCHS[1]}"),
        practice.Check(
            "FINDING: 'mouse' wins the slot more often than 'dog', and that is the objective working",
            base["nearest"]["mouse"] > base["nearest"][TARGET],
            f"nearest word to 'cat' across the {seeds} seeds: {dict(base['nearest'])} at "
            f"{EPOCHS[0]} epochs, {dict(more['nearest'])} at {EPOCHS[1]}. Skip-gram raises the "
            f"probability of words that co-occur with 'cat', and 'mouse' does. 'dog' shares the "
            f"slots 'cat' fills instead -- a substitutability judgement from a co-occurrence model"),
        practice.Check(
            "CONTROL: the seed, not the corpus, is what the check is measuring",
            0 < base["written"] < seeds and 0 < more["nearest"][TARGET] < seeds,
            f"one corpus, one configuration, {seeds} seeds, and the answer to the exercise's "
            f"question changes {seeds - base['written']} times at {EPOCHS[0]} epochs and "
            f"{seeds - more['written']} at {EPOCHS[1]}. A single run reports whichever answer its "
            f"seed drew, and the lesson's `main()` fixes seed=42 without saying that it matters"),
        practice.Check(
            "CONTROL: the negative sampler is uniform, and it never runs short here",
            samp["short"] == 0,
            f"`rng.integers(0, vocab_size, size=k_neg * 2)` filtered down to {CONFIG['k_neg']} came "
            f"up short {samp['short']} times in {samp['draws']} draws, so the `[:k_neg]` truncation "
            f"is safe at this vocabulary size. It is uniform over types, though: '{samp['top'][0]}' "
            f"occurs {samp['top'][1]} times in the corpus and is drawn as a negative exactly as "
            f"often as each of the {samp['hapax']} words that occur once, where word2vec specifies "
            f"a unigram^0.75 distribution"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
