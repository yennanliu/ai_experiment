"""Exercise 2 — the threshold leaves nothing to measure.

    **Medium.** Add subsampling of frequent words. Words with frequency above
    `10^-5` are dropped from training pairs with probability proportional to
    their frequency. Measure the effect on rare-word similarity.

Reading of the exercise: 10^-5 is word2vec's published threshold, chosen
against a corpus of a billion tokens, and this lesson's corpus has 96. Every
one of its 43 word types is above it -- the rarest by a factor of 1042 -- so
the clause that reads like a filter selects the entire vocabulary, and
word2vec's keep probability sqrt(t/f) tops out at 0.031. Two of 96 tokens
survive and no training pair does. The last sentence of the exercise has
nothing left to measure.

Raising the threshold three orders of magnitude makes it a real experiment, and
the experiment does not say what the exercise implies. The effect on rare-word
similarity is non-monotone: at t=0.03 a rare word's nearest neighbour is a word
it truly co-occurs with 0.798 of the time against 0.738 unsubsampled, and at
t=0.01 that falls to 0.638 -- below doing nothing at all -- while the obvious
proxy, the share of training pairs touching a rare word, rises the whole way,
0.311 to 0.382 to 0.447. The quantity subsampling maximizes and the quantity it
is for come apart past the peak, and at t=0.01 one rare word is deleted from
the vocabulary outright.

Two smaller mismatches. "Dropped from training pairs" is not what word2vec
does: it drops tokens, which pulls distant words into the window and mints
pairs the corpus never contained -- 10 of them here, `mat`-`sat` and `dog`-`rug`
among them. And "probability proportional to their frequency" is a third rule
again. word2vec's `1 - sqrt(t/f)` is clipped at zero, so a word below the
threshold is never dropped; a proportional rule has no floor and drops the
rarest words too, which is the opposite of the point.

Structure: `frequencies` is the unigram distribution, `keep` is word2vec's
probability and `linear` the exercise's wording. `drop_tokens` subsamples the
token stream and `drop_pairs` the pair list, which is the distinction the
exercise's wording erases. `hit_rate` trains and asks, for each hapax, whether
its nearest neighbour is a word it actually appears beside.
"""

from __future__ import annotations

import collections
import math
import pathlib

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "03-word-embeddings-word2vec"

SIBLING = "ex01_a_nine_in_twelve_check.py"
NAMED, TRIED, SEEDS, WINDOW = 1e-5, (0.03, 0.01), 4, 2
CONFIG = {"dim": 16, "window": WINDOW, "k_neg": 5, "lr": 0.05, "epochs": 200}

keep = lambda t, f: min(1.0, math.sqrt(t / f))                                   # noqa: E731
linear = lambda f, top: f / top                                                  # noqa: E731
share = lambda pairs, rare: sum(1 for a, b in pairs if a in rare or b in rare) / len(pairs)


def frequencies(docs) -> tuple:
    counts = collections.Counter(token for doc in docs for token in doc)
    total = sum(counts.values())
    return counts, {w: c / total for w, c in counts.items()}, total  # the unigram distribution


def drop_tokens(np, docs, freq, t, seed) -> list:
    """word2vec's subsampling: remove tokens, which widens every surviving window."""
    rng = np.random.default_rng(seed)
    return [[w for w in doc if rng.random() < keep(t, freq[w])] for doc in docs]


def drop_pairs(np, pairs, freq, t, seed) -> list:
    """The exercise's wording: remove pairs, which cannot widen anything."""
    rng = np.random.default_rng(seed)
    return [(a, b) for a, b in pairs if rng.random() < keep(t, freq[a]) * keep(t, freq[b])]


def context(docs) -> dict:
    """Every word's true corpus neighbours inside the window."""
    seen = collections.defaultdict(set)
    for doc in docs:
        for i, word in enumerate(doc):
            seen[word].update(doc[max(0, i - WINDOW):i] + doc[i + 1:i + WINDOW + 1])
    return seen


def hit_rate(ref, train_docs, rare, truth, seed) -> tuple:
    """Of the rare words that survive, how many have a true neighbour ranked first."""
    vocab, weights = ref.train(train_docs, seed=seed, **CONFIG)
    live = [w for w in sorted(rare) if w in vocab]
    hits = sum(ref.nearest(vocab, weights, weights[vocab[w]], topk=1,
                           exclude={vocab[w]})[0][0] in truth[w] for w in live)
    return hits, len(live)


def counted(ref, np, docs, freq, arms, base, rare) -> dict:
    """Everything measurable without training: what each threshold leaves behind."""
    thinned = drop_tokens(np, docs, freq, NAMED, 0)
    grams = {name: ref.skipgram_pairs(rows, window=WINDOW) for name, rows in arms.items()}
    return {"above": sum(f > NAMED for f in freq.values()), "survive": sum(map(len, thinned)),
            "keep_max": max(keep(NAMED, f) for f in freq.values()),
            "named_pairs": len(ref.skipgram_pairs(thinned, window=WINDOW)),
            "kept": {t: len(grams[t]) for t in TRIED}, "n_widened": len(set(grams[TRIED[0]]) - base),
            "share": {name: share(rows, rare) for name, rows in grams.items()},
            "widened": sorted(set(grams[TRIED[0]]) - base)[:3]}


def trained(ref, arms, rare, truth) -> dict:
    scored = {name: [hit_rate(ref, docs, rare, truth, s) for s in range(SEEDS)]
              for name, docs in arms.items()}
    return {"hit": {n: sum(h / k for h, k in rows) / SEEDS for n, rows in scored.items()},
            "live": {n: min(k for _, k in rows) for n, rows in scored.items()}}


def solve():
    try:
        import numpy as np
    except ImportError as exc:                      # pragma: no cover - T1 needs numpy
        raise practice.Skip(f"needs numpy: uv sync --extra math ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    sentences = practice.load_module(pathlib.Path(__file__).with_name(SIBLING)).SENTENCES
    docs = [ref.tokenize(sentence) for sentence in sentences]
    counts, freq, total = frequencies(docs)
    rare = {w for w, c in counts.items() if c == 1}
    pairs, top = ref.skipgram_pairs(docs, window=WINDOW), max(freq.values())
    arms = {"base": docs} | {t: drop_tokens(np, docs, freq, t, 0) for t in TRIED}
    return dict(counted(ref, np, docs, freq, arms, set(pairs), rare),
                **trained(ref, arms, rare, context(docs)), types=len(counts), tokens=total,
                min_f=min(freq.values()), pairs=len(pairs), rare=len(rare),
                by_pairs=len(drop_pairs(np, pairs, freq, TRIED[0], 0)),
                floor={"w2v": 1 - keep(TRIED[0], top), "rare_linear": linear(min(freq.values()), top),
                       "rare_w2v": 1 - keep(TRIED[0], min(freq.values()))})


def verify(result):
    hit, share_, kept = result["hit"], result["share"], result["kept"]
    lo, hi = TRIED
    return [
        practice.Check(
            "ANSWER: at the threshold the exercise names, no training pair survives",
            result["above"] == result["types"] and result["named_pairs"] == 0,
            f"the corpus is {result['tokens']} tokens over {result['types']} types, all "
            f"{result['above']} above {NAMED:g} -- the rarest by a factor of "
            f"{result['min_f'] / NAMED:.0f}. word2vec's keep probability sqrt(t/f) tops out at "
            f"{result['keep_max']:.4f}, so {result['survive']} tokens and {result['named_pairs']} "
            f"of {result['pairs']} pairs remain. Nothing is left to measure"),
        practice.Check(
            "MECHANISM: 10^-5 is calibrated for a billion tokens, and this corpus has 96",
            result["min_f"] > 1000 * NAMED and kept[lo] > kept[hi] > 0,
            f"a threshold is a frequency, so it is a statement about corpus size. Raised three "
            f"orders of magnitude the experiment is real: t={lo} keeps {kept[lo]} of "
            f"{result['pairs']} pairs, t={hi} keeps {kept[hi]}. Below that it is not aggressive, "
            f"it is total"),
        practice.Check(
            "FINDING: the effect on rare-word similarity is non-monotone -- it helps, then hurts",
            hit[lo] > hit["base"] > hit[hi],
            f"a rare word's nearest neighbour is one it truly co-occurs with {hit['base']:.3f} of "
            f"the time unsubsampled, {hit[lo]:.3f} at t={lo}, {hit[hi]:.3f} at t={hi} -- below "
            f"doing nothing. The exercise says to measure the effect as though it had a sign; over "
            f"{SEEDS} seeds and {result['rare']} hapax words it has two"),
        practice.Check(
            "FINDING: the obvious proxy rises the whole way, including where the outcome falls",
            share_[hi] > share_[lo] > share_["base"] and hit[hi] < hit[lo],
            f"the share of pairs touching a rare word goes {share_['base']:.3f} -> {share_[lo]:.3f} "
            f"-> {share_[hi]:.3f} as t falls, monotonically, while the hit rate turns over. t={hi} "
            f"has the best rare-word coverage and the worst rare-word embeddings, having deleted "
            f"{result['pairs'] - kept[hi]} of {result['pairs']} pairs to get there -- and one rare "
            f"word with them ({result['live'][hi]} of {result['live']['base']} left in vocabulary)"),
        practice.Check(
            "FINDING: dropping tokens is not dropping pairs -- it mints pairs the corpus never had",
            result["n_widened"] > 0,
            f"word2vec removes tokens, pulling distant words inside the window: at t={lo} that "
            f"mints {result['n_widened']} pair types the corpus never had, including "
            f"{result['widened']}. Dropping pairs instead, as the exercise words it, leaves "
            f"{result['by_pairs']} against {kept[lo]} and can create nothing"),
        practice.Check(
            "CONTROL: 'proportional to frequency' has no floor, and word2vec's rule is all floor",
            result["floor"]["rare_w2v"] == 0.0 and result["floor"]["rare_linear"] > 0,
            f"word2vec's 1 - sqrt(t/f) is clipped at zero, so at t={lo} the most frequent word is "
            f"dropped with probability {result['floor']['w2v']:.3f} and the rarest with "
            f"{result['floor']['rare_w2v']:.3f} -- never. Proportional to frequency drops the rarest "
            f"word with probability {result['floor']['rare_linear']:.3f}, the opposite of the point"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
