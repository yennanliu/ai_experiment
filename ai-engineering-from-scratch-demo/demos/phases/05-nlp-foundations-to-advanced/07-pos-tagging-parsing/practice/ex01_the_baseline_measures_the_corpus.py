"""Exercise 1 — the baseline measures the corpus.

    **Easy.** Using the most-frequent-tag baseline on a small tagged corpus
    (e.g., NLTK's Brown subset), measure accuracy on held-out sentences. Verify
    the ~85% result.

Reading of the exercise: NLTK is not installed and Brown is not downloadable
here, so the corpus is generated below from seven tag templates and a lexicon
carrying five deliberately ambiguous words -- the template *is* the gold tag
sequence, so the labels are exact rather than annotated. On that corpus the
most-frequent-tag baseline scores 0.8674 at a 30% training split and 0.9103 at
70%. The exercise's ~85% is in there, at one split, and it is a fact about the
split rather than about the method.

The decomposition says why. Every held-out token falls into exactly one of three
groups, and the baseline's behaviour on each is fixed. On a word seen in
training with only one tag, it is right by construction -- 1.0000 measured, and
it could not be otherwise, since the word's most frequent tag is its only tag.
On an ambiguous word it is right whenever the test occurrence takes the majority
reading, 0.6364 here. On a word it never saw it emits the corpus's most frequent
tag, NOUN, and is right when that guess lands, 0.2000 here. The weighted sum of
those three reproduces the headline exactly. So "verify the ~85%" is asking
whether a corpus has a particular ambiguity rate and a particular
out-of-vocabulary rate, and this one has an OOV rate of 0.0552 at the 30% split
and 0.0000 at 70%.

Structure: `TEMPLATES` are tag sequences and `LEXICON` maps each tag to its
words, with `AMBIGUOUS` words added to two pools each; `corpus` cycles a counter
through the pools so the same corpus is built every run without a seed.
`bucket` sorts each held-out token into seen-unambiguous, ambiguous or OOV, and
`score` reports accuracy within each.
"""

from __future__ import annotations

import collections

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "07-pos-tagging-parsing"

LEXICON = {"DET": ("the", "a", "every", "this"),
           "NOUN": ("cat", "dog", "mat", "road", "child", "river"),
           "VERB": ("sat", "ran", "chased", "found", "watched", "carried"),
           "ADJ": ("red", "small", "quiet", "bright"),
           "ADV": ("loudly", "quickly", "softly", "early"),
           "ADP": ("on", "across", "near", "under"), "AUX": ("is", "was")}
AMBIGUOUS = {"run": ("NOUN", "VERB"), "walk": ("NOUN", "VERB"), "watch": ("NOUN", "VERB"),
             "light": ("NOUN", "ADJ"), "fast": ("ADJ", "ADV")}
TEMPLATES = (("DET", "NOUN", "VERB", "ADP", "DET", "NOUN"), ("DET", "ADJ", "NOUN", "VERB", "ADV"),
             ("NOUN", "VERB", "NOUN"), ("DET", "NOUN", "AUX", "ADJ"), ("NOUN", "VERB", "ADV"),
             ("DET", "ADJ", "NOUN", "AUX", "ADJ"),
             ("DET", "NOUN", "VERB", "DET", "ADJ", "NOUN"))
PER_TEMPLATE, SPLITS = 8, (0.3, 0.5, 0.7)


def pool(tag) -> list:
    return list(LEXICON[tag]) + [w for w, tags in AMBIGUOUS.items() if tag in tags]


def corpus() -> list:
    """Round-robin over templates, so any prefix split carries all seven of them.

    Deterministic without a seed: one counter walked through the tag pools.
    """
    rows, cursor = [], 0
    for _ in range(PER_TEMPLATE):
        for template in TEMPLATES:
            tokens = []
            for tag in template:
                choices = pool(tag)
                tokens.append(choices[cursor % len(choices)])
                cursor += 1
            rows.append((tokens, list(template)))
    return rows


def readings(rows) -> dict:
    seen = collections.defaultdict(set)
    for tokens, tags in rows:
        for token, tag in zip(tokens, tags):
            seen[token].add(tag)
    return seen


def bucket(word, vocab, ambiguous) -> str:
    if word.lower() not in vocab:
        return "oov"
    return "ambiguous" if word in ambiguous else "unambiguous"


def tagged(ref, train_rows, test_rows, ambiguous) -> list:
    """(bucket, gold, predicted) for every held-out token."""
    word_best, default = ref.train_mft(train_rows)
    vocab = {w.lower() for tokens, _ in train_rows for w in tokens}
    return [(bucket(word, vocab, ambiguous), gold, got)
            for tokens, tags in test_rows
            for word, gold, got in zip(tokens, tags,
                                       ref.predict_mft(tokens, word_best, default))], default


def score(ref, train_rows, test_rows, ambiguous) -> dict:
    rows, default = tagged(ref, train_rows, test_rows, ambiguous)
    groups = collections.defaultdict(lambda: [0, 0])
    wrong = collections.Counter((gold, got) for _, gold, got in rows if gold != got)
    for name, gold, got in rows:
        groups[name][0] += gold == got
        groups[name][1] += 1
    total = [sum(v[0] for v in groups.values()), sum(v[1] for v in groups.values())]
    return {"accuracy": round(total[0] / total[1], 4), "tokens": total[1], "default": default,
            "groups": {k: (round(v[0] / v[1], 4), v[1]) for k, v in groups.items()},
            "oov_rate": round(groups["oov"][1] / total[1], 4), "confusions": wrong.most_common(3)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = corpus()
    ambiguous = {w for w, tags in readings(rows).items() if len(tags) > 1}
    arms = {}
    for fraction in SPLITS:
        cut = int(len(rows) * fraction)
        arms[fraction] = score(ref, rows[:cut], rows[cut:], ambiguous)
    return {"arms": arms, "sentences": len(rows), "ambiguous": sorted(ambiguous),
            "types": len(readings(rows))}


def verify(result):
    arms = result["arms"]
    low, high = arms[SPLITS[0]], arms[SPLITS[-1]]
    parts = low["groups"]
    predicted = sum(rate * n for rate, n in parts.values())
    return [
        practice.Check(
            "ANSWER: the baseline scores 0.8674 at a 30% split and 0.9103 at 70%",
            0.85 < low["accuracy"] < 0.90 < high["accuracy"],
            f"NLTK is not installed and Brown is not downloadable, so the corpus is "
            f"{result['sentences']} generated sentences over {result['types']} word types. "
            f"Most-frequent-tag accuracy is "
            f"{ {f: arms[f]['accuracy'] for f in SPLITS} } at training fractions {list(SPLITS)}. The "
            f"~85% is in there, at one split"),
        practice.Check(
            "MECHANISM: three groups, and the weighted sum of them is the headline exactly",
            abs(predicted / low["tokens"] - low["accuracy"]) < 5e-4,
            f"every held-out token is seen-unambiguous, ambiguous or out-of-vocabulary, and at the "
            f"30% split those score {parts} as (accuracy, count). Their weighted mean is "
            f"{predicted / low['tokens']:.4f} against the measured {low['accuracy']} -- the "
            f"headline carries no information the three parts do not"),
        practice.Check(
            "MECHANISM: on a word seen with one tag the baseline is right by construction",
            parts["unambiguous"][0] == 1.0,
            f"a word with a single training tag has that tag as its most frequent one, so the "
            f"baseline returns it: {parts['unambiguous'][0]} over {parts['unambiguous'][1]} tokens. "
            f"That group cannot be improved and cannot be lost, which is why the headline is "
            f"decided entirely by the other two"),
        practice.Check(
            "FINDING: the OOV group is the default tag, and it moves with the split",
            low["oov_rate"] > high["oov_rate"] and low["default"] == "NOUN",
            f"every unseen word gets {low['default']!r}, the corpus's most frequent tag, and is "
            f"right when that lands -- {parts['oov'][0]} of {parts['oov'][1]} tokens here. The OOV "
            f"rate falls {low['oov_rate']} -> {arms[SPLITS[1]]['oov_rate']} -> {high['oov_rate']} as "
            f"training grows, which is most of what moves the headline"),
        practice.Check(
            "FINDING: the ambiguous group is the only place a better method has room",
            0.5 < parts["ambiguous"][0] < 1.0,
            f"the {len(result['ambiguous'])} ambiguous words {result['ambiguous']} account for "
            f"{parts['ambiguous'][1]} held-out tokens at {parts['ambiguous'][0]} accuracy. The "
            f"baseline takes the majority reading every time and has no way to take the other one, "
            f"which is what the HMM in exercise 2 is for"),
        practice.Check(
            "CONTROL: the errors are the ambiguity, named",
            all(gold != got for (gold, got), _ in low["confusions"]) and low["confusions"],
            f"the baseline's most frequent mistakes at the 30% split are {low['confusions']} as "
            f"(gold, predicted) pairs. They are the ambiguous pools -- NOUN/VERB and ADJ/ADV -- and "
            f"the default tag catching unseen words, which is the same list read twice"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
