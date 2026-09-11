"""Exercise 2 — the ratio it reports is memorisation, and the 32k arm cannot be built.

    **Medium.** Compare token counts on 100 English Wikipedia sentences between
    `cl100k_base`, `o200k_base`, and a SentencePiece BPE you train with
    vocab=32k. Report the compression ratio of each.

Reading of the exercise: `tiktoken` and `sentencepiece` are both absent, so the
two OpenAI arms cannot be measured. The third arm cannot be built either, and
that is not an installation problem. Vocabulary size is a ceiling the corpus
sets, not a parameter you pass: 20 encyclopedic sentences admit **566** merges
before `pair_counts` runs dry, for a total vocabulary of **593** -- 54x short of
32,000. The ceiling grows about 23 tokens per added sentence here, so a 32k
vocabulary needs on the order of 1,400 sentences, not 100.

What the exercise would report is memorisation. At the merge ceiling the trained
tokenizer scores exactly **1.0000** tokens per word on its own training corpus,
because BPE run to exhaustion is a word tokenizer over the corpus vocabulary. The
same tokenizer scores **3.1519** on ten held-out sentences, of which 53 of 67
word types are unseen -- the case subword tokenization exists for, and the case
the exercise never asks about. The two sides start equal at zero merges and
diverge at every step after.

And the comparison is not available even with the libraries installed, because
the two sides do not encode the same string. `word_counts` matches `[a-zA-Z]+`,
which erases 80 characters of this corpus -- every numeral, so `196 BC`, `1969`,
`8,849` and `1928` are simply gone -- while tiktoken is byte-level and lossless
by construction. Chars-per-token compares 1341 surviving characters against 1670
actual ones.

Structure: `TRAIN` and `HELD` are one sentence per line; `cost` totals encoded
pieces for a word-count table; `curve` records tokens-per-word on both sides
across merge counts; `ceilings` reports the largest vocabulary a prefix of the
corpus can reach.
"""

from __future__ import annotations

import importlib.util
import re

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "19-subword-tokenization"

UNAVAILABLE = ("tiktoken", "sentencepiece", "transformers")
ASKED_VOCAB = 32000
TRAIN = """The Rosetta Stone is a granodiorite stele inscribed with a decree issued in Memphis in 196 BC.
Photosynthesis converts light energy into chemical energy stored in carbohydrate molecules.
The Apollo 11 mission landed the first humans on the Moon on 20 July 1969.
Mount Everest, at 8,849 metres, is Earth's highest mountain above sea level.
The printing press was introduced to Europe by Johannes Gutenberg around 1440.
The Amazon rainforest covers roughly 5,500,000 square kilometres across nine nations.
Penicillin was discovered by Alexander Fleming in 1928 at St Mary's Hospital.
The Pacific Ocean is the largest and deepest of Earth's oceanic divisions.
DNA carries genetic instructions for the development and functioning of living organisms.
The Industrial Revolution transformed manufacturing processes between 1760 and 1840.
The Sahara is the largest hot desert in the world, covering much of North Africa.
The human brain contains approximately 86 billion neurons connected by synapses.
Antarctica is the coldest, driest and windiest of Earth's seven continents.
Plate tectonics explains the large-scale motion of Earth's lithosphere over geological time.
The Nile is a north-flowing river in northeastern Africa, about 6,650 kilometres long.
The Roman Empire reached its greatest territorial extent under the emperor Trajan.
Evolution by natural selection was described by Charles Darwin and Alfred Russel Wallace.
The atmosphere of Earth is composed mostly of nitrogen and oxygen by volume.
The Mediterranean Sea connects to the Atlantic Ocean through the Strait of Gibraltar.
Glaciers form where snowfall exceeds melting over many years and compacts into ice."""
HELD = """The Colosseum in Rome is the largest ancient amphitheatre ever built.
Ribosomes translate messenger RNA into chains of amino acids inside the cell.
The Hubble Space Telescope was launched into low Earth orbit in 1990.
Coral reefs support about a quarter of all marine species despite covering little seabed.
Superconductors conduct electric current with exactly zero electrical resistance.
The Magna Carta was agreed by King John of England in June 1215.
Cryptography protects information by transforming it into an unreadable form."""


def cost(ref, counts, merges):
    """Total encoded pieces for a word-count table under one merge list."""
    return sum(len(ref.encode_bpe(word, merges)) * freq for word, freq in counts.items())


def curve(ref, train, held, ceiling):
    """Tokens per word on both sides, across merge counts up to the ceiling."""
    rows = {}
    for n in (0, 100, 200, 400, ceiling):
        merges = ref.train_bpe(TRAIN, n)[0] if n else []
        rows[n] = tuple(round(cost(ref, c, merges) / sum(c.values()), 4) for c in (train, held))
    return rows


def ceilings(ref):
    """Largest vocabulary reachable from the first k sentences."""
    lines = TRAIN.splitlines()
    out = {}
    for k in (6, 13, 20):
        text = "\n".join(lines[:k])
        merges = ref.train_bpe(text, 10**6)[0]
        alphabet = {char for word in ref.word_counts(text) for char in word}
        out[k] = len(alphabet) + 1 + len(merges)
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    train, held = ref.word_counts(TRAIN), ref.word_counts(HELD)
    merges = ref.train_bpe(TRAIN, 10**6)[0]
    alphabet = {char for word in train for char in word} | {"</w>"}
    reach = ceilings(ref)
    kept = sum(len(word) * freq for word, freq in train.items())
    return {
        "absent": [n for n in UNAVAILABLE if importlib.util.find_spec(n) is None],
        "merges": len(merges),
        "vocab": len(alphabet) + len(merges),
        "asked": ASKED_VOCAB,
        "reach": reach,
        "per_sentence": round((reach[20] - reach[13]) / 7, 1),
        "rows": curve(ref, train, held, len(merges)),
        "unseen": sum(1 for word in held if word not in train),
        "held_types": len(held),
        "raw_chars": len(TRAIN),
        "kept_chars": kept,
        "dropped": len([c for c in TRAIN if not c.isalpha() and not c.isspace()]),
        "numerals": re.findall(r"[0-9][0-9,]*", TRAIN),
        "chars_per_token": round(kept / cost(ref, train, merges), 4),
    }


def verify(result):
    top, rows = result["merges"], result["rows"]
    needed = round(result["asked"] / result["per_sentence"] / 100) * 100
    return [
        practice.Check(
            "ANSWER: the 32k arm is not a setting this corpus can honour",
            result["vocab"] < result["asked"],
            f"{result['absent']} are all absent, so the two OpenAI arms cannot be measured -- but "
            f"the third cannot be built either: 20 sentences admit {top} merges before "
            f"`pair_counts` runs dry, a total vocabulary of {result['vocab']} against the "
            f"{result['asked']} requested. Vocabulary size is a ceiling the corpus sets",
        ),
        practice.Check(
            "MECHANISM: and the ceiling grows with text, not with the request",
            result["reach"][6] < result["reach"][13] < result["reach"][20],
            f"the first 6, 13 and 20 sentences reach {list(result['reach'].values())} -- about "
            f"{result['per_sentence']} tokens per added sentence. At that rate 32,000 needs on "
            f"the order of {needed} sentences, against the 100 the exercise specifies",
        ),
        practice.Check(
            "FINDING: the ratio the exercise reports is memorisation",
            rows[top][0] == 1.0 and rows[top][1] > rows[top][0],
            f"at the ceiling the tokenizer scores exactly {rows[top][0]} tokens per word on its "
            f"own training text -- BPE run to exhaustion is a word tokenizer over the corpus "
            f"vocabulary -- and {rows[top][1]} on ten held-out sentences. The compression ratio "
            "measured where the exercise measures it says the text was in the training set",
        ),
        practice.Check(
            "MECHANISM: the gap opens as training continues, it does not close",
            rows[100][1] - rows[100][0] < rows[top][1] - rows[top][0],
            f"train and held-out tokens per word run {[rows[n] for n in (0, 100, 200, 400)]} and "
            f"{rows[top]}: identical at zero merges, then diverging at every step. "
            f"{result['unseen']} of {result['held_types']} held-out word types are unseen, which "
            "is the case subword tokenization exists for, and it is the case never reported",
        ),
        practice.Check(
            "FINDING: the two sides do not encode the same string, so no ratio compares them",
            result["kept_chars"] < result["raw_chars"],
            f"`word_counts` matches `[a-zA-Z]+`, dropping {result['dropped']} non-alphabetic "
            f"characters including every numeral -- {result['numerals'][:5]} and the rest. "
            f"Chars-per-token here is {result['chars_per_token']} over {result['kept_chars']} "
            f"surviving characters of {result['raw_chars']}; tiktoken is byte-level and lossless. "
            "The ratios would be computed against different inputs",
        ),
        practice.Check(
            "CONTROL: at zero merges the two sides agree, which is what makes the rest a gap",
            abs(rows[0][0] - rows[0][1]) < 0.1,
            f"a character tokenizer scores {rows[0][0]} and {rows[0][1]} tokens per word, within "
            "1% -- mean word length plus the `</w>` marker, and English is English on both sides. Every "
            "later difference is the merge list fitting the training corpus and nothing else",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
