"""Exercise 5 — resampling measures the branch table, and never flags a wrong field.

    Add "confidence scores" to your extraction pipeline. For each extracted
    field, estimate how confident the model is (based on token probabilities,
    or by running extraction 3 times and measuring consistency). Flag
    low-confidence fields for human review.

Reading of the exercise: the first option is closed here, so the second is
implemented exactly as written -- three extractions per input, per-field
agreement as the confidence, and anything below 1.0 flagged. The corpus is
twenty labelled descriptions generated from a template table, ten of which the
lesson's branch table matches and ten of which it does not.

**ANSWER: nothing is ever flagged.** `simulate_llm_extraction` is a pure
function of `(text, attempt)`, so three runs at the same attempt are
bit-identical and every field on every input scores confidence 1.00. The
mechanism the exercise offers as a fallback measures determinism, and this
extractor is deterministic.

**FINDING: the first option is not available.** The extractor returns a `str`
-- there are no token probabilities anywhere in the lesson to read -- so
"based on token probabilities" has nothing to attach to.

**FINDING: sweeping `attempt` 0-1-2, the only variation the simulator has,
moves exactly one field on exactly one family.** The Sony branch drops
`categories` at `attempt >= 1`, so that field scores 0.67 -- majority agreement
over three runs -- and everything else stays 1.00. Across the corpus that flags
4 fields of 80. What the confidence measures is the branch table.

**FINDING: consistency cannot see a confidently wrong field.** On the ten
descriptions the branch table misses, the fallback returns
`{"product": "Unknown", ...}` three times out of three -- confidence 1.00 on a
field that is certainly wrong. Across the corpus 0 fields of 80 are flagged and
50 are wrong, so the flag and the error have no overlap at all.

**CONTROL: perturb the input instead of the model.** Three paraphrases per
description -- lowercased, brand removed, a distractor category appended -- flag
32 fields, of which 28 are wrong: four false positives against resampling's
zero true ones. Consistency is informative when what varies is the input the
extractor actually reads, and uninformative when it is the seed of a pure
function.

Structure: `make_corpus` generates the labelled pairs, `confidence` is the
exercise's three-run agreement, and `PERTURB` is the control's paraphrases.
"""

from __future__ import annotations

import collections
import contextlib
import io
import json

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "03-structured-outputs"
FIELDS = ["product", "price", "in_stock", "categories"]
CATALOGUE = [("Sony WH-1000XM5", "audio", "headphones"),
             ("MacBook Pro 16", "computers", "laptop"),
             ("Keychron Q1", "peripherals", "keyboard"),
             ("Dell U2724D", "monitors", "monitor"),
             ("Anker 737", "power", "power bank")]
PERTURB = [lambda t, n: t.lower(),
           lambda t, n: t.replace(n, "it"),
           lambda t, n: t + " Compare it to a keyboard."]


def make_corpus(n=20):
    corpus = []
    for i in range(n):
        name, category, noun = CATALOGUE[i % len(CATALOGUE)]
        text = (f"The {name} {noun} is priced at ${99 + 50 * (i % 7)} and is "
                f"{'available now' if i % 2 == 0 else 'sold out'}.")
        corpus.append((text, name, {"product": name, "price": float(99 + 50 * (i % 7)),
                                    "in_stock": i % 2 == 0, "categories": [category]}))
    return corpus


def extract(ref, text, attempt=0):
    with contextlib.redirect_stdout(io.StringIO()):
        raw = ref.simulate_llm_extraction(text, ref.PRODUCT_SCHEMA, attempt)
    return json.loads(raw)


def confidence(runs):
    """Per-field agreement over the runs: the exercise's second option, as written."""
    scores = {}
    for key in FIELDS:
        seen = collections.Counter(json.dumps(run.get(key), sort_keys=True) for run in runs)
        scores[key] = round(seen.most_common(1)[0][1] / len(runs), 2)
    return scores


def sweep(ref, text, name, mode):
    """Three runs: same input three times, three attempts, or three paraphrases."""
    if mode == "resample":
        return [extract(ref, text) for _ in range(3)]
    if mode == "attempt":
        return [extract(ref, text, a) for a in (0, 1, 2)]
    return [extract(ref, change(text, name)) for change in PERTURB]


def score_corpus(ref, corpus, mode):
    flagged, wrong, both = 0, 0, 0
    for text, name, want in corpus:
        runs = sweep(ref, text, name, mode)
        marks = confidence(runs)
        for key in FIELDS:
            low, bad = marks[key] < 1.0, runs[0].get(key) != want[key]
            flagged, wrong, both = flagged + low, wrong + bad, both + (low and bad)
    return {"flagged": flagged, "wrong": wrong, "both": both}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    corpus = make_corpus()
    sony = corpus[0]
    return {
        "fields": len(corpus) * len(FIELDS),
        "resample": score_corpus(ref, corpus, "resample"),
        "attempt": score_corpus(ref, corpus, "attempt"),
        "perturb": score_corpus(ref, corpus, "perturb"),
        "sony_resample": confidence(sweep(ref, sony[0], sony[1], "resample")),
        "sony_attempt": confidence(sweep(ref, sony[0], sony[1], "attempt")),
        "returns": type(ref.simulate_llm_extraction("sony", ref.PRODUCT_SCHEMA)).__name__,
        "identical": len({json.dumps(r, sort_keys=True)
                          for r in sweep(ref, sony[0], sony[1], "resample")}),
    }


def verify(result):
    resample, attempt, perturb = result["resample"], result["attempt"], result["perturb"]
    return [
        practice.Check(
            "ANSWER: three runs are bit-identical, so nothing is ever flagged",
            all([resample["flagged"] == 0, result["identical"] == 1,
                 set(result["sony_resample"].values()) == {1.0}]),
            f"`simulate_llm_extraction` is a pure function of (text, attempt), so the "
            f"three runs collapse to {result['identical']} distinct document and every "
            f"field scores {sorted(set(result['sony_resample'].values()))}. Across "
            f"{result['fields']} fields, {resample['flagged']} are flagged for review",
        ),
        practice.Check(
            "FINDING: the first option the exercise offers is not available",
            result["returns"] == "str",
            f"`simulate_llm_extraction` returns a {result['returns']}, and no function in "
            "the lesson carries logprobs, so 'based on token probabilities' has nothing "
            "to attach to. Only the consistency branch of the exercise can be built",
        ),
        practice.Check(
            "FINDING: sweeping attempt moves one field on one family",
            all([result["sony_attempt"]["categories"] == 0.67,
                 attempt["flagged"] < result["fields"] // 4]),
            f"attempts 0, 1, 2 are the only variation the simulator has, and the Sony "
            f"branch uses it to drop `categories`: {result['sony_attempt']}. Across the "
            f"corpus that flags {attempt['flagged']} of {result['fields']} fields. The "
            "confidence is a measurement of the branch table",
        ),
        practice.Check(
            "FINDING: consistency cannot see a confidently wrong field",
            all([resample["wrong"] > 0, resample["both"] == 0]),
            f"{resample['wrong']} of {result['fields']} fields are wrong and "
            f"{resample['flagged']} are flagged, overlap {resample['both']}. On the "
            "descriptions the branch table misses, the fallback returns 'Unknown' three "
            "times of three -- confidence 1.00 on a field that is certainly wrong",
        ),
        practice.Check(
            "CONTROL: perturbing the input flags fields, and they are the wrong ones",
            all([perturb["flagged"] > 30, perturb["both"] > 0.8 * perturb["flagged"]]),
            f"three paraphrases -- lowercased, brand removed, distractor appended -- flag "
            f"{perturb['flagged']} fields, of which {perturb['both']} are wrong: "
            f"{perturb['flagged'] - perturb['both']} false positives, against "
            f"{resample['both']} true positives for resampling. "
            "Consistency is informative when what varies is the input the extractor "
            "actually reads, and uninformative when it is the seed of a pure function",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
