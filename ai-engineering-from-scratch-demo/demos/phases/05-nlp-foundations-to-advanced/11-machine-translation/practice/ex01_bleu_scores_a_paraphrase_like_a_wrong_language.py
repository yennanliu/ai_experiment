"""Exercise 1 — BLEU scores a paraphrase like a wrong language.

    **Easy.** Translate a 5-sentence English paragraph to French and back to
    English using `nllb-200-distilled-600M`. Measure how close the round-trip is
    to the original. You should see semantic preservation with word-choice drift.

Reading of the exercise: transformers and torch are absent and no NLLB
checkpoint is downloadable here, so the five round-tripped sentences are written
below rather than generated -- each one the original with the word-choice drift
the exercise predicts, `approved` becoming `accepted`, `arrived at` becoming
`got to`. Three controls sit beside them: the sentence returned unchanged, an
off-target French generation, and an unrelated English sentence.

The measurement is where this goes. Of the five drifted sentences, three score
`simple_bleu` **0.0** -- and so do the off-target French output and the sentence
about bananas. BLEU as the lesson ships it gives one number, zero, to a correct
paraphrase, to a translation into the wrong language, and to a sentence with no
relation to the source at all. The other metric in the same file separates all
eight: chrF puts the drifted sentences at 52.1 to 80.5 and the off-target and
unrelated ones at 40.9 and 16.9, with no overlap between the groups.

The cause is in the lesson's own note. `simple_bleu` has no smoothing, so a
single zero-precision order zeroes the geometric mean, and word-choice drift
destroys 4-grams first. It is not that BLEU scores these low; it is that BLEU
cannot score them at all, and the exercise asks for exactly the property it is
blind to.

Structure: `PARAGRAPH` holds the five originals with their round trips and the
three controls, each labelled by kind. `score` runs the lesson's own
`simple_bleu` and `chrf` over one pair; `by_kind` collects the two metrics per
label so the groups can be compared rather than the sentences.
"""

from __future__ import annotations

import importlib.util

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "11-machine-translation"

SOURCE = "The committee approved the proposal without further discussion."
PARAGRAPH = (
    (SOURCE, "The committee accepted the proposal without additional discussion.", "drift"),
    ("She arrived at the station shortly before the train departed.",
     "She got to the station just before the train left.", "drift"),
    ("The report was published on the first of March.",
     "The report was released on March the first.", "drift"),
    ("Heavy rain delayed the flight by two hours.",
     "Heavy rain postponed the flight by two hours.", "drift"),
    ("He explained the decision to the assembled journalists.",
     "He explained the decision to the gathered reporters.", "drift"),
    (SOURCE, SOURCE, "identical"),
    (SOURCE, "Le comite a approuve la proposition sans discussion.", "off-target"),
    (SOURCE, "Bananas are grown in tropical climates near the equator.", "unrelated"),
)
KINDS = ("drift", "identical", "off-target", "unrelated")
MODELS = ("transformers", "torch", "sacrebleu", "ctranslate2")


def score(ref, original, returned) -> dict:
    return {"bleu": round(ref.simple_bleu(returned, original), 1),
            "chrf": round(ref.chrf(returned, original), 1)}


def by_kind(ref) -> dict:
    rows = {kind: [] for kind in KINDS}
    for original, returned, kind in PARAGRAPH:
        rows[kind].append(score(ref, original, returned))
    return rows


def span(group, key="chrf") -> tuple:
    return min(row[key] for row in group), max(row[key] for row in group)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = by_kind(ref)
    return {
        "missing": [m for m in MODELS if importlib.util.find_spec(m) is None],
        "rows": rows, "sentences": len(PARAGRAPH), "note": ref.simple_bleu_note(),
        "zeroed": {kind: sum(row["bleu"] == 0.0 for row in group) for kind, group in rows.items()},
        "sizes": {kind: len(group) for kind, group in rows.items()},
        "drift_chrf": span(rows["drift"]),
        "wrong_chrf": span(rows["off-target"] + rows["unrelated"]),
        "scored": [row["bleu"] for row in rows["drift"] if row["bleu"] > 0],
    }


def verify(result):
    rows, zeroed, sizes = result["rows"], result["zeroed"], result["sizes"]
    drift_lo, drift_hi = result["drift_chrf"]
    wrong_lo, wrong_hi = result["wrong_chrf"]
    return [
        practice.Check(
            "ANSWER: three of five round trips score BLEU 0.0, and so do both wrong controls",
            zeroed["drift"] == 3 and zeroed["off-target"] == zeroed["unrelated"] == 1,
            f"{result['missing']} are absent and no NLLB checkpoint is downloadable, so the round "
            f"trips are written rather than generated. `simple_bleu` returns 0.0 on "
            f"{zeroed['drift']} of {sizes['drift']} drifted sentences, on the off-target French "
            f"output and on the unrelated one: {rows['drift']} against {rows['off-target']} and "
            f"{rows['unrelated']}"),
        practice.Check(
            "MECHANISM: one number for a good paraphrase, a wrong language and an unrelated sentence",
            zeroed["drift"] + zeroed["off-target"] + zeroed["unrelated"] == 5,
            f"five of the eight pairs score exactly 0.0, and they include the three the exercise "
            f"expects to preserve meaning. A metric that cannot order a paraphrase above a sentence "
            f"in the wrong language is not measuring how close the round trip is; it is reporting "
            f"that some 4-gram is missing"),
        practice.Check(
            "FINDING: chrF separates the two groups with no overlap",
            drift_lo > wrong_hi,
            f"the drifted sentences score chrF {drift_lo} to {drift_hi} and the off-target and "
            f"unrelated ones {wrong_lo} to {wrong_hi} -- every good round trip above every bad one. "
            f"The same file ships both metrics and only one of them answers the question the "
            f"exercise asks"),
        practice.Check(
            "MECHANISM: the lesson says why, in the function next to it",
            "smoothing" in result["note"] and "0.0" in result["note"],
            f"`simple_bleu_note()` reads: {result['note']!r}. Word-choice drift destroys 4-grams "
            f"first, so the highest order goes to zero while unigram and bigram precision stay "
            f"high, and the unsmoothed geometric mean takes the whole score with it"),
        practice.Check(
            "FINDING: the two round trips BLEU does score are the ones that changed one word",
            len(result["scored"]) == 2 and min(result["scored"]) > 50,
            f"the drifted sentences BLEU scores above zero are {result['scored']} -- both of them "
            f"single-word substitutions that leave most 4-grams intact. What survives the metric is "
            f"not what preserved the meaning, it is what preserved the word order"),
        practice.Check(
            "CONTROL: on an identical round trip both metrics agree at 100",
            rows["identical"][0] == {"bleu": 100.0, "chrf": 100.0},
            f"the unchanged sentence scores {rows['identical'][0]} on both, so neither metric is "
            f"broken and the disagreement below is about drift specifically. That is the case the "
            f"exercise says will not occur -- 'semantic preservation with word-choice drift' is "
            f"everything except this row"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
