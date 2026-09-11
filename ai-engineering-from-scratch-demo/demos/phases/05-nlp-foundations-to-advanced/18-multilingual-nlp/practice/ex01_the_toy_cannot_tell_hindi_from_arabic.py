"""Exercise 1 — the toy cannot tell Hindi from Arabic.

    **Easy.** Run the zero-shot classification pipeline on 10 sentences per
    language across English, French, Hindi, and Arabic. Report accuracy on each.
    You should see strong French, decent Hindi, variable Arabic.

Reading of the exercise: there is no classification pipeline in the lesson --
`code/main.py` is a three-feature language similarity table and a formula -- and
transformers is not installed, so no accuracies are reported here. What the
lesson does provide is a prediction about the same three languages, and it
disagrees with the exercise.

`similarity` counts matches over word order, script and family. English against
French is 2 of 3, so `simulate_transfer_accuracy` gives 0.7500. English against
Hindi is 0 of 3 and English against Arabic is 0 of 3, so both give **0.4500** --
the same number. The exercise predicts "decent Hindi, variable Arabic", a
difference; the lesson's own model predicts they are indistinguishable, and the
strong-French half is the only part the two agree on.

That is a property of the feature set rather than of these three languages.
Three binary-ish features admit four similarity values -- 0.0000, 0.3333, 0.6667
and 1.0000 -- and `simulate_transfer_accuracy` maps them one-to-one onto 0.4500,
0.6000, 0.7500 and 0.9000, so the second function carries exactly the information
of the first, rescaled. Eleven languages collapse into seven distinct profiles,
and five pairs -- english/german, french/spanish, french/italian, spanish/italian
and hindi/marathi -- are identical under it, meaning the table cannot say that
Hindi and Marathi are different languages.

The lesson names its own limitation ("real similarity comes from qWALS /
lang2vec, not a 3-feature toy") and then draws a conclusion from the toy anyway
in its last line. The conclusion happens to be right; the exercise's ordering,
drawn from the same source, is not.

Structure: `pairs` enumerates every unordered language pair. `resolution`
collects the distinct values of both functions and checks the map between them;
`profiles` counts how many of the eleven languages the feature set can tell
apart; `english_transfer` reads off the three predictions the exercise names.
"""

from __future__ import annotations

import collections
import itertools

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "18-multilingual-nlp"

NAMED = ("french", "hindi", "arabic")
SOURCE = "english"
UNAVAILABLE = ("transformers", "torch", "sentencepiece")


def pairs(languages) -> list:
    return list(itertools.combinations(languages, 2))


def resolution(ref, languages) -> dict:
    sims = sorted({round(ref.similarity(a, b), 4) for a, b in pairs(languages)})
    accs = sorted({round(ref.simulate_transfer_accuracy(a, b), 4) for a, b in pairs(languages)})
    return {"similarity": sims, "accuracy": accs, "bijection": len(sims) == len(accs)}


def profiles(ref, languages) -> dict:
    seen = collections.Counter(tuple(sorted(ref.LANGUAGE_FEATURES[name].items()))
                               for name in languages)
    return {"languages": len(languages), "distinct": len(seen),
            "identical": [(a, b) for a, b in pairs(languages) if ref.similarity(a, b) == 1.0]}


def english_transfer(ref) -> dict:
    return {name: {"similarity": round(ref.similarity(SOURCE, name), 4),
                   "accuracy": round(ref.simulate_transfer_accuracy(name, SOURCE), 4)}
            for name in NAMED}


def solve():
    import importlib.util
    ref = parity.load_reference(PHASE, LESSON, "main")
    languages = list(ref.LANGUAGE_FEATURES)
    transfer = english_transfer(ref)
    return {
        "unavailable": [m for m in UNAVAILABLE if importlib.util.find_spec(m) is None],
        "resolution": resolution(ref, languages), "profiles": profiles(ref, languages),
        "transfer": transfer, "features": len(ref.LANGUAGE_FEATURES[SOURCE]),
        "symmetric": all(ref.similarity(a, b) == ref.similarity(b, a)
                         for a, b in pairs(languages)),
        "ties": len({row["accuracy"] for row in transfer.values()}),
        "has_pipeline": any(hasattr(ref, name) for name in ("classify", "zero_shot", "pipeline")),
        "exports": sorted(n for n in dir(ref) if not n.startswith("_") and n != "main"),
    }


def verify(result):
    transfer, res, prof = result["transfer"], result["resolution"], result["profiles"]
    return [
        practice.Check(
            "ANSWER: the lesson's own model gives Hindi and Arabic the same number",
            transfer["hindi"]["accuracy"] == transfer["arabic"]["accuracy"]
            < transfer["french"]["accuracy"],
            f"{result['unavailable']} are all absent and the lesson ships no classifier -- it "
            f"exports {result['exports']} -- so no accuracies are measured. Its own prediction for "
            f"an English source reads {transfer}: French {transfer['french']['accuracy']}, Hindi "
            f"and Arabic both {transfer['hindi']['accuracy']}. The exercise predicts a difference "
            f"between the last two"),
        practice.Check(
            "MECHANISM: three features admit four values, and the second function is the first rescaled",
            len(res["similarity"]) == 4 and res["bijection"],
            f"`similarity` counts matches over {result['features']} features, so it takes the "
            f"values {res['similarity']}, and `simulate_transfer_accuracy` maps them one-to-one "
            f"onto {res['accuracy']}. The second function carries exactly the information of the "
            f"first, so calling one a simulated accuracy adds a unit and nothing else"),
        practice.Check(
            "FINDING: eleven languages collapse into seven profiles",
            prof["distinct"] < prof["languages"] and len(prof["identical"]) == 5,
            f"{prof['languages']} languages have {prof['distinct']} distinct feature profiles, and "
            f"{len(prof['identical'])} pairs score similarity 1.0: {prof['identical']}. Under this "
            f"table Hindi and Marathi are the same language, and so are English and German"),
        practice.Check(
            "MECHANISM: so the resolution the exercise's question needs is not available",
            result["ties"] < len(NAMED),
            f"the exercise asks for three accuracies and expects three answers; the model produces "
            f"{result['ties']} distinct values across {list(NAMED)}. Hindi at 0 of "
            f"{result['features']} shared features and Arabic at 0 of {result['features']} cannot "
            f"be ordered by a function of those features"),
        practice.Check(
            "FINDING: similarity is symmetric and transfer is not",
            result["symmetric"],
            f"`similarity(a, b) == similarity(b, a)` for every pair, so the model says an "
            f"English-trained classifier transfers to Hindi exactly as well as a Hindi-trained one "
            f"transfers to English. Zero-shot transfer is not symmetric in practice, and nothing in "
            f"a feature-match count can represent that"),
        practice.Check(
            "CONTROL: the lesson names this limitation and then draws a conclusion from it anyway",
            transfer["french"]["similarity"] > 0.5,
            f"the file's closing note says real similarity comes from qWALS or lang2vec rather than "
            f"a three-feature toy, and its next line concludes that Hindi is a better source than "
            f"English for Marathi. That conclusion is right; the strong-French part of the "
            f"exercise's ordering is also right, at similarity "
            f"{transfer['french']['similarity']}. The part that fails is the part the toy has no "
            f"resolution for"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
