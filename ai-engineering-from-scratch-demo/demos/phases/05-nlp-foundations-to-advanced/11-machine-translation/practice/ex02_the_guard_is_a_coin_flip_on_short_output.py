"""Exercise 2 — the guard is a coin flip on short output.

    **Medium.** Implement a language-ID check on translation outputs using
    `fasttext lid.176` or `langdetect`. Integrate into the MT call so off-target
    generations are caught before returning.

Reading of the exercise: neither library is installed and neither model is
downloadable, so the identifier is built here -- character trigram profiles over
four short language samples, scored by mean log probability with a floor. On
held-out sentences at full length it is right 20 times out of 20 across English,
French, Spanish and German, so a guard on ordinary output is sound.

The integration the exercise asks for is where it stops being sound. Truncate
the same sentences and the identifier degrades faster than the intuition
suggests: 1.0000 at 32 characters, 0.8000 at 16, 0.5500 at 8 and 0.5000 at 4,
against a chance level of 0.2500. A guard placed on the MT call sees whatever
the model emitted, and a model that has gone off target frequently emits very
little -- a copied source fragment, a repeated token, an empty string. The
shortest outputs are both the most likely to be wrong and the ones the check
cannot read.

The cost is symmetric and worth stating as a rate. At 8 characters the
identifier is wrong on 9 of 20 held-out sentences, and 3 of those 9 are correct
English outputs called something else -- 3 of the 5 English sentences in the set.
Integrating this check without a length floor rejects most good short
translations to catch the bad ones. The margin between the top two languages is
the readable signal -- 0.4762 at 8 characters against 0.8619 at full length -- so
the guard can tell when it is guessing, and the exercise's phrasing ("caught
before returning") has no room for that third outcome.

Structure: `SAMPLES` are the four training strings and `HELD_OUT` the labelled
test sentences, all ASCII so no encoding difference does the work. `profile`
builds a trigram distribution, `identify` returns the best language and its
margin over the runner-up, and `sweep` scores the whole held-out set at a
sequence of truncation lengths.
"""

from __future__ import annotations

import collections
import importlib.util
import math

from harness import practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "11-machine-translation"

ORDER, FLOOR = 3, 1e-4
PREFIXES = (4, 8, 16, 32, 1000)
LIBRARIES = ("fasttext", "langdetect", "langid", "pycld3")
SAMPLES = {
    "en": "the quick brown fox jumps over the lazy dog and the committee approved the proposal "
          "without further discussion she arrived at the station shortly before the train departed "
          "the report was published on the first of march heavy rain delayed the flight",
    "fr": "le comite a approuve la proposition sans discussion supplementaire elle est arrivee a la "
          "gare peu avant le depart du train le rapport a ete publie le premier mars la pluie a "
          "retarde le vol de deux heures les chats courent dans le jardin",
    "es": "el comite aprobo la propuesta sin mas discusion ella llego a la estacion poco antes de "
          "que saliera el tren el informe fue publicado el primero de marzo la lluvia retraso el "
          "vuelo dos horas los gatos corren en el jardin con los perros",
    "de": "der ausschuss genehmigte den vorschlag ohne weitere diskussion sie kam kurz vor der "
          "abfahrt des zuges am bahnhof an der bericht wurde am ersten maerz veroeffentlicht "
          "starker regen verzoegerte den flug um zwei stunden die katzen laufen im garten",
}
HELD_OUT = {
    "en": ("the meeting starts at nine", "please send the documents",
           "he left without saying anything", "we agreed on the terms",
           "the results were published"),
    "fr": ("la reunion commence a neuf heures", "veuillez envoyer les documents",
           "il est parti sans rien dire", "nous sommes d accord sur les termes",
           "les resultats ont ete publies"),
    "es": ("la reunion empieza a las nueve", "por favor envie los documentos",
           "se fue sin decir nada", "acordamos los terminos", "los resultados fueron publicados"),
    "de": ("die sitzung beginnt um neun", "bitte senden sie die dokumente",
           "er ging ohne etwas zu sagen", "wir haben uns auf die bedingungen geeinigt",
           "die ergebnisse wurden veroeffentlicht"),
}


def profile(text) -> dict:
    counts = collections.Counter(text[i:i + ORDER] for i in range(len(text) - ORDER + 1))
    total = sum(counts.values())
    return {gram: n / total for gram, n in counts.items()}


def identify(models, text) -> tuple:
    grams = [text[i:i + ORDER] for i in range(len(text) - ORDER + 1)]
    if not grams:
        return None, 0.0
    scores = {lang: sum(math.log(table.get(g, FLOOR)) for g in grams) / len(grams)
              for lang, table in models.items()}
    ranked = sorted(scores.values(), reverse=True)
    return max(scores, key=scores.get), round(ranked[0] - ranked[1], 4)


def sweep(models, cut) -> dict:
    hits, margins, wrong = 0, [], collections.Counter()
    for language, rows in HELD_OUT.items():
        for sentence in rows:
            guess, margin = identify(models, sentence[:cut])
            hits += guess == language
            margins.append(margin)
            if guess != language:
                wrong[language] += 1
    total = sum(len(rows) for rows in HELD_OUT.values())
    return {"accuracy": round(hits / total, 4), "wrong": total - hits,
            "margin": round(sum(margins) / len(margins), 4),
            "false_alarms": wrong["en"], "total": total}


def solve():
    models = {lang: profile(text) for lang, text in SAMPLES.items()}
    return {
        "missing": [m for m in LIBRARIES if importlib.util.find_spec(m) is None],
        "grid": {cut: sweep(models, cut) for cut in PREFIXES},
        "chance": round(1 / len(SAMPLES), 4), "languages": sorted(SAMPLES),
        "empty": identify(models, ""),
    }


def verify(result):
    grid, chance = result["grid"], result["chance"]
    full, short = grid[PREFIXES[-1]], grid[PREFIXES[1]]
    accuracies = [grid[cut]["accuracy"] for cut in PREFIXES]
    return [
        practice.Check(
            "ANSWER: at full length the guard is right on every held-out sentence",
            full["accuracy"] == 1.0,
            f"{result['missing']} are all absent, so the identifier is character trigram profiles "
            f"over four short samples. On {full['total']} held-out sentences across "
            f"{result['languages']} it scores {full['accuracy']} with a mean margin of "
            f"{full['margin']} over the runner-up. A guard on ordinary MT output is sound"),
        practice.Check(
            "MECHANISM: it degrades with output length, and fast",
            accuracies == sorted(accuracies) and grid[4]["accuracy"] < 0.6,
            f"truncating the same sentences gives accuracy {dict(zip(PREFIXES, accuracies))} at "
            f"{list(PREFIXES)} characters, against a chance level of {chance}. The check reads "
            f"whatever the model emitted, and there is no length at which it is told how much to "
            f"trust itself"),
        practice.Check(
            "FINDING: the shortest outputs are the ones most likely to need the guard",
            grid[8]["accuracy"] < 0.6,
            f"a model that has gone off target frequently emits very little -- a copied source "
            f"fragment, a repeated token, an empty string. At 8 characters the identifier scores "
            f"{grid[8]['accuracy']}, so the outputs most likely to be wrong are the ones the check "
            f"cannot read"),
        practice.Check(
            "FINDING: the cost is symmetric -- good translations get rejected too",
            short["false_alarms"] > 0,
            f"at {PREFIXES[1]} characters the identifier is wrong on {short['wrong']} of "
            f"{short['total']} sentences, and {short['false_alarms']} of those are correct English "
            f"outputs called something else. Integrating this check without a length floor rejects "
            f"good translations at a rate of "
            f"{short['false_alarms'] / len(HELD_OUT['en']):.0%} to catch the bad ones"),
        practice.Check(
            "MECHANISM: the margin is the signal the guard could have used and the exercise has no slot for",
            short["margin"] < full["margin"],
            f"the gap between the best and second-best language falls from {full['margin']} at full "
            f"length to {short['margin']} at {PREFIXES[1]} characters. The identifier knows when it "
            f"is guessing; 'caught before returning' is a two-way decision and this is a three-way "
            f"one -- pass, reject, or decline to answer"),
        practice.Check(
            "CONTROL: an empty generation has no trigrams and no answer at all",
            result["empty"] == (None, 0.0),
            f"on an empty string the identifier returns {result['empty']} -- there is nothing to "
            f"profile. An MT call that returns empty is the clearest off-target failure there is, "
            f"and it is the one input for which a language check has no output to give"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
