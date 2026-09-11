"""Exercise 3 — the answer is a lookup and the ties decide it.

    **Hard.** Compare English-source and Hindi-source fine-tuning for a Hindi
    classification task. Use 500 target-language examples for few-shot
    fine-tuning under both regimes. Report which source produces better Hindi
    accuracy and by how much. This is the LANGRANK thesis in miniature.

Reading of the exercise: the comparison it describes is already answered by the
lesson's table, without training anything. `simulate_transfer_accuracy` is a
deterministic function of `similarity`, so a Hindi target with a Hindi-family
source returns 0.9000 and with an English source returns 0.4500 -- "by how much"
is 0.4500, available from a dictionary lookup, and the 500 examples change
nothing because the function does not take them.

That is the exercise being true to its source rather than a defect: the LANGRANK
thesis is that source-language choice can be predicted from typological features
instead of measured, and the lesson implements exactly that prediction. What the
implementation adds, and the exercise inherits, is a resolution problem
underneath it. Three features give four similarity levels, so the ranking over
ten candidate sources contains large ties -- Urdu's best source is a three-way
tie, Japanese's a four-way one -- and `rank_source_languages` breaks them with a
stable sort, which means by the order of the candidate list. Reverse the
candidate list and Japanese's recommended source changes from Hindi to Urdu, with
no change to any feature.

For the exercise's own target the answer is stable: Hindi's best source is
Marathi at similarity 1.0000 under either ordering, because only one language
matches all three features. But "Marathi is the best source for Hindi" and
"Hindi and Marathi are the same language" are the same statement in this table,
so the recommendation the exercise asks for is a restatement of the fact that
the feature set cannot separate them.

Structure: `rankings` reads the lesson's own `rank_source_languages` over every
target under two candidate orderings. `tie_widths` counts how many sources share
the top score for each target, and `gap` reads the English-versus-best
comparison the exercise asks for directly off the table.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "18-multilingual-nlp"

TARGET, SOURCE, EXAMPLES = "hindi", "english", 500


def rankings(ref, languages, order) -> dict:
    return {target: ref.rank_source_languages(target, order) for target in languages}


def tie_widths(ranked) -> dict:
    widths = {}
    for target, rows in ranked.items():
        best = rows[0][1]
        widths[target] = len([name for name, score in rows if score == best])
    return widths


def winners(ranked) -> dict:
    return {target: rows[0][0] for target, rows in ranked.items()}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    languages = list(ref.LANGUAGE_FEATURES)
    forward = rankings(ref, languages, languages)
    reverse = rankings(ref, languages, list(reversed(languages)))
    best = forward[TARGET][0]
    return {
        "languages": len(languages), "examples": EXAMPLES,
        "best_source": best[0], "best_similarity": round(best[1], 4),
        "best_accuracy": round(ref.simulate_transfer_accuracy(TARGET, best[0]), 4),
        "english_similarity": round(ref.similarity(TARGET, SOURCE), 4),
        "english_accuracy": round(ref.simulate_transfer_accuracy(TARGET, SOURCE), 4),
        "ties": tie_widths(forward),
        "flipped": [t for t in languages if winners(forward)[t] != winners(reverse)[t]],
        "winners": {t: (winners(forward)[t], winners(reverse)[t])
                    for t in languages if winners(forward)[t] != winners(reverse)[t]},
        "identical": ref.similarity(TARGET, best[0]) == 1.0,
        "arity": ref.simulate_transfer_accuracy.__code__.co_argcount,
        "signature": list(ref.simulate_transfer_accuracy.__code__.co_varnames[:2]),
    }


def verify(result):
    gap = round(result["best_accuracy"] - result["english_accuracy"], 4)
    ties, flipped = result["ties"], result["flipped"]
    return [
        practice.Check(
            "ANSWER: 0.4500, and it comes from a dictionary lookup rather than a fine-tune",
            gap == 0.45 and result["arity"] == 2,
            f"for a {TARGET} target the lesson's table gives its best source "
            f"{result['best_source']!r} at similarity {result['best_similarity']} and accuracy "
            f"{result['best_accuracy']}, against {SOURCE!r} at {result['english_similarity']} and "
            f"{result['english_accuracy']} -- a gap of {gap}. "
            f"`simulate_transfer_accuracy{tuple(result['signature'])}` takes "
            f"{result['arity']} arguments and neither is the {result['examples']} examples, so the "
            f"number does not move when they are supplied"),
        practice.Check(
            "MECHANISM: that is the LANGRANK thesis implemented, not a shortcut around it",
            result["best_similarity"] > result["english_similarity"],
            f"the thesis is that source-language choice can be predicted from typological features "
            f"instead of measured, and the lesson implements that prediction directly. The exercise "
            f"asks for the comparison the prediction already makes, so the honest answer to 'report "
            f"which source and by how much' is to read the table"),
        practice.Check(
            "FINDING: underneath it the ranking is mostly ties",
            max(ties.values()) >= 4 and ties[TARGET] == 1,
            f"three features give four similarity levels, so the top of each ranking is shared: "
            f"{ties} sources tie for first, per target. Urdu has a three-way tie and Japanese a "
            f"four-way one; {TARGET} is one of the few with a unique best source"),
        practice.Check(
            "MECHANISM: and the tie-break is the order of the candidate list",
            flipped,
            f"`rank_source_languages` sorts by score with a stable sort, so equal scores keep their "
            f"input order. Reversing the candidate list changes the recommended source for "
            f"{flipped}: {result['winners']}. No feature changed"),
        practice.Check(
            "FINDING: for this target the recommendation restates the table's own blind spot",
            result["identical"] and result["best_similarity"] == 1.0,
            f"{result['best_source']!r} scores {result['best_similarity']} against {TARGET}, which "
            f"means the feature set finds no difference between them at all. 'Marathi is the best "
            f"source for Hindi' and 'this table cannot tell Hindi from Marathi' are the same "
            f"statement, so the recommendation is stable for the reason that makes it least "
            f"informative"),
        practice.Check(
            "CONTROL: the English comparison is the one the table can make confidently",
            result["english_similarity"] == 0.0,
            f"{SOURCE!r} shares none of the three features with {TARGET} -- word order, script and "
            f"family all differ -- so it sits at the bottom of the scale with no tie to break and "
            f"no resolution needed. The half of the exercise's comparison that is trustworthy here "
            f"is the half where the answer was obvious"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
