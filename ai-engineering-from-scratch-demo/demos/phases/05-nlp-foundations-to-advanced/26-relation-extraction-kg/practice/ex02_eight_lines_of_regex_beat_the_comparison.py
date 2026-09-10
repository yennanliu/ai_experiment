"""Exercise 2 — eight lines of regex beat the comparison.

    **Medium.** Use REBEL (or a small LLM) on the same sentences. Compare
    triples. Which extractor has higher precision? Higher recall?

Reading of the exercise: `transformers` is absent and no LLM is reachable, so
REBEL cannot be run. What the question is really asking -- whether the choice of
*system* is what sets precision and recall -- can be answered without it, by
moving the shipped extractor along its own frontier.

Two edits to `code/main.py`'s six patterns. A **guard** drops any match in a
sentence containing a denial or hedge cue. A **relaxation** lets the object run
over consecutive capitalised words instead of stopping after one, and adds a
passive form and a relative-clause form. Over exercise 1's thirteen sentences:

| variant | precision | recall | F1 |
|---|---|---|---|
| as shipped | 0.5000 | 0.4545 | 0.4762 |
| + guard | **0.8333** | 0.4545 | 0.5882 |
| + relaxation | 0.6667 | **0.7273** | 0.6957 |
| both | **1.0000** | **0.7273** | **0.8421** |

The shipped extractor is dominated by a variant built from its own six patterns:
precision doubles, recall rises by 60%, and F1 goes from 0.4762 to 0.8421. Which
extractor has higher precision is a question about which regexes someone wrote.

The two edits are independent, which is the part the exercise's framing hides.
The guard moves precision and leaves recall untouched at 0.4545; the relaxation
moves recall and drags precision up as a side effect, because two of its new
matches replace truncated objects that were counted wrong. A system comparison
reports one point per system and cannot separate these.

And the shipped extractor has no confidence score, so there is no operating curve
to compare against -- `extract` returns a list, every element equally asserted.
Comparing two systems at one unknown operating point each is the comparison the
exercise asks for, and it has no answer that survives editing a regex.

Structure: exercise 1's labelled sentences are loaded rather than copied; `CUES`
is the hedge list, `RELAXED` the widened pattern set, `run` applies one variant,
and `score` reports precision, recall and F1.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "26-relation-extraction-kg"

EX1 = practice.load_module(
    pathlib.Path(__file__).resolve().parent / "ex01_precision_is_no_higher_than_recall.py")
ROWS = EX1.ROWS
UNAVAILABLE = ("transformers", "torch", "openai", "anthropic")
CUES = ("denied", "unclear", "whether", "believes", "suggest", "reports", "nobody", " not ")
NAME = r"([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)?)"
PHRASE = r"([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)*)"
RELAXED = tuple((re.compile(pattern), relation) for pattern, relation in (
    (NAME + r" was born in " + PHRASE, "P19"),
    (NAME + r" founded " + PHRASE, "P112"),
    (NAME + r" (?:became|is|was) CEO of " + PHRASE, "P169"),
    (NAME + r" works? (?:at|for) " + PHRASE, "P108"),
    (NAME + r" (?:studied|graduated) (?:at|from) " + PHRASE, "P69"),
    (NAME + r" (?:acquired|bought) " + PHRASE, "P1830"),
    (NAME + r" was (?:acquired|bought) by " + NAME, "~P1830"),
    (NAME + r", who works? (?:at|for) " + PHRASE, "P108"),
))


def matched(pattern, relation, text, index):
    """The triples one pattern contributes to one sentence, reversing marked relations."""
    reversed_form = relation.startswith("~")
    label = relation[1:] if reversed_form else relation
    pairs = [(m.group(2), m.group(1)) if reversed_form else (m.group(1), m.group(2))
             for m in pattern.finditer(text)]
    return {(index, subject, label, obj) for subject, obj in pairs}


def run(ref, guard=False, relaxed=False):
    """Every (sentence, subject, relation, object) one variant returns."""
    patterns = RELAXED if relaxed else tuple(ref.PATTERNS)
    found = set()
    for i, (text, _) in enumerate(ROWS):
        if guard and any(cue in text.lower() for cue in CUES):
            continue
        for pattern, relation in patterns:
            found |= matched(pattern, relation, text, i)
    return found


def score(predicted, gold):
    """Precision, recall and F1 of one variant."""
    hits = len(predicted & gold)
    precision = hits / len(predicted) if predicted else 0.0
    recall = hits / len(gold)
    harmonic = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"predicted": len(predicted), "precision": round(precision, 4),
            "recall": round(recall, 4), "f1": round(harmonic, 4)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    gold = EX1.gold_triples()
    arms = {"as shipped": run(ref), "guard": run(ref, guard=True),
            "relaxation": run(ref, relaxed=True),
            "both": run(ref, guard=True, relaxed=True)}
    return {
        "absent": [n for n in UNAVAILABLE if importlib.util.find_spec(n) is None],
        "gold": len(gold),
        "rows": {name: score(found, gold) for name, found in arms.items()},
        "patterns": len(ref.PATTERNS),
        "relaxed_patterns": len(RELAXED),
        "scores": sorted({round(t.get("score", 0.0), 4) for t in ref.extract(ROWS[0][0])}),
        "keys": sorted(ref.extract(ROWS[0][0])[0]),
    }


def verify(result):
    rows = result["rows"]
    shipped, both = rows["as shipped"], rows["both"]
    return [
        practice.Check(
            "ANSWER: the shipped extractor is dominated by a variant built from its own patterns",
            both["precision"] > shipped["precision"] and both["recall"] > shipped["recall"],
            f"{result['absent']} are all absent, so REBEL cannot be run -- but the question is "
            f"answerable without it. Over exercise 1's sentences: {rows}. Precision doubles, "
            f"recall rises by {round((both['recall'] / shipped['recall'] - 1) * 100)}%, and F1 "
            f"goes {shipped['f1']} to {both['f1']}",
        ),
        practice.Check(
            "MECHANISM: the guard moves precision and leaves recall exactly where it was",
            rows["guard"]["recall"] == shipped["recall"]
            and rows["guard"]["precision"] > shipped["precision"],
            f"dropping any match in a sentence containing {list(CUES[:4])} and three more cues "
            f"takes precision {shipped['precision']} to {rows['guard']['precision']} with recall "
            f"unchanged at {rows['guard']['recall']}. It removes only triples that were wrong",
        ),
        practice.Check(
            "MECHANISM: the relaxation moves recall and drags precision with it",
            rows["relaxation"]["recall"] > shipped["recall"]
            and rows["relaxation"]["precision"] > shipped["precision"],
            f"letting the object run over consecutive capitalised words, plus a passive form and "
            f"a relative-clause form, gives {rows['relaxation']}. Precision rises too because two "
            "new matches replace truncated objects that were being counted wrong",
        ),
        practice.Check(
            "FINDING: so the two edits are independent, and a system comparison cannot see that",
            rows["guard"]["recall"] < rows["relaxation"]["recall"],
            f"{result['patterns']} patterns become {result['relaxed_patterns']}, and the two "
            "changes move different axes. 'Which extractor has higher precision, which higher "
            "recall' reports one point per system and attributes both numbers to the system",
        ),
        practice.Check(
            "FINDING: there is no operating curve to compare against",
            "score" not in result["keys"],
            f"a triple carries {result['keys']} -- no confidence, no threshold. `extract` returns "
            "a list in which every element is equally asserted, so each variant is a single point "
            "and the comparison has no axis to move along",
        ),
        practice.Check(
            "CONTROL: the whole frontier comes from the same six relations",
            result["relaxed_patterns"] - result["patterns"] == 2,
            f"the relaxed set adds {result['relaxed_patterns'] - result['patterns']} patterns and "
            f"widens one capture group. The gain from {shipped['f1']} to {both['f1']} needs no "
            "model, no training data and no new relation type",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
