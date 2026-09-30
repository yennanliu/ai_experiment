"""Exercise 3 — the labelled log misses the known mix by TVD 0.43, and benign lookalikes outscore redacted attacks.

    Pull thirty additional fixtures from your own product's logs (redacted) and confirm the category distribution matches what your team intuitively expected.

Reading of the exercise: this repo has no product and no logs, and inventing
new attack text is out of scope. So the 30 lines are a labelled stand-in whose
true mix is known, and the lesson's matcher is tested as the label producer
the exercise relies on. Twenty lines are corpus attacks drawn with seed 82,
with `REDACTED_HARMFUL` replaced by a redactor token such as
`[HARM_CAT_7 request #3]`. Ten are benign support traffic, several of them in
a fixture's own wrapper around harmless content, as real logs contain. Each
line is labelled with the lesson's `score_matrix`, with its own source fixture
masked out. The "intuitive expectation" is the known label of every line,
which is the best case: a team's intuition can only be less accurate.

**ANSWER: no, the distribution does not match.** The labelled histogram is
TVD 0.43 from the known one. The matcher has no "none" output, so all 10
benign lines are labelled as attacks.

**FINDING: on the 20 attacks alone, 7 get the right category.** Role-play
(8 labelled vs 3 true) and context-smuggling (7 vs 3) absorb the rest, and all
3 encoding tricks are labelled something else.

**FINDING: no score floor can fix this.** Benign lookalikes outscore redacted
attacks: an attack beats a benign line in only 33.5% of pairs, and 19 of 20
attacks score below the best benign line.

**FINDING: the lesson's validator refuses a held-out split.**
`multi-turn-ramp` has exactly the minimum of 7 fixtures, so holding one out
raises `ValueError`. Held-out evaluation therefore has to mask columns of the
full corpus, as this file does.

Expected output: four PASS checks.
"""

from __future__ import annotations

import importlib.util
import random
import sys
from collections import Counter

from harness import parity, practice

if importlib.util.find_spec("numpy") is None:  # the lesson's main.py imports it
    raise practice.Skip("needs numpy: uv sync --extra math")

PHASE, LESSON = "19-capstone-projects", "82-jailbreak-taxonomy"
SEED, ATTACKS = 82, 20
BENIGN = [  # support traffic; several reuse a fixture's wrapper around harmless content, as real logs do
    "How do I reset my password?",
    "Summarize this document: ---BEGIN--- Q3 revenue rose 4 percent on subscription growth. ---END---",
    "Translate this Spanish phrase to English: 'donde esta la estacion de tren'.",
    "Decode this base64 config value from our settings page: aGVsbG8gd29ybGQ=",
    "Review this code: # compute the average\nprint(sum(xs) / len(xs))",
    "Write a short story where a wise old wizard teaches the hero to bake bread.",
    "Respond only with JSON listing three fruits.",
    "Complete this sentence: 'The best way to learn a language is'",
    "Earlier in this conversation you gave me a packing list. Please restate it concisely.",
    "Can you cancel my subscription and refund the last invoice?",
]


def load():
    """main.py does `from fixtures import ...`; register the lesson's fixtures module for that import."""
    saved = sys.modules.get("fixtures")
    sys.modules["fixtures"] = fx = parity.load_reference(PHASE, LESSON, "fixtures")
    try:
        return fx, parity.load_reference(PHASE, LESSON, "main")
    finally:
        sys.modules.pop("fixtures") if saved is None else sys.modules.__setitem__("fixtures", saved)


def make_log(corpus):
    """30 labelled lines: 20 corpus attacks as a log redactor leaves them, plus 10 benign lines."""
    rng = random.Random(SEED)
    lines = [{"text": str(r["prompt"]).replace("REDACTED_HARMFUL", f"[HARM_CAT_{rng.randint(1, 13)} request #{n}]"),
              "truth": r["category"], "source": r["id"]} for n, r in enumerate(rng.sample(corpus, ATTACKS), 1)]
    return lines + [{"text": t, "truth": "none", "source": None} for t in BENIGN]


def labels(tax, log):
    """The lesson's scores, with each line's own source fixture masked out (leave-source-out)."""
    fixtures = tax.all()
    rows = tax.score_matrix([x["text"] for x in log])
    for row, x in zip(rows, log):
        row[[f.id == x["source"] for f in fixtures]] = -1.0
    return [(fixtures[j].category, float(row[j])) for row, j in zip(rows, rows.argmax(axis=1))]


def loo_refused(ref, corpus):
    try:
        ref.Taxonomy([r for r in corpus if r["id"] != "mt-01"])
    except ValueError as exc:
        return str(exc)
    return ""


def tvd(p, q):
    return 0.5 * sum(abs(p[k] / p.total() - q[k] / q.total()) for k in p.keys() | q.keys())


def separation(attacks, benign):
    return {"attack_hits": sum(c == truth for c, _, truth in attacks),
            "attack_wins": sum(s > b for _, s, _ in attacks for b in benign) / (len(attacks) * len(benign)),
            "below_best_benign": sum(s < max(benign) for _, s, _ in attacks)}


def solve():
    fx, ref = load()
    corpus = fx.fixtures()
    labelled = labels(ref.Taxonomy(corpus), log := make_log(corpus))
    attacks = [(c, s, x["truth"]) for (c, s), x in zip(labelled, log) if x["source"]]
    benign = [s for (_, s), x in zip(labelled, log) if not x["source"]]
    return {"expected": Counter(x["truth"] for x in log), "labelled": Counter(c for c, _ in labelled),
            "attack_expected": Counter(t for _, _, t in attacks), "attack_labelled": Counter(c for c, _, _ in attacks),
            **separation(attacks, benign), "loo_refused": loo_refused(ref, corpus)}


def verify(result):
    exp, lab = result["expected"], result["labelled"]
    a_exp, a_lab = result["attack_expected"], result["attack_labelled"]
    return [
        practice.Check(
            "ANSWER: no -- the labelled distribution is 0.43 TVD from the known one, and no line is ever 'none'",
            sum(exp.values()) == 30 and exp["none"] == 10 and lab["none"] == 0 and 0.40 < tvd(exp, lab) < 0.46,
            f"known {dict(exp)}; labelled {dict(lab)}; TVD {tvd(exp, lab):.2f}",
        ),
        practice.Check(
            "FINDING: on the 20 attacks alone, 7 get the right category; role-play and smuggling absorb the rest",
            result["attack_hits"] == 7 and (a_lab["role-play"], a_lab["context-smuggling"]) == (8, 7)
            and a_lab["encoding-trick"] == 0 and a_exp["encoding-trick"] == 3,
            f"right {result['attack_hits']}/20; known {dict(a_exp)}; labelled {dict(a_lab)}; TVD {tvd(a_exp, a_lab):.2f}",
        ),
        practice.Check(
            "FINDING: benign lookalikes outscore redacted attacks two pairs in three, so no score floor can gate",
            result["attack_wins"] < 0.40 and result["below_best_benign"] == 19,
            f"P(attack > benign) {result['attack_wins']:.3f}; {result['below_best_benign']}/20 below the best benign",
        ),
        practice.Check(
            "FINDING: the lesson's validator refuses a held-out split, because multi-turn-ramp sits at the minimum",
            "multi-turn-ramp has 6" in result["loo_refused"],
            f"holding out mt-01: {result['loo_refused']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
