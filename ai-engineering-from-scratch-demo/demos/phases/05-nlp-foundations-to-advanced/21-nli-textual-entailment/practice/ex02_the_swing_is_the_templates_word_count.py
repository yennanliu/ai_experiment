"""Exercise 2 — the swing is the template's word count.

    **Medium.** Compare the zero-shot template `"This text is about {label}"`
    against `"The topic is {label}"` and `"{label}"` on 100 AG News headlines.
    Report accuracy swing.

Reading of the exercise: `transformers` is absent and AG News does not ship, so
the scorer is the lesson's own `predict_nli` over 16 headlines written to the
four AG News classes. The reported swing is **0.000 across all three templates**
-- and the reason is that the measurement is degenerate, not that templates do
not matter. Every label scores overlap **0.000** against every headline, because
the class name never appears literally in the text and lexical overlap is the
only evidence there is. Argmax then returns whichever label is first in the list,
so all three templates score 4/16: the share of that first class, which is chance.

On five headlines that do contain their class name, the templates separate
completely. Under the classifier's own entailment threshold the entailment rate
runs **1/5, 4/5, 5/5** from the longest template to the shortest, an 80-point
swing with no change to the text.

The mechanism is arithmetic. `lexical_overlap` divides by the hypothesis's
content-word count, so a template's ceiling is `k / (k + e)` where `k` is the
label's token count and `e` the content words the template adds: two for `This
text is about`, one for `The topic is`, none for the bare label. For a one-token
label that is 0.33, 0.50 and 1.00, and `predict_nli`'s threshold sits at 0.50 --
between the first two. Template choice is a threshold crossing dressed as a
prompt.

Label length enters the same denominator. `Sci/Tech` tokenizes to two words, so
under `This text is about {label}` it reaches 0.50 and returns entailment while
every one-token label maxes at 0.33 and returns neutral. The same template gives
different verdicts to different classes on identical evidence.

And the swing depends on a decision rule the exercise never names: **0 points
under argmax, 80 points under the entailment threshold**, on the same data and
the same three templates.

Structure: `HEADLINES` are written to the four classes without naming them,
`LITERAL` name theirs; `rank` scores every label under one template; `by_argmax`
and `by_threshold` are the two decision rules; `ceiling` is the arithmetic bound.
"""

from __future__ import annotations

import importlib.util

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "21-nli-textual-entailment"

UNAVAILABLE = ("transformers", "torch", "datasets")
LABELS = ("World", "Sports", "Business", "Sci/Tech")
TEMPLATES = ("This text is about {}", "The topic is {}", "{}")
HEADLINES = """Peace talks between the two nations resume in Geneva this week|World
Refugees cross the border as fighting escalates in the capital|World
The prime minister survives a no confidence vote in parliament|World
Election observers report irregularities at polling stations|World
United beat City in a thrilling derby at the weekend|Sports
The champion retained her title after a five set final|Sports
The club signed a striker for a record transfer fee|Sports
Olympic organisers unveil the schedule for the summer games|Sports
Shares fell sharply after the earnings report disappointed|Business
The central bank raised interest rates by a quarter point|Business
A merger between the two retailers was approved by regulators|Business
Oil prices climbed on supply concerns in the gulf|Business
Researchers report a breakthrough in quantum error correction|Sci/Tech
The space agency launched a probe toward the outer planets|Sci/Tech
A new chip design promises lower power for mobile devices|Sci/Tech
Astronomers detect water vapour on a distant exoplanet|Sci/Tech"""
LITERAL = """The sports desk previews the weekend fixtures|Sports
Business leaders meet to discuss the trade deal|Business
World leaders gather for the climate summit|World
Sci tech spending rose across the sector|Sci/Tech
A tech firm reported record revenue|Sci/Tech"""
ROWS = tuple(tuple(line.split("|")) for line in HEADLINES.splitlines())
NAMED = tuple(tuple(line.split("|")) for line in LITERAL.splitlines())


def rank(ref, headline, template):
    """Overlap and predicted relation for every candidate label."""
    return {label: ref.predict_nli(headline, template.format(label)) for label in LABELS}


def by_argmax(ref, rows, template):
    """The usual zero-shot rule: highest-scoring label wins, first on a tie."""
    hits = 0
    for headline, gold in rows:
        scored = rank(ref, headline, template)
        hits += max(LABELS, key=lambda label: scored[label][1]) == gold
    return hits


def by_threshold(ref, rows, template):
    """The classifier's own rule: does the gold label come back as entailment."""
    return sum(1 for headline, gold in rows if rank(ref, headline, template)[gold][0] == "entailment")


def ceiling(ref, template, label):
    """The best overlap this template can reach for this label, k / (k + e)."""
    hypothesis = ref.content_words(ref.tokenize(template.format(label)))
    return round(len(ref.content_words(ref.tokenize(label))) / len(hypothesis), 2)


def arms(ref, rows):
    """Both decision rules and the observed overlap values, per template."""
    return {
        "argmax": {t: by_argmax(ref, rows, t) for t in TEMPLATES},
        "threshold": {t: by_threshold(ref, rows, t) for t in TEMPLATES},
        "overlaps": {t: sorted({round(v[1], 3) for h, _ in rows for v in rank(ref, h, t).values()})
                     for t in TEMPLATES},
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    plain, named = arms(ref, ROWS), arms(ref, NAMED)
    return {
        "absent": [n for n in UNAVAILABLE if importlib.util.find_spec(n) is None],
        "n": len(ROWS),
        "named_n": len(NAMED),
        "argmax": plain["argmax"],
        "overlaps": plain["overlaps"],
        "named_argmax": named["argmax"],
        "named_threshold": named["threshold"],
        "first_label_share": sum(1 for _, gold in ROWS if gold == LABELS[0]),
        "ceilings": {t: {label: ceiling(ref, t, label) for label in ("Sports", "Sci/Tech")}
                     for t in TEMPLATES},
    }


def verify(result):
    named, ceilings = result["named_threshold"], result["ceilings"]
    swing = (max(named.values()) - min(named.values())) / result["named_n"] * 100
    return [
        practice.Check(
            "ANSWER: the swing on realistic headlines is 0.000, and so is every score",
            len(set(result["argmax"].values())) == 1 and result["overlaps"][TEMPLATES[0]] == [0.0],
            f"{result['absent']} are all absent, so the scorer is the lesson's own `predict_nli` "
            f"over {result['n']} headlines written to the four AG News classes. Every label "
            f"scores overlap {result['overlaps'][TEMPLATES[0]]} against every headline -- the "
            f"class name never appears in the text -- so all three templates score "
            f"{list(result['argmax'].values())}",
        ),
        practice.Check(
            "MECHANISM: that number is the first label's share, not a skill",
            set(result["argmax"].values()) == {result["first_label_share"]},
            f"with every score tied at zero, argmax returns whichever label the list puts first, "
            f"so accuracy is exactly the {LABELS[0]} share of the corpus: "
            f"{result['first_label_share']}/{result['n']}. Reordering `LABELS` would change it "
            "and nothing about the templates would have changed",
        ),
        practice.Check(
            "FINDING: on headlines that name their class the templates separate completely",
            len(set(named.values())) == len(TEMPLATES),
            f"over {result['named_n']} headlines containing their own class name, entailment "
            f"under the three templates runs {list(named.values())} -- a {swing:.0f}-point swing "
            "from the longest template to the shortest, with no change to the text",
        ),
        practice.Check(
            "MECHANISM: because the template's own words sit in the denominator",
            ceilings[TEMPLATES[0]]["Sports"] < 0.5 <= ceilings[TEMPLATES[1]]["Sports"],
            f"`lexical_overlap` divides by the hypothesis's content-word count, so a template's "
            f"ceiling is k/(k+e) for a k-token label and e added words: "
            f"{[ceilings[t]['Sports'] for t in TEMPLATES]} for a one-token label. `predict_nli` "
            "thresholds at 0.50, which falls between the first two. The prompt is a threshold "
            "crossing in disguise",
        ),
        practice.Check(
            "FINDING: label length enters the same denominator",
            ceilings[TEMPLATES[0]]["Sci/Tech"] > ceilings[TEMPLATES[0]]["Sports"],
            f"`Sci/Tech` tokenizes to two words, so under {TEMPLATES[0]!r} it reaches "
            f"{ceilings[TEMPLATES[0]]['Sci/Tech']} and returns entailment while every one-token "
            f"label maxes at {ceilings[TEMPLATES[0]]['Sports']} and returns neutral. One template, "
            "two verdicts, identical evidence",
        ),
        practice.Check(
            "CONTROL: the swing has no value until the decision rule is named",
            len(set(result["named_argmax"].values())) == 1,
            f"the same five headlines under argmax score {list(result['named_argmax'].values())} "
            f"-- 0 points of swing -- against {swing:.0f} points under the entailment threshold. "
            "The exercise asks for the swing without saying which rule reads the scores",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
