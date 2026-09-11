"""Exercise 1 — the contexts quote the knowledge base.

    **Easy.** Implement the prior+context disambiguator in `code/main.py` on 10
    ambiguous mentions (Paris, Jordan, Apple). Hand-label the correct entity.
    Measure accuracy.

Reading of the exercise: the disambiguator ships, and on `main()`'s own eleven
cases it scores **11 of 11** with the prior and 10 of 11 without. Hand-labelling
a second set of eleven -- same eleven entities, contexts written without reusing
the KB's wording -- gives **5 of 11** with the prior and 7 of 11 without.

The lesson's contexts quote the knowledge base. "Paris is the capital of France
and home to the Eiffel Tower" shares 6 of the 9 tokens in `Q90`'s description;
across the eleven cases the gold description contributes 1 to 6 matching tokens
to a sentence of about a dozen. The accuracy being measured is how the evaluation
sentences were written.

The prior's contribution changes sign between the two sets. It rescues one case
on the quoting set and costs two on the paraphrase set, where it pulls four
mentions to the most popular reading. Whether the prior helps is a property of
the eval, not of the method.

The one case it rescues is the one where the context carries no signal at all.
"Jordan scored 45 points against the Lakers last night" shares exactly one token
with `Q41421`'s description, and that token is `jordan`, which every candidate
shares equally. All four Jaccard scores tie, and the prior returns the most
popular entity -- which is the answer, and also what a system with no context
model at all would have said.

That system scores 4 of 11 on both sets. On the paraphrase set the full
disambiguator is one case above ignoring the context entirely.

`use_prior=False` is not the ablation it looks like: it sets every candidate's
prior to 1.0, so the score becomes `jac + 0.1` and the ranking is Jaccard alone.
And an unrecognised mention returns `(None, 0.0)` -- there is no NIL.

Structure: `QUOTING` is `main()`'s own set, `PARAPHRASE` the hand-labelled one,
both as mention/context/entity lines; `score` runs the lesson's disambiguator
over a set; `popular` is the argmax-prior baseline; `overlap` counts shared
tokens with the gold description.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "25-entity-linking"

QUOTING = """Jordan|Jordan scored 45 points against the Lakers last night.|Q41421
Jordan|Jordan borders Syria, Iraq, and Saudi Arabia in the Middle East.|Q810
Jordan|Jordan starred in the superhero movie Black Panther.|Q254110
Jordan|Jordan's work on variational inference shaped machine learning.|Q3308285
Paris|Paris is the capital of France and home to the Eiffel Tower.|Q90
Paris|Paris Texas is a small city in Lamar County.|Q663094
Paris|Paris Hilton is a television personality and heiress.|Q55411
Apple|Apple announced the new iPhone at their Cupertino event.|Q312
Apple|An apple a day keeps the doctor away; the fruit is rich in fiber.|Q89
Python|Python is a popular programming language used for data science.|Q28865
Python|The python is a large nonvenomous snake found in Asia.|Q83320"""
PARAPHRASE = """Jordan|Jordan hit the game winner in the fourth quarter at the United Center.|Q41421
Jordan|Jordan lies south of Damascus and shares a long desert frontier.|Q810
Jordan|Jordan gave a raw performance as the antagonist in that Marvel sequel.|Q254110
Jordan|Jordan supervised a generation of graduate students at Berkeley.|Q3308285
Paris|Paris flooded when the Seine burst its banks in the spring.|Q90
Paris|Paris sits between Dallas and the Oklahoma border on the old rail line.|Q663094
Paris|Paris launched a perfume line and appeared on reality television.|Q55411
Apple|Apple posted record quarterly revenue from services and wearables.|Q312
Apple|The apple bruised in the crate before it reached the market stall.|Q89
Python|Python ships with a standard library and a package installer.|Q28865
Python|The python coiled around the branch and waited for prey.|Q83320"""
SETS = {"quoting": tuple(tuple(line.split("|")) for line in QUOTING.splitlines()),
        "paraphrase": tuple(tuple(line.split("|")) for line in PARAPHRASE.splitlines())}


def score(ref, cases, use_prior=True):
    """Correct links and the misses, as (mention, predicted, gold)."""
    hits, misses = 0, []
    for mention, context, gold in cases:
        predicted = ref.disambiguate(mention, context, use_prior=use_prior)[0]
        hits += predicted == gold
        if predicted != gold:
            misses.append((mention, predicted, gold))
    return hits, misses


def popular(ref, cases):
    """The argmax-prior baseline: pick the most popular reading, ignore the context."""
    best = {alias: max(qs, key=lambda q: ref.PRIORS[q]) for alias, qs in ref.ALIAS_INDEX.items()}
    return sum(1 for mention, _, gold in cases if best[mention.lower()] == gold)


def overlap(ref, cases):
    """Tokens the gold description shares with the context, and the description's size."""
    return [(len(ref.tokenize(context) & ref.tokenize(ref.KB_DESC[gold])),
             len(ref.tokenize(ref.KB_DESC[gold]))) for _, context, gold in cases]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = {}
    for name, cases in SETS.items():
        with_prior, missed = score(ref, cases)
        rows[name] = {"prior": with_prior, "flat": score(ref, cases, use_prior=False)[0],
                      "popular": popular(ref, cases), "misses": missed,
                      "overlap": overlap(ref, cases), "n": len(cases)}
    tied = SETS["quoting"][0]
    return {
        "rows": rows,
        "tie_case": tied[1],
        "tie_scores": {q: round(ref.PRIORS[q], 2)
                       for q in ref.ALIAS_INDEX["jordan"]},
        "tie_overlap": [len(ref.tokenize(tied[1]) & ref.tokenize(ref.KB_DESC[q]))
                        for q in ref.ALIAS_INDEX["jordan"]],
        "unknown": ref.disambiguate("Berlin", "Berlin is the capital of Germany."),
    }


def verify(result):
    quoting, para = result["rows"]["quoting"], result["rows"]["paraphrase"]
    shares = [f"{a}/{b}" for a, b in quoting["overlap"]]
    return [
        practice.Check(
            "ANSWER: 11 of 11 on the lesson's own cases, 5 of 11 on hand-labelled ones",
            quoting["prior"] > para["prior"],
            f"the disambiguator scores {quoting['prior']}/{quoting['n']} with the prior on "
            f"`main()`'s own eleven cases and {para['prior']}/{para['n']} on eleven contexts "
            "written about the same entities without reusing the KB's wording",
        ),
        practice.Check(
            "MECHANISM: because the lesson's contexts quote the knowledge base",
            sum(a for a, _ in quoting["overlap"]) > sum(a for a, _ in para["overlap"]),
            f"tokens the gold description shares with its context, of the description's size: "
            f"{shares} on the quoting set against "
            f"{[f'{a}/{b}' for a, b in para['overlap']]} on the paraphrase set. The accuracy "
            "being measured is how the evaluation sentences were written",
        ),
        practice.Check(
            "FINDING: the prior's contribution changes sign between the two sets",
            quoting["prior"] > quoting["flat"] and para["prior"] < para["flat"],
            f"with the prior against without: {quoting['prior']} and {quoting['flat']} on the "
            f"quoting set, {para['prior']} and {para['flat']} on the paraphrase set. It rescues "
            "one case on one and costs two on the other, pulling mentions to the popular reading",
        ),
        practice.Check(
            "MECHANISM: and the case it rescues is the one with no context signal at all",
            result["tie_overlap"] == [1, 1, 1, 1],
            f"'{result['tie_case']}' shares {result['tie_overlap']} tokens with the four Jordan "
            "descriptions -- one each, and that token is `jordan`, which every candidate shares. "
            f"The Jaccard scores tie and the prior {result['tie_scores']} returns the most "
            "popular entity, which is what a system with no context model would have said",
        ),
        practice.Check(
            "FINDING: that system scores 4 of 11 on both sets",
            quoting["popular"] == para["popular"] and para["prior"] - para["popular"] <= 1,
            f"argmax-prior with the context discarded scores {quoting['popular']} and "
            f"{para['popular']}. On the paraphrase set the full disambiguator, at "
            f"{para['prior']}, is one case above ignoring the context entirely",
        ),
        practice.Check(
            "CONTROL: `use_prior=False` is a uniform prior, and there is no NIL",
            result["unknown"] == (None, 0.0),
            "it sets every candidate's prior to 1.0, so the score becomes `jac + 0.1` and the "
            f"ranking is Jaccard alone -- an ablation of the weighting, not of the prior. And an "
            f"unrecognised mention returns {result['unknown']}: no candidate set, no NIL, and a "
            "score indistinguishable from a confident zero",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
