"""Exercise 3 — the fallback is the small half.

    **Hard.** Build a lemmatizer that uses WordNet as a lookup table but falls
    back to your Porter stemmer when WordNet has no entry. Measure accuracy on a
    tagged corpus against plain WordNet and plain Porter.

Reading of the exercise: neither `nltk` nor `spacy` is installed and the lesson
ships no tagged corpus, so both halves of "measure accuracy on a tagged corpus
against plain WordNet" are supplied here and said so. The corpus is 48
hand-labelled (word, POS, lemma) triples below; WordNet is its own published
morphy algorithm -- the detachment rules of morphy(7WN) -- run over a
vocabulary, and the vocabulary used is the answer key's own range, which is the
most favourable dictionary morphy can be given on this corpus. Its 27/48 is
therefore a ceiling, not an estimate.

That substitution is what makes the exercise's premise checkable, and the
premise is wrong in a specific way. WordNet is not a lookup table; morphy is a
rule engine gated on a dictionary membership test, and its rules are close
cousins of Porter's. So "fall back to Porter when WordNet has no entry" falls
back to nearly the same rules with the membership test removed, and it fires
exactly on the rows where that test was the only thing protecting the answer.
The measurement says how much that is worth: 4 rows out of 48, all four of them
exercise 2's double-consonant rule, against 15 rows for the exception list that
real WordNet ships and that "lookup table" quietly stands in for.

Structure: `CORPUS` is the labelled fixture and `VOCAB` its range. `morphy`
applies the published detachment rules in their published order and returns the
first candidate that is in the vocabulary, or `None` -- abstention is the point
of it. `stemmer` is the lesson's own `stem_step_1a` followed by exercise 2's
step 1b, imported rather than re-implemented; `hybrid` is the exercise's
construction; `EXC` is a 15-entry stand-in for WordNet's exception list, and
`accuracy` scores an arm against the key.
"""

from __future__ import annotations

import pathlib

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "01-text-processing"
SIBLING = pathlib.Path(__file__).with_name("ex02_the_four_rules_the_wording_omits.py")

CORPUS = (
    ("walked", "VERB", "walk"), ("watched", "VERB", "watch"), ("played", "VERB", "play"),
    ("stopped", "VERB", "stop"), ("planned", "VERB", "plan"), ("carried", "VERB", "carry"),
    ("walking", "VERB", "walk"), ("running", "VERB", "run"), ("hoping", "VERB", "hope"),
    ("sitting", "VERB", "sit"), ("studying", "VERB", "study"), ("being", "VERB", "be"),
    ("runs", "VERB", "run"), ("watches", "VERB", "watch"), ("carries", "VERB", "carry"),
    ("goes", "VERB", "go"), ("ran", "VERB", "run"), ("went", "VERB", "go"),
    ("were", "VERB", "be"), ("was", "VERB", "be"), ("is", "VERB", "be"), ("ate", "VERB", "eat"),
    ("took", "VERB", "take"), ("brought", "VERB", "bring"), ("cats", "NOUN", "cat"),
    ("boxes", "NOUN", "box"), ("buses", "NOUN", "bus"), ("cities", "NOUN", "city"),
    ("dishes", "NOUN", "dish"), ("watches", "NOUN", "watch"), ("children", "NOUN", "child"),
    ("men", "NOUN", "man"), ("feet", "NOUN", "foot"), ("mice", "NOUN", "mouse"),
    ("geese", "NOUN", "goose"), ("teeth", "NOUN", "tooth"), ("gas", "NOUN", "gas"),
    ("class", "NOUN", "class"), ("analysis", "NOUN", "analysis"), ("news", "NOUN", "news"),
    ("series", "NOUN", "series"), ("lens", "NOUN", "lens"), ("better", "ADJ", "good"),
    ("best", "ADJ", "good"), ("larger", "ADJ", "large"), ("largest", "ADJ", "large"),
    ("happier", "ADJ", "happy"), ("faster", "ADJ", "fast"),
)
VOCAB = frozenset(lemma for _, _, lemma in CORPUS)
UNINFLECTED = tuple(w for w, pos, lemma in CORPUS if pos == "NOUN" and w == lemma)
# morphy(7WN)'s detachment rules, in their published order
RULES = {"NOUN": (("s", ""), ("ses", "s"), ("xes", "x"), ("zes", "z"), ("ches", "ch"),
                  ("shes", "sh"), ("men", "man"), ("ies", "y")),
         "VERB": (("s", ""), ("ies", "y"), ("es", "e"), ("es", ""), ("ed", "e"), ("ed", ""),
                  ("ing", "e"), ("ing", "")),
         "ADJ": (("er", ""), ("est", ""), ("er", "e"), ("est", "e"))}
EXC = {("ran", "VERB"): "run", ("went", "VERB"): "go", ("were", "VERB"): "be",
       ("was", "VERB"): "be", ("is", "VERB"): "be", ("ate", "VERB"): "eat",
       ("took", "VERB"): "take", ("brought", "VERB"): "bring", ("children", "NOUN"): "child",
       ("feet", "NOUN"): "foot", ("mice", "NOUN"): "mouse", ("geese", "NOUN"): "goose",
       ("teeth", "NOUN"): "tooth", ("better", "ADJ"): "good", ("best", "ADJ"): "good"}

accuracy = lambda arm: sum(arm(w, p) == lemma for w, p, lemma in CORPUS)          # noqa: E731
rate = lambda hits: f"{hits}/{len(CORPUS)} = {hits / len(CORPUS):.3f}"            # noqa: E731


def morphy(word: str, pos: str, exceptions=()) -> str | None:
    """Rules gated on a membership test; `None` where WordNet would have no entry."""
    if (word, pos) in exceptions:
        return exceptions[(word, pos)]
    if word in VOCAB:
        return word
    for suffix, replacement in RULES.get(pos, ()):
        candidate = word[: -len(suffix)] + replacement
        if word.endswith(suffix) and candidate in VOCAB:
            return candidate
    return None


def arms_for(ref, stemmer) -> dict:
    """The four arms the exercise names, plus morphy with an exception list."""
    return {"lesson": ref.lemmatize, "porter": lambda w, _p: stemmer(w),
            "wordnet": lambda w, pos: morphy(w, pos) or w,
            "hybrid": lambda w, pos: morphy(w, pos) or stemmer(w),
            "wordnet+exc": lambda w, pos: morphy(w, pos, EXC) or w}


def membership(arms) -> dict:
    """Scores on the nouns that are already lemmas -- what the dictionary test buys."""
    return {name: sum(arms[name](w, "NOUN") == w for w in UNINFLECTED)
            for name in ("lesson", "wordnet")}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    step_1b = practice.load_module(SIBLING).porter
    stemmer = lambda word: step_1b(ref.stem_step_1a(word))                        # noqa: E731
    arms = arms_for(ref, stemmer)
    abstains = [(w, p, lemma) for w, p, lemma in CORPUS if morphy(w, p) is None]
    return {
        "acc": {name: accuracy(arm) for name, arm in arms.items()},
        "abstains": len(abstains), "n": len(CORPUS), "membership": membership(arms),
        "rescued": [w for w, _p, lemma in abstains if stemmer(w) == lemma],
        "damaged": [ref.lemmatize(w, "NOUN") for w in UNINFLECTED],
        "case": [ref.lemmatize(w, p) for w, p in (("Cats", "NOUN"), ("Dogs", "NOUN"),
                                                  ("Walking", "VERB"), ("Walked", "VERB"))],
    }


def verify(result):
    acc, n, rescued = result["acc"], result["n"], result["rescued"]
    return [
        practice.Check(
            "ANSWER: hybrid 31/48 beats plain WordNet 27 and plain Porter 14, on a supplied corpus",
            acc["hybrid"] > acc["wordnet"] > acc["porter"] > acc["lesson"],
            f"neither nltk nor spacy is installed and no tagged corpus ships, so the corpus is {n} "
            f"hand-labelled triples and WordNet is its published morphy rules over a vocabulary. "
            f"lesson {rate(acc['lesson'])}, porter {rate(acc['porter'])}, wordnet "
            f"{rate(acc['wordnet'])}, hybrid {rate(acc['hybrid'])}. The vocabulary is the answer "
            f"key's own range -- the best dictionary morphy can have here -- so 27 is its ceiling"),
        practice.Check(
            "MECHANISM: the hybrid's score is a sum, because the fallback only fires on abstentions",
            acc["hybrid"] == acc["wordnet"] + len(rescued),
            f"morphy abstains on {result['abstains']} of {n} rows and the stemmer is consulted on "
            f"exactly those, so the hybrid is {acc['wordnet']} + {len(rescued)} = {acc['hybrid']} "
            f"by construction. It cannot change a row WordNet covers, however wrong that row is"),
        practice.Check(
            "FINDING: the whole fallback gain is exercise 2's double-consonant rule",
            sorted(rescued) == ["planned", "running", "sitting", "stopped"],
            f"the {len(rescued)} rows the stemmer rescues are {sorted(rescued)} -- every one a "
            f"doubled final consonant, which morphy cannot reach because 'stopp' and 'runn' are not "
            f"words. Porter scores {acc['porter']}/{n} overall but {len(rescued)}/"
            f"{result['abstains']} on the abstentions: it is weakest exactly where it is called"),
        practice.Check(
            "FINDING: the exercise asks for the small half -- the exception list is worth 15, not 4",
            acc["wordnet+exc"] - acc["wordnet"] > 3 * (acc["hybrid"] - acc["wordnet"]),
            f"a {len(EXC)}-entry exception list of the kind real WordNet ships takes morphy from "
            f"{acc['wordnet']} to {acc['wordnet+exc']}, +{acc['wordnet+exc'] - acc['wordnet']} rows, "
            f"against +{acc['hybrid'] - acc['wordnet']} for the Porter fallback. 'WordNet as a "
            f"lookup table' names that list and then asks for the other repair"),
        practice.Check(
            "FINDING: WordNet is rules plus a membership test, and the membership test is the part missing",
            result["membership"]["wordnet"] == len(UNINFLECTED) and result["membership"]["lesson"] == 0,
            f"on the {len(UNINFLECTED)} nouns that are already lemmas, morphy scores "
            f"{result['membership']['wordnet']}/{len(UNINFLECTED)} by checking the word itself first; "
            f"the lesson's lemmatize scores {result['membership']['lesson']}, returning "
            f"{result['damaged']} -- its NOUN branch strips a final 's' unconditionally, though "
            f"stem_step_1a two functions above guards 'ss' explicitly"),
        practice.Check(
            "CONTROL: the lesson's lemmatizer scores below its own stemmer, and is case-inconsistent",
            acc["lesson"] < acc["porter"] and result["case"] == ["cat", "Dog", "Walk", "walked"],
            f"on lemmatization -- the task the lesson says lemmatizers win -- lemmatize scores "
            f"{acc['lesson']}/{n} and stem_step_1a + step 1b scores {acc['porter']}/{n}. It also "
            f"disagrees with itself about case: {result['case']} for Cats, Dogs, Walking, Walked. "
            f"The table path lowercases, the suffix paths do not, the final fallback does"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
