"""Exercise 2 — the four rules the wording omits.

    **Medium.** Implement Porter step 1b. If a word contains a vowel and ends in
    `ed` or `ing`, remove it. Handle the double-consonant rule (`hopping -> hop`,
    not `hopp`).

Reading of the exercise: implemented exactly as written, this reproduces 5 of
the 16 worked examples Porter published for step 1b, and both cases the wording
names -- `hopping -> hop` and the vowel test -- are among the 5 it gets right,
so the stated acceptance criterion cannot see the other 11. Porter's own
examples are the grading set here because his definition ships them attached to
the rules they pin down (the file the lesson's Further Reading links to), so
each miss names the omitted rule rather than a taste. Four omissions. The vowel
test is on the *stem*, not on the word: `bled` and `sing` carry their only
vowel inside the suffix, and read literally the rule strips it, which on the
two-letter words `ed` and `ing` returns the empty string. The double-consonant
rule has an exception for l, s and z that the wording drops, so `falling`,
`hissing` and `fizzed` lose a letter. `eed` is its own rule gated on measure,
which the wording has no notion of. And three cleanups restore an `e` the
suffix removal took. The fourth is not pedantry: `stem_step_1a` in the lesson's
own code guards `ss` explicitly, and a literal step 1b un-guards it two lines
later -- `process`/`processing` stop sharing a stem.

Structure: `cons`, `vowelly`, `measure`, `dbl` and `cvc` are Porter's four
conditions, written once and shared by both arms. `porter` is step 1b in full
and `literal` is the exercise's sentence transcribed; `GOLD` is Porter's
published example table and `score` counts exact matches against it. `families`
runs three `-ss` verb families through the lesson's own `stem_step_1a` and then
each arm, and the corpus for the aggregate is the lesson's own `docs/en.md`,
tokenized by the lesson's own `tokenize`.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "01-text-processing"

VOWELS, SUFFIXES, KEEP = "aeiou", (2, 3), "lsz"
FAMILIES = tuple(b + s for b in ("process", "miss", "pass") for s in ("", "es", "ing", "ed"))
# Porter's own worked examples for step 1b, in his order
GOLD = (("feed", "feed"), ("agreed", "agree"), ("plastered", "plaster"), ("bled", "bled"),
        ("motoring", "motor"), ("sing", "sing"), ("conflated", "conflate"), ("sized", "size"),
        ("troubled", "trouble"), ("hopping", "hop"), ("tanned", "tan"), ("falling", "fall"),
        ("hissing", "hiss"), ("fizzed", "fizz"), ("failing", "fail"), ("filing", "file"))
GROUPS = {"stem-scope": ("bled", "sing"), "eed": ("feed", "agreed"),
          "l/s/z": ("falling", "hissing", "fizzed"),
          "restore-e": ("conflated", "troubled", "sized", "filing")}  # 2 + 2 + 3 + 4 = the 11

vowelly = lambda w: any(not cons(w, i) for i in range(len(w)))                  # noqa: E731
dbl = lambda w: len(w) > 1 and w[-1] == w[-2] and cons(w, len(w) - 1)           # noqa: E731
score = lambda fn: [w for w, gold in GOLD if fn(w) != gold]                     # noqa: E731
stems = lambda fn, ref, ws: {w: fn(ref.stem_step_1a(w)) for w in ws}            # noqa: E731


def cons(word: str, i: int) -> bool:
    """Porter: a consonant is any letter but aeiou, and y unless a consonant precedes it."""
    if word[i] in VOWELS:
        return False
    return i == 0 or not cons(word, i - 1) if word[i] == "y" else True


def measure(word: str) -> int:
    flags = [cons(word, i) for i in range(len(word))]
    return sum(1 for i in range(1, len(flags)) if flags[i] and not flags[i - 1])


def cvc(word: str) -> bool:
    return (len(word) > 2 and cons(word, len(word) - 1) and not cons(word, len(word) - 2)
            and cons(word, len(word) - 3) and word[-1] not in "wxy")


def cleanup(stem: str) -> str:
    """The three rules that fire after a successful ed/ing removal (AT/BL/IZ, *d, m=1 *o)."""
    if stem.endswith(("at", "bl", "iz")):
        return stem + "e"
    if dbl(stem) and stem[-1] not in KEEP:
        return stem[:-1]
    return stem + "e" if measure(stem) == 1 and cvc(stem) else stem


def porter(word: str) -> str:
    """Step 1b in full: the eed gate, the stem vowel test, and the three cleanups."""
    if word.endswith("eed"):
        return word[:-1] if measure(word[:-3]) > 0 else word
    for n in SUFFIXES:
        if word[-n:] in ("ed", "ing") and vowelly(word[:-n]):
            return cleanup(word[:-n])
    return word


def literal(word: str) -> str:
    """The exercise's sentence, transcribed: vowel anywhere in the word, no exceptions."""
    for n in SUFFIXES:
        if word[-n:] in ("ed", "ing") and any(c in VOWELS for c in word):
            return word[:-n - 1] if dbl(word[:-n]) else word[:-n]
    return word


def probes() -> dict:
    """The literal reading on the groups that name each omitted rule, and on step 1b's own phase."""
    return {"groups": {name: [literal(w) for w in words] for name, words in GROUPS.items()},
            "empty": [literal(w) for w in ("ing", "ed")],
            "phase": [(w, porter(w)) for w in ("treated", "during")]}


def corpus(ref) -> tuple:
    """The lesson's own prose as word types, tokenized by the lesson's own tokenizer."""
    types = sorted({t.lower() for t in ref.tokenize(parity.doc_text(PHASE, LESSON, "en"))
                    if t.isalpha()})
    return types, [w for w in types if w.endswith(("ed", "ing"))]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    types, candidates = corpus(ref)
    return dict(probes(), gold=len(GOLD), types=len(types), cand=len(candidates),
                wrong={"literal": score(literal), "porter": score(porter)},
                named=[literal("hopping"), porter("hopping")],
                family={name: stems(fn, ref, FAMILIES)
                        for name, fn in (("literal", literal), ("porter", porter))},
                guarded=[ref.stem_step_1a(w) for w in ("process", "miss", "pass")],
                split=[w for w in candidates if literal(w) != porter(w)])


def verify(result):
    wrong, family, total = result["wrong"], result["family"], result["gold"]
    kept = {name: len(set(rows.values())) for name, rows in family.items()}
    return [
        practice.Check(
            "ANSWER: step 1b in full reproduces all 16 of Porter's examples; the wording gives 5",
            not wrong["porter"] and len(wrong["literal"]) == 11,
            f"scored against the example table Porter published with the rules, the full step is "
            f"{total}/{total} and the exercise's sentence transcribed is "
            f"{total - len(wrong['literal'])}/{total}, missing {wrong['literal']}. Both cases the "
            f"wording names pass either way: hopping -> {result['named'][0]} in both arms"),
        practice.Check(
            "MECHANISM: the vowel test is on the stem, not the word, and can empty a word",
            result["groups"]["stem-scope"] == ["bl", "s"] and result["empty"] == ["", ""],
            f"bled and sing carry their only vowel inside the suffix, so 'contains a vowel' read "
            f"over the whole word strips it: {result['groups']['stem-scope']}. Porter's *v* asks the "
            f"stem, which is why both are in his table. On 'ing' and 'ed' themselves the literal "
            f"rule returns {result['empty']} -- an empty token"),
        practice.Check(
            "FINDING: the double-consonant rule carries an l/s/z exception the wording drops",
            result["groups"]["l/s/z"] == ["fal", "his", "fiz"],
            f"the rule is (*d and not (*L or *S or *Z)) -> single letter. Without the exception "
            f"falling, hissing and fizzed become {result['groups']['l/s/z']}; the exercise names "
            f"hopping, which needs the rule, and none of the three that need the exception"),
        practice.Check(
            "FINDING: the lesson's own stem_step_1a guards 'ss', and a literal step 1b un-guards it",
            result["guarded"] == ["process", "miss", "pass"] and kept["literal"] == 2 * kept["porter"],
            f"stem_step_1a returns {result['guarded']} untouched -- an explicit `endswith('ss')` "
            f"rule. Run the same 3 families of 4 forms through it and then step 1b: full Porter "
            f"leaves {kept['porter']} stems, the literal reading {kept['literal']}, splitting every "
            f"base form off its own inflections: {sorted(set(family['literal'].values()))}"),
        practice.Check(
            "FINDING: eed and the three e-restorations are the remaining 6 misses",
            result["groups"]["eed"] == ["fe", "agre"]
            and result["groups"]["restore-e"] == ["conflat", "troubl", "siz", "fil"],
            f"eed is its own measure-gated rule, which the wording has no notion of: feed and agreed "
            f"become {result['groups']['eed']} instead of feed and agree. AT/BL/IZ -> +e and (m=1 "
            f"and *o) -> +e restore a letter the removal took: conflated, troubled, sized and filing "
            f"become {result['groups']['restore-e']}"),
        practice.Check(
            "CONTROL: step 1b is a phase, not a stemmer -- grading it against full Porter misgrades it",
            dict(result["phase"]) == {"treated": "treate", "during": "dure"}
            and 0 < len(result["split"]) < result["cand"],
            f"the correct step-1b output for treated and during is {result['phase']} -- the restored "
            f"e is step 5a's to remove, and the exercise does not ask for step 5a. On the lesson's "
            f"own prose ({result['types']} word types, {result['cand']} ending in ed/ing) the two "
            f"readings disagree on {len(result['split'])}: string -> str, thing -> th, speed -> spe"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
