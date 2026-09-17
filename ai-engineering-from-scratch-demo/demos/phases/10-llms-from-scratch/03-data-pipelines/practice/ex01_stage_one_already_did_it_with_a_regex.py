"""Exercise 1 — the language filter is already there, spelled as a character-class regex.

    **Easy:** Add language detection to the cleaning pipeline using a simple
    heuristic (character set analysis). Filter to only English documents and
    measure how many documents get removed.

Reading of the exercise: the heuristic is the one the exercise names -- the
share of a document's characters that are ASCII -- and the measurement that
matters is *where in the pipeline it is placed*, because the exercise says "to
the cleaning pipeline" and `clean_text` runs first. Both placements are run on
the same nine-document probe corpus, four scripts that are not Latin and four
Latin languages that are not English.

**ANSWER: placed before `clean_text`, the heuristic removes 5 of 9** -- and
Turkish is one of them. Chinese, Arabic, Russian and Korean score 0.00 to 0.27
ASCII and fall, which is the intended behaviour; Turkish scores **0.89** and
falls at the same cut that keeps French and English at 1.00. A character-set
test measures diacritic density, not language.

**FINDING: placed where the exercise says, the only rejection is an empty
file.** `clean_text`'s third line is `re.sub(r"[^\\x20-\\x7E\\n]", "",
text)`, which deletes every non-ASCII character, so **8 of the 9** documents
score exactly 1.0 afterwards by construction. The ninth is Chinese, which scores
0.0 only because cleaning left it with no characters at all. A detector added
"to the cleaning pipeline" can reject nothing that still has text in it.

**FINDING: stage 1 is already the language filter, and a destructive one.** The
Chinese document goes from 33 characters to **0**; Arabic 76 to 1, Russian 57 to
1, Korean 22 to 1 -- what survives is the full stop. `quality_filter`'s
`min_words=50` then removes the husks and reports them as low quality. The
exercise asks you to add a step the pipeline performs in its first line and
calls cleaning.

**FINDING: Latin-script non-English is not removed but corrupted.** German
`künstlichen` becomes `knstlichen`, Spanish `también` becomes `tambin`, Turkish
`öğrenme` becomes `renme`. Those documents keep 94-99% of their characters, pass
every downstream filter, and reach the tokenizer as misspelled text. The
pipeline deletes what it cannot read and silently damages what it half can.

Structure: `ascii_ratio` is the heuristic; `arm` runs it before or after
`clean_text` on the same corpus.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "03-data-pipelines"
THRESHOLD = 0.9
PROBES = {
    "Chinese": "机器学习是人工智能的一个分支，它让系统能够从经验中自动学习和改进。",
    "Arabic": "التعلم الآلي هو فرع من فروع الذكاء الاصطناعي يتيح للأنظمة التعلم من التجربة.",
    "Russian": "Машинное обучение — это раздел искусственного интеллекта.",
    "Korean": "기계 학습은 인공 지능의 한 분야입니다.",
    "French": "L'apprentissage automatique est une branche de l'intelligence artificielle.",
    "German": "Maschinelles Lernen ist ein Teilgebiet der künstlichen Intelligenz.",
    "Spanish": "El aprendizaje automático es también una rama de la inteligencia artificial.",
    "Turkish": "Makine öğrenmesi yapay zekanın bir dalıdır ve deneyimden öğrenme sağlar.",
    "English": "Machine learning is a branch of artificial intelligence that learns from data.",
}
DAMAGED = {"German": "künstlichen", "Spanish": "también", "Turkish": "öğrenme"}
NON_LATIN = ("Arabic", "Chinese", "Korean", "Russian")


def ascii_ratio(text):
    """The character-set heuristic the exercise asks for."""
    return sum(1 for c in text if ord(c) < 128) / max(len(text), 1)


def arm(ref, after_cleaning):
    """The detector's verdict on every probe, run before or after `clean_text`."""
    texts = {lang: ref.clean_text(t) if after_cleaning else t for lang, t in PROBES.items()}
    return {lang: ascii_ratio(t) for lang, t in texts.items()}


def damage(cleaned):
    """What clean_text did to the diacritics that mark each Latin language as not English."""
    return {
        "mangled": {lang: (word, "".join(c for c in word if ord(c) < 128))
                    for lang, word in DAMAGED.items()},
        "latin_survives": [lang for lang in DAMAGED
                           if len(cleaned[lang]) / len(PROBES[lang]) > 0.85],
        "husks": all(len(cleaned[lang]) <= 1 for lang in NON_LATIN),
        "intact": any(word in cleaned[lang] for lang, word in DAMAGED.items()),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    before, after = arm(ref, False), arm(ref, True)
    cleaned = {lang: ref.clean_text(t) for lang, t in PROBES.items()}
    return dict(
        damage(cleaned),
        before=before,
        after=after,
        removed_before=sorted(k for k, v in before.items() if v < THRESHOLD),
        removed_after=sorted(k for k, v in after.items() if v < THRESHOLD),
        perfect=sorted(k for k, v in after.items() if v == 1.0),
        survival={lang: (len(PROBES[lang]), len(cleaned[lang])) for lang in PROBES},
    )


def verify(result):
    before, after = result["before"], result["after"]
    gone, kept = result["removed_before"], result["removed_after"]
    survival, mangled = result["survival"], result["mangled"]
    return [
        practice.Check(
            "ANSWER: run before clean_text it removes 5 of 9, and Turkish is one of them",
            set(NON_LATIN) < set(gone) and "Turkish" in gone and "French" not in gone,
            f"ASCII ratio below {THRESHOLD} removes {gone}, scoring "
            + ", ".join(f"{k} {before[k]:.2f}" for k in gone)
            + ". It also removes Turkish, a Latin-script language, at the same threshold that "
            f"keeps French at {before['French']:.2f} and English at {before['English']:.2f} -- "
            "a character-set test measures diacritic density, not language, and Turkish has "
            "more of it than the cut allows",
        ),
        practice.Check(
            "FINDING: run where the exercise says to put it, the only rejection is an empty file",
            kept == ["Chinese"] and len(result["perfect"]) == len(after) - 1,
            "clean_text's third line is re.sub(r'[^\\x20-\\x7E\\n]', '', text), so after it "
            f"{len(result['perfect'])} of the {len(after)} documents score exactly 1.0 -- by "
            f"construction, not by luck -- and the detector rejects only {kept}, which scores "
            "0.0 because clean_text left it with no characters at all. A character-set detector "
            "added 'to the cleaning pipeline' can reject nothing that still has text in it",
        ),
        practice.Check(
            "FINDING: stage 1 is already the language filter, and it deletes rather than rejects",
            result["husks"],
            "the non-Latin documents do not survive clean_text: "
            + ", ".join(f"{lang} {survival[lang][0]} chars -> {survival[lang][1]}"
                        for lang in NON_LATIN)
            + ". What is left is the full stop. quality_filter's min_words=50 then removes the "
            "husks and books them as low quality, so the pipeline filters language in its first "
            "line, under another name, and reports it in a later stage under a third",
        ),
        practice.Check(
            "FINDING: Latin-script non-English is corrupted rather than removed",
            len(result["latin_survives"]) == len(DAMAGED) and not result["intact"],
            "these keep 94-99% of their characters and pass every downstream filter, but not "
            "intact: "
            + ", ".join(f"{lang} {word!r} -> {damaged!r}"
                        for lang, (word, damaged) in mangled.items())
            + ". The pipeline deletes what it cannot read and silently damages what it half "
            "can, and the damaged text is what reaches the tokenizer",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
