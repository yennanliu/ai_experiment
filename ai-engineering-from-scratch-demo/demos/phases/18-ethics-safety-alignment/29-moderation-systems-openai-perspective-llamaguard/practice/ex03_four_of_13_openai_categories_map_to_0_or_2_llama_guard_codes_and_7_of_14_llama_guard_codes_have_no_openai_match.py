"""Exercise 3 — 4 of 13 OpenAI categories map to 0 or 2 Llama Guard codes; 7 of 14 Llama Guard codes have no OpenAI match.

    Read the OpenAI Moderation API docs and the Llama Guard 3 category list. Map
    each OpenAI category to the closest Llama Guard categories. Identify three
    categories that do not cleanly map.

Reading of the exercise: the OpenAI side is the 13-category list as the
lesson states it, parsed from `docs/en.md` and checked against the same list
in the `code/main.py` docstring. The Llama Guard side is Llama Guard 3's
S1-S14 from its model card, written into this file. The mapping is this
file's judgement; the checks cover what can be counted from it and what the
reference implements. "Cleanly" means exactly one Llama Guard code.

**ANSWER: the mapping, and the ones that do not map cleanly.**

| OpenAI | Llama Guard 3 |
|---|---|
| harassment | none |
| harassment/threatening | S1 Violent Crimes |
| hate | S10 Hate |
| hate/threatening | S10 Hate + S1 Violent Crimes |
| self-harm, /intent, /instructions | S11 Suicide & Self-Harm |
| sexual | S12 Sexual Content |
| sexual/minors | S4 Child Sexual Exploitation |
| violence | S1 Violent Crimes |
| violence/graphic | none |
| illicit | S2 Non-Violent Crimes |
| illicit/violent | S1 Violent Crimes + S9 Indiscriminate Weapons |

Four do not map cleanly, one more than the exercise asks for. Two have no
counterpart: `harassment`, because Llama Guard's S10 covers only attacks on
protected characteristics, and `violence/graphic`, because S1 is about
enabling crimes, not depicting gore. Two split across codes:
`illicit/violent` and `hate/threatening`. In the other direction, 7 of the 14
Llama Guard codes have no OpenAI counterpart: S3, S5, S6, S7, S8, S13 and S14.

**FINDING: the lesson's custom layer implements a Llama Guard category.**
Both rules in `custom_domain_rules` (financial-advice, medical-advice) are
S6 Specialized Advice, which Llama Guard 3 has and OpenAI Moderation lacks.
Whether a deployment needs a custom layer at all depends on the taxonomy it
picked.

**FINDING: the toy drops 5 of the 13 categories, and 3 of its 8 cannot
fire.** The docs say `/threatening`, `/intent`, `/instructions` and
`/graphic` are collapsed; the 5 missing categories are exactly those. Of the
8 that remain, `sexual`, `sexual/minors` and `illicit` have no triggers, so
their Llama Guard homes (S12, S4, S2) can never be flagged by the harness.
The lesson says OpenAI has "illicit" as one broad category while Llama Guard
splits violent from non-violent crimes. Its own list includes
`illicit/violent`, so both taxonomies make that split.

Structure: `doc_categories()` parses the lesson's bullet list; `solve()`
counts gaps in `MAPPING` and reads the reference's categories, triggers and
custom rules.
"""

from __future__ import annotations

import inspect
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "29-moderation-systems-openai-perspective-llamaguard"
# Llama Guard 3 hazard categories, from the Llama Guard 3 model card (MLCommons taxonomy)
LLAMA_GUARD_3 = {
    "S1": "Violent Crimes", "S2": "Non-Violent Crimes", "S3": "Sex-Related Crimes",
    "S4": "Child Sexual Exploitation", "S5": "Defamation", "S6": "Specialized Advice",
    "S7": "Privacy", "S8": "Intellectual Property", "S9": "Indiscriminate Weapons",
    "S10": "Hate", "S11": "Suicide & Self-Harm", "S12": "Sexual Content", "S13": "Elections",
    "S14": "Code Interpreter Abuse",
}
# closest Llama Guard 3 codes for each OpenAI category; () = no counterpart
MAPPING = {
    "harassment": (), "harassment/threatening": ("S1",),
    "hate": ("S10",), "hate/threatening": ("S10", "S1"),
    "self-harm": ("S11",), "self-harm/intent": ("S11",), "self-harm/instructions": ("S11",),
    "sexual": ("S12",), "sexual/minors": ("S4",),
    "violence": ("S1",), "violence/graphic": (),
    "illicit": ("S2",), "illicit/violent": ("S1", "S9"),
}
SUFFIXES = ("/threatening", "/intent", "/instructions", "/graphic")
ADVICE_PROMPTS = ("should i invest in memecoins", "diagnose my headache")


def doc_categories(doc):
    """The bullet list under 'Categories (13 booleans in the response schema)'."""
    block = doc.split("Categories (13 booleans in the response schema):")[1].strip().split("\n\n")[0]
    return [c.strip() for line in block.splitlines() for c in line.lstrip("- ").split(",")]


def source_categories(ref):
    """The same list, as the docstring under CATEGORIES in code/main.py states it."""
    text = inspect.getsource(ref).split("returns 13 booleans:")[1].split(". We collapse")[0]
    return [c.strip() for c in " ".join(text.split()).split(",")]


def gaps(openai):
    """OpenAI categories with no / several Llama Guard codes, and uncovered Llama Guard codes."""
    covered = {code for codes in MAPPING.values() for code in codes}
    return {
        "no_home": [c for c in openai if not MAPPING[c]],
        "split": [c for c in openai if len(MAPPING[c]) > 1],
        "lg_orphans": [f"{k} {v}" for k, v in LLAMA_GUARD_3.items() if k not in covered],
    }


def toy(ref, openai):
    """What the reference harness keeps of the 13, and what of it can fire."""
    return {
        "collapsed": [c for c in openai if c not in ref.CATEGORIES],
        "suffixed": [c for c in openai if c.endswith(SUFFIXES)],
        "toy_extra": [c for c in ref.CATEGORIES if c not in openai],
        "untriggered": [c for c in ref.CATEGORIES if c not in ref.CATEGORY_TRIGGERS],
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    doc = parity.doc_text(PHASE, LESSON, "en")
    openai = doc_categories(doc)
    return {
        "doc_openai": openai, "src_openai": source_categories(ref),
        "doc_lg_count": int(re.search(r"(\d+) MLCommons hazard", doc)[1]),
        **gaps(openai), **toy(ref, openai),
        "custom": [ref.custom_domain_rules(p)[1].split(":")[0] for p in ADVICE_PROMPTS],
        "illicit_split_in_doc": "illicit/violent" in openai,
    }


def verify(result):
    r, openai = result, result["doc_openai"]
    orphans = [o.split()[0] for o in r["lg_orphans"]]
    return [
        practice.Check(
            "the mapping covers the lesson's 13 OpenAI categories and 14 Llama Guard codes",
            sorted(openai) == sorted(r["src_openai"]) == sorted(MAPPING)
            and (len(openai), r["doc_lg_count"]) == (13, len(LLAMA_GUARD_3)),
            f"docs list {len(openai)}, same set as the code docstring; docs say "
            f"{r['doc_lg_count']} Llama Guard categories",
        ),
        practice.Check(
            "ANSWER: 4 of 13 do not map cleanly -- 2 have no counterpart, 2 split",
            (r["no_home"], r["split"])
            == (["harassment", "violence/graphic"], ["hate/threatening", "illicit/violent"]),
            f"no counterpart {r['no_home']}; split {r['split']}",
        ),
        practice.Check(
            "ANSWER: 7 of 14 Llama Guard codes have no OpenAI counterpart",
            orphans == ["S3", "S5", "S6", "S7", "S8", "S13", "S14"], f"{r['lg_orphans']}",
        ),
        practice.Check(
            "FINDING: the lesson's custom layer implements a Llama Guard category",
            r["custom"] == ["financial-advice", "medical-advice"] and "S6" in orphans,
            f"custom rules fire as {r['custom']}; S6 has no OpenAI counterpart",
        ),
        practice.Check(
            "FINDING: the toy drops 5 of the 13 categories, and 3 of its 8 cannot fire",
            (len(r["collapsed"]), r["toy_extra"], r["untriggered"], r["illicit_split_in_doc"])
            == (5, [], ["sexual", "sexual/minors", "illicit"], True) and r["collapsed"] == r["suffixed"],
            f"collapsed {r['collapsed']}; no triggers {r['untriggered']}; "
            f"'illicit/violent' in the docs' own list: {r['illicit_split_in_doc']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
