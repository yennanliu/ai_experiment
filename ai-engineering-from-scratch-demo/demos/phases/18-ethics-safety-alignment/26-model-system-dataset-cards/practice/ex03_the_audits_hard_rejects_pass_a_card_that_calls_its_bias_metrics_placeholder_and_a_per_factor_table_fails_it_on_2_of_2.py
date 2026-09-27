"""Exercise 3 — the audit's hard rejects pass a card that calls its bias metrics placeholder and a per-factor table fails it on 2 of 2.

    Read Oreamuno et al. 2023 on the 0.3% adoption rate. Propose one
    structural change to the model card specification that would increase
    ethical-considerations adoption.

Reading of the exercise: adoption itself cannot be measured offline, so the
proposal is tested on what it changes: which cards pass. "Adoption" that a
one-line disclaimer satisfies is not adoption, so the lesson's own card and
its own audit skill are the test bench: does the rule accept the placeholder
section the lesson ships, and can an honest card satisfy it?

**ANSWER: make Ethical Considerations a table generated from the Factors
section -- one row per listed factor, each with a risk, a measured value (or
an explicit "not measurable: <reason>") and a mitigation -- and fail
validation on any factor without a complete row.** The template fills in the
row keys, so the author meets a table with named blanks instead of an
optional free-text heading; an empty row is a visible validation error, not
a silent omission. That is the structural change: the section stops being
optional prose and becomes a function of a section authors already fill in.

**FINDING: the lesson's audit passes the lesson's placeholder card.** The
skill file's three hard rejects (no Ethical Considerations; dataset without
datasheet; "bias-tested" without disaggregation) fire 0 of 3 times on the
generated model card, although that card's Ethical Considerations says its
bias metrics are placeholder. A presence rule counts it as adopted. The
per-factor rule fails it: Factors lists gender and age bucket, and neither
appears in Ethical Considerations (0 of 2) or names a Quantitative Analysis
line (0 of 2; the card says only "group0 vs group1").

**FINDING: the table exposes a datasheet mismatch no presence rule sees.**
The datasheet documents 1 binary sensitive attribute, the model card lists
2 factors, so the age-bucket row cannot be measured from the documented data.
A card carrying Exercise 2's measured gender row and "not measurable:
datasheet has no age field" for age passes the rule, so the rule is
satisfiable -- and the unmeasurable factor is now written down, not hidden.

Structure: `factors()` reads the card's sensitive attributes;
`hard_rejects()` implements the skill file's three rules; `validate()` is
the proposed per-factor rule.
"""

from __future__ import annotations

import pathlib
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "26-model-system-dataset-cards"
HERE = pathlib.Path(__file__).resolve().parent
EX01 = practice.load_module(next(HERE.glob("ex01_*.py")))
EX02 = practice.load_module(next(HERE.glob("ex02_*.py")))
MEASURED = re.compile(r".*\d|not measurable")      # a number, or an explicit reason


def factors(card):
    """The sensitive attributes listed under Factors, parentheticals dropped."""
    line = next(b for b in EX01.sections(card)["Factors"] if b.startswith("Sensitive"))
    return [re.sub(r"\(.*?\)", "", f).strip() for f in line.split(":", 1)[1].split(",")]


def hard_rejects(ref, card):
    """The skill file's hard-reject bullets, and which of them fire on `card`."""
    skill = (parity.lesson_dir(PHASE, LESSON) / "outputs" / "skill-card-audit.md").read_text()
    rules = re.findall(r"(?m)^- (.+)$", skill.split("Hard rejects:", 1)[1].split("\n\n", 1)[0])
    secs, low = EX01.sections(card), card.lower()
    fired = ["Ethical Considerations" not in secs,
             "datasheet" in low and not ref.datasheet().strip(),
             "bias-tested" in low and "disaggregat" not in low]
    return rules, fired


def validate(card):
    """Proposed rule: every factor has a 'factor | risk | measured | mitigation' row."""
    cells = [[c.strip() for c in b.split("|")] for b in EX01.sections(card)["Ethical Considerations"]]
    rows = {c[0]: c[2] for c in cells if len(c) == 4 and all(c)}
    return {f: bool(MEASURED.match(rows.get(f, ""))) for f in factors(card)}


def mentions(card, section):
    text = " ".join(EX01.sections(card)[section]).lower()
    return [f for f in factors(card) if f.lower() in text]


def honest_card(card):
    """The reference card with its Ethical Considerations replaced by the per-factor table."""
    rows, gap, half, _ = EX02.table(*EX02.measured()[::3])
    table = (f"- gender | selection-rate gap | parity gap {gap:+} +/- {half} (n={rows[0]['n']}"
             f"/{rows[1]['n']}) | re-measure on deployment data\n"
             "- age bucket | unmeasured group harm | not measurable: datasheet has no age field "
             "| add age to the datasheet or drop the factor\n")
    head, rest = card.split("## Ethical Considerations\n", 1)
    return head + "## Ethical Considerations\n" + table + "\n## " + rest.split("\n## ", 1)[1]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    card = ref.model_card()
    rules, fired = hard_rejects(ref, card)
    composition = " ".join(EX01.sections(ref.datasheet())["Composition"])
    return {
        "factors": factors(card), "rules": len(rules), "fired": sum(fired),
        "admits": "placeholder" in " ".join(EX01.sections(card)["Ethical Considerations"]),
        "ec_mentions": mentions(card, "Ethical Considerations"),
        "qa_mentions": mentions(card, "Quantitative Analysis"),
        "reference": validate(card), "honest": validate(honest_card(card)),
        "datasheet_attrs": int(re.search(r"(\d+) binary sensitive attribute", composition).group(1)),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: a per-factor Ethical Considerations table, validated against Factors",
            result["factors"] == ["gender", "age bucket"]
            and result["reference"] == {"gender": False, "age bucket": False}
            and result["honest"] == {"gender": True, "age bucket": True},
            f"factors {result['factors']}; rule on the reference card {result['reference']}, "
            f"on the per-factor card {result['honest']}",
        ),
        practice.Check(
            "FINDING: the lesson's audit passes the lesson's placeholder card",
            result["rules"] == 3 and result["fired"] == 0 and result["admits"]
            and result["ec_mentions"] == [] and result["qa_mentions"] == [],
            f"{result['fired']} of {result['rules']} hard rejects fire; card admits placeholder: "
            f"{result['admits']}; factors named in Ethical Considerations {result['ec_mentions']}, "
            f"in Quantitative Analysis {result['qa_mentions']}",
        ),
        practice.Check(
            "FINDING: the table exposes a datasheet mismatch no presence rule sees",
            result["datasheet_attrs"] == 1 < len(result["factors"]) == 2,
            f"datasheet documents {result['datasheet_attrs']} sensitive attribute; model card "
            f"lists {len(result['factors'])} factors",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
