"""Exercise 1 — four sections are placeholder-only and the Quantitative Analysis is typed in by hand.

    Run `code/main.py`. Inspect the generated cards. Identify sections that
    are weak (placeholder-only) and specify what evidence would strengthen
    them.

Reading of the exercise: "placeholder-only" is made mechanical so the audit
is repeatable: a bullet is a placeholder when it is `N/A`, a pointer
elsewhere ("see ..."), a self-declared placeholder, a bare disclaimer ("not
validated") or a dead end ("goes nowhere"). A section is weak when every
bullet is one; partial when some are. The three cards are also held against
the section lists the lesson itself gives for each format, and the card's
numbers against the code that is supposed to have produced them.

**ANSWER: four sections are placeholder-only -- model card Metrics,
Training Data and Ethical Considerations, and system card Regulatory
Alignment -- and two more are partial: Security Capabilities (1 of 3
bullets, prompt injection `N/A`) and Incident Response (1 of 2).** The
evidence that would strengthen each: Metrics needs definitions, thresholds
and the evaluation split; Training Data needs the datasheet's counts, seed
and label rule inline; Ethical Considerations needs one row per listed
factor with the measured gap and a mitigation (Exercise 3); Regulatory
Alignment needs the reasoning behind each `N/A` (which article, why out of
scope); prompt injection needs a measured test result even for a
non-generative model (Exercise 5); incident response needs an owner and a
response time.

**FINDING: the model card is missing one of the lesson's nine Mitchell
sections.** The lesson lists nine; the generator emits eight -- there is no
Evaluation Data section. The datasheet covers all seven Gebru sections and
the system card all five of the lesson's system-card items.

**FINDING: the Quantitative Analysis looks like the strongest section and is
typed in by hand.** Its three figures (0.97, +0.03, -0.01) are string
literals in `model_card()`; `main.py` imports nothing but `__future__` and
trains nothing, and the card's own Ethical Considerations says "Bias metrics
are placeholder". The datasheet's "fixed seed" names no seed, and "regenerated
on every run" describes a dataset the code never generates. The evidence is
the run itself: Exercise 2 regenerates the data from the datasheet and
measures the numbers.

Structure: `sections()` parses a card; `lesson_items()` reads a format's
canonical list out of the lesson; `audit()` classifies every bullet.
"""

from __future__ import annotations

import inspect
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "26-model-system-dataset-cards"
PLACEHOLDER = re.compile(r"\bN/A\b|placeholder|\bsee\b|not validated|goes nowhere", re.I)
HEADINGS = {"model_card": "### Model Cards", "datasheet": "### Datasheets for Datasets",
            "system_card": "### System Cards"}


def cards(ref):
    return {name: getattr(ref, name)() for name in HEADINGS}


def sections(card):
    """{heading: [bullet, ...]} for one generated card."""
    out, head = {}, None
    for line in card.strip().splitlines():
        if line.startswith("## "):
            head = line[3:]
            out[head] = []
        elif line.startswith("- ") and head:
            out[head].append(line[2:])
    return out


def lesson_items(heading):
    """The canonical section list the lesson gives under `heading`."""
    body = parity.doc_text(PHASE, LESSON).split(heading, 1)[1].split("\n#", 1)[0]
    return [m.group(1) for m in re.finditer(r"(?m)^- (?:\*\*)?([^\n]+)", body)]


def first_word(text):
    return re.split(r"[\s.:(*]", text.strip().lower(), maxsplit=1)[0]


def missing(card, heading):
    """Canonical items whose first word is neither a heading nor a bullet label."""
    secs = sections(card)
    have = {first_word(h) for h in secs} | {first_word(b) for bs in secs.values() for b in bs}
    return [i for i in lesson_items(heading) if first_word(i) not in have]


def audit(card):
    """(weak, partial): sections where all / some bullets are placeholders."""
    weak, partial = [], []
    for head, bullets in sections(card).items():
        hits = sum(bool(PLACEHOLDER.search(b)) for b in bullets)
        if hits == len(bullets):
            weak.append(head)
        elif hits:
            partial.append(f"{head} {hits}/{len(bullets)}")
    return weak, partial


def per_card(docs):
    return {k: {"audit": audit(docs[k]), "canonical": len(lesson_items(h)),
                "emitted": len(sections(docs[k])), "missing": missing(docs[k], h)}
            for k, h in HEADINGS.items()}


def provenance(ref, docs):
    """Where the model card's numbers and the datasheet's seed come from."""
    qa = sections(docs["model_card"])["Quantitative Analysis"]
    seed_line = next(b for b in sections(docs["datasheet"])["Collection Process"] if "seed" in b)
    return {
        "qa_numbers": [re.search(r"[-+]?\d\.\d+", b).group() for b in qa],
        "literal": all(b in inspect.getsource(ref.model_card) for b in qa),
        "imports": re.findall(r"(?m)^(?:from|import) (\S+)", inspect.getsource(ref)),
        "admits": "placeholder" in " ".join(sections(docs["model_card"])["Ethical Considerations"]),
        "seed_digits": re.findall(r"\d+", seed_line),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    docs = cards(ref)
    return {"cards": per_card(docs), **provenance(ref, docs)}


def verify(result):
    c = result["cards"]
    a, miss = ({k: v[f] for k, v in c.items()} for f in ("audit", "missing"))
    counts = [(v["canonical"], v["emitted"]) for v in c.values()]
    prov = (result["qa_numbers"], result["literal"], result["imports"], result["admits"],
            result["seed_digits"])
    return [
        practice.Check(
            "ANSWER: four sections are placeholder-only and two are partial",
            a == {"model_card": (["Metrics", "Training Data", "Ethical Considerations"], []),
                  "datasheet": ([], []),
                  "system_card": (["Regulatory Alignment"],
                                  ["Security Capabilities 1/3", "Incident Response 1/2"])},
            f"(weak, partial) per card: {a}",
        ),
        practice.Check(
            "FINDING: the model card is missing one of the lesson's nine Mitchell sections",
            counts == [(9, 8), (7, 7), (5, 5)]
            and miss == {"model_card": ["Evaluation data."], "datasheet": [], "system_card": []},
            f"(lesson sections, emitted headings) for model card, datasheet, system card: "
            f"{counts}; uncovered: {miss}",
        ),
        practice.Check(
            "FINDING: the Quantitative Analysis is typed in by hand",
            prov == (["0.97", "+0.03", "-0.01"], True, ["__future__"], True, []),
            f"figures {result['qa_numbers']} are literals in model_card(); main.py imports "
            f"{result['imports']}; Ethical Considerations admits placeholder: {result['admits']}; "
            f"digits in the datasheet's seed line: {result['seed_digits']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
