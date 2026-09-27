"""Exercise 1 — the reference fills two of the three columns, and its PF ladder has v1's four levels where v2 has two.

    Read RSP v3.0, PF v2, and FSF v3.0. Compile a table of each lab's CBRN
    threshold, each's AI R&D threshold, and each's required pre-deployment
    evaluation.

Reading of the exercise: the table is compiled from the three primary
documents, read 2026-09-27: the RSP v3.0 PDF (effective 2026-02-24), the PF v2
PDF and announcement (2025-04-15), and the FSF v3.0 PDF (2025-09-22). Each
cell is then set against what the lesson ships -- `code/main.py`'s LABS table,
the lesson page and its skill file -- so the table doubles as a diff of the
reference against its sources.

**ANSWER:**

| lab | CBRN threshold | AI R&D threshold | pre-deployment evaluation |
|---|---|---|---|
| RSP v3.0 | non-novel chem/bio weapons production (people with basic technical backgrounds); novel chem/bio (moderately resourced expert-backed teams) | automated R&D in key domains, operationalized as compressing two years of 2018-2024 AI progress into one | no prespecified evals; a Risk Report every 3-6 months, externally reviewed when it covers highly capable models and is significantly redacted |
| PF v2 | Biological and Chemical, High and Critical | AI Self-improvement, High and Critical | Capabilities Report + Safeguards Report, reviewed by the Safety Advisory Group; High needs safeguards before deployment, Critical during development too |
| FSF v3.0 | CBRN uplift level 1 (security level 2) | ML R&D acceleration level 1 (SL3), ML R&D automation level 1 (SL4) | a safety case per CCL reached, reviewed by a governance function before external launch, and before large-scale internal deployment for the ML R&D CCLs |

**FINDING: the reference cannot fill the third column.** LABS has six fields
and none is an evaluation. The lesson's per-lab sections mention evaluation
0 / 1 / 0 times, and the one OpenAI mention is a tracking criterion
("Empirical evaluation possible"), not a required evaluation.

**FINDING: the reference's PF ladder is PF v1's.** main.py gives PF v2
"Low / Medium / High / Critical"; v2 keeps only High and Critical. It also
says AI R&D "Critical definitions pending"; v2 defines Critical for AI
Self-improvement.

**FINDING: the lesson's Anthropic AI R&D names are not RSP v3.0's.** The page
lists a five-rung ASL-1..ASL-5+ ladder with ASL-4 = "AI R&D-2" (entry-level
research), and the skill file maps AI R&D-4 to "substantially accelerate
scaling". The v3.0 PDF contains "AI R&D-2" zero times and "ASL-4" zero
times. "AI R&D-4" appears only in its v2.2 changelog, where it is the
entry-level threshold. Its Appendix B says future levels are defined by the
argument required, not by ASL control lists.

**FINDING: FSF has no "Bioweapon Uplift CCL".** main.py and the page use that
name; v3.0 calls it CBRN uplift level 1. The page lists four CCL domains and
omits ML R&D automation level 1 and the exploratory misalignment
(instrumental reasoning) levels.

Structure: `PRIMARY` holds the compiled table; `reference_view()` reads LABS,
the page and the skill file.
"""

from __future__ import annotations

import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "18-frontier-safety-frameworks-rsp-pf-fsf"
# Primary sources, read 2026-09-27. Column order: CBRN, AI R&D, pre-deployment evaluation.
PRIMARY = {
    "RSP v3.0": ("non-novel / novel chem-bio weapons production",
                 "automated R&D: two years of 2018-2024 progress in one",
                 "Risk Report every 3-6 months; external review if highly capable and redacted"),
    "PF v2": ("Biological and Chemical: High, Critical", "AI Self-improvement: High, Critical",
              "Capabilities + Safeguards Reports; Safety Advisory Group review"),
    "FSF v3.0": ("CBRN uplift level 1", "ML R&D acceleration level 1; ML R&D automation level 1",
                 "per-CCL safety case reviewed before external or large internal deployment"),
}
PF_V2_LEVELS = ("High", "Critical")
RSP_V3_COUNTS = {"AI R&D-2": 0, "ASL-4": 0, "AI R&D-4": 1}     # occurrences in the v3.0 PDF


def section(doc, heading):
    return doc.split(f"### {heading}")[1].split("\n### ")[0]


def reference_view(ref):
    doc = parity.doc_text(PHASE, LESSON)
    skill = (parity.lesson_dir(PHASE, LESSON) / "outputs" / "skill-framework-diff.md").read_text()
    labs = {lab["name"].split(" (")[0].split(" ", 1)[1]: lab for lab in ref.LABS}
    return {
        "labs": list(labs), "fields": list(ref.LABS[0]),
        "eval_mentions": [len(re.findall("evaluat", section(doc, h), re.I))
                          for h in ("Anthropic", "OpenAI", "DeepMind")],
        "openai_eval": re.search(r"[^.\n]*[Ee]valuation[^.\n]*", section(doc, "OpenAI")).group(0),
        "pf_levels": labs["PF v2"]["tier_structure"].split(" per ")[0].split(" / "),
        "pf_ai_rd": labs["PF v2"]["ai_rd_threshold"],
        "asl_ladder": re.findall(r"^- (ASL-[^:]+):", doc, re.M),
        "asl4": re.search(r"^- ASL-4: ([^;]+);", doc, re.M).group(1),
        "skill_rd4": re.search(r'"([^"]+)" \(Anthropic AI R&D-4\)', skill).group(1),
        "fsf_cbrn": labs["FSF v3.0"]["cbrn_threshold"],
        "fsf_ccls": [b.split(" (")[0].split(":")[0]
                     for b in re.findall(r"^- (.+)$", section(doc, "DeepMind"), re.M)[:4]],
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    view = reference_view(ref)
    view["table"], view["shape"] = PRIMARY, [len(v) for v in PRIMARY.values()]
    view["pf_extra"] = [x for x in view["pf_levels"] if x not in PF_V2_LEVELS]
    view["filled"] = sum(any(k in f for f in view["fields"]) for k in ("cbrn", "ai_rd", "eval"))
    return view


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: a 3 x 3 table, one row per lab the reference names",
            (r["labs"], r["shape"]) == (list(r["table"]), [3, 3, 3]),
            f"rows {list(r['table'])} match LABS {r['labs']}",
        ),
        practice.Check(
            "FINDING: the reference cannot fill the third column",
            (r["filled"], r["eval_mentions"], r["openai_eval"].strip("* "))
            == (2, [0, 1, 0], "Empirical evaluation possible"),
            f"LABS fields {r['fields']} fill {r['filled']} of 3 columns; page sections mention "
            f"evaluation {r['eval_mentions']} times, the one being {r['openai_eval'].strip('* ')!r}",
        ),
        practice.Check(
            "FINDING: the reference's PF ladder is PF v1's",
            (r["pf_levels"], r["pf_extra"], r["pf_ai_rd"].split("; ")[-1])
            == (["Low", "Medium", "High", "Critical"], ["Low", "Medium"], "Critical definitions pending"),
            f"main.py PF levels {r['pf_levels']} vs v2 {PF_V2_LEVELS}; AI R&D cell {r['pf_ai_rd']!r}",
        ),
        practice.Check(
            "FINDING: the lesson's Anthropic AI R&D names are not RSP v3.0's",
            (r["asl_ladder"], r["asl4"], r["skill_rd4"])
            == (["ASL-1", "ASL-2", "ASL-3", "ASL-4", "ASL-5+"], "AI R&D-2 crossing threshold",
                "substantially accelerate scaling"),
            f"page ladder {r['asl_ladder']}, ASL-4 = {r['asl4']!r}, skill file AI R&D-4 = "
            f"{r['skill_rd4']!r}; occurrences in the v3.0 PDF: {RSP_V3_COUNTS}",
        ),
        practice.Check(
            "FINDING: FSF has no 'Bioweapon Uplift CCL'",
            (r["fsf_cbrn"], r["fsf_ccls"]) == ("Bioweapon Uplift CCL", [
                "Bioweapon Uplift", "Cyber Uplift", "ML R&D Acceleration", "Harmful Manipulation"]),
            f"main.py {r['fsf_cbrn']!r} vs v3.0 {PRIMARY['FSF v3.0'][0]!r}; page CCLs {r['fsf_ccls']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
