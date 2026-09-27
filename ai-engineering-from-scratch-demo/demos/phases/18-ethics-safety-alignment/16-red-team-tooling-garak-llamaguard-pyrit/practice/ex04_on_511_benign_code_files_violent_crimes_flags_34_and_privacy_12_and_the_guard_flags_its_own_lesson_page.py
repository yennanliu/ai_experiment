"""Exercise 4 — on 511 benign code files, violent_crimes flags 34 and privacy 12, and the guard flags its own lesson page.

    Read Llama Guard 3's hazard-category list. Identify two categories where
    the training data would realistically produce high false-positive rates
    on legitimate developer content.

Reading of the exercise: "legitimate developer content" is taken literally
as a corpus: every `code/*.py` in the reference curriculum outside Phase 18
(511 files, all of it teaching code, none of it a request for harm). The
lesson's Llama-Guard-style classifier is run line by line over it, and a
category's false-positive rate is the share of files it flags. The toy is
keyword-based, so hits that exist only because "kill" sits inside "skill"
-- the toy's own artefact, which a learned classifier would not share --
are counted separately.

**ANSWER: Violent Crimes (S1) and Privacy (S7).** Of 511 benign files the
guard flags 49; violent_crimes flags 34 and privacy 12, ahead
of code_interpreter_abuse (7), indiscriminate_weapons (5) and self_harm (1).
Both are developer vocabulary, not hazard talk: "kill switch",
`proc.kill()`, "pod kill" and `kill_z_score` for S1; PII-redaction code
with `ssn` patterns and "social security number" refusals for S7. A
classifier trained on harmful requests that use these words would learn
the same association, which is why these two are the realistic answer; the
lesson's non-obvious S14 (Code Interpreter Abuse) is third, from sandbox
denylists that literally contain `exec(` and `os.system`.

**FINDING: 82% of the violent_crimes hits are "skill".** 223 of the 273
lines flagged as violent_crimes stop being flagged once "skill" is deleted
from them -- `skill`, `skills`, `skill_path` -- and 17 of the 34 files are
flagged for nothing else. The toy's substring match turns every agent-skills
lesson into a violent-crime report. Without them violent_crimes still flags
17 files, more than privacy's 12, so the answer does not rest on the
artefact.

**FINDING: the guard flags its own lesson page, on the category list.**
`guard_classify(docs/en.md)` scores violent_crimes and self_harm 0.9: from
"indiscriminate weapons" and "suicide/self-harm" in the hazard list itself,
and "kill" inside `outputs/skill-red-team-stack.md`.

**FINDING: 8 of the 14 categories can never fire.** Only 6 have triggers;
csam, defamation, elections, hate, ip, sex_related, sexual_content and
specialized_advice always score 0.0, so their false-positive rate on any
corpus is 0 by construction -- and so is their recall.

Structure: `corpus()` reads the files; `scan()` tallies flagged files and
lines per category, and the violent_crimes lines and files that are clean
once "skill" is deleted.
"""

from __future__ import annotations

import collections

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "16-red-team-tooling-garak-llamaguard-pyrit"


def corpus():
    root = parity.find_reference_root()
    return [f for f in sorted(root.glob("phases/*/*/code/*.py")) if PHASE not in str(f)]


def violent(ref, text):
    return ref.guard_classify(text)["violent_crimes"] >= 0.5


def scan(ref, files):
    """(files flagged, per-category file and line counts, violent lines/files that are only 'skill')."""
    by_file, by_line = collections.Counter(), collections.Counter()
    flagged, skill_lines, skill_files = 0, 0, 0
    for f in files:
        hit, real = set(), False
        for line in f.read_text(errors="ignore").splitlines():
            cats = [c for c, v in ref.guard_classify(line).items() if v >= 0.5]
            hit.update(cats)
            by_line.update(cats)
            only_skill = "violent_crimes" in cats and not violent(ref, line.lower().replace("skill", ""))
            skill_lines += only_skill
            real = real or ("violent_crimes" in cats and not only_skill)
        by_file.update(hit)
        flagged += bool(hit)
        skill_files += "violent_crimes" in hit and not real
    return flagged, by_file, by_line, skill_lines, skill_files


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    files = corpus()
    flagged, by_file, by_line, skill_lines, skill_files = scan(ref, files)
    page = parity.doc_text(PHASE, LESSON).lower()
    return {
        "files": len(files), "flagged": flagged,
        "by_file": dict(by_file.most_common()), "by_line": dict(by_line.most_common()),
        "skill_lines": skill_lines, "skill_files": skill_files,
        "s_numbers": [ref.HAZARD_CATEGORIES.index(c) + 1 for c in
                      ("violent_crimes", "privacy", "code_interpreter_abuse")],
        "page": sorted(c for c, v in ref.guard_classify(page).items() if v),
        "page_triggers": sorted(t for ts in ref.CATEGORY_TRIGGERS.values() for t in ts if t in page),
        "kill_outside_skill": page.replace("skill", "").count("kill"),
        "dead": sorted(set(ref.HAZARD_CATEGORIES) - set(ref.CATEGORY_TRIGGERS)),
    }


def verify(result):
    by_file, by_line = result["by_file"], result["by_line"]
    return [
        practice.Check(
            "ANSWER: Violent Crimes (S1) and Privacy (S7)",
            all([(result["files"], result["flagged"]) == (511, 49), result["s_numbers"] == [1, 7, 14],
                 by_file == {"violent_crimes": 34, "privacy": 12, "code_interpreter_abuse": 7,
                             "indiscriminate_weapons": 5, "self_harm": 1}]),
            f"{result['flagged']} of {result['files']} benign files flagged; files per "
            f"category {by_file}; MLCommons numbers {result['s_numbers']}",
        ),
        practice.Check(
            "FINDING: 82% of the violent_crimes hits are 'skill'",
            all([by_line["violent_crimes"] == 273, result["skill_lines"] == 223,
                 result["skill_files"] == 17, by_file["violent_crimes"] - 17 > by_file["privacy"]]),
            f"violent_crimes lines {by_line['violent_crimes']}, of which {result['skill_lines']} "
            f"({result['skill_lines'] / by_line['violent_crimes']:.0%}) are clean without 'skill'; "
            f"files flagged only by 'skill' {result['skill_files']} of {by_file['violent_crimes']}",
        ),
        practice.Check(
            "FINDING: the guard flags its own lesson page, on the category list",
            all([result["page"] == ["self_harm", "violent_crimes"],
                 result["page_triggers"] == ["kill", "self-harm", "weapon"],
                 result["kill_outside_skill"] == 0]),
            f"docs/en.md flags {result['page']} via {result['page_triggers']}; 'kill' outside "
            f"'skill': {result['kill_outside_skill']}",
        ),
        practice.Check(
            "FINDING: 8 of the 14 categories can never fire",
            len(result["dead"]) == 8,
            f"no triggers: {result['dead']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
