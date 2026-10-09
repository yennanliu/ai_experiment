"""Exercise 1 — Step 2 installs 8 of the 13 recommended extensions, and the 13 never prompt.

    Install VS Code and all extensions listed in Step 2

Reading of the exercise: installing an editor is a host action CI cannot take,
so the install is replaced by its specification: the `code --install-extension`
lines of Step 2, parsed from `docs/en.md`, checked as well-formed marketplace IDs
and against the lesson's own `code/vscode/extensions.json`, which the doc says
VS Code will offer to install for you.

**ANSWER: 8 `code --install-extension` commands, all well-formed and
duplicate-free**, one per row of Step 2's "What each one does" table.

**FINDING: Step 2 installs 8 of the 13 recommended extensions.** The 5 the doc
never installs are Dev Containers, the two Jupyter cell-tag/slideshow
extensions, YAML and Even Better TOML. A learner who follows Step 2 and skips
the prompt ends up without them.

**FINDING: the prompt never comes.** The doc says the list lives at
`code/.vscode/extensions.json` and "When you open the project folder, VS Code
will prompt you". The lesson ships `code/vscode/` -- no dot -- and VS Code only
reads workspace recommendations from `.vscode/` at the folder it opens; there is
no `.vscode` directory anywhere in the lesson. The zh doc repeats the same path.

**CONTROL: extensions.json is strict JSON** with one key, `recommendations`.

Structure: `step2` reads the doc's install block; the checks diff the two lists.
"""

from __future__ import annotations

import json
import re

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "08-editor-setup"
EXT_ID = re.compile(r"^[a-z0-9][a-z0-9-]*\.[a-z0-9][a-z0-9-]*$")


def step2(doc):
    section = doc.split("### Step 2", 1)[1].split("### Step 3", 1)[0]
    ids = re.findall(r"(?m)^code --install-extension (\S+)$", section)
    rows = re.findall(r"(?m)^\| (?!Extension|-)[^|]+\|", section)
    return ids, len(rows)


def solve():
    root = parity.lesson_dir(PHASE, LESSON)
    raw = json.loads((root / "code" / "vscode" / "extensions.json").read_text(encoding="utf-8"))
    doc = parity.doc_text(PHASE, LESSON)
    ids, rows = step2(doc)
    return {
        "step2": ids,
        "table_rows": rows,
        "keys": sorted(raw),
        "recommended": raw["recommendations"],
        "dot_vscode": [p.name for p in root.rglob(".vscode")],
        "well_formed": all(EXT_ID.match(i) for i in ids + raw["recommendations"]),
        "unique": len(ids) == len(set(ids)),
        "plain_vscode": (root / "code" / "vscode").is_dir(),
        "doc_path": doc.count("code/.vscode/"),
        "zh_path": parity.doc_text(PHASE, LESSON, "zh").count("code/.vscode/"),
    }


def verify(result):
    ids, rec = result["step2"], result["recommended"]
    extra = sorted(set(rec) - set(ids))
    return [
        practice.Check(
            "ANSWER: 8 well-formed, duplicate-free install commands, one per table row",
            result["unique"] and result["well_formed"] and len(ids) == result["table_rows"] == 8,
            f"Step 2 installs {len(ids)}: {', '.join(ids)}; its table has "
            f"{result['table_rows']} rows",
        ),
        practice.Check(
            "FINDING: Step 2 installs 8 of the 13 recommended extensions",
            set(ids) <= set(rec) and len(rec) == 13 and len(extra) == 5,
            f"extensions.json recommends {len(rec)}; Step 2 never installs {', '.join(extra)}",
        ),
        practice.Check(
            "FINDING: the recommendations sit in code/vscode/, where VS Code never looks",
            (result["doc_path"], result["zh_path"], result["plain_vscode"], result["dot_vscode"])
            == (2, 2, True, []),
            f"docs/en.md cites code/.vscode/ {result['doc_path']} times (zh "
            f"{result['zh_path']}); the lesson has code/vscode/ and "
            f"{len(result['dot_vscode'])} .vscode directories, so no prompt appears",
        ),
        practice.Check(
            "CONTROL: extensions.json is strict JSON with one key",
            result["keys"] == ["recommendations"] and result["well_formed"],
            f"json.loads keys {result['keys']}; all {len(rec)} IDs are publisher.name",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
