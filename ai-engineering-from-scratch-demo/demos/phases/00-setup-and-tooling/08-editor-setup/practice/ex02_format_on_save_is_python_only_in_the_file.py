"""Exercise 2 — the file pastes cleanly, but its formatOnSave is Python-only, the doc's global.

    Copy the `settings.json` from this lesson into your VS Code config

Reading of the exercise: copying a file into a user's editor profile is a host
action CI cannot take, so "copy" is read as "would the copy work, and is it the
config the doc describes?". The lesson's `code/vscode/settings.json` is parsed
with `json.loads`, and the two JSON snippets the doc says are "the key settings"
(Step 3) and the terminal setup (Step 4) are parsed and compared key by key.

**ANSWER: yes, as-is.** The file is strict JSON -- no comments, no trailing
commas, though the doc labels its snippets `jsonc` -- with 30 top-level keys,
so it can replace an empty user `settings.json` or be merged key by key.

**FINDING: copying the file and copying the doc's snippet give different
editors.** Step 3 sets `"editor.formatOnSave": true` at the top level, which
formats every language on save. The file never sets it there: it appears only
inside `"[python]"`, so after copying the file a YAML, JSON or Markdown file is
never formatted on save. 4 of the snippet's 5 keys match the file's top level.

**CONTROL: everything the file configures is installed by the recommendations.**
Each extension-scoped key prefix (`python.analysis` for Pylance,
`black-formatter`, `ruff`, `gitlens`) and the Python default formatter belong
to an extension in `extensions.json`; the Step 4 terminal snippet matches the
file on all 4 keys.

Structure: `snippets` reads the doc's ```jsonc blocks; the checks compare them.
"""

from __future__ import annotations

import json
import re

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "08-editor-setup"
PREFIX_OWNER = {
    "python.analysis.": "ms-python.vscode-pylance",
    "black-formatter.": "ms-python.black-formatter",
    "ruff.": "charliermarsh.ruff",
    "gitlens.": "eamodio.gitlens",
}


def snippets(doc):
    return [json.loads(b) for b in re.findall(r"```jsonc\n(.*?)```", doc, re.S)]


def solve():
    vscode = parity.lesson_dir(PHASE, LESSON) / "code" / "vscode"
    raw = (vscode / "settings.json").read_text(encoding="utf-8")
    settings = json.loads(raw)
    recommended = json.loads((vscode / "extensions.json").read_text())["recommendations"]
    key_settings, terminal = snippets(parity.doc_text(PHASE, LESSON))
    owners = {
        PREFIX_OWNER[p] for k in settings for p in PREFIX_OWNER if k.startswith(p)
    } | {settings["[python]"]["editor.defaultFormatter"]}
    return {
        "keys": len(settings),
        "comments": len(re.findall(r"(?m)^\s*//", raw)),
        "doc_snippet": key_settings,
        "file_top": {k: settings.get(k) for k in key_settings},
        "python_block": settings["[python]"],
        "terminal_match": sum(settings.get(k) == v for k, v in terminal.items()),
        "terminal_keys": len(terminal),
        "owners": sorted(owners),
        "missing_owner": sorted(owners - set(recommended)),
    }


def verify(result):
    doc, top = result["doc_snippet"], result["file_top"]
    match = [k for k in doc if top[k] == doc[k]]
    return [
        practice.Check(
            "ANSWER: settings.json is strict JSON with 30 keys, ready to copy",
            result["keys"] == 30 and result["comments"] == 0,
            f"json.loads reads {result['keys']} top-level keys and the file has "
            f"{result['comments']} comment lines",
        ),
        practice.Check(
            "FINDING: the doc's formatOnSave is global, the file's is Python-only",
            doc.get("editor.formatOnSave") is True
            and top["editor.formatOnSave"] is None
            and result["python_block"].get("editor.formatOnSave") is True
            and len(match) == 4,
            f"Step 3's snippet sets editor.formatOnSave at the top level; the file sets it only "
            f"in [python]; {len(match)} of {len(doc)} snippet keys match the file's top level",
        ),
        practice.Check(
            "CONTROL: every extension the settings configure is recommended",
            not result["missing_owner"]
            and result["terminal_match"] == result["terminal_keys"] == 4,
            f"settings configure {', '.join(result['owners'])}; none missing from "
            f"extensions.json; Step 4 terminal snippet matches {result['terminal_match']} of "
            f"{result['terminal_keys']} keys",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
