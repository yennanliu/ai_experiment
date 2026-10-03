"""Exercise 3 — Black formats only on Cmd+S: the 1-second auto-save skips format-on-save.

    Open a Python file and verify that Pylance shows type hints and Black formats on save

Reading of the exercise: there is no editor in CI, so "verify" is read as
deriving, from the lesson's own `code/vscode/settings.json`, exactly which type
hints Pylance will draw and exactly which saves will run Black, using VS Code's
documented semantics for each setting (quoted in `VSCODE_SAYS`). The test file
the learner should open is then the one that separates the cases.

**ANSWER: return-type hints only, and Black on an explicit save.** Pylance runs
`basic` type checking with `inlayHints.functionReturnTypes` on and
`inlayHints.variableTypes` off, so `def f(): return 1` shows `-> int` while
`x = f()` shows no `: int` -- 1 of the 2 inlay hints is on. Black is the
`[python]` default formatter with `formatOnSave` on, at line length 88.

**FINDING: the auto-save the lesson sells disables the auto-format it sells.**
The file also sets `files.autoSave: afterDelay` with a 1000 ms delay. VS Code
documents `editor.formatOnSave` as running only if "the file must not be saved
after delay", and `codeActionsOnSave: "explicit"` (organize imports) as running
only on explicit saves. So an edit left alone is saved unformatted within a
second, and the doc's two promises -- "Never think about formatting again" and
"You will forget to save" -- cannot both hold: the save you forget is exactly
the one Black never sees. The verification passes only if you press Cmd+S.

**CONTROL: the formatter settings agree with each other.** Black's
`--line-length 88` equals the first ruler and Black's own default (so the arg is
a no-op), and `ms-python.black-formatter` is in the recommendations.

Structure: `saves` classifies auto vs explicit saves; the checks read settings.
"""

from __future__ import annotations

import json

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "08-editor-setup"
BLACK_DEFAULT = 88
VSCODE_SAYS = {
    "editor.formatOnSave": "Format a file on save. A formatter must be available, the file "
    "must not be saved after delay, and the editor must not be shutting down.",
    "explicit": "Triggers Code Actions when explicitly saved.",
}


def saves(settings):
    """Which save kinds run the formatter and the code actions, per VS Code's semantics."""
    py = settings["[python]"]
    after_delay = settings.get("files.autoSave") == "afterDelay"
    kinds = {"explicit": True, "auto": not after_delay}
    actions = set(py.get("editor.codeActionsOnSave", {}).values())
    return {
        kind: {
            "format": py.get("editor.formatOnSave", False) and ok,
            "actions": bool(actions) and (ok or "always" in actions),
        }
        for kind, ok in kinds.items()
    }


def solve():
    vscode = parity.lesson_dir(PHASE, LESSON) / "code" / "vscode"
    settings = json.loads((vscode / "settings.json").read_text(encoding="utf-8"))
    recommended = json.loads((vscode / "extensions.json").read_text())["recommendations"]
    hints = {k.rsplit(".", 1)[1]: v for k, v in settings.items() if ".inlayHints." in k}
    args = settings["black-formatter.args"]
    doc = parity.doc_text(PHASE, LESSON)
    return {
        "mode": settings["python.analysis.typeCheckingMode"],
        "hints": hints,
        "formatter": settings["[python]"]["editor.defaultFormatter"],
        "saves": saves(settings),
        "delay": settings["files.autoSaveDelay"],
        "line_length": int(args[args.index("--line-length") + 1]),
        "rulers": settings["editor.rulers"],
        "recommended": recommended,
        "doc_promises": all(s in doc for s in ("Never think about formatting", "forget to save")),
    }


def verify(result):
    hints, s = result["hints"], result["saves"]
    return [
        practice.Check(
            "ANSWER: return-type hints only, and Black on an explicit save",
            result["mode"] == "basic"
            and hints == {"functionReturnTypes": True, "variableTypes": False}
            and result["formatter"] == "ms-python.black-formatter"
            and s["explicit"]["format"],
            f"typeCheckingMode {result['mode']}; inlay hints {hints}; formatter "
            f"{result['formatter']} runs on an explicit save",
        ),
        practice.Check(
            "FINDING: the 1-second auto-save skips format-on-save and organize-imports",
            not s["auto"]["format"]
            and not s["auto"]["actions"]
            and result["delay"] == 1000
            and result["doc_promises"],
            f"files.autoSave afterDelay at {result['delay']} ms; VS Code: formatOnSave "
            f"'{VSCODE_SAYS['editor.formatOnSave']}'; 'explicit': "
            f"'{VSCODE_SAYS['explicit']}'. Auto-saves: {s['auto']}; explicit: {s['explicit']}",
        ),
        practice.Check(
            "CONTROL: line length 88 agrees with the ruler and Black's default",
            result["line_length"] == result["rulers"][0] == BLACK_DEFAULT
            and result["formatter"] in result["recommended"],
            f"--line-length {result['line_length']}, rulers {result['rulers']}, Black's "
            f"default {BLACK_DEFAULT}; {result['formatter']} is recommended",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
