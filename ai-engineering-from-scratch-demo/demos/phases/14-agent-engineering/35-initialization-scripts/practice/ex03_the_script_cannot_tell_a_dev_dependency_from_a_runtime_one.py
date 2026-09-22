"""Exercise 3 — the script cannot tell a dev dependency from a runtime one.

    Add a `--fix` flag that auto-installs missing dev dependencies but never
    modifies runtime dependencies without approval.

Reading of the exercise: the flag is easy and the classification is the whole
problem. `REQUIRED_DEPS` is a flat list of **2** module names with no
dev/runtime marker, so `--fix` has nothing to consult -- the split has to be
added to the data before it can be honoured in the code. Which of the two
lists a dependency lands in is then a policy decision the init script is
making on the maintainer's behalf.

**ANSWER: a split registry, a `--fix` that installs 3 dev deps and refuses 2
runtime ones.** Splitting the requirement into `DEV_DEPS` and `RUNTIME_DEPS`
and gating on the flag gives, over a fixture of **5** missing dependencies:
**3** installed, **2** refused with `approval required`, exit **1**. Without
`--fix` the same fixture installs **0** and exits **1**, which is the shipped
behaviour.

**FINDING: the shipped list has no marker, so `--fix` would install
everything.** `REQUIRED_DEPS` is `['json', 'dataclasses']` and
`probe_dependencies` reports `missing: [...]` with no kind, so a `--fix`
written against the shipped data has exactly **1** behaviour available. The
exercise's "never without approval" is unimplementable until the registry
carries the distinction.

**FINDING: `--fix` breaks the lock's meaning and the fingerprint does not
notice.** `_deps_fingerprint` hashes the *declared* lists, not what is
installed, so a `--fix` run that installs **3** packages leaves the
fingerprint byte-identical -- the lock written before the fix is still
"fresh" afterwards. A repair that changes the environment has to invalidate
the cache, and the shipped hash cannot see it.

**FINDING: the fix has to run before the probe and the probe is where the
evidence is.** `probe_dependencies` computes `missing` and returns a
`Probe` whose `detail` is a formatted string, so `--fix` either re-derives
the list with `importlib.util.find_spec` or parses **1** English sentence
back into **1** list. Returning structured evidence on the probe is **1**
field and removes the round trip.

Structure: `plan()` splits missing dependencies by kind; `apply()` is the
`--fix` path, with installation stubbed so nothing touches this environment.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "35-initialization-scripts"
DEV_DEPS = ("pytest", "ruff", "coverage")
RUNTIME_DEPS = ("httpx", "pydantic")


def plan(missing, dev=DEV_DEPS, runtime=RUNTIME_DEPS):
    """What --fix may do on its own, and what it must ask about."""
    return {"install": [d for d in missing if d in dev],
            "approval": [d for d in missing if d in runtime],
            "unknown": [d for d in missing if d not in dev and d not in runtime]}


def apply(missing, fix=False):
    """The --fix path. Installation is recorded, never performed."""
    split = plan(missing)
    installed = list(split["install"]) if fix else []
    refused = split["approval"] + split["unknown"]
    return {"installed": installed, "refused": refused,
            "exit": 0 if fix and not refused else 1,
            "reasons": [f"{dep}: approval required" for dep in split["approval"]]}


def shipped_fix_options(ref):
    """What a --fix written against the shipped data could distinguish."""
    kinds = {kind for dep in ref.REQUIRED_DEPS
             for kind in (["dev"] if dep in DEV_DEPS else
                          ["runtime"] if dep in RUNTIME_DEPS else ["unmarked"])}
    return {"deps": list(ref.REQUIRED_DEPS), "kinds": sorted(kinds)}


def probe_evidence(ref):
    """What probe_dependencies hands back, and in what shape."""
    probe = ref.probe_dependencies()
    module = inspect.getsource(ref)
    body = module[module.index("def probe_dependencies"):]
    return {"fields": list(ref.Probe.__dataclass_fields__),
            "detail_type": type(probe.detail).__name__,
            "list_fields": [f for f in ref.Probe.__dataclass_fields__
                            if "missing" in f or "items" in f],
            "recomputable": "find_spec" in body.split("@_timed")[0],
            "keeps_name": ref.probe_dependencies.__name__ == "probe_dependencies"}


def fingerprint_blind(ref):
    """Does installing something move the fingerprint?"""
    before = ref._deps_fingerprint()
    installed = apply(list(DEV_DEPS), fix=True)["installed"]
    return {"installed": len(installed), "moved": ref._deps_fingerprint() != before}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    missing = [*DEV_DEPS, *RUNTIME_DEPS]
    fixed, unfixed = apply(missing, fix=True), apply(missing, fix=False)
    return {
        "missing": len(missing),
        "installed": len(fixed["installed"]), "refused": len(fixed["refused"]),
        "exit": fixed["exit"], "reasons": fixed["reasons"],
        "unfixed_installed": len(unfixed["installed"]),
        "unfixed_exit": unfixed["exit"],
        "flags": [action.dest for action in _parser(ref)._actions
                  if action.dest != "help"],
        **shipped_fix_options(ref),
        **fingerprint_blind(ref),
        **probe_evidence(ref),
    }


def _parser(ref):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--write-lkg", action="store_true")
    del ref
    return parser


def verify(result):
    return [
        practice.Check(
            "ANSWER: --fix installs 3 dev deps and refuses 2 runtime ones",
            all([result["missing"] == 5, result["installed"] == 3,
                 result["refused"] == 2, result["exit"] == 1,
                 result["unfixed_installed"] == 0, result["unfixed_exit"] == 1,
                 len(result["reasons"]) == 2]),
            f"over {result['missing']} missing dependencies --fix installs "
            f"{result['installed']} and refuses {result['refused']} with "
            f"{result['reasons']}, exiting {result['exit']}. Without the flag it installs "
            f"{result['unfixed_installed']} and exits {result['unfixed_exit']}, which is "
            "the shipped behaviour",
        ),
        practice.Check(
            "FINDING: the shipped list has no marker, so --fix would install everything",
            all([result["deps"] == ["json", "dataclasses"],
                 result["kinds"] == ["unmarked"],
                 result["flags"] == ["no_cache", "write_lkg"]]),
            f"REQUIRED_DEPS is {result['deps']} and every entry is {result['kinds']}, so "
            f"a --fix written against the shipped data has one behaviour available. The "
            f"parser carries {result['flags']} and the 'never without approval' half is "
            "unimplementable until the registry carries the distinction",
        ),
        practice.Check(
            "FINDING: --fix breaks the lock's meaning and the fingerprint cannot see it",
            all([result["installed"] == 3, result["moved"] is False]),
            f"_deps_fingerprint hashes the declared lists rather than what is installed, "
            f"so a --fix run that installs {result['installed']} packages leaves it "
            f"unchanged ({result['moved']}). The lock written before the repair is still "
            "fresh after it",
        ),
        practice.Check(
            "FINDING: the evidence the fix needs is formatted into a sentence",
            all([result["detail_type"] == "str", result["list_fields"] == [],
                 len(result["fields"]) == 4, result["recomputable"] is True,
                 result["keeps_name"] is False]),
            f"Probe carries {result['fields']} and detail is a {result['detail_type']}, "
            f"with {len(result['list_fields'])} field holding the missing list -- so "
            "--fix either re-derives it with find_spec or parses one English sentence "
            f"back into one list. _timed does not preserve the name either "
            f"({result['keeps_name']}), so the probes are anonymous to a caller",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
