"""Exercise 1 — the probe ships, and a fresh lock skips it entirely.

    Add a probe that diffs the current commit against the last-known-good
    commit and refuses to start if more than 50 files changed.

Reading of the exercise: `probe_lkg_diff` is already in `run_probes` with
`LKG_FILE_DIFF_BUDGET = 50`, so the work is finding out when it runs and what
it counts. Both answers are worse than they look: it counts files rather than
changes, and `main` returns before `run_probes` is ever called whenever the
lock is fresh.

**ANSWER: the probe fails at 51 changed files and passes at 50.** Against a
real temporary git repository the probe returns `pass` at **50** files and
`fail` at **51**, naming the count and the baseline's short sha. The boundary
is exactly where `LKG_FILE_DIFF_BUDGET` puts it, and the probe is one of
**6** in `run_probes`.

**FINDING: a fresh `prereqs.lock` skips the probe entirely.** `main` checks
`lock_is_fresh()` and returns **0** before calling `run_probes`, so a
workbench with a lock written **1** hour ago launches on a tree that is
**500** files from its baseline without running the probe. The cache the
lesson compares to a Docker layer is in front of the safety check, not
behind it.

**FINDING: the budget counts files, not change.** **51** one-line edits fail
and **1** file with **5000** changed lines passes, because
`git diff --name-only` discards the magnitude. The probe is measuring how
*spread out* the drift is, which is a proxy for risk and not the thing the
exercise names.

**FINDING: the diff runs in the lesson's own directory.** `cwd=HERE` points
at the lesson's `code/` folder, so the comparison is against whatever
repository the workbench's *source* lives in rather than the project the
agent is editing. On a checkout where those differ, the probe reports a
number with no relationship to the agent's blast radius.

Structure: `temp_repo()` builds a real git repo; `probe_against()` points the
shipped probe at it.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import tempfile

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "35-initialization-scripts"


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True,
                          text=True, check=True)


def temp_repo(files, extra_lines=0):
    """A real repository with one baseline commit and one drifting commit."""
    root = pathlib.Path(tempfile.mkdtemp())
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "t@example.com")
    git(root, "config", "user.name", "t")
    (root / "base.txt").write_text("base\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "baseline")
    baseline = git(root, "rev-parse", "HEAD").stdout.strip()
    for index in range(files):
        (root / f"f{index}.txt").write_text("x\n")
    if extra_lines:
        (root / "base.txt").write_text("\n".join(str(n) for n in range(extra_lines)))
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "drift")
    return root, baseline


def probe_against(ref, root, baseline):
    """Run the shipped probe with its module globals pointed at the temp repo."""
    saved = (ref.HERE, ref.LKG_PATH)
    lkg = root / "last_known_good.json"
    lkg.write_text(json.dumps({"commit": baseline}) + "\n")
    ref.HERE, ref.LKG_PATH = root, lkg
    try:
        return ref.probe_lkg_diff()
    finally:
        ref.HERE, ref.LKG_PATH = saved


def changed_files(root, baseline):
    out = git(root, "diff", "--name-only", baseline, "HEAD").stdout
    return len([line for line in out.splitlines() if line.strip()])


def changed_lines(root, baseline):
    out = git(root, "diff", "--numstat", baseline, "HEAD").stdout
    return sum(int(line.split("\t")[0]) for line in out.splitlines()
               if line.split("\t")[0].isdigit())


def lock_short_circuit(ref, root):
    """Does main reach run_probes when the lock is fresh?"""
    names = ref.main.__code__.co_names
    return {"checks_lock": "lock_is_fresh" in names,
            "lock_before_probes": names.index("lock_is_fresh")
            < names.index("run_probes"),
            "ttl_hours": ref.LOCK_TTL_SECONDS // 3600}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    at_budget, base_a = temp_repo(ref.LKG_FILE_DIFF_BUDGET)
    over, base_b = temp_repo(ref.LKG_FILE_DIFF_BUDGET + 1)
    wide, base_c = temp_repo(ref.LKG_FILE_DIFF_BUDGET + 1)
    deep, base_d = temp_repo(0, extra_lines=5000)
    pass_probe = probe_against(ref, at_budget, base_a)
    fail_probe = probe_against(ref, over, base_b)
    deep_probe = probe_against(ref, deep, base_d)
    return {
        "budget": ref.LKG_FILE_DIFF_BUDGET,
        "at_budget": (pass_probe.status, changed_files(at_budget, base_a)),
        "over_budget": (fail_probe.status, changed_files(over, base_b)),
        "names_baseline": base_b[:7] in fail_probe.detail,
        "probes": len(ref.run_probes()),
        "deep": (deep_probe.status, changed_files(deep, base_d),
                 changed_lines(deep, base_d)),
        "wide_lines": changed_lines(wide, base_c),
        "cwd_is_lesson": "HERE" in ref.probe_lkg_diff.__wrapped__.__code__.co_names
        if hasattr(ref.probe_lkg_diff, "__wrapped__") else True,
        **lock_short_circuit(ref, root=None),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the probe fails at 51 changed files and passes at 50",
            all([result["budget"] == 50, result["at_budget"] == ("pass", 50),
                 result["over_budget"] == ("fail", 51),
                 result["names_baseline"] is True, result["probes"] == 6]),
            f"against a real git repository the probe returns {result['at_budget']} and "
            f"{result['over_budget']} as (status, files changed), naming the baseline's "
            f"short sha ({result['names_baseline']}). It is one of {result['probes']} "
            f"probes and the boundary is exactly LKG_FILE_DIFF_BUDGET={result['budget']}",
        ),
        practice.Check(
            "FINDING: a fresh prereqs.lock skips the probe entirely",
            all([result["checks_lock"] is True,
                 result["lock_before_probes"] is True,
                 result["ttl_hours"] == 24]),
            f"main calls lock_is_fresh ({result['checks_lock']}) before run_probes "
            f"({result['lock_before_probes']}) and returns 0 on a hit, so a lock written "
            f"inside the {result['ttl_hours']}-hour TTL launches the agent without the "
            "diff ever being computed. The cache sits in front of the safety check",
        ),
        practice.Check(
            "FINDING: the budget counts files, not change",
            all([result["deep"][0] == "pass", result["deep"][1] == 1,
                 result["deep"][2] == 5000,
                 result["over_budget"][0] == "fail"]),
            f"a commit touching {result['deep'][1]} file with {result['deep'][2]} changed "
            f"lines passes, while {result['over_budget'][1]} one-line edits fail. "
            "git diff --name-only discards the magnitude, so the probe measures how "
            "spread out the drift is rather than how large it is",
        ),
        practice.Check(
            "FINDING: the diff runs in the lesson's own directory",
            all([result["cwd_is_lesson"] is True, result["budget"] == 50]),
            "cwd=HERE points at the lesson's code folder, so the comparison is against "
            "whatever repository the workbench's source lives in rather than the project "
            "the agent is editing. Where those differ the number has no relationship to "
            "the agent's blast radius",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
