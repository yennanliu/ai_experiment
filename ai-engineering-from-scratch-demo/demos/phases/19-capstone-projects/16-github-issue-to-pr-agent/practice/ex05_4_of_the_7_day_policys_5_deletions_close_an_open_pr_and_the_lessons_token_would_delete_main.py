"""Exercise 5 -- 4 of the 7-day policy's 5 deletions close an open PR, and the lesson's token would let it delete main.

    Add a retention policy: PR branches older than 7 days without merge get deleted automatically.

Reading of the exercise: "PR branches" are the agent's own branches, named
the way the lesson's TypeScript stub names them (`agent/issue-<n>`, read from
`code/ts/src/agent.ts`). "Older than 7 days" is measured from the branch's
last push, strictly greater than 7.0 days. "Without merge" means the branch
was never merged. The policy runs over a seeded repo of 40 agent branches
(age 0-21 days, 40% merged, 60% of the rest with an open PR, last push
somewhere between creation and now), one more agent branch at exactly 7.0
days, and `main` (pushed today), `release/1.x` and two stale human
branches. Every deletion is first put to the lesson's
`InstallationToken.can`. GitHub's rule (read 2026-09-29): "If the branch is
associated with at least one open pull request, deleting the branch closes
the pull requests" (docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-and-deleting-branches-within-your-repository).
Its built-in setting deletes head branches only "after pull requests are
merged" (.../managing-the-automatic-deletion-of-branches), so the unmerged
case needs a job of its own.

**ANSWER: `retention()` deletes 5 of the 40 agent branches.** The
exercise's rule, applied as written, deletes the unmerged agent branches
whose last push is more than 7 days old. A branch at exactly 7.0 days is
kept.

**FINDING: as written, the rule closes open PRs.** 4 of the 5 deleted
branches head an open PR, so the job closes them unmerged. Skipping branches
with an open PR deletes 1. The clock matters as well: measured from
creation instead of last push, the same rule deletes 17.

**FINDING: the lesson's token guards two strings, not branches.** `can`
refuses only `force_push` and actions starting `write:main`. It allows
`delete:refs/heads/main`, `force-push` and `write:.github/workflows`, and it
never reads its own `permissions` map. Without the `agent/` prefix, the rule
also deletes the 3 stale human branches (`release/1.x`, `feature/login`,
`spike/cache`), and `can` approves all 3. Only a push to `main` in the last
7 days keeps `main` itself off the list.
"""

from __future__ import annotations

import inspect
import random
import re

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "16-github-issue-to-pr-agent"
DAYS = 7.0


def repo_branches():
    rng = random.Random(16)
    out = []
    for i in range(40):
        age = rng.uniform(0, 21)
        merged = rng.random() < 0.4
        out.append({"name": f"agent/issue-{800 + i}", "created": age, "pushed": age * rng.random(),
                    "merged": merged, "open_pr": not merged and rng.random() < 0.6})
    out.append({"name": "agent/issue-900", "created": DAYS, "pushed": DAYS, "merged": False, "open_pr": False})
    for name, age in [("main", 0.2), ("release/1.x", 40.0), ("feature/login", 9.0), ("spike/cache", 30.0)]:
        out.append({"name": name, "created": 200.0, "pushed": age, "merged": False, "open_pr": False})
    return out


def retention(branches, prefix="agent/", clock="pushed", keep_open=False):
    return [b for b in branches
            if b["name"].startswith(prefix) and not b["merged"] and b[clock] > DAYS
            and not (keep_open and b["open_pr"])]


def token_side(ref, branches):
    token = ref.InstallationToken.mint("acme/widget")
    probes = ["force_push", "force-push", "write:main", "delete:refs/heads/main",
              "write:.github/workflows", "write:refs/heads/main"]
    unscoped = [b["name"] for b in retention(branches, prefix="") if not b["name"].startswith("agent/")]
    return {
        "can": {a: token.can(a) for a in probes},
        "reads_permissions": "self.permissions" in inspect.getsource(ref.InstallationToken.can),
        "unscoped_humans": unscoped,
        "approved": sum(token.can(f"delete:refs/heads/{n}") for n in unscoped),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    agent_ts = (parity.lesson_dir(PHASE, LESSON) / "code" / "ts" / "src" / "agent.ts").read_text()
    branches = repo_branches()
    deleted = retention(branches)
    return {
        "ts_branch": re.search(r"`(agent/issue-)\$\{issueNumber\}`", agent_ts).group(1),
        "agent_branches": sum(b["name"].startswith("agent/") for b in branches) - 1,
        "deleted": len(deleted),
        "boundary_kept": "agent/issue-900" not in {b["name"] for b in deleted},
        "closes_open_prs": sum(b["open_pr"] for b in deleted),
        "keep_open": len(retention(branches, keep_open=True)),
        "by_created": len(retention(branches, clock="created")),
        **token_side(ref, branches),
    }


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: the rule deletes 5 of 40 agent branches, keeping the one at exactly 7.0 days",
            (r["ts_branch"], r["agent_branches"], r["deleted"], r["boundary_kept"]) == ("agent/issue-", 40, 5, True),
            f"branch prefix {r['ts_branch']!r}; deleted {r['deleted']}/{r['agent_branches']}; "
            f"7.0-day branch kept {r['boundary_kept']}",
        ),
        practice.Check(
            "FINDING: as written 4 of the 5 deletions close an open PR; skipping them deletes 1; creation clock 17",
            (r["closes_open_prs"], r["keep_open"], r["by_created"]) == (4, 1, 17),
            f"open PRs closed {r['closes_open_prs']}; keep-open variant deletes {r['keep_open']}; "
            f"by creation date {r['by_created']}",
        ),
        practice.Check(
            "FINDING: can() guards two strings, never reads permissions, and approves deleting main",
            r["can"] == {"force_push": False, "force-push": True, "write:main": False,
                         "delete:refs/heads/main": True, "write:.github/workflows": True,
                         "write:refs/heads/main": True}
            and (r["reads_permissions"], r["unscoped_humans"], r["approved"])
            == (False, ["release/1.x", "feature/login", "spike/cache"], 3),
            f"can: {r['can']}; reads permissions {r['reads_permissions']}; unscoped rule deletes "
            f"{r['unscoped_humans']}, token approves {r['approved']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
