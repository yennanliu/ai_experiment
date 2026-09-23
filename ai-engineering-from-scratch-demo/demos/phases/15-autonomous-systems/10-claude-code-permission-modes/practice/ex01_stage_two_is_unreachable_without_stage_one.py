"""Exercise 1 — Stage 2 is unreachable without Stage 1.

    Run `code/main.py`. Which synthetic action type is never flagged by
    Stage 1 but always caught by Stage 2? Which is caught by neither?

Reading of the exercise: the first question presumes an edge the architecture
does not have, so it is answered by checking the call graph before looking at
the trajectory. The second question has an answer and it is the last three
actions.

**ANSWER: nothing, and the credential composite.** `classify` calls `stage2`
only inside the branch where Stage 1 flagged, so the set "never flagged by
Stage 1, caught by Stage 2" is empty for every possible action -- not just
for these. What neither stage catches is actions **13-15**: read
`~/.aws/credentials`, write it to `/tmp`, `git push`. Each passes Stage 1 on
the fast path; none is ever shown to Stage 2.

**FINDING: the shipped run never reaches a human.** Of **15** actions, **10**
are approved on the Stage 1 fast path, **5** are flagged, and of those **1**
is cleared by the allowlist and **4** are blocked -- **0** escalate. The HITL
path exists and the demonstration does not exercise it.

**FINDING: nine of the twelve Stage 1 keywords can only escalate.** Stage 2
has rules for `rm -rf`, `sudo` and `chmod 777` and for two shapes of `curl `;
`; dd `, `chown `, `iptables`, `kubectl delete`, `drop table`, `exec('`,
`base64 -d`, `aws s3 rb` and a bare `curl ` all fall through to the default
verdict, which is HITL. In an unattended `auto` run that is **9** of **12**
keywords stopping the run rather than deciding it.

**FINDING: the only rule Stage 2 adds is subtraction.** Its repetition rule
is the same three lines as Stage 1's, so it detects nothing Stage 1 did not
already suspect; the one verdict it reaches that Stage 1 could not is
`APPROVE`, on the host allowlist. Stage 2 is a false-positive filter with
block rules attached, which is a different thing from a second detector.

Structure: `verdicts()` replays the shipped trajectory through both stages;
`keyword_fates()` asks Stage 2 what it would say about each Stage 1 keyword.
"""

from __future__ import annotations

import inspect
import re

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "10-claude-code-permission-modes"

COMPOSITE = (13, 14, 15)          # the three actions the lesson's headline is about


def trajectory(ref):
    """The trajectory `main` builds, read back out of its source."""
    source = inspect.getsource(ref.main)
    rows = []
    for tool, payload in _pairs(source):
        rows.append(ref.Action(tool, payload))
    return rows


def _pairs(source):
    body = source[source.index("traj = ["):source.index("]\n    classify")]
    return re.findall(r'Action\(\s*"([^"]+)",\s*"([^"]*)"', body)


def verdicts(ref, actions):
    """(stage-1 flag, stage-2 verdict or None) for each action, in order."""
    history, rows = [], []
    for action in actions:
        flagged = ref.stage1(action, history)
        rows.append((flagged, ref.stage2(action, history)[0].value if flagged else None))
        history.append(action)
    return rows


def keyword_fates(ref):
    """What Stage 2 says about an action carrying each Stage 1 keyword."""
    return [ref.stage2(ref.Action("shell", f"echo {keyword} something"), [])[0].value
            for keyword in ref.STAGE1_FLAG_KEYWORDS]


def tally(rows):
    """How the trajectory splits across the two stages."""
    verdicts_seen = [verdict for _flagged, verdict in rows]
    return {
        "fast_path": verdicts_seen.count(None),
        "flagged": len(rows) - verdicts_seen.count(None),
        "approved_s2": verdicts_seen.count("approve"),
        "blocked_s2": verdicts_seen.count("block"),
        "hitl_s2": verdicts_seen.count("hitl"),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    actions = trajectory(ref)
    rows = verdicts(ref, actions)
    fates = keyword_fates(ref)
    driver = inspect.getsource(ref.classify)
    return {
        "actions": len(actions),
        **tally(rows),
        "composite": [rows[index - 1] for index in COMPOSITE],
        "stage2_guarded": driver.index("stage2(") > driver.index("if not s1_flag"),
        "keywords": len(ref.STAGE1_FLAG_KEYWORDS),
        "escalating_keywords": sum(1 for fate in fates if fate == "hitl"),
        "blocking_keywords": sum(1 for fate in fates if fate == "block"),
        "repetition_shared": "t.payload == a.payload" in inspect.getsource(ref.stage1)
        and "t.payload == a.payload" in inspect.getsource(ref.stage2),
        "approve_rules": inspect.getsource(ref.stage2).count("return Verdict.APPROVE"),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: nothing, and the credential composite",
            all([result["stage2_guarded"],
                 result["composite"] == [(False, None)] * 3,
                 result["actions"] == 15]),
            f"classify calls stage2 only inside the flagged branch, so the first "
            f"question's set is empty for every possible action; of "
            f"{result['actions']} actions the composite at {list(COMPOSITE)} is "
            f"{result['composite']} -- fast-pathed, never shown to Stage 2",
        ),
        practice.Check(
            "FINDING: the shipped run never reaches a human",
            all([result["fast_path"] == 10, result["flagged"] == 5,
                 result["approved_s2"] == 1, result["blocked_s2"] == 4,
                 result["hitl_s2"] == 0]),
            f"{result['fast_path']} approved on the fast path, {result['flagged']} "
            f"flagged, of which {result['approved_s2']} cleared and "
            f"{result['blocked_s2']} blocked -- {result['hitl_s2']} escalations, so the "
            "HITL path is never exercised",
        ),
        practice.Check(
            "FINDING: nine of the twelve Stage 1 keywords can only escalate",
            all([result["keywords"] == 12, result["escalating_keywords"] == 9,
                 result["blocking_keywords"] == 3]),
            f"{result['blocking_keywords']} of {result['keywords']} keywords reach an "
            f"automated block and {result['escalating_keywords']} fall through to the "
            "default HITL verdict -- in an unattended run those stop it rather than "
            "decide it",
        ),
        practice.Check(
            "FINDING: the only rule Stage 2 adds is subtraction",
            all([result["repetition_shared"], result["approve_rules"] == 1]),
            "the repetition rule is the same test in both stages, and the one verdict "
            f"Stage 2 reaches that Stage 1 could not is its {result['approve_rules']} "
            "APPROVE on the host allowlist -- a false-positive filter, not a second "
            "detector",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
