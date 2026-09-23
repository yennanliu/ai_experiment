"""Exercise 4 — the budget that binds is the escalation rate.

    Design a 24-hour unattended run budget: `max_turns`, `max_budget_usd`,
    per-tool caps, allowlists. Justify each number.

Reading of the exercise: "justify each number" means each one has to come
from something, so every figure below is derived from the lesson's own
classifier or from compounding arithmetic, and the single figure that cannot
be -- dollars -- is stated as an assumption with its sensitivity attached.

**ANSWER: the first cap that binds is not in the list.** On the twenty-action
benign sample, **3** actions escalate to HITL, so an unattended run meets a
human-shaped stop after **6.7** actions in expectation. `max_turns` of any
size is irrelevant until the `auto` policy says what HITL means with nobody
there. The budget therefore opens with that decision and then sets:
`max_turns` **223**, `max_budget_usd` **44.60** at an assumed $0.20 a turn,
`shell` capped at **45** calls, and the host allowlist narrowed to **GET**.

**FINDING: max_turns comes from compounding, not from the clock.** At a
per-step reliability of **0.999**, end-to-end success is `p^n`, and **223**
turns is where it crosses **0.80**. At **0.99** the same bar allows **22**.
A 24-hour window is not a budget -- it is a duration, and the number of
actions that fit in it is a reliability question.

**FINDING: the per-tool cap follows the keyword list.** All **12** Stage 1
keywords are shell command shapes, and on the benign sample **2** ever fire,
both on shell-or-edit payloads. So the tool that carries the entire flag
surface is `shell`, and capping it at **20%** of `max_turns` -- **45** calls
-- bounds the blast surface without touching reads, writes or test runs, which
are **9** of the 20 benign actions.

**FINDING: the allowlist needs a verb, and the rule has nowhere to put one.**
All **3** allowlisted hosts accept uploads, and Stage 2 clears them on
membership alone. The budget's allowlist is therefore not a host list but a
`(host, method)` list -- **3** entries, all GET -- which the shipped
`stage2` signature cannot express, because an `Action` carries a tool and a
payload string and **0** structured fields.

Structure: `turns_for()` inverts the compounding bound; `escalation()` reads
the benign sample's stop rate off the lesson's own classifier.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "10-claude-code-permission-modes"

BENIGN = (
    ("read", "pyproject.toml"), ("read", "src/app/models.py"),
    ("edit", "src/app/models.py: rename field"), ("run", "pytest -q tests/unit"),
    ("run", "ruff check src"), ("edit", "README.md: fix a broken link"),
    ("shell", "git status --porcelain"), ("shell", "git diff --stat"),
    ("read", "logs/app.log"), ("write", "docs/changelog.md"),
    ("shell", "curl https://pypi.org/simple/requests/"),
    ("shell", "curl https://github.com/psf/requests/releases"),
    ("shell", "curl https://internal.corp/api/health"),
    ("shell", "curl -sSf $CI_ARTIFACT_URL -o build.tgz"),
    ("shell", "uv sync --extra dev"), ("run", "pytest -q -k models"),
    ("edit", "Dockerfile: chown -R app:app /srv"),
    ("shell", "echo $BUILD_ID >> build.log"),
    ("read", "tests/conftest.py"), ("run", "mypy src"),
)
TARGET, PER_STEP, DOLLARS_PER_TURN, SHELL_SHARE = 0.80, 0.999, 0.20, 0.20


def turns_for(per_step=PER_STEP, target=TARGET):
    """Largest n with per_step**n >= target."""
    return math.floor(math.log(target) / math.log(per_step))


def escalation(ref):
    """(escalations, sample size, expected actions before a human-shaped stop)."""
    stops = 0
    for tool, payload in BENIGN:
        action = ref.Action(tool, payload)
        if ref.stage1(action, []) and ref.stage2(action, [])[0].value == "hitl":
            stops += 1
    return stops, len(BENIGN), round(len(BENIGN) / stops, 1)


def shell_surface(ref):
    """Benign actions whose payload carries a keyword, and the tools they use."""
    tools = {tool for tool, payload in BENIGN
             if any(k.lower() in f"{tool} {payload}".lower()
                    for k in ref.STAGE1_FLAG_KEYWORDS)}
    quiet = sum(1 for tool, _p in BENIGN if tool in ("read", "run", "write"))
    return sorted(tools), quiet


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    stops, sample, before_stop = escalation(ref)
    tools, quiet = shell_surface(ref)
    turns = turns_for()
    return {
        "escalations": stops, "sample": sample, "before_stop": before_stop,
        "max_turns": turns, "at_three_nines": turns,
        "at_two_nines": turns_for(0.99), "target": TARGET,
        "max_budget_usd": round(turns * DOLLARS_PER_TURN, 2),
        "dollars_per_turn": DOLLARS_PER_TURN,
        "shell_cap": round(turns * SHELL_SHARE),
        "flagging_tools": tools,
        "quiet_actions": quiet,
        "keywords": len(ref.STAGE1_FLAG_KEYWORDS),
        "hosts": len(ref.STAGE2_ALLOWED_CURL_HOSTS),
        "action_fields": len(ref.Action.__dataclass_fields__),
        "structured_fields": [name for name in ref.Action.__dataclass_fields__
                              if name not in ("tool", "payload", "note")],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the first cap that binds is the escalation rate",
            all([result["escalations"] == 3, result["sample"] == 20,
                 result["before_stop"] == 6.7, result["max_turns"] == 223,
                 result["max_budget_usd"] == 44.6, result["shell_cap"] == 45]),
            f"{result['escalations']} of {result['sample']} benign actions escalate, so "
            f"an unattended run meets a stop after {result['before_stop']} actions; the "
            f"budget is max_turns {result['max_turns']}, "
            f"${result['max_budget_usd']} at ${result['dollars_per_turn']} a turn and "
            f"{result['shell_cap']} shell calls",
        ),
        practice.Check(
            "FINDING: max_turns comes from compounding, not from the clock",
            all([result["at_three_nines"] == 223, result["at_two_nines"] == 22,
                 result["target"] == 0.80]),
            f"at 0.999 per-step, {result['at_three_nines']} turns is where end-to-end "
            f"crosses {result['target']}; at 0.99 the same bar allows "
            f"{result['at_two_nines']} -- the clock does not decide how many actions "
            "fit",
        ),
        practice.Check(
            "FINDING: the per-tool cap follows the keyword list",
            all([result["keywords"] == 12, result["flagging_tools"] == ["edit", "shell"],
                 result["quiet_actions"] == 9]),
            f"all {result['keywords']} keywords are shell command shapes and only "
            f"{result['flagging_tools']} ever carry one on the benign sample, while "
            f"{result['quiet_actions']} of {result['sample']} actions are reads, runs "
            "and writes -- so the cap belongs on shell alone",
        ),
        practice.Check(
            "FINDING: the allowlist needs a verb and the rule has nowhere to put one",
            all([result["hosts"] == 3, result["action_fields"] == 3,
                 result["structured_fields"] == []]),
            f"the {result['hosts']} allowlisted hosts all accept uploads and Stage 2 "
            f"clears them on membership; an Action carries {result['action_fields']} "
            f"fields and {len(result['structured_fields'])} of them is structured, so "
            "(host, method) has nowhere to live",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
