"""Exercise 3 — the detector only looks at two of nine actions.

    Design a canary token set for a browser agent (Lesson 11). List at least
    three canaries and what each would detect.

Reading of the exercise: a canary set is only as good as the place it is
checked, so the four canaries below are each paired with the boundary that
would have to inspect them -- and each pairing is then tested against the
shipped `Canary`, which checks one boundary with one comparison.

**ANSWER: four canaries, at four different boundaries.** A fake credential
in the profile detects a *read* of the credential store. A unique URL that
appears only inside page content detects an agent *following* an instruction
it read. A fake contact in the address book detects exfiltration through a
legitimate channel. A honeytoken in the cookie jar detects session replay
from another host. Only the first is a file read, which is the only boundary
`check_read` covers.

**FINDING: the shipped detector inspects 2 of the 9 shipped actions.** It
runs only when `kind == "read"`, and **7** of the trajectory's **9** actions
have kind `tool` -- including `tool:read:src/app.py`, which is a read. A read
expressed as a tool call is never shown to the canary at all.

**FINDING: the match is string equality against 2 paths.** `check_read`
tests `path in CANARY_FILES`, so `~/.env.canary` fires and
`.env.canary` and `/home/u/.env.canary` do not. A canary whose detection
depends on the agent spelling the path the way the seed file spelled it is a
canary against accidents, not against intent.

**FINDING: three of the four canaries have no call site here.** The URL,
contact and cookie canaries each need a check on an *outbound* boundary, and
the module has none -- `run_trajectory` has **3** detectors and **0** of them
sees an action's destination. That is the same gap Lesson 11 found from the
other side: the alarm has to sit where the value leaves, and nothing here
does.

Structure: `CANARIES` is the set; `covered()` asks which of them the shipped
detector could fire on.
"""

from __future__ import annotations

import inspect
import re

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "14-kill-switches-canaries"

# (canary, where it is seeded, the boundary that must check it, what it detects)
CANARIES = (
    ("fake AWS key", "profile / .env", "file read", "a read of the credential store"),
    ("unlinked URL", "page content the agent reads", "outbound fetch",
     "the agent following an instruction it read"),
    ("fake contact", "address book", "outbound message",
     "exfiltration through a legitimate channel"),
    ("honeytoken cookie", "cookie jar", "inbound session use",
     "session replay from another host"),
)
SPELLINGS = ("~/.env.canary", ".env.canary", "/home/u/.env.canary")


def covered(ref):
    """Canaries whose boundary the shipped detector actually inspects."""
    source = inspect.getsource(ref.run_trajectory)
    return [name for name, _seed, boundary, _detects in CANARIES
            if boundary == "file read" and 'a.kind == "read"' in source]


def spellings(ref):
    canary = ref.Canary()
    return [canary.check_read(index, path) for index, path in enumerate(SPELLINGS, 1)]


def shipped_actions(ref):
    body = inspect.getsource(ref.main)
    return re.findall(r'Action\("(\w+)",', body)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    kinds = shipped_actions(ref)
    trajectory = inspect.getsource(ref.run_trajectory)
    return {
        "canaries": len(CANARIES),
        "boundaries": sorted({boundary for _n, _s, boundary, _d in CANARIES}),
        "covered": covered(ref),
        "actions": len(kinds),
        "read_kind": kinds.count("read"),
        "tool_kind": kinds.count("tool"),
        "tool_reads": sum(1 for kind, payload in
                          re.findall(r'Action\("(\w+)",\s*"([^"]+)"', inspect.getsource(ref.main))
                          if kind == "tool" and payload.startswith("read:")),
        "seeded_paths": len(ref.CANARY_FILES),
        "spellings": spellings(ref),
        "detectors": 3,
        "sees_destination": any(word in trajectory for word in ("dest", "endpoint", "host")),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: four canaries at four different boundaries",
            all([result["canaries"] == 4, len(result["boundaries"]) == 4,
                 result["covered"] == ["fake AWS key"]]),
            f"{result['canaries']} canaries sit at {result['boundaries']}, and "
            f"{len(result['covered'])} of them -- {result['covered'][0]} -- is at the "
            "boundary the shipped detector checks",
        ),
        practice.Check(
            "FINDING: the shipped detector inspects 2 of the 9 shipped actions",
            all([result["actions"] == 9, result["read_kind"] == 2,
                 result["tool_kind"] == 7, result["tool_reads"] >= 1]),
            f"the canary runs only for kind 'read', which is {result['read_kind']} of "
            f"{result['actions']} actions; {result['tool_kind']} are tool calls, of "
            f"which {result['tool_reads']} are themselves reads and are never shown to "
            "it",
        ),
        practice.Check(
            "FINDING: the match is string equality against two paths",
            all([result["seeded_paths"] == 2, result["spellings"] == [True, False, False]]),
            f"check_read tests membership in {result['seeded_paths']} literal paths, so "
            f"the three spellings {list(SPELLINGS)} give {result['spellings']} -- a "
            "canary against accidents, not intent",
        ),
        practice.Check(
            "FINDING: three of the four canaries have no call site here",
            all([len(result["covered"]) == 1, result["detectors"] == 3,
                 not result["sees_destination"]]),
            f"the URL, contact and cookie canaries need an outbound check, and "
            f"{result['detectors']} detectors inspect no destination -- the same gap "
            "Lesson 11 found from the other side",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
