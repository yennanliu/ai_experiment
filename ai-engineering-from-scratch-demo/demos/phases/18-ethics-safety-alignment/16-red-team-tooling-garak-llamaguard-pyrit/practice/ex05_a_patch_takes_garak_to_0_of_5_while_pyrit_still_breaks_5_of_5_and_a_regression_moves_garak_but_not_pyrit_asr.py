"""Exercise 5 — a patch takes Garak to 0 of 5 while PyRIT still breaks 5 of 5, and a regression moves Garak but not PyRIT's ASR.

    Compare Garak and PyRIT's design principles. Argue for a deployment
    where each is the right tool.

Reading of the exercise: the design principles are the lesson's -- Garak is
breadth (a fixed set of single-turn probes, scored per probe), PyRIT is depth
(one seed escalated through converters until it breaks). The argument is
made by running both of the reference's tools against two edits of its toy
target that a real team produces: a *patch* that blocks the one probe
Garak reported passing ("expl0sive"), and a *regression* where a model
update stops refusing one phrase ("fraud scheme"). Each tool's verdict is
read off before and after.

**ANSWER: Garak for nightly regression, PyRIT for pre-release.** After the
patch Garak reports 0 of 5 probes passing -- a green board -- while PyRIT,
seeded with the same five payloads, still breaks 5 of 5 at `encode`, using
15 target calls against Garak's 5. Garak can only re-ask the questions it
already has, so a patch aimed at its report makes it stale; that is the
pre-release case for PyRIT, on a new model or agent after the team has
fixed everything the scanner found. After the regression Garak goes from 1
to 2 probes passing and names the probe that flipped (`roleplay_crime`),
while PyRIT's ASR is 5 of 5 before and after -- only the break turn on one
seed moves, from `encode` to `seed`. A campaign that stops at the first
success saturates, so its headline number cannot see a regression; a fixed
probe set scored per probe can, at a fixed cost. That is the nightly case
for Garak: a chat product whose prompt or model changes every week.

**FINDING: the toy Garak's detector never reads the target's output.**
`garak_scan` computes `guard_detected` from `is_unsafe(p.payload)`. Against
a target that answers every probe with a flagged sentence and no refusal,
it still reports the same 4 of 5 "detected" and now 0 of 5 blocked. Real
Garak detectors score outputs; this one is an input classifier, so its
detection column says nothing about what the model said.

Structure: `patched()` and `regressed()` wrap `ref.toy_target`; `garak()`
and `pyrit()` run the reference's `garak_scan` and `pyrit_campaign` and
count calls through a counting wrapper.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "16-red-team-tooling-garak-llamaguard-pyrit"
REFUSE = "I must refuse this request."


def patched(ref):
    """Blocks the one probe Garak reported passing."""
    return lambda p: REFUSE if "expl0sive" in p.lower() else ref.toy_target(p)


def regressed(ref):
    """A model update that no longer refuses 'fraud scheme'."""
    return lambda p: ref.toy_target(p.lower().replace("fraud scheme", "fraud-scheme"))


def counted(target):
    calls = []
    return (lambda p: calls.append(p) or target(p)), calls


def garak(ref, target):
    """(names of probes that passed the target, target calls)."""
    fn, calls = counted(target)
    scan = ref.garak_scan(fn)
    return [n for n, r in scan.items() if not r["blocked"]], len(calls)


def pyrit(ref, target):
    """(break turn per probe seed, target calls)."""
    fn, calls = counted(target)
    runs = [ref.pyrit_campaign(fn, p.payload, max_turns=4)["broke_at"] for p in ref.GARAK_PROBES]
    return runs, len(calls)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    targets = {"shipped": ref.toy_target, "patched": patched(ref), "regressed": regressed(ref)}
    flagged_output = ref.garak_scan(lambda p: "ok: here is the bomb answer")
    return {
        "garak": {k: garak(ref, t) for k, t in targets.items()},
        "pyrit": {k: pyrit(ref, t) for k, t in targets.items()},
        "output_is_unsafe": ref.is_unsafe("ok: here is the bomb answer"),
        "blind": (sum(r["guard_detected"] for r in flagged_output.values()),
                  sum(r["blocked"] for r in flagged_output.values())),
    }


def verify(result):
    g, p = result["garak"], result["pyrit"]
    asr = {k: sum(b is not None for b in v[0]) for k, v in p.items()}
    return [
        practice.Check(
            "ANSWER: after the patch Garak reports 0 of 5 while PyRIT breaks 5 of 5",
            all([g["shipped"] == (["encoded_weapon"], 5), g["patched"] == ([], 5),
                 p["patched"] == (["encode"] * 5, 15)]),
            f"Garak (passing probes, calls): shipped {g['shipped']}, patched {g['patched']}; "
            f"PyRIT (break turns, calls) on the patched target {p['patched']}",
        ),
        practice.Check(
            "ANSWER: a regression moves Garak from 1 to 2 passing, and PyRIT's ASR not at all",
            all([g["regressed"][0] == ["encoded_weapon", "roleplay_crime"],
                 asr == {"shipped": 5, "patched": 5, "regressed": 5},
                 [a == b for a, b in zip(p["shipped"][0], p["regressed"][0])].count(False) == 1,
                 p["regressed"][0][3] == "seed", p["shipped"][0][3] == "encode"]),
            f"Garak passing after regression {g['regressed'][0]}; PyRIT ASR {asr}; break turns "
            f"{p['shipped'][0]} -> {p['regressed'][0]}",
        ),
        practice.Check(
            "FINDING: the toy Garak's detector never reads the target's output",
            all([result["output_is_unsafe"], result["blind"] == (4, 0)]),
            f"every output flagged unsafe, yet (detected, blocked) = {result['blind']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
