"""Exercise 2 — refinement mutates the object every parent is holding.

    Implement per-skill version pinning. When a parent skill composes child
    `crafting@1`, a refinement to `crafting@2` must not silently upgrade the
    parent.

Reading of the exercise: `depends_on` is `tuple[str, ...]` -- names, with no
room for a version -- and `register(dedup=True)` edits the *existing* `Skill`
object in place rather than storing a new one. So there is nothing for a pin
to point at: the old behaviour is not kept anywhere. Pinning therefore means
adding an archive first, and the measurement is what the shipped library
loses at the moment of refinement.

**ANSWER: `name@version` in `depends_on`, resolved against an archive.**
After `crafting` is refined to v2, the pinned parent still runs the v1
behaviour -- log line `ran crafting v1` and result `crafted with 3 ore` --
while the unpinned parent, composed from the same library, reads
`crafted with 5 ore`. Same call, same instant, two answers.

**FINDING: the refinement edits the object in place.** `register` with
`dedup=True` assigns `existing.fn = skill.fn` on the `Skill` the parent
already holds a name for, so the upgrade is not a new version the parent
could decline -- it is the same identity with different behaviour. The
library keeps **1** object for `crafting` before and after.

**FINDING: `history` is documentation, not a rollback.** It stores the old
`code` string and **0** copies of the old `fn`, so after a refinement the
previous behaviour is unrecoverable from the library: **1** history entry, of
type `str`, and nothing callable.

**FINDING: `dedup=False` rewinds the version counter.** Registering
`crafting` a third time with `dedup=False` replaces the entry with a fresh
`Skill` at version **1** and drops its **1**-entry history. Two ways to write
a skill, and one of them silently makes v3 look like v1.

Structure: `PinnedLibrary` keeps an archive keyed `name@version` and resolves
pinned `depends_on` entries against it; everything else is the lesson's.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "10-skill-libraries-voyager"


def craft_v1(context):
    return "crafted with 3 ore"


def craft_v2(context):
    return "crafted with 5 ore"


def parent_fn(context):
    return "parent done"


class PinnedLibrary:
    """The lesson's library plus an archive, so `name@version` can resolve."""

    def __init__(self, ref):
        self.ref, self.lib, self.archive = ref, ref.SkillLibrary(), {}

    def register(self, skill):
        note = self.lib.register(skill)
        stored = self.lib.get(skill.name)
        self.archive[f"{skill.name}@{stored.version}"] = (stored.version, stored.fn)
        return note

    def execute(self, name, context=None):
        context = {"log": []} if context is None else context
        skill = self.lib.get(name)
        for dep in skill.depends_on:
            version, fn = self.archive[dep]
            context["log"].append(f"ran {dep.split('@')[0]} v{version}: {fn(context)}")
        context["log"].append(f"ran {name}: {skill.fn(context)}")
        return context


def child(ref, fn):
    return ref.Skill(name="crafting", description="craft a tool from ore",
                     code="craft()", fn=fn, tags=("craft",))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    pinned = PinnedLibrary(ref)
    pinned.register(child(ref, craft_v1))
    pinned.register(ref.Skill(name="build_kit", description="build a starter kit",
                              code="kit()", fn=parent_fn,
                              depends_on=("crafting@1",)))
    loose = ref.SkillLibrary()
    loose.register(child(ref, craft_v1))
    loose.register(ref.Skill(name="build_kit", description="build a starter kit",
                             code="kit()", fn=parent_fn, depends_on=("crafting",)))
    before = loose.get("crafting")
    pinned.register(child(ref, craft_v2))
    loose.register(child(ref, craft_v2))
    after = loose.get("crafting")
    pinned_log = pinned.execute("build_kit")["log"]
    loose_log = loose.execute("build_kit")["log"]
    history = list(after.history)
    loose.register(child(ref, craft_v1), dedup=False)
    reset = loose.get("crafting")
    return {
        "pinned_line": pinned_log[0], "loose_line": loose_log[0],
        "same_object": before is after, "versions": (1, after.version),
        "archive": sorted(pinned.archive),
        "history_len": len(history), "history_types": sorted({type(h).__name__
                                                              for h in history}),
        "fn_history": [h for h in history if callable(h)],
        "reset_version": reset.version, "reset_history": len(reset.history),
        "depends_type": str(ref.Skill.__dataclass_fields__["depends_on"].type),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the pinned parent still runs v1 after the child is refined",
            all(["crafting v1" in result["pinned_line"],
                 "3 ore" in result["pinned_line"], "5 ore" in result["loose_line"],
                 result["archive"] == ["build_kit@1", "crafting@1", "crafting@2"]]),
            f"after crafting is refined the pinned parent logs "
            f"{result['pinned_line']!r} and the unpinned one {result['loose_line']!r} -- "
            f"same call, same instant, two answers. The archive holds "
            f"{result['archive']}, which is what a pin can point at",
        ),
        practice.Check(
            "FINDING: the refinement edits the object in place",
            all([result["same_object"] is True, result["versions"] == (1, 2),
                 "str" in result["depends_type"]]),
            f"register(dedup=True) assigns existing.fn on the object the library already "
            f"holds, so before and after the refinement are the same instance "
            f"({result['same_object']}) at versions {result['versions']}. depends_on is "
            f"{result['depends_type']}, which has no room for the version to pin to",
        ),
        practice.Check(
            "FINDING: history is documentation, not a rollback",
            all([result["history_len"] == 1, result["history_types"] == ["str"],
                 result["fn_history"] == []]),
            f"history holds {result['history_len']} entry of type "
            f"{result['history_types'][0]} and {len(result['fn_history'])} callables, so "
            "the previous behaviour is unrecoverable from the library. The code string "
            "survives and the thing that runs does not",
        ),
        practice.Check(
            "FINDING: dedup=False rewinds the version counter",
            all([result["reset_version"] == 1, result["reset_history"] == 0,
                 result["versions"][1] == 2]),
            f"registering crafting a third time with dedup=False replaces the entry with "
            f"a fresh Skill at version {result['reset_version']} and "
            f"{result['reset_history']} history entries, after it had reached version "
            f"{result['versions'][1]}. Two ways to write a skill, one of which makes v3 "
            "look like v1",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
