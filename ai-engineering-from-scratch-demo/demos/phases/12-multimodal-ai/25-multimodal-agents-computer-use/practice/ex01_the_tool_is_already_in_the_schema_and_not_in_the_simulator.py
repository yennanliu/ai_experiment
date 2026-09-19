"""Exercise 1 — the tool is already in the schema, and not in the simulator.

    Extend the action schema with a `screenshot_region` tool (crop + zoom). What
    tasks benefit?

Reading of the exercise: the schema is read before anything is added to it, and
`screenshot_region` is already there -- so the exercise's premise is wrong and the
real gap is one level down, in `apply_action`. The "what tasks benefit" half is
then answered with the patch arithmetic from Lesson 12.01, because crop-and-zoom
is a resolution argument and resolution is measurable.

**ANSWER: it is already declared.** `ACTION_SCHEMA` has **10** entries and
`screenshot_region: ["x0", "y0", "x1", "y1"]` is one of them. What is missing is
not the declaration.

**FINDING: `apply_action` names 4 of the 10 and changes state for 3.** click,
type and select transition; `done` is a deliberate no-op; and **scroll, drag,
hover, navigate, wait and screenshot_region** fall through to an unchanged copy
with no error raised. **60%** of the schema is a declaration with no
implementation behind it.

**ANSWER: what benefits is anything whose target is smaller than a patch.** At
1920x1080 and patch 14 a full screenshot is **10,549** tokens (Lesson 12.02). Fit
that into a 2,048-token budget and the screen is downscaled by **0.4406** -- to
846x476 -- so a 14-pixel UI label lands at **6.2** pixels, under half a patch and
unreadable in principle. Crop a 200x100 region instead and it is **98** patches
at native resolution: dense tables, small-text forms, mobile UIs, chart axis
labels.

**FINDING: and the tool cannot help this lesson's own benchmark.** Both tasks are
fixed plans with hard-coded coordinates -- `run_task` iterates a list and reads no
observation -- so there is no point at which a cropped view could change an
action. The simulator has no input for the tool to produce.

Structure: `handled` inspects `apply_action`'s source for the actions it names,
`unchanged` runs every action against a fresh state, and `label_pixels` scales a
UI element into a token budget.
"""

from __future__ import annotations

import inspect
import math

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "25-multimodal-agents-computer-use"
SCREEN, PATCH = (1920, 1080), 14
BUDGET = 2048
LABEL_PIXELS = 14
CROP = (200, 100)
DELIBERATE_NOOP = "done"


def full_tokens(screen=SCREEN, patch=PATCH):
    return (screen[0] // patch) * (screen[1] // patch)


def scale_for(budget=BUDGET, screen=SCREEN, patch=PATCH):
    return round(math.sqrt(budget / full_tokens(screen, patch)), 4)


def label_pixels(budget=BUDGET, label=LABEL_PIXELS):
    return round(label * scale_for(budget), 1)


def crop_tokens(crop=CROP, patch=PATCH):
    return (crop[0] // patch) * (crop[1] // patch)


def handled(ref):
    source = inspect.getsource(ref.apply_action)
    return [name for name in ref.ACTION_SCHEMA if f'"{name}"' in source]


def unchanged(ref):
    """Actions that leave a fresh state untouched, with plausible arguments."""
    base = ref.BrowserState()
    args = {"x": 1, "y": 1, "text": "t", "option_index": 3, "direction": "d",
            "amount": 1, "x0": 0, "y0": 0, "x1": 1, "y1": 1, "url": "u",
            "ms": 1, "success": True, "explanation": "e", "element_desc": "Book"}
    out = []
    for name in ref.ACTION_SCHEMA:
        after = ref.apply_action(base, {"action": name, **args})
        if (after.url, after.page, after.filled) == (base.url, base.page, base.filled):
            out.append(name)
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    named = handled(ref)
    inert = unchanged(ref)
    return {
        "schema_size": len(ref.ACTION_SCHEMA),
        "has_region": "screenshot_region" in ref.ACTION_SCHEMA,
        "region_params": ref.ACTION_SCHEMA.get("screenshot_region"),
        "named": named, "named_count": len(named),
        "unimplemented": [n for n in ref.ACTION_SCHEMA if n not in named],
        "unimplemented_pct": round(
            (len(ref.ACTION_SCHEMA) - len(named)) / len(ref.ACTION_SCHEMA) * 100),
        "inert": inert,
        "deliberate_noop": DELIBERATE_NOOP in named and DELIBERATE_NOOP in inert,
        "state_changing": len(named) - 1,
        "full_tokens": full_tokens(), "budget": BUDGET,
        "scale": scale_for(), "label_after": label_pixels(),
        "under_a_patch": label_pixels() < PATCH,
        "crop_tokens": crop_tokens(),
        "reads_observation": "state" in inspect.signature(ref.run_task).parameters,
        "run_task_branches": inspect.getsource(ref.run_task).count("if "),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: it is already declared -- 10 entries, screenshot_region among them",
            all([result["schema_size"] == 10, result["has_region"],
                 result["region_params"] == ["x0", "y0", "x1", "y1"]]),
            f"ACTION_SCHEMA has {result['schema_size']} entries and screenshot_region: "
            f"{result['region_params']} is one of them. The exercise's premise is that the "
            "tool is missing; what is missing is not the declaration",
        ),
        practice.Check(
            "FINDING: apply_action names 4 of the 10 and changes state for 3",
            all([result["named"] == ["click", "type", "select", "done"],
                 result["named_count"] == 4, result["state_changing"] == 3,
                 result["unimplemented_pct"] == 60,
                 "screenshot_region" in result["unimplemented"],
                 result["deliberate_noop"]]),
            f"apply_action names {result['named']} and {result['unimplemented']} fall through "
            f"to an unchanged copy with no error. {result['unimplemented_pct']}% of the "
            f"schema is a declaration with nothing behind it, and one of the four named "
            f"({DELIBERATE_NOOP}) is a deliberate no-op",
        ),
        practice.Check(
            "ANSWER: what benefits is anything whose target is smaller than a patch",
            all([result["full_tokens"] == 10_549, result["scale"] == 0.4406,
                 result["label_after"] == 6.2, result["under_a_patch"],
                 result["crop_tokens"] == 98]),
            f"a {SCREEN[0]}x{SCREEN[1]} screenshot is {result['full_tokens']:,} tokens at "
            f"patch {PATCH}; fitting {result['budget']:,} downscales it by "
            f"{result['scale']}, so a {LABEL_PIXELS}-pixel label lands at "
            f"{result['label_after']} pixels -- under half a patch. A {CROP[0]}x{CROP[1]} "
            f"crop is {result['crop_tokens']} patches at native resolution: dense tables, "
            "small-text forms, mobile UIs, chart axis labels",
        ),
        practice.Check(
            "FINDING: the tool cannot help this lesson's own benchmark",
            all([not result["reads_observation"], result["run_task_branches"] == 0]),
            f"run_task takes a task and iterates its plan with {result['run_task_branches']} "
            "branches, reading no observation at any point -- both benchmark tasks are fixed "
            "sequences with hard-coded coordinates. There is no point at which a cropped view "
            "could change an action, so the simulator has no input for the tool to produce",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
