"""Exercise 3 — the mode it asks for is a wire name, not a canonical one.

    Implement `tool_choice` conversion: map a canonical
    `ToolChoice(mode="force", tool_name="x")` into all three provider shapes.
    Then map `mode="any"` and `mode="none"`. Check the lesson's diff table.

Reading of the exercise: all three modes are run through all three converters,
and `mode="any"` is not special-cased away when it fails -- the failure is the
result. The conversion is already implemented in the lesson, so what the
exercise actually asks for is a check of it, and the check is what turns up the
vocabulary problem.

**ANSWER: `force` and `none` convert cleanly; `any` raises in all three.**
`force` gives OpenAI `{"type": "function", "function": {"name": "x"}}`,
Anthropic `{"type": "tool", "name": "x"}` and Gemini `ANY` plus
`allowed_function_names`. `none` gives the bare string `"none"`, `{"type":
"none"}` and `{"mode": "NONE"}`. `any` gives **`ValueError: any`** from **3 of
3**.

**FINDING: "any" is Anthropic's wire name for the mode the lesson calls
`required`.** The canonical vocabulary is `auto / none / required / force`, and
`tool_choice_anthropic` maps `required` **to** `{"type": "any"}`. So the
exercise asks you to feed a provider's output back in as canonical input. The
mode it means exists; the name it uses is the one on the other side of the
translation.

**FINDING: which is exactly the bug a canonical layer exists to prevent, and the
lesson's own table is where it leaks.** `required` is the only one of the four
modes whose Anthropic spelling differs from its canonical name, and it is the
one the exercise gets wrong. Reading the diff table left to right gives `auto`,
`none`, `any`, `tool` down the Anthropic column -- two of which are canonical
names and two of which are not.

**FINDING: and `force` is the only mode that needs the tool name.** The other
three ignore `tool_name` entirely: passing one through `auto`, `none` or
`required` changes **0** of the nine payloads. Gemini is the only provider whose
`force` shape is a *modifier* on another mode -- `ANY` plus a one-element
allow-list -- rather than a distinct type, so Gemini's `force` and `required`
differ by one key while the other two differ by their whole shape.

Structure: `convert` runs one canonical choice through all three converters and
records either the payload or the exception, `MODES` is the four the lesson
supports plus the one the exercise names, and `ignores_name` re-runs each mode
with a tool name attached.
"""

from __future__ import annotations

import json

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "02-function-calling-deep-dive"
CANONICAL = ("auto", "none", "required", "force")
ASKED = ("force", "any", "none")
TOOL = "get_weather"


def converters(ref):
    return {"openai": ref.tool_choice_openai, "anthropic": ref.tool_choice_anthropic,
            "gemini": ref.tool_choice_gemini}


def convert(ref, mode, tool_name=None):
    """Each provider's payload for one canonical mode, or the exception it raised."""
    choice = ref.ToolChoice(mode=mode, tool_name=tool_name)
    out = {}
    for provider, function in converters(ref).items():
        try:
            out[provider] = json.dumps(function(choice))
        except ValueError as error:
            out[provider] = f"ValueError: {error}"
    return out


def ignores_name(ref):
    """Modes whose payload is unchanged by supplying a tool_name."""
    return [mode for mode in CANONICAL
            if convert(ref, mode) == convert(ref, mode, TOOL)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    asked = {mode: convert(ref, mode, TOOL if mode == "force" else None)
             for mode in ASKED}
    anthropic_column = [json.loads(convert(ref, mode, TOOL)["anthropic"])["type"]
                        for mode in CANONICAL]
    gemini_required = json.loads(convert(ref, "required")["gemini"])
    gemini_force = json.loads(convert(ref, "force", TOOL)["gemini"])
    return {
        "asked": asked,
        "any_errors": sum(value.startswith("ValueError")
                          for value in asked["any"].values()),
        "force_ok": all(not value.startswith("ValueError")
                        for value in asked["force"].values()),
        "none_ok": all(not value.startswith("ValueError")
                       for value in asked["none"].values()),
        "canonical_modes": list(CANONICAL),
        "anthropic_column": anthropic_column,
        "required_maps_to_any": anthropic_column[CANONICAL.index("required")] == "any",
        "shared_names": [mode for mode in CANONICAL if mode in anthropic_column],
        "ignores_name": ignores_name(ref),
        "gemini_force_extra": sorted(set(gemini_force["function_calling_config"])
                                     - set(gemini_required["function_calling_config"])),
        "gemini_same_mode":
            gemini_force["function_calling_config"]["mode"]
            == gemini_required["function_calling_config"]["mode"],
    }


def verify(result):
    asked = result["asked"]
    return [
        practice.Check(
            "ANSWER: force and none convert cleanly; any raises in all three",
            all([result["force_ok"], result["none_ok"], result["any_errors"] == 3,
                 asked["force"]["anthropic"] == '{"type": "tool", "name": "get_weather"}',
                 asked["none"]["openai"] == '"none"']),
            f"force gives {asked['force']}; none gives {asked['none']}; and any gives "
            f"ValueError from {result['any_errors']} of 3 -- {asked['any']}",
        ),
        practice.Check(
            "FINDING: 'any' is Anthropic's wire name for the mode the lesson calls required",
            all([result["canonical_modes"] == ["auto", "none", "required", "force"],
                 result["required_maps_to_any"], result["any_errors"] == 3]),
            f"the canonical vocabulary is {result['canonical_modes']}, and "
            f"tool_choice_anthropic maps required TO {{'type': 'any'}}. So the exercise asks "
            "you to feed a provider's output back in as canonical input: the mode it means "
            "exists, but the name it uses is the one on the other side of the translation",
        ),
        practice.Check(
            "FINDING: the lesson's own diff table is where the vocabulary leaks",
            all([result["anthropic_column"] == ["auto", "none", "any", "tool"],
                 result["shared_names"] == ["auto", "none"]]),
            f"reading the diff table down the Anthropic column gives "
            f"{result['anthropic_column']} -- {result['shared_names']} are also canonical "
            f"names and the other two are not. required is the only mode whose Anthropic "
            "spelling differs from its canonical name, and it is the one the exercise gets "
            "wrong",
        ),
        practice.Check(
            "FINDING: force is the only mode that needs the tool name",
            all([result["ignores_name"] == ["auto", "none", "required"],
                 result["gemini_same_mode"],
                 result["gemini_force_extra"] == ["allowed_function_names"]]),
            f"supplying a tool_name changes nothing for {result['ignores_name']} -- 0 of "
            f"those nine payloads move. Gemini is the only provider whose force shape is a "
            f"modifier on another mode: its force and required carry the same mode value and "
            f"differ only by {result['gemini_force_extra']}, while OpenAI and Anthropic change "
            "shape entirely",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
