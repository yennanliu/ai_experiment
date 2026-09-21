"""Exercise 3 — sensitive is a property of elements, and typing has none.

    Add a confirmation gate for actions tagged `sensitive=True`. Log every
    denied confirmation.

Reading of the exercise: the gate is shipped -- `needs_confirmation` plus
`human_confirm` -- and the log is not: a denial becomes the string
`"DENIED BY HUMAN: ..."` inside a trace tuple, which no caller can count
without parsing prose. So the work is the audit log, and writing one exposes
what the gate can and cannot see.

**ANSWER: a structured audit log, and it records 2 denials in 5 actions.**
Each entry carries the action kind, its arguments, the element, the verdict
and the outcome. Against the shipped trace the same run yields **5** tuples
whose second field is a string, from which **2** denials are recoverable only
by matching the prefix `"DENIED BY HUMAN"`.

**FINDING: `type` is never gated, whatever it is typed into.** `sensitive`
lives on `Element`, and the `type` branch of `assess` never calls
`element_at` -- it has no coordinates to call it with. Typing a card number
into an element flagged `sensitive=True` returns `allow=True,
needs_confirmation=False`, so **0** of **3** typing actions reach the human
while **2** of **2** clicks on the same element do.

**FINDING: the human is asked to approve a label.** The confirmation callback
receives `verdict.reason`, which is
`"label 'buy_button' is sensitive; confirm required"` -- **0** of the
action's **2** arguments appear in it. Two clicks 60 pixels apart produce
byte-identical prompts, so the reviewer cannot tell which purchase they are
approving.

**FINDING: a denial and a block are the same outcome and different events.**
`run_agent` appends to one list either way, and the shipped trace
distinguishes them only by a prefix. In the audit log they separate: **2**
human denials, **1** classifier block, **2** executions -- and the denial
count is the number a rollout gate would watch.

Structure: `audit()` re-runs the lesson's loop recording structure instead of
prose; `SCRIPT` is the five actions both versions see.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "21-computer-use-agents"
SCRIPT = (("click", {"x": 140, "y": 115}), ("click", {"x": 140, "y": 215}),
          ("type", {"text": "4111 1111 1111 1111"}), ("click", {"x": 9, "y": 9}),
          ("click", {"x": 140, "y": 215}))


def world(ref):
    screen = ref.Screen(
        elements=[ref.Element("btn_search", "search_button", 100, 100, 80, 30),
                  ref.Element("btn_buy", "buy_button", 100, 200, 80, 30,
                              sensitive=True),
                  ref.Element("fld_card", "card_field", 50, 60, 200, 30,
                              sensitive=True)],
        dom_text="Search for products and buy with one click.")
    guard = ref.SafetyClassifier(
        allowed_labels=("search_button", "buy_button", "card_field"))
    return screen, guard


def audit(ref, screen, guard, approve):
    """The lesson's loop, recording structure where it recorded prose."""
    log = []
    for kind, args in SCRIPT:
        verdict = guard.assess(ref.Action(kind, args), screen)
        element = screen.element_at(args["x"], args["y"]) if kind == "click" else None
        entry = {"kind": kind, "args": args, "reason": verdict.reason,
                 "element": element.eid if element else None,
                 "confirmed": None, "outcome": "executed"}
        if not verdict.allow:
            entry["outcome"] = "blocked"
        elif verdict.needs_confirmation:
            entry["confirmed"] = approve(verdict.reason)
            entry["outcome"] = "executed" if entry["confirmed"] else "denied"
        log.append(entry)
    return log


def shipped_trace(ref, screen, guard, approve):
    actions = [ref.Action(kind, args) for kind, args in SCRIPT]
    return ref.run_agent(actions, screen, guard, human_confirm=approve)


def typing_gate(ref, screen, guard):
    """Typing into an element flagged sensitive, and clicking the same one."""
    typed = guard.assess(ref.Action("type", {"text": "4111 1111 1111 1111"}), screen)
    clicked = guard.assess(ref.Action("click", {"x": 140, "y": 75}), screen)
    return {"typed_allow": typed.allow, "typed_confirm": typed.needs_confirmation,
            "clicked_confirm": clicked.needs_confirmation,
            "element": screen.element_at(140, 75).eid}


def prompts(ref, screen, guard):
    seen = [guard.assess(ref.Action("click", {"x": x, "y": 215}), screen).reason
            for x in (110, 170)]
    return {"identical": seen[0] == seen[1], "text": seen[0],
            "args_present": sum(str(v) in seen[0] for v in (110, 170, 215))}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    screen, guard = world(ref)
    log = audit(ref, screen, guard, lambda reason: False)
    trace = shipped_trace(ref, screen, guard, lambda reason: False)
    outcomes = [entry["outcome"] for entry in log]
    return {
        "entries": len(log), "outcomes": outcomes,
        "denied": outcomes.count("denied"), "blocked": outcomes.count("blocked"),
        "executed": outcomes.count("executed"),
        "trace_len": len(trace),
        "trace_denials": sum(result.startswith("DENIED BY HUMAN")
                             for _, result in trace),
        "trace_types": sorted({type(result).__name__ for _, result in trace}),
        "fields": sorted(log[0]),
        "typing": typing_gate(ref, screen, guard),
        "typed_gated": sum(entry["confirmed"] is not None for entry in log
                           if entry["kind"] == "type"),
        "click_gated": sum(entry["confirmed"] is not None for entry in log
                           if entry["kind"] == "click"),
        "prompt": prompts(ref, screen, guard),
    }


def verify(result):
    typing, prompt = result["typing"], result["prompt"]
    return [
        practice.Check(
            "ANSWER: a structured audit log recording 2 denials in 5 actions",
            all([result["entries"] == 5, result["denied"] == 2,
                 result["trace_len"] == 5, result["trace_denials"] == 2,
                 result["trace_types"] == ["str"],
                 result["fields"] == ["args", "confirmed", "element", "kind",
                                      "outcome", "reason"]]),
            f"the log carries {result['fields']} per action and records "
            f"{result['denied']} denials in {result['entries']} actions. The shipped "
            f"trace yields {result['trace_len']} tuples whose second field is a "
            f"{result['trace_types'][0]}, from which the same {result['trace_denials']} "
            "are recoverable only by matching a prefix",
        ),
        practice.Check(
            "FINDING: type is never gated, whatever it is typed into",
            all([typing["typed_allow"] is True, typing["typed_confirm"] is False,
                 typing["clicked_confirm"] is True, typing["element"] == "fld_card",
                 result["typed_gated"] == 0, result["click_gated"] == 2]),
            f"sensitive lives on Element and the type branch never calls element_at -- it "
            f"has no coordinates to call it with. Typing a card number returns "
            f"allow={typing['typed_allow']}, needs_confirmation={typing['typed_confirm']} "
            f"while a click on {typing['element']} returns "
            f"{typing['clicked_confirm']}: {result['typed_gated']} typing actions reach "
            f"the human against {result['click_gated']} clicks",
        ),
        practice.Check(
            "FINDING: the human is asked to approve a label",
            all([prompt["identical"] is True, prompt["args_present"] == 0,
                 "buy_button" in prompt["text"]]),
            f"the callback receives verdict.reason -- {prompt['text']!r} -- in which "
            f"{prompt['args_present']} of the action's arguments appear. Two clicks 60 "
            f"pixels apart produce byte-identical prompts ({prompt['identical']}), so the "
            "reviewer cannot tell which purchase they are approving",
        ),
        practice.Check(
            "FINDING: a denial and a block are the same outcome and different events",
            all([result["blocked"] == 1, result["denied"] == 2,
                 result["executed"] == 2,
                 result["outcomes"] == ["executed", "denied", "executed", "blocked",
                                        "denied"]]),
            f"run_agent appends to one list either way and the trace separates them by a "
            f"prefix. In the audit log they are three outcomes: {result['denied']} human "
            f"denials, {result['blocked']} classifier block, {result['executed']} "
            "executions -- and the denial count is what a rollout gate watches",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
