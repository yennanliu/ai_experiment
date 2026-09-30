"""Exercise 2 — slot templates match the regex fixer on 5 of 5 demo drafts, and a misspelled fix is a silent no-op in the lesson's.

    Replace the regex fixer with a templating fixer that takes named slots. Demonstrate one rule rewritten under the new design.

Reading of the exercise: every fix becomes one shape, `{where, template,
slots, pattern}`. `where` is append, prepend or replace. `template` is a
`str.format` string. `slots` maps each `{name}` to a literal or to
`{"from": "group_name"}`, the name of the regex group that matched. The
lesson's three operations translate to templates with no slots, so the
existing constitution runs unchanged. `template_fixer` validates every fix
when it is built: an unknown `where`, a missing pattern, or a template
slot that is not declared raises ValueError. The rule rewritten is
`no-pii-in-examples`. Its single `[example-contact]` replacement becomes
`[{label}-{kind}]`, with the pattern split into named `email` and `phone`
groups, and the phone alternative also takes a leading `(` or `+`.

**ANSWER: with the slot-free translation, the templating fixer gives the
same revised text and second-pass violations as the lesson's `Fixer` on
5/5 demo drafts.** With the rewritten PII rule only the PII draft changes,
to "Example user: [example-email]. Here is how to look them up." Four
contact probes become `[example-email]` or `[example-phone]`, with 0
violations left on the second pass.

**FINDING: the lesson's replacement leaves a stray `(` or `+` on 2/4
probes, and the second pass does not notice.** "Call (555) 123-4567"
becomes "Call ([example-contact]" and "+1 555 123 4567" becomes
"+[example-contact]". The phone regex starts at `\\b`, which cannot sit
before `(` or `+`. The engine reports 0 violations on both results.

**FINDING: a misspelled fix operation is a silent no-op in the lesson.**
`Fixer` checks three keys with `if/elif` and ignores anything else, so
`append_if_mising` returns the draft unchanged and the violation stays.
The templating fixer rejects it when built, and it rejects a template
that uses an undeclared `{x}` slot.
"""
import re
import string
import sys
from unittest import mock

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "86-constitutional-rules-engine"
WHERE = ("append", "prepend", "replace")
PII_FIX = {"where": "replace", "template": "[{label}-{kind}]", "slots": {"label": "example", "kind": {"from": "group_name"}},
           "pattern": r"(?P<email>\b[\w.+-]+@[\w-]+\.[\w.-]+\b)"
                      r"|(?P<phone>(?<![\w(+])(?:\+\d{1,3}[ .-]?|\d{1,3}[ .-])?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}\b)"}
PII_DRAFTS = ["Example user: lee@example.com.", "Call (555) 123-4567 for an example.",
              "Example: +1 555 123 4567", "Sample: ops@acme.test or 555.123.4567"]
PII_NEW = ["Example user: [example-email].", "Call [example-phone] for an example.",
           "Example: [example-phone]", "Sample: [example-email] or [example-phone]"]


def from_regex_fix(fix):
    """The lesson's three operations, restated as templates with no slots; anything else passes through."""
    if "replace_regex" in fix:
        return {"where": "replace", "pattern": fix["replace_regex"]["pattern"],
                "template": fix["replace_regex"]["replacement"]}
    op = next((op for op in ("append_if_missing", "prepend_if_missing") if op in fix), None)
    return {"where": op.split("_")[0], "template": fix[op]} if op else fix


def check_spec(name, spec):
    """Refuse at load time what the lesson's Fixer would silently skip."""
    slots = spec.get("slots", {})
    used = {f for _, f, _, _ in string.Formatter().parse(spec.get("template", "")) if f}
    bad_slot = any(v != {"from": "group_name"} and not isinstance(v, str) for v in slots.values())
    no_pattern = (spec.get("where"), "pattern" in spec) == ("replace", False)
    if any([spec.get("where") not in WHERE, "template" not in spec, no_pattern, bad_slot, used - set(slots)]):
        raise ValueError(f"rule {name}: bad fix {spec!r} (undeclared slots: {sorted(used - set(slots))})")


def render(spec, match=None):
    """A slot is a literal string or {"from": "group_name"}: which named group of the pattern matched."""
    slots = spec.get("slots", {})
    return spec["template"].format(**{k: v if isinstance(v, str) else match.lastgroup for k, v in slots.items()})


def apply_fixes(specs, text, violations):
    out = text
    for spec in (specs[v.rule_name] for v in violations if v.rule_name in specs):
        if spec["where"] == "replace":
            out = re.sub(spec["pattern"], lambda m, s=spec: render(s, m), out, flags=re.IGNORECASE)
            continue
        piece = render(spec)
        if piece.strip() and piece.strip() not in out:
            out = out.rstrip() + piece if spec["where"] == "append" else piece + out.lstrip()
    return out


def template_fixer(rules):
    """Validate every fix up front; return apply(text, violations) -> revised."""
    specs = {r["name"]: r["fix"] if "where" in r["fix"] else from_regex_fix(r["fix"]) for r in rules if "fix" in r}
    [check_spec(name, spec) for name, spec in specs.items()]
    return lambda text, violations: apply_fixes(specs, text, violations)


def run(main, fixer, drafts):
    """(revised text, violations left on the second pass) per draft."""
    engine = main.Engine()
    revised = [fixer(d, engine.evaluate(d).violations()) for d in drafts]
    return [(t, len(engine.evaluate(t).violations())) for t in revised]


def rejects(rules):
    try:
        template_fixer(rules)
    except ValueError as exc:
        return str(exc)


def solve():
    # main.py does `from yaml_subset import load_yaml`; register the lesson's module for that import only
    with mock.patch.dict(sys.modules, {"yaml_subset": parity.load_reference(PHASE, LESSON, "yaml_subset")}):
        main = parity.load_reference(PHASE, LESSON, "main")
    rules, old = main.Engine().rules(), main.Fixer(main.Engine().rules()).apply
    rewritten = [dict(r, fix=PII_FIX) if r["name"] == "no-pii-in-examples" else r for r in rules]
    demo = [d["draft"] for d in main._DEMO_DRAFTS]
    typo = [dict(rules[0], fix={"append_if_mising": " Try the help page."})]
    return {
        "demo": [run(main, f, demo) for f in (old, template_fixer(rules), template_fixer(rewritten))],
        "pii": [run(main, f, PII_DRAFTS) for f in (old, template_fixer(rewritten))],
        "typo_old": main.Fixer(typo).apply(demo[0], main.Engine(rules=typo).evaluate(demo[0]).violations()) == demo[0],
        "typo_new": rejects(typo), "undeclared": rejects([dict(rules[0], fix={"where": "append", "template": "{x}"})]),
    }


def verify(result):
    old, same, new = result["demo"]
    changed = [i for i, (a, b) in enumerate(zip(old, new)) if a != b]
    (pii_old, pii_new), left = ([[t for t, _ in r] for r in result["pii"]], [n for r in result["pii"] for _, n in r])
    return [
        practice.Check(
            "ANSWER: slot-free templates match the lesson Fixer on 5/5 demo drafts; the PII rewrite labels email/phone",
            (same == old, changed, new[2], pii_new, left) ==
            (True, [2], ("Example user: [example-email]. Here is how to look them up.", 0), PII_NEW, [0] * 8),
            f"demo draft 3 -> {new[2][0]!r}; PII probes -> {pii_new}",
        ),
        practice.Check(
            "FINDING: the lesson's phone regex leaves '(' and '+' behind on 2/4 probes, and the engine passes both",
            pii_old[1:3] == ["Call ([example-contact] for an example.", "Example: +[example-contact]"],
            f"lesson Fixer: {pii_old}; violations left on the second pass: {left[:4]}",
        ),
        practice.Check(
            "FINDING: a misspelled fix op is a silent no-op in the lesson's Fixer; the template fixer refuses it at load",
            (result["typo_old"], None in (result["typo_new"], result["undeclared"])) == (True, False),
            f"lesson Fixer returns the draft unchanged; new: {result['typo_new']!r}; {result['undeclared']!r}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
