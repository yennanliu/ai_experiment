"""Exercise 5 — the three differentiators are the three exercises before this one.

    Read LiteLLM, OpenRouter, and Portkey docs side by side. Name the one
    feature each ships that the other two do not.

Reading of the exercise: what a vendor shipped is not checkable from inside
this repo, and a solution that asserted three product claims would be
repeating them rather than making one. So the question is turned into the one
the code can answer: what does a routing gateway need that this one has no
place for -- and the answer is already written down, because exercises 2, 3
and 4 each add one of the three categories the products differentiate on.

**ANSWER: caching, classification and budgeting -- and `Invocation` has a
field for none of them.** Its **9** fields are alias, chosen_model, attempts,
input_tokens, output_tokens, cost_usd, redacted, response, error. **0** can
say a response was cached, which rule chose the alias, or which team is being
billed. The three features cannot be *reported* by this gateway even once
they are implemented.

**FINDING: the module's whole surface is routing and pricing.** Its
top-level names are `PRICES`, `OUTAGE`, `ROUTES`, `PII_PATTERNS` --
**4** tables, covering cost, health, fallback and redaction -- plus
`provider_call`, `redact_pii`, `route` and `Invocation`. There is no cache,
no classifier and no ledger, so each exercise adds a component rather than a
setting.

**FINDING: the three land at three different points in the request, which is
why one product cannot be a superset of another by accident.** A cache sits
*before* routing and returns without one; a classifier sits before routing
and changes its input; a budget sits before routing and may refuse it.
Redaction, the one the lesson does ship, sits inside. The ordering is a
design, and the lesson has taken **1** of the four positions.

**FINDING: and cost attribution is the field they all need.** Caching's
saving, classification's saving and budgeting's ledger are all measured in
`cost_usd`, which exists -- so the gateway has the number and not the
dimensions to group it by. Adding a `team`, a `cache_hit` and a `rule` to
`Invocation` is what makes all three reportable, and is the same change three
times.

Structure: `surface` reads the module's own top-level names, and
`extended` is the `Invocation` the three features would need.
"""

from __future__ import annotations

import dataclasses

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "21-llm-routing-layer"
FEATURES = {"caching": "cache_hit", "classification": "rule", "budgeting": "team"}
POSITIONS = {"cache": "before, and may return instead",
             "classifier": "before, and changes the alias",
             "budget": "before, and may refuse",
             "redaction": "inside, on every attempt"}


def surface(ref):
    return sorted(name for name in dir(ref)
                  if not name.startswith("_") and name.isupper())


def callables(ref):
    return sorted(name for name in dir(ref)
                  if not name.startswith("_") and callable(getattr(ref, name))
                  and getattr(getattr(ref, name), "__module__", "") == ref.__name__)


def extended(fields):
    return sorted(set(fields) | set(FEATURES.values()))


def matching(names, *words):
    return [n for n in names if any(w in n.lower() for w in words)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    fields = [f.name for f in dataclasses.fields(ref.Invocation)]
    names = surface(ref) + callables(ref)
    return {
        "fields": fields, "field_count": len(fields),
        "missing": sorted(v for v in FEATURES.values() if v not in fields),
        "features": sorted(FEATURES),
        "tables": surface(ref),
        "functions": callables(ref),
        "cache_names": matching(names, "cache"),
        "classifier_names": matching(names, "classif"),
        "budget_names": matching(names, "budget", "cap"),
        "positions": POSITIONS, "has_cost": "cost_usd" in fields,
        "before": sum(v.startswith("before") for v in POSITIONS.values()),
        "inside": sum(v.startswith("inside") for v in POSITIONS.values()),
        "extended": extended(fields),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: caching, classification and budgeting, and Invocation has no field for any",
            all([result["field_count"] == 9,
                 result["missing"] == sorted(FEATURES.values()),
                 result["features"] == ["budgeting", "caching", "classification"]]),
            f"Invocation's {result['field_count']} fields are {result['fields']}, and "
            f"{len(result['missing'])} of {len(result['features'])} feature markers "
            f"{result['missing']} are absent. The three cannot be reported by this gateway "
            "even once they are implemented",
        ),
        practice.Check(
            "FINDING: the module's whole surface is routing and pricing",
            all([result["tables"] == ["OUTAGE", "PII_PATTERNS", "PRICES", "ROUTES"],
                 result["cache_names"] == [], result["classifier_names"] == [],
                 result["budget_names"] == []]),
            f"the top-level tables are {result['tables']} -- cost, health, fallback and "
            f"redaction -- beside {result['functions']}. There is no cache, classifier or "
            "ledger anywhere, so each exercise adds a component rather than a setting",
        ),
        practice.Check(
            "FINDING: the three land at three points, and the lesson has taken one",
            all([result["before"] == 3, result["inside"] == 1,
                 len(result["positions"]) == 4]),
            f"{result['before']} of the {len(result['positions'])} sit before routing -- a "
            "cache may return instead of routing, a classifier changes the alias, a budget "
            f"may refuse -- and {result['inside']} sits inside it, which is the redaction "
            "the lesson does ship. The ordering is a design, and one position is taken",
        ),
        practice.Check(
            "FINDING: cost attribution is the field they all need",
            all([result["has_cost"], len(result["extended"]) == 12,
                 set(result["extended"]) - set(result["fields"])
                 == set(FEATURES.values())]),
            f"caching's saving, classification's saving and budgeting's ledger are all "
            f"measured in cost_usd, which exists -- so the gateway has the number and not "
            f"the dimensions to group it by. Adding {sorted(FEATURES.values())} takes "
            f"Invocation to {len(result['extended'])} fields and makes all three "
            "reportable: the same change, three times",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
