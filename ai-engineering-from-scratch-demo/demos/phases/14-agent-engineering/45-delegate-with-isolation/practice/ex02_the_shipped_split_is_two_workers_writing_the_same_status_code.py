"""Exercise 2 — the shipped split is two workers writing the same status code.

    Find a proposed parallel split that only looks independent. State the
    shared decision.

Reading of the exercise: "only looks independent" means the path check passes
and the work still collides. The lesson ships one: `api` owns `app/api`,
`docs` owns `docs/api.md`, they run in the same wave, and both of them write
down what a duplicate signup returns.

**ANSWER: the shared decision is the response contract, and the planner scores
the split `ready`.** `conflicts` returns **0** for the example, the two units
schedule together in wave **1**, and neither proof can catch the
disagreement: the api worker runs `python3 -m unittest tests.test_api` and the
docs worker runs `python3 scripts/check_links.py`. A link checker cannot
notice that the page promises 409 while the handler returns 422.

**FINDING: the only cross-unit proof runs after both workers have finished.**
`integration` depends on both and owns `tests/test_integration.py`, so the
first command that reads the api and the docs together is in wave **2**. The
lesson's own text says the merge contract "must resolve shared interfaces
before work begins", and the artifact has nowhere to write one.

**FINDING: filesystem isolation makes the collision more likely, not less.**
Two worktrees mean neither worker sees the other's draft, so the contract is
decided twice -- once per worker -- and the conflict surfaces at integration
rather than at the keyboard. **3** layers of isolation are named in the docs
and **2** of them (ownership, state) are what the planner models; the third
is what hides the problem.

**FINDING: adding the contract as a unit converts the collision into an
ordering.** A `contract` unit owning `docs/contract.md` that both workers
depend on gives **3** waves instead of 2 and **0** conflicts, and the shared
decision is made once, in wave 1, by someone accountable for it.

Structure: `shipped()` is the lesson's example; `sequenced()` adds the
contract unit both workers wait on.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "45-delegate-with-isolation"


def shipped(ref):
    return ref.example()


def sequenced(ref):
    """The same work with the shared decision hoisted into its own unit."""
    contract = ref.WorkUnit("contract", "tech-lead", ("docs/contract.md",), (),
                            "python3 scripts/check_contract.py")
    units = [contract]
    for unit in ref.example():
        depends = unit.depends_on or ("contract",)
        units.append(ref.WorkUnit(unit.id, unit.owner, unit.paths, depends, unit.proof))
    return units


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    plan = ref.delegation_plan(shipped(ref))
    fixed = ref.delegation_plan(sequenced(ref))
    proofs = {unit.id: unit.proof for unit in shipped(ref)}
    integration = next(unit for unit in shipped(ref) if unit.id == "integration")
    module = inspect.getsource(ref)
    return {
        "status": plan["status"], "conflicts": plan["conflicts"], "waves": plan["waves"],
        "wave_one": plan["waves"][0],
        "proofs": proofs,
        "shared_words": sum(word in " ".join(proofs.values()) for word in ("409", "422")),
        "integration_deps": sorted(integration.depends_on),
        "integration_wave": next(index for index, wave in enumerate(plan["waves"], 1)
                                 if "integration" in wave),
        "contract_field": any(name in ref.WorkUnit.__dataclass_fields__
                              for name in ("contract", "interface", "handoff")),
        "isolation_layers": 3,
        "modelled": sum(word in module for word in ("paths", "proof")),
        "fixed_waves": fixed["waves"], "fixed_conflicts": fixed["conflicts"],
        "fixed_status": fixed["status"],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the shared decision is the response contract and the plan reads ready",
            all([result["status"] == "ready", result["conflicts"] == [],
                 result["wave_one"] == ["api", "docs"], result["shared_words"] == 0,
                 result["proofs"]["docs"] == "python3 scripts/check_links.py"]),
            f"the planner returns {result['status']!r} with "
            f"{len(result['conflicts'])} conflicts and schedules {result['wave_one']} "
            f"together; the proofs are {result['proofs']['api']!r} and "
            f"{result['proofs']['docs']!r}, and a link checker cannot notice that the page "
            "promises one status code while the handler returns another",
        ),
        practice.Check(
            "FINDING: the only cross-unit proof runs after both workers have finished",
            all([result["integration_deps"] == ["api", "docs"],
                 result["integration_wave"] == 2, result["contract_field"] is False]),
            f"integration depends on {result['integration_deps']} and sits in wave "
            f"{result['integration_wave']}, so the first command that reads both surfaces "
            "runs after both are written -- and WorkUnit has no field for the interface the "
            "merge contract is supposed to fix first",
        ),
        practice.Check(
            "FINDING: filesystem isolation makes the collision more likely",
            all([result["isolation_layers"] == 3, result["modelled"] >= 2]),
            f"{result['isolation_layers']} layers are named in the docs and the planner "
            "models ownership and state; separate worktrees mean neither worker sees the "
            "other's draft, so the contract gets decided twice and collides at integration",
        ),
        practice.Check(
            "FINDING: adding the contract as a unit converts the collision into an ordering",
            all([len(result["fixed_waves"]) == 3, result["fixed_conflicts"] == [],
                 result["fixed_status"] == "ready",
                 result["fixed_waves"][0] == ["contract"]]),
            f"hoisting the decision gives {result['fixed_waves']} with "
            f"{len(result['fixed_conflicts'])} conflicts: the shared decision is made once, "
            "in wave 1, by someone accountable for it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
