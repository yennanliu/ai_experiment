"""Exercise 2 — restart is safe only below (N-1)/N load, and the reference auto-approves a limit change.

    Define three "safe" auto-remediation actions for your service. Justify each.

Reading of the exercise: "your service" is the lesson's own -- the vLLM
checkout-summary pool behind `code/main.py`. "Safe" is made checkable: an
action is safe when a policy gate can decide it from state alone, the same
automation can undo it, and its blast radius has a stated bound. Each of the
three actions is a guarded function, and the justification is the bound the
guard computes.

**ANSWER: restart one pod, revert the last deploy, scale within [2, 8] --
each with a guard, and the guard is the justification.**
- *Restart one pod* is safe only when the survivors can absorb its traffic:
  load u on N pods puts u*N/(N-1) on each of the rest, so the bound is
  u <= (N-1)/N. At N=4 that is 0.75; at 80% load the survivors run at 106.7%,
  and the restart turns one bad pod into a pool overload. At N=2 the bound is
  0.5, and at N=1 a restart is an outage. One pod at a time, not the same pod
  twice in 30 minutes.
- *Revert the last deploy* only when it landed inside the 60 minutes before
  onset and carries no migration: an image revert is undone by redeploying,
  and a schema change is not.
- *Scale within [2, 8], at most +2 per step*: the bound caps the cost of a
  wrong call at 2 GPUs.

**FINDING: the reference's proposed action is two actions, and the second is
on the lesson's own "not safe" list.** `supervisor()` always proposes "restart
pod + lower --gpu-memory-utilization". Lowering vLLM's memory fraction is
"modify resource limits", which the lesson lists as broad. The gate allows the
restart and denies the flag change. The reference auto-approves both whenever
two agents share a key (exercise 1), and RB-017's own evidence files it as
"safe action: restart pod + lower --gpu-memory-utilization to 0.85". vLLM preallocates its KV cache at
startup, so the change only lands with a restart. For an 8B bf16 model
(16 GB) on an 80 GB GPU, ignoring activations, 0.90 -> 0.85 shrinks the KV
budget from 56 to 52 GB, 7.1% fewer cached tokens: a capacity cut shipped as a
fix.

Structure: one guard per action returns (allowed, reason); anything else is
off the allowlist and denied.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "23-sre-for-ai"
POOL_MIN, POOL_MAX, STEP = 2, 8, 2


def survivor_load(u, n):
    return u * n / (n - 1) if n > 1 else float("inf")


def restart(state):
    u, n = state["util"], state["pods"]
    if state.get("restarted_recently"):
        return False, "same pod restarted in the last 30 min"
    ok = survivor_load(u, n) <= 1.0
    return ok, f"survivors at {survivor_load(u, n):.0%}"


def revert(state):
    if state["migration"]:
        return False, "deploy carries a migration"
    ok = 0 <= state["deploy_age_min"] <= 60
    return ok, f"deploy {state['deploy_age_min']} min before onset"


def scale(state):
    frm, to = state["pods"], state["target"]
    ok = POOL_MIN <= to <= POOL_MAX and abs(to - frm) <= STEP
    return ok, f"{frm} -> {to} within [{POOL_MIN}, {POOL_MAX}] step {STEP}"


def gate(action, state):
    guards = {"restart pod": restart, "revert deploy": revert, "scale pool": scale}
    guard = guards.get(action)
    return guard(state) if guard else (False, f"'{action}' is not on the allowlist")


def kv_budget(frac, gpu_gb=80, weights_gb=16):
    return frac * gpu_gb - weights_gb


def scenarios():
    """Each guard on one state it allows and states it denies."""
    return {
        "restart": [gate("restart pod", {"util": u, "pods": 4}) for u in (0.7, 0.8)],
        "revert": [gate("revert deploy", {"migration": m, "deploy_age_min": a})
                   for m, a in ((False, 20), (False, 3 * 1440), (True, 20))],
        "scale": [gate("scale pool", {"pods": frm, "target": to})
                  for frm, to in ((4, 6), (4, 12), (8, 10))],
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    proposed = ref.supervisor([ref.metric_agent(""), ref.runbook_agent("")])["proposed_action"]
    parts = ["restart pod" if p.startswith("restart") else p for p in proposed.split(" + ")]
    return {
        "bounds": {n: round((n - 1) / n, 2) for n in (1, 2, 4, 8)},
        "at_80": round(survivor_load(0.8, 4), 3), **scenarios(),
        "proposed": proposed, "parts": {p: gate(p, {"util": 0.5, "pods": 4})[0] for p in parts},
        "kv": (kv_budget(0.90), kv_budget(0.85)),
        "runbook": [e for e in ref.runbook_agent("").evidence if e.startswith("safe action")],
    }


def verify(result):
    kv = result["kv"]
    cut = 1 - kv[1] / kv[0]
    oks = {k: [ok for ok, _ in result[k]] for k in ("restart", "revert", "scale")}
    return [
        practice.Check(
            "ANSWER: restart one pod, revert the last deploy, scale within [2, 8], each guarded",
            all([result["bounds"] == {1: 0.0, 2: 0.5, 4: 0.75, 8: 0.88},
                 result["at_80"] == 1.067, oks["restart"] == [True, False],
                 oks["revert"] == oks["scale"] == [True, False, False]]),
            f"restart bound (N-1)/N {result['bounds']}; 80% on 4 pods puts survivors at "
            f"{result['at_80']:.1%}; revert {result['revert']}; scale {result['scale']}",
        ),
        practice.Check(
            "FINDING: the reference's second action is on the lesson's not-safe list",
            all([result["parts"] == {"restart pod": True, "lower --gpu-memory-utilization": False},
                 round(cut, 3) == 0.071, "gpu-memory-utilization" in result["runbook"][0]]),
            f"'{result['proposed']}' splits into {result['parts']}; 0.90 -> 0.85 on 80 GB "
            f"with 16 GB of weights shrinks the KV budget {kv[0]:.0f} -> {kv[1]:.0f} GB, {cut:.1%}; "
            f"RB-017 files it as {result['runbook']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
