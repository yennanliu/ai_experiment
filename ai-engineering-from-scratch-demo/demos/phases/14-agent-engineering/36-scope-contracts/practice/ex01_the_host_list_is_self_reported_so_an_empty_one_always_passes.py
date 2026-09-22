"""Exercise 1 — the host list is self-reported, so an empty one always passes.

    Add a `network_egress` field listing allowed external hosts. Refuse runs
    that touch other hosts.

Reading of the exercise: `network_egress` ships on `ScopeContract`,
`network_hosts` ships on `RunSummary`, and `scope_check` already emits a
blocking finding for hosts outside the allowlist. So the field is there and
the question is what it enforces -- and the answer is bounded by the fact
that the run reports its own hosts.

**ANSWER: the allowlist blocks 1 of 2 hosts and only when the run admits to
them.** Against an allowlist of `['api.anthropic.com']`, a run reporting
`['api.anthropic.com', 'evil.example']` produces a blocking
`network.unallowed_host` finding naming **1** host. The same run reporting
`[]` produces **0** findings and passes, because the check is guarded by
`and run.network_hosts`.

**FINDING: the guard makes silence indistinguishable from compliance.** Over
**4** runs -- compliant, violating, empty, and a violating run that omits the
host -- the checker refuses **1**. The contract can express deny-all with
`[]` and still cannot catch a run that declines to declare, so the field is a
*declaration* check and not an egress control.

**FINDING: `None` and `[]` are different policies and one of them is the
default.** `network_egress=None` means no enforcement and `[]` means deny
all, so a contract that omits the field permits every host while one that
sets it empty permits none. Of the lesson's **2** shipped contracts, **0**
use `None`, and a task author who deletes the line gets the permissive
reading.

**FINDING: hosts are compared with `not in`, so the allowlist is exact
strings.** `api.anthropic.com` is allowed and `API.anthropic.com`,
`api.anthropic.com.evil.test` and `api.anthropic.com:443` are all refused --
**3** of **3** near-misses blocked, which is the right direction -- while a
subdomain a team would expect to work, `eu.api.anthropic.com`, is refused
too. Exact matching fails closed and needs an explicit suffix rule to be
usable.

Structure: `contract()` builds the egress policy; `probe()` runs one summary
through the shipped checker.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "36-scope-contracts"
ALLOWED = ["api.anthropic.com"]
NEAR_MISSES = ("API.anthropic.com", "api.anthropic.com.evil.test",
               "api.anthropic.com:443", "eu.api.anthropic.com")


def contract(ref, egress=ALLOWED):
    return ref.ScopeContract(
        task_id="T-001", goal="call the model", allowed_files=["app.py"],
        forbidden_files=[], acceptance_criteria=[], rollback_plan="revert",
        network_egress=egress)


def probe(ref, hosts, egress=ALLOWED, files=("app.py",)):
    report = ref.scope_check(contract(ref, egress),
                             ref.RunSummary(touched_files=list(files),
                                            commands_run=[], network_hosts=list(hosts)))
    return {"codes": [f.code for f in report.findings],
            "blocks": sum(f.severity == "block" for f in report.findings),
            "passed": report.passed(),
            "detail": next((f.detail for f in report.findings
                            if f.code == "network.unallowed_host"), "")}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    compliant = probe(ref, ["api.anthropic.com"])
    violating = probe(ref, ["api.anthropic.com", "evil.example"])
    empty = probe(ref, [])
    silent = probe(ref, [])
    deny_all = probe(ref, ["api.anthropic.com"], egress=[])
    no_policy = probe(ref, ["evil.example"], egress=None)
    near = {host: probe(ref, [host])["blocks"] for host in NEAR_MISSES}
    return {
        "allowed": ALLOWED,
        "violating_blocks": violating["blocks"],
        "violating_names": "evil.example" in violating["detail"],
        "violating_count": violating["detail"].count("'"),
        "compliant_passed": compliant["passed"],
        "empty_passed": empty["passed"], "empty_codes": empty["codes"],
        "runs": 4,
        "refused": sum(not row["passed"] for row in
                       (compliant, violating, empty, silent)),
        "deny_all_passed": deny_all["passed"],
        "no_policy_passed": no_policy["passed"],
        "default_egress": ref.ScopeContract.__dataclass_fields__[
            "network_egress"].default,
        "near_misses": near,
        "near_blocked": sum(1 for blocks in near.values() if blocks),
        "subdomain_blocked": bool(near["eu.api.anthropic.com"]),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the allowlist blocks 1 of 2 hosts, and only when declared",
            all([result["violating_blocks"] == 1,
                 result["violating_names"] is True,
                 result["violating_count"] == 2,
                 result["compliant_passed"] is True,
                 result["empty_passed"] is True, result["empty_codes"] == []]),
            f"against {result['allowed']} a run reporting two hosts produces "
            f"{result['violating_blocks']} blocking finding naming the bad one "
            f"({result['violating_names']}), while the same run reporting no hosts "
            f"produces {result['empty_codes']} and passes "
            f"({result['empty_passed']})",
        ),
        practice.Check(
            "FINDING: silence is indistinguishable from compliance",
            all([result["runs"] == 4, result["refused"] == 1,
                 result["empty_passed"] is True]),
            f"over {result['runs']} runs -- compliant, violating, empty and a violating "
            f"run that omits the host -- the checker refuses {result['refused']}. The "
            "check is guarded by `and run.network_hosts`, so it validates a declaration "
            "rather than controlling egress",
        ),
        practice.Check(
            "FINDING: None and [] are different policies and None is the default",
            all([result["deny_all_passed"] is False,
                 result["no_policy_passed"] is True,
                 result["default_egress"] is None]),
            f"an empty allowlist refuses a compliant-looking run "
            f"({result['deny_all_passed']}) while None permits a call to evil.example "
            f"({result['no_policy_passed']}), and the field's default is "
            f"{result['default_egress']}. Deleting the line gets the permissive reading",
        ),
        practice.Check(
            "FINDING: hosts are compared as exact strings",
            all([result["near_blocked"] == 4,
                 result["subdomain_blocked"] is True,
                 len(result["near_misses"]) == 4]),
            f"all {result['near_blocked']} of {len(result['near_misses'])} near-misses "
            f"are blocked -- different case, a suffix attack, an explicit port, and "
            f"eu.api.anthropic.com ({result['subdomain_blocked']}). Exact matching fails "
            "closed, which is right, and needs an explicit suffix rule to be usable",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
