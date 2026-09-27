"""Exercise 3 — an integrity scope violation needs no output boundary, so output controls stop none of it.

    Aim Labs' Scope Violation framework has three boundaries: retrieval,
    scope, output. Construct a fourth CVE-class attack that exploits a
    different boundary combination.

Reading of the exercise: EchoLeak is retrieval -> scope *read* -> output. The
fourth class here is retrieval -> scope *write*: untrusted retrieved content
makes the agent change privileged state (a config flag), and nothing ever
crosses the output boundary. This is the shape of CVE-2025-53773, modelled
abstractly: a placeholder instruction and a boolean "wrote config" effect,
no real payload. Both attacks run through a toy agent built on the reference's
`State`, inbox and marker, with three independent gates, over all 8 on/off
combinations of the gates.

**ANSWER: the integrity attack gets through 2 of the 8 gate sets: none, and
the output gate alone.** EchoLeak gets through only with no gate at all (1 of
8), because it needs all three boundaries and any one gate breaks the chain.
The integrity attack never renders anything, so an output control (the ex02
allowlist, or CamoLeak's no-rendering fix) stops none of it; only the retrieval
gate or the scope gate does.

| gates on | EchoLeak leaks | integrity attack writes config |
|---|---|---|
| none | yes | yes |
| output only | no | yes |
| retrieval or scope (any set containing one) | no | no |

**FINDING: the reference cannot express this class.** The naive agent's only
tool is `render_image` and neither agent has a write path, so the reference
trace reconstruction can only show confidentiality leaks.

**FINDING: "all three must be prevented independently" is really a claim about
attack classes.** For EchoLeak alone any single gate suffices; the independence
matters because the next CVE skips a boundary the last fix guarded.

Structure: `run()` is the toy agent; `grid()` runs both attacks under every
gate set.
"""

from __future__ import annotations

import inspect
import itertools
import re
import urllib.parse as up

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "25-echoleak-cves-for-ai"
GATES = ("retrieval", "scope", "output")
ALLOWED_HOSTS = {"contoso.sharepoint.com", "res.cdn.office.net", "upload.wikimedia.org"}
ATTACKS = {
    "echoleak": {"reads_private": True, "writes_config": False, "renders": True},
    "integrity": {"reads_private": False, "writes_config": True, "renders": False},
}


def leak_url(ref):
    data = ",".join(e["body"] for e in ref.INBOX_PRIVATE if "MFA code" in e["body"])
    return f"https://signed.microsoft.com/img?data={data}"


def output_ok(url, gates):
    """The output boundary passes the render URL (only the output gate can stop it)."""
    return not ("output" in gates and up.urlsplit(url).hostname not in ALLOWED_HOSTS)


def effect(attack, privileged, url, gates):
    leak = privileged and attack["renders"] and attack["reads_private"] and output_ok(url, gates)
    return leak, privileged and attack["writes_config"]


def run(ref, marker, attack, gates):
    """(private data rendered, config written) for one attack under one gate set."""
    body = f"Q4 update. {marker} <placeholder instruction>]"
    triggered = marker in body and "retrieval" not in gates
    privileged = triggered and "scope" not in gates
    return effect(attack, privileged, leak_url(ref), gates)


def grid(ref, marker):
    out = {}
    for n in range(4):
        for gates in itertools.combinations(GATES, n):
            leak, _ = run(ref, marker, ATTACKS["echoleak"], gates)
            _, wrote = run(ref, marker, ATTACKS["integrity"], gates)
            out["+".join(gates) or "none"] = (leak, wrote)
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    naive, defended = inspect.getsource(ref.naive_copilot), inspect.getsource(ref.scope_separated_copilot)
    marker = re.search(r'if "(\[\w+:)" in body', naive).group(1)
    return {
        "marker": marker, "grid": grid(ref, marker),
        "tools": sorted(set(re.findall(r'"tool": "(\w+)"', naive + defended))),
        "write_path": any(w in naive + defended for w in ("write", "config", "setting")),
    }


def verify(result):
    g = result["grid"]
    leaks = [k for k, (leak, _) in g.items() if leak]
    writes = [k for k, (_, wrote) in g.items() if wrote]
    return [
        practice.Check(
            "ANSWER: the integrity attack passes 2 of 8 gate sets (none, output only); EchoLeak 1",
            len(g) == 8 and writes == ["none", "output"] and leaks == ["none"],
            f"marker {result['marker']!r}; (EchoLeak leaks, config written) per gate set: {g}",
        ),
        practice.Check(
            "FINDING: the reference cannot express this class",
            result["tools"] == ["render_image"] and not result["write_path"],
            f"tools named in either agent: {result['tools']}; any write path: {result['write_path']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
