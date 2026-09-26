"""Exercise 5 — seven fields reach five frameworks, but the logging clauses sit under retention and nothing is written.

    Map your LLM audit log fields (Phase 17 · 25) to at least three framework
    controls.

Reading of the exercise: "your audit log" is Lesson 25's `AuditEntry`, its 10
fields exactly as shipped. Each field is mapped to the clauses whose
requirement it is evidence for. The mapping is then compared with this
lesson's CONTROL_MAP, and checked against what the Lesson 25 code actually
does with an entry.

**ANSWER: 7 of 10 fields map to 7 clauses across 5 frameworks.** timestamp,
user and tenant are evidence for HIPAA 164.312(b) audit controls, ISO 27001
A.8.15 logging, SOC 2 CC7 and EU AI Act Art. 12 record-keeping. model feeds
SOC 2 CC8 and Art. 12. The two hashes are GDPR Art. 25 minimisation: they are
kept in place of the raw text. guardrail_trips feeds CC7, A.8.16 monitoring
and Art. 12. input_tokens, output_tokens and cost_usd map to no control; they
are FinOps fields. In `main()`'s own output cost_usd is 0.0012 on all 3
entries while input tokens run 11, 6 and 7.

**FINDING: the map files the logging clauses under retention, not under
access logging.** The logging clauses sit in CONTROL_MAP's "audit log
retention" row: A.8.15, 164.312(b) and SOC 2 CC7 all appear there, and all
three are in this mapping. The "access logging" row cites ISO A.5.15-5.18 and
HIPAA 164.312(a), whose heading in the CFR is "Standard: Access control". Not
one of those clauses is in this mapping. A team that claims "access logging"
from the map is citing its access-control clauses as log evidence.

**FINDING: the lesson's log is not appended, retained or tamper-evident.** The
Lesson 25 docstring says it "Appends to an immutable audit log on every call".
`audit_log_call` returns a JSON string, and the module opens no file. Nothing
records the SOC 2 1-year or HIPAA 6-year retention its docs give. Change the
user on an entry and nothing can tell. Adding one `prev` field, a hash chain,
makes the same edit detectable at the edited entry, and that is the property
A.8.15 asks for when it says logs are protected.

**FINDING: "Model + version" is logged as a model name.** Lesson 25's docs
ask for model and version. The field holds "anthropic/claude-3.7-sonnet", with
no snapshot or date, so Art. 12 or CC8 evidence cannot say which build served a
call.

Structure: FIELD_MAP is the deliverable; `chain()` / `broken_at()` are the
smallest tamper-evidence the log needs.
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import io
import json
import re
import warnings

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "26-compliance-frameworks"
SEC, MODEL = "25-security-secrets-audit", "anthropic/claude-3.7-sonnet"
LOGGING = ["HIPAA §164.312(b)", "ISO 27001 A.8.15", "SOC 2 CC7", "EU AI Act Art. 12"]
FIELD_MAP = {
    **dict.fromkeys(("timestamp", "user", "tenant"), LOGGING),
    "model": ["SOC 2 CC8", "EU AI Act Art. 12"],
    **dict.fromkeys(("prompt_hash", "response_hash"), ["GDPR Art. 25"]),
    "guardrail_trips": ["SOC 2 CC7", "ISO 27001 A.8.16", "EU AI Act Art. 12"],
    **dict.fromkeys(("input_tokens", "output_tokens", "cost_usd"), []),
}


def chain(lines):
    out = []
    for line in lines:
        prev = hashlib.sha256(out[-1].encode()).hexdigest()[:12] if out else "0" * 12
        out.append(json.dumps(dict(json.loads(line), prev=prev), sort_keys=True))
    return out


def broken_at(lines):
    prevs = ["0" * 12] + [hashlib.sha256(x.encode()).hexdigest()[:12] for x in lines]
    bad = [i - 1 for i, x in enumerate(lines) if json.loads(x)["prev"] != prevs[i]]
    return bad[0] if bad else None


def tamper(lines):
    edited = dict(json.loads(lines[1]), user="user_999")
    return [lines[0], json.dumps(edited, sort_keys=True), *lines[2:]]


def integrity(sec, lines):
    return {
        "claims": "Appends to an immutable audit log" in " ".join(sec.__doc__.split()),
        "writes": bool(re.search(r"open\(|\.write\(", inspect.getsource(sec))),
        "plain_detect": broken_at(chain(tamper(lines))),
        "chained_detect": broken_at(tamper(chain(lines))),
    }


def mapping(control_map):
    cites = {c for v in FIELD_MAP.values() for c in v}
    return {
        "cites": sorted(cites),
        "mapped": [f for f, v in FIELD_MAP.items() if v],
        "frameworks": sorted({re.split(r" (?:§|A\.|CC|Art)", c)[0] for c in cites}),
        "in_access": sorted(cites & set(control_map["access logging"])),
        "in_retention": sorted(cites & set(control_map["audit log retention"])),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    sec = parity.load_reference(PHASE, SEC, "main")
    out = io.StringIO()
    with contextlib.redirect_stdout(out), warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # its datetime.utcnow()
        sec.main()
    lines = [x for x in out.getvalue().splitlines() if x.startswith("{")]
    entries = [json.loads(x) for x in lines]
    return {
        **integrity(sec, lines),
        **mapping(ref.CONTROL_MAP),
        "fields": list(sec.AuditEntry.__dataclass_fields__),
        "costs": sorted({e["cost_usd"] for e in entries}),
        "tokens": [e["input_tokens"] for e in entries],
        "models": {e["model"] for e in entries},
        "version_doc": "Model + version" in parity.doc_text(PHASE, SEC),
    }


def verify(result):
    sizes = [len(result[k]) for k in ("fields", "mapped", "cites", "frameworks")]
    sizes.append(len(set(result["tokens"])))
    return [
        practice.Check(
            "ANSWER: 7 of 10 fields map to 7 clauses across 5 frameworks",
            (sizes, result["costs"]) == ([10, 7, 7, 5, 3], [0.0012]),
            f"mapped {result['mapped']} to {result['cites']}; tokens and cost map to none, "
            f"and cost_usd is {result['costs']} on entries of {result['tokens']} tokens",
        ),
        practice.Check(
            "FINDING: the map files the logging clauses under retention, not access logging",
            (result["in_access"], len(result["in_retention"])) == ([], 3),
            "of this mapping, 'access logging' (ISO A.5.15-5.18, HIPAA 164.312(a)) shares "
            f"{result['in_access']}; 'audit log retention' shares {result['in_retention']}",
        ),
        practice.Check(
            "FINDING: the lesson's log is not appended, retained or tamper-evident",
            [result[k] for k in ("claims", "writes", "plain_detect", "chained_detect")]
            == [True, False, None, 1],
            "docstring claims an immutable append, source opens no file; editing entry 1 "
            f"before chaining is detected at {result['plain_detect']}, after chaining at "
            f"entry {result['chained_detect']}",
        ),
        practice.Check(
            'FINDING: "Model + version" is logged as a model name',
            result["version_doc"] and result["models"] == {MODEL},
            f"docs ask for model + version; every entry logs {result['models']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
