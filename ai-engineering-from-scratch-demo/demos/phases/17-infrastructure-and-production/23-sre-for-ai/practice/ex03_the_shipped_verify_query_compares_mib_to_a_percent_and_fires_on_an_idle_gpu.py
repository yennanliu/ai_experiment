"""Exercise 3 — the shipped verify query compares MiB to a percent, and fires on an idle GPU.

    Write a structured runbook template: sections, required fields,
    verification commands.

Reading of the exercise: a template is only structured if a machine can
reject a runbook that does not fill it, so the template ships with a
validator. It is rendered to markdown, parsed back, and then used for the two
jobs the lesson gives structured runbooks: carrying verification commands
that can actually be run, and feeding retrieval.

**ANSWER: five sections -- meta, symptom, hypothesis, verify, act -- and 15
required fields.** Every verify step is a `command` plus an `expect` with a
unit, and every act step names its guard and its `rollback`. RB-017 rewritten
in the template passes the validator and survives a markdown round trip
unchanged. The reference's RB-017 -- the runbook agent's evidence, "runbook",
"last applied" and "safe action" -- fills 3 of the 15 fields. It has no
symptom query, no verify step and no rollback.

**FINDING: the shipped verification query would fire on an idle GPU.** The
metric agent's evidence is "DCGM_FI_DEV_FB_USED >= 97% for 240s". dcgm-exporter
documents that field as "Framebuffer memory used (in MiB)", so a unit-checked
`expect` rejects the comparison. Read literally it compares MiB to 97: a GPU
holding nothing but 16 GB of weights reads 16384, and the condition holds from
the moment the model loads. The command that means 97% is `FB_USED /
(FB_USED + FB_FREE) >= 0.97`, which is 0.2 on that GPU.

**FINDING: the runbook agent retrieves RB-017 for every incident; the
structured symptom field retrieves it only when the symptom is present.**
`runbook_agent` ignores its input and returns RB-017 at 0.88 for the checkout
incident and for a DNS incident alike. Matching the log agent's evidence
against the template's `log_pattern` retrieves RB-017 for checkout, because
'kv_cache_allocation_failed' appears in both. For the DNS incident it retrieves
nothing, and "no runbook" is itself the signal to page a human.

Structure: `TEMPLATE` lists each section's required fields; `validate()`,
`render()` and `parse()` work over plain dicts.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "23-sre-for-ai"
TEMPLATE = {
    "meta": ["id", "service", "owner", "last_verified"],
    "symptom": ["alert", "log_pattern", "query"],
    "hypothesis": ["cause", "evidence"],
    "verify": ["command", "expect", "unit"],
    "act": ["action", "guard", "rollback"],
}
UNITS = {"DCGM_FI_DEV_FB_USED": "MiB", "DCGM_FI_DEV_FB_FREE": "MiB"}
RB017 = {
    "meta": {"id": "RB-017", "service": "checkout-summary (vLLM)", "owner": "ml-platform",
             "last_verified": "2026-01-14"},
    "symptom": {"alert": "checkout 5xx > 5%", "log_pattern": "kv_cache_allocation_failed",
                "query": 'sum(rate(http_requests_total{code=~"5..",svc="checkout"}[5m]))'},
    "hypothesis": {"cause": "KV cache OOM under burst concurrency",
                   "evidence": "FB used ratio >= 0.97 for 240s before onset"},
    "verify": {"command": "DCGM_FI_DEV_FB_USED / (DCGM_FI_DEV_FB_USED + DCGM_FI_DEV_FB_FREE)",
               "expect": ">= 0.97", "unit": "ratio"},
    "act": {"action": "restart pod", "guard": "survivor load <= 1.0 (exercise 2)",
            "rollback": "none needed; pod is recreated from the same image"},
}


def validate(book):
    missing = [f"{s}.{f}" for s, fields in TEMPLATE.items() for f in fields
               if not book.get(s, {}).get(f)]
    metric = book.get("verify", {}).get("command", "").split(" ")[0]
    unit = book.get("verify", {}).get("unit")
    if metric in UNITS and unit not in (UNITS[metric], "ratio"):
        missing.append(f"verify.unit: {metric} is {UNITS[metric]}, not {unit}")
    return missing


def render(book):
    return "\n".join(f"## {s}\n" + "\n".join(f"- {k}: {v}" for k, v in book[s].items())
                     for s in TEMPLATE)


def parse(text):
    book, section = {}, None
    for line in text.splitlines():
        if line.startswith("## "):
            section = book.setdefault(line[3:], {})
        elif line.startswith("- "):
            key, value = line[2:].split(": ", 1)
            section[key] = value
    return book


def shipped(ref):
    """The reference RB-017, i.e. the runbook agent's evidence, placed in the template."""
    ev = dict(e.split(": ", 1) for e in ref.runbook_agent("").evidence)
    return {"meta": {"id": ev["runbook"], "last_verified": ev["last applied"]},
            "act": {"action": ev["safe action"]}}


def retrieve(log_evidence, books):
    return [b["meta"]["id"] for b in books
            if any(b["symptom"]["log_pattern"] in e for e in log_evidence)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    fb = ref.metric_agent("").evidence[0]
    incidents = {"checkout": "High error rate in /checkout/generate-summary",
                 "dns": "DNS resolution failures in payments"}
    return {
        "fields": sum(map(len, TEMPLATE.values())), "ours": validate(RB017),
        "round_trip": parse(render(RB017)) == RB017,
        "shipped_missing": validate(shipped(ref)),
        "fb_evidence": fb, "literal": validate({**RB017, "verify": {
            "command": "DCGM_FI_DEV_FB_USED", "expect": ">= 97", "unit": "%"}}),
        "idle": (16 * 1024 >= 97, round(16 * 1024 / (80 * 1024), 2)),
        "agent": {k: ref.runbook_agent(v).evidence[0] for k, v in incidents.items()},
        "retrieved": {k: retrieve(ref.log_agent(v).evidence, [RB017])
                      for k, v in incidents.items()},
    }


def verify(result):
    fields, missing = result["fields"], result["shipped_missing"]
    return [
        practice.Check(
            "ANSWER: five sections, 15 required fields; the shipped RB-017 fills 3",
            fields == 15 and result["ours"] == [] and result["round_trip"]
            and fields - len(missing) == 3,
            f"our RB-017 validates and round-trips; the reference's is missing "
            f"{len(missing)} of {fields}: {missing}",
        ),
        practice.Check(
            "FINDING: the shipped verification query would fire on an idle GPU",
            "DCGM_FI_DEV_FB_USED >= 97%" in result["fb_evidence"] and result["literal"]
            == ["verify.unit: DCGM_FI_DEV_FB_USED is MiB, not %"] and result["idle"] == (True, 0.2),
            f"'{result['fb_evidence']}' is rejected {result['literal']}; 16 GB of weights "
            f"reads 16384 MiB >= 97 = {result['idle'][0]}, ratio {result['idle'][1]}",
        ),
        practice.Check(
            "FINDING: the runbook agent retrieves RB-017 for every incident",
            set(result["agent"].values()) == {"runbook: RB-017"}
            and result["retrieved"] == {"checkout": ["RB-017"], "dns": []},
            f"runbook_agent: {result['agent']}; symptom-field retrieval: {result['retrieved']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
