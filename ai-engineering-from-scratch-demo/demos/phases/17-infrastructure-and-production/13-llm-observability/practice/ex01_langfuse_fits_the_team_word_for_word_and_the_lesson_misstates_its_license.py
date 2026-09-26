"""Exercise 1 — Langfuse fits the team word for word, and the lesson misstates its license.

    Your team on LangChain wants OSS self-hosted observability. Pick Langfuse
    or Opik and justify.

Reading of the exercise: the team's three words are three hard filters --
LangChain integration, an OSS license, self-hosting -- and a pick is justified
only if it survives all three and then wins on something the filters do not
test. The facts come from the lesson's own tool sections, cross-checked
against each project's repository (read 2026-09-26); the lesson's `code/`
models retention, not tools, so the filter runs over those facts.

**ANSWER: Langfuse -- unless the team already runs Comet, then Opik.** Both
pass every hard filter: both ship a LangChain integration
(`langfuse.langchain.CallbackHandler`, `opik.integrations.langchain.OpikTracer`),
both are OSS, and both self-host with Docker. LangSmith, the LangChain-native
choice, is out: the lesson says it self-hosts on Enterprise only. What breaks
the tie is the lesson's own sweet spots: Langfuse's is "LangSmith-class
features but must self-host or stay on OSS license" -- this team, word for
word -- while Opik's is "ML teams already on Comet".

**FINDING: the lesson states Langfuse's license three ways.** Its Langfuse
section says "Core Apache / MIT", the summary "MIT-licensed core" and Key Terms
"MIT OSS". The repository's LICENSE is MIT outside its `ee/` directories,
which hold 13 commercially licensed features: 8 in the web app (admin API,
audit-log viewer, billing, multi-tenant SSO, SSO settings, Salesforce sync,
UI customization, verified domains) and 5 in the worker, among them
`dataRetention`. Opik's LICENSE is Apache-2.0 with no such carve-out. A team
self-hosting Langfuse on the MIT core therefore gets traces, evals, prompts
and datasets, but SSO and automatic data retention are the places where the
two licenses differ -- and retention is what this lesson's sampling section
is about.

**FINDING: the free tier the lesson quotes cannot serve this team, which
is why self-hosting is the requirement.** 50K events a month, even at one event
per trace, covers 72 minutes of the reference simulator's 1M-trace day --
1/600 of the month.

Structure: `FACTS` holds each tool's attributes with where they were read;
`passes()` applies the three filters and `pick()` the tie-break.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "13-llm-observability"
FREE_EVENTS_PER_MONTH = 50_000
# Read 2026-09-26: LICENSE files and SDK trees of langfuse/langfuse(-python), comet-ml/opik.
# web/src/ee/features and worker/src/ee of langfuse/langfuse, read 2026-09-26.
EE = """admin-api audit-log-viewer billing multi-tenant-sso sfdc-sync sso-settings
    ui-customization verified-domains cloudSpendAlerts cloudUsageMetering dataRetention
    meteringDataPostgresExport usageThresholds""".split()
FIELDS = ("license", "oss", "self_host", "langchain")
FACTS = {
    tool: dict(zip(FIELDS, row))
    for tool, row in {
        "Langfuse": (
            "MIT core + ee/",
            True,
            True,
            "langfuse.langchain.CallbackHandler",
        ),
        "Opik": ("Apache-2.0", True, True, "opik.integrations.langchain.OpikTracer"),
        "LangSmith": ("commercial", False, False, "native"),
    }.items()
}


def section(doc, heading):
    """Body of a `### heading` section of the lesson page."""
    body = doc.split(f"### {heading}", 1)[1]
    return body.split("\n### ", 1)[0]


def passes(tool):
    fact = FACTS[tool]
    return bool(fact["langchain"]) and fact["oss"] and fact["self_host"]


def pick(on_comet=False):
    survivors = [tool for tool in FACTS if passes(tool)]
    return "Opik" if on_comet and "Opik" in survivors else survivors[0], survivors


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    doc = parity.doc_text(PHASE, LESSON)
    langfuse, opik = (
        section(doc, "Langfuse — OSS balance"),
        section(doc, "Opik (Comet)"),
    )
    langsmith = section(doc, "LangSmith")
    day = ref.simulate_day(ref.STRATEGIES[0])["retained"]
    return {
        "pick": pick(),
        "pick_comet": pick(on_comet=True)[0],
        "doc_langsmith_selfhost": "Self-host only on Enterprise" in langsmith,
        "sweet": (
            "must self-host or stay on OSS license" in langfuse,
            "already on Comet" in opik,
        ),
        "license_lines": (
            "Core Apache / MIT" in langfuse,
            "MIT-licensed core" in doc,
            "MIT OSS with similar feature set" in doc,
        ),
        "opik_doc": "Apache 2.0, fully OSS" in opik,
        "free_minutes": FREE_EVENTS_PER_MONTH / day * 24 * 60,
        "free_share": FREE_EVENTS_PER_MONTH / (day * 30),
    }


def verify(result):
    chosen, survivors = result["pick"]
    return [
        practice.Check(
            "ANSWER: Langfuse -- unless the team already runs Comet, then Opik",
            all(
                [
                    chosen == "Langfuse",
                    survivors == ["Langfuse", "Opik"],
                    result["pick_comet"] == "Opik",
                    result["doc_langsmith_selfhost"],
                    all(result["sweet"]),
                ]
            ),
            f"survivors of LangChain + OSS + self-host: {survivors}; LangSmith self-hosts on "
            f"Enterprise only; tie-break by the lesson's sweet spots gives {chosen}, "
            f"or {result['pick_comet']} for a Comet shop",
        ),
        practice.Check(
            "FINDING: the lesson states Langfuse's license three ways",
            all(result["license_lines"])
            and result["opik_doc"]
            and len(EE) == 13
            and "dataRetention" in EE,
            "'Core Apache / MIT', 'MIT-licensed core' and 'MIT OSS' all appear; the repo is "
            f"MIT outside ee/, which holds {len(EE)} commercial features {EE}; Opik is "
            f"{FACTS['Opik']['license']} with no ee/ carve-out",
        ),
        practice.Check(
            "FINDING: the free tier cannot serve this team, which is why self-hosting is the requirement",
            round(result["free_minutes"]) == 72
            and round(1 / result["free_share"]) == 600,
            f"{FREE_EVENTS_PER_MONTH:,} events/month covers {result['free_minutes']:.0f} minutes "
            f"of the reference's 1M-trace day, 1/{1 / result['free_share']:.0f} of the month",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
