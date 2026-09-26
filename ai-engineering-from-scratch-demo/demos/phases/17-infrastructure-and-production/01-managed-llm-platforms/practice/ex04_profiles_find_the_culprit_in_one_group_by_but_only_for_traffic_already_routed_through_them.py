"""Exercise 4 — profiles find the culprit in one group-by, but only for traffic already routed through them.

    You discover your Bedrock bill is up 4x this month with no traffic change.
    Without Application Inference Profiles, how would you find the culprit? With
    profiles, how long does it take?

Reading of the exercise: "no traffic change" means the invocation count is
flat, so the 4x has to be tokens per request. A synthetic month is built --
four features on one Claude model, two of them behind one shared IAM role --
priced at the reference's Bedrock row ($3/$15 per M), and in it one feature
starts stuffing the whole ticket thread into its prompt; its new prompt size
is solved for so the bill is exactly 4x. Each attribution view is then asked
the same question -- which features could explain the rise? -- and "how
long" is measured as what each view has to read.

**ANSWER: without profiles, go model -> role -> request body; with
profiles, one group-by.** The bill by model shows invocations flat at
85,000/day and input tokens up, which rules traffic out and every feature
in: all four are on the same model. Model invocation logging -- if it was
on before the spike -- grouped by caller role narrows it to the shared app
role, still 2 candidates; separating those needs the prompts themselves,
fingerprinted across 2,550,000 logged requests. With profiles tagged per
feature, one cost group-by over 120 daily rows names `support/summarize`,
whose prompt went from 3,000 to 89,250 tokens.

**FINDING: profiles cannot answer for this month if they are created after
the spike.** They attribute only calls that name the profile's ARN. Created
on day 20 to investigate, they cover 11 of 30 days -- 36.7% of a month
whose daily spend is flat; the other 63.3% stays in the unattributed model line. If invocation
logging was off too, the requests are gone and the model view -- 4
candidates -- is all there is.

**FINDING: the reference has no attribution to run.** Its Bedrock row's
"attribution" is the string "A (Application Inference Profiles)"; nothing
in the module takes a caller, a tag or a feature. And the lesson's
"CloudWatch breaks out cost per profile" is not how AWS describes it:
Bedrock's inference-profile page (read 2026-09-26) sends usage metrics to
CloudWatch and costs to cost allocation tags in AWS Billing.

Structure: `month()` builds per-feature daily totals; `candidates()` asks a
grouping which of its groups grew and returns the features behind them.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "01-managed-llm-platforms"
DAYS, SPIKE, CULPRIT = 30, 4.0, "support/summarize"
# feature: (IAM role, requests/day, input tokens/request, output tokens/request)
FEATURES = {
    "search/rerank": ("role-search", 20_000, 2_000, 200),
    "support/chat": ("role-app", 10_000, 1_500, 400),
    CULPRIT: ("role-app", 5_000, 3_000, 150),
    "ops/tagging": ("role-ops", 50_000, 500, 20),
}


def daily_cost(p, requests, tok_in, tok_out):
    return requests * (tok_in * p.per_mtok_input + tok_out * p.per_mtok_output) / 1e6


def spiked_prompt(p):
    """Input tokens/request for the culprit that makes the whole bill SPIKE x."""
    base = sum(daily_cost(p, *f[1:]) for f in FEATURES.values())
    _, requests, tok_in, _ = FEATURES[CULPRIT]
    return tok_in + round((SPIKE - 1) * base * 1e6 / (requests * p.per_mtok_input))


def month(p, spiked):
    rows = {}
    for name, (role, requests, tok_in, tok_out) in FEATURES.items():
        if spiked and name == CULPRIT:
            tok_in = spiked_prompt(p)
        rows[name] = {
            "model": "claude",
            "role": role,
            "requests": requests,
            "cost": daily_cost(p, requests, tok_in, tok_out),
            "tok_in": tok_in,
        }
    return rows


def candidates(before, after, key):
    """Features behind every group (model, role or feature) whose cost grew."""

    def label(name, row):
        return name if key == "feature" else row[key]

    def group(rows):
        out = {}
        for name, row in rows.items():
            out[label(name, row)] = out.get(label(name, row), 0) + row["cost"]
        return out

    grown = {g for g, cost in group(after).items() if cost > 1.5 * group(before)[g]}
    return sorted(n for n, r in after.items() if label(n, r) in grown)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    bedrock = ref.PLATFORMS[0]
    before, after = month(bedrock, False), month(bedrock, True)
    total = [sum(r["cost"] for r in m.values()) for m in (before, after)]
    created = 20  # profiles created on this day of the month, after the spike
    return {
        "ratio": total[1] / total[0],
        "requests": [sum(r["requests"] for r in m.values()) for m in (before, after)],
        "views": {
            k: candidates(before, after, k) for k in ("model", "role", "feature")
        },
        "prompt": (before[CULPRIT]["tok_in"], after[CULPRIT]["tok_in"]),
        "rows": (
            DAYS * sum(r["requests"] for r in after.values()),
            DAYS * len(FEATURES),
        ),
        "covered": (DAYS - created + 1) / DAYS,
        "attribution": bedrock.attribution,
        "claim": "CloudWatch breaks out cost per profile"
        in parity.doc_text(PHASE, LESSON),
    }


def verify(result):
    v, rows = result["views"], result["rows"]
    return [
        practice.Check(
            "ANSWER: without profiles, go model -> role -> request body; with profiles, one group-by",
            round(result["ratio"], 9) == SPIKE
            and result["requests"][0] == result["requests"][1]
            and len(v["model"]) == 4
            and len(v["role"]) == 2
            and v["feature"] == [CULPRIT],
            f"bill x{result['ratio']:.1f} at {result['requests'][1]:,} requests/day both months; "
            f"candidates by model {len(v['model'])}, by role {v['role']}, by profile "
            f"{v['feature']}; prompt {result['prompt'][0]:,} -> {result['prompt'][1]:,} tokens; "
            f"{rows[0]:,} log rows against {rows[1]} profile rows",
        ),
        practice.Check(
            "FINDING: profiles cannot answer for this month if they are created after the spike",
            round(result["covered"], 3) == 0.367,
            f"profiles made on day 20 cover {result['covered']:.1%} of the month; the rest "
            "stays in the model line, and without invocation logs that view has 4 candidates",
        ),
        practice.Check(
            "FINDING: the reference has no attribution to run",
            isinstance(result["attribution"], str) and result["claim"],
            f"Bedrock's attribution is the string {result['attribution']!r}; AWS routes "
            "profile costs through cost allocation tags, not CloudWatch",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
