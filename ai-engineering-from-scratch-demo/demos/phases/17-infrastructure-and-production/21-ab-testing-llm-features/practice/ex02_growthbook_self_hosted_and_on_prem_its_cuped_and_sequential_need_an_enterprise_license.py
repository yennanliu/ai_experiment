"""Exercise 2 — GrowthBook, self-hosted, and on-prem its CUPED and sequential need an Enterprise license.

    Pick Statsig or GrowthBook for a healthcare-regulated on-prem customer.

Reading of the exercise: "on-prem" is a hard requirement -- the experiment
platform, its metadata store and the metric data stay on the customer's
infrastructure -- and "healthcare-regulated" means PHI, so a vendor that
touches it needs a HIPAA BAA and the deployment needs SSO and audit logs.
Each platform fact is taken from its vendor's own pages (fetched 2026-09-26)
and the pick is computed from them, next to the lesson's two criteria.

**ANSWER: GrowthBook, self-hosted.** It ships as one Docker image plus a
MongoDB and reads metrics from the customer's own warehouse, so nothing
leaves the network. Statsig is eliminated by the first requirement: even its
Warehouse Native mode runs compute in your warehouse while experiments are
"managed using the Statsig console", and its docs describe no self-hosted
control plane. No BAA is needed at all if GrowthBook is self-hosted and no
data reaches GrowthBook Inc.

**FINDING: the lesson's two criteria never ask the deciding question.** It
says to pick "based on warehouse-SQL preference and whether 'acquired by
OpenAI' matters". A team with no warehouse preference that does not mind
OpenAI ownership gets Statsig from those two -- and Statsig cannot be
installed on-prem. The deployment requirement decides before either
criterion is read.

**FINDING: "open-source (MIT) ... CUPED, SRM, Bonferroni, BH" is not one
product.** The repo is MIT except three `enterprise` directories under the
GrowthBook Enterprise License. GrowthBook's docs say "CUPED is available on
Pro and Enterprise plans" and the same of sequential testing, and Pro is
cloud-only: self-hosting is the free open-source edition (1 project) or a
custom Enterprise agreement. Audit logs, custom OIDC SSO and a BAA are
Enterprise too. The MIT build keeps SRM, multiple-testing corrections and
both engines. So all 4 of this customer's needs -- CUPED, sequential, audit
logs, SSO -- mean a self-hosted Enterprise license, or CUPED in its own SQL;
the reference code offers no help there either: it has no CUPED function.

Structure: `PLATFORMS` records the verified facts; `pick()` filters on hard
requirements, `lesson_pick()` applies only the lesson's two criteria.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "21-ab-testing-llm-features"
# vendor docs/pricing, fetched 2026-09-26; GrowthBook values are the license a
# SELF-HOSTED install needs -- Pro (CUPED, sequential) is cloud-only
PLATFORMS = {
    "Statsig": {
        "self_host": None,
        "warehouse": "native mode, hosted console",
        "openai_owned": True,
    },
    "GrowthBook": {
        "self_host": "MIT",
        "warehouse": "native",
        "openai_owned": False,
        "cuped": "Enterprise",
        "sequential": "Enterprise",
        "audit_logs": "Enterprise",
        "oidc_sso": "Enterprise",
        "baa": "Enterprise",
        "srm": "MIT",
        "corrections": "MIT",
    },
}
CUSTOMER = {"on_prem": True, "needs": ("cuped", "sequential", "audit_logs", "oidc_sso")}


def pick(customer):
    """Platforms that can run on-prem, with the license each needed feature requires."""
    out = {}
    for name, facts in PLATFORMS.items():
        if customer["on_prem"] and not facts["self_host"]:
            continue
        out[name] = {need: facts.get(need) for need in customer["needs"]}
    return out


def lesson_pick(prefers_warehouse_sql, minds_openai):
    """The lesson's rule: warehouse-SQL posture and acquisition stance, nothing else."""
    return "GrowthBook" if prefers_warehouse_sql or minds_openai else "Statsig"


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    doc = parity.doc_text(PHASE, LESSON)
    gb = PLATFORMS["GrowthBook"]
    return {
        "pick": pick(CUSTOMER),
        "lesson": lesson_pick(False, False),
        "lesson_statsig_on_prem": PLATFORMS[lesson_pick(False, False)]["self_host"],
        "doc_criteria": "warehouse-SQL preference" in doc
        and "Open-source (MIT)" in doc,
        "mit_features": sorted(
            k for k, v in gb.items() if v == "MIT" and k != "self_host"
        ),
        "ref_cuped": [name for name in dir(ref) if "cuped" in name.lower()],
    }


def verify(result):
    gb = result["pick"].get("GrowthBook", {})
    return [
        practice.Check(
            "ANSWER: GrowthBook, self-hosted",
            list(result["pick"]) == ["GrowthBook"],
            f"on-prem leaves {list(result['pick'])}; Statsig has no self-hosted control plane",
        ),
        practice.Check(
            "FINDING: the lesson's two criteria never ask the deciding question",
            result["doc_criteria"]
            and result["lesson"] == "Statsig"
            and result["lesson_statsig_on_prem"] is None,
            "no warehouse preference and indifferent to OpenAI gives "
            f"{result['lesson']}, which cannot be installed on-prem",
        ),
        practice.Check(
            "FINDING: 'open-source (MIT) ... CUPED' is not one product",
            set(gb.values()) == {"Enterprise"}
            and len(gb) == 4
            and result["mit_features"] == ["corrections", "srm"]
            and result["ref_cuped"] == [],
            f"the customer's needs require {gb}; MIT keeps {result['mit_features']}; "
            f"the reference has CUPED functions {result['ref_cuped']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
