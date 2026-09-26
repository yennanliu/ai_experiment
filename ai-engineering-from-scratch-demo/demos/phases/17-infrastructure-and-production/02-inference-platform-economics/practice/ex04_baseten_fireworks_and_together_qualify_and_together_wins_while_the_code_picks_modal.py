"""Exercise 4 — Baseten, Fireworks and Together qualify and Together wins, while the code picks Modal.

    A regulated customer requires SOC 2 Type II + HIPAA + dedicated GPUs.
    Which three platforms are viable and which one wins on FinOps?

Reading of the exercise: "viable" means the vendor publicly claims both SOC 2
Type II and HIPAA and sells a reserved, dedicated GPU. "Wins on FinOps" means
the lowest cost of one H100 reserved for 24 hours. Two versions are compared:
the answer the lesson's own data gives, and the answer from vendor pages
fetched 2026-09-26.

**ANSWER: Baseten, Fireworks and Together are viable, and Together wins at
$95.76/day.**

- **Baseten** says "SOC 2 Type II" and "HIPAA compliant" on its pricing page.
- **Fireworks**' 2023 post says its platform "is both SOC 2 Type II and HIPAA
  compliant".
- **Together**'s July 2025 post reports a completed SOC 2 Type 2 examination
  and says it "adheres to the stringent requirements of HIPAA, including ...
  business associate agreements".

At list H100 prices one reserved day costs **$95.76 on Together** ($3.99/hr),
$156.00 on Baseten ($6.50) and $192.00 on Fireworks ($8.00).

Two vendors do not qualify:

- **Anyscale's** certifications page lists SOC 2 Type 2 and does not mention
  HIPAA.
- **Modal** offers HIPAA on its Enterprise plan only, and it bills serverless
  per-second containers rather than reserved GPUs.

**FINDING: the lesson's own data admits one vendor and picks Modal.** In
`VENDORS`, the dedicated (per-minute) vendors are Baseten, Modal and Anyscale,
and only Baseten's notes mention SOC 2 or HIPAA. A reserved day there costs
$691.20 on Modal, $792.00 on Baseten and $864.00 on Anyscale, so the code's
FinOps winner is Modal. The shipped skill
`outputs/skill-inference-platform-picker.md` refuses Modal for regulated work
and says "Suggest Baseten". So the code and the skill contradict each other.
The lesson never names three viable platforms.

**FINDING: the code's dedicated prices are 5-7x the list prices.** $0.55/min
is $33.00/hr for Baseten, 5.1x its listed $6.50. Modal's $0.48/min is
$28.80/hr, 7.3x its listed $0.001097/s ($3.95/hr). And
`Vendor` has no GPU type, so the code cannot say which GPU the minute buys.

Structure: `VIABLE` and `LISTED_H100` hold what the fetched pages say;
`reserved_day()` prices 1440 minutes through the reference `cost_per_day`.
"""

from __future__ import annotations

import dataclasses

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "02-inference-platform-economics"
# vendor pages fetched 2026-09-26: (SOC 2 Type II, HIPAA, reserved dedicated GPU)
CLAIMS = {
    "Baseten": (True, True, True),
    "Fireworks": (True, True, True),
    "Together": (True, True, True),
    "Modal": (True, "Enterprise plan only", False),
    "Anyscale": (True, False, True),
}
LISTED_H100 = {"Together": 3.99, "Baseten": 6.50, "Fireworks": 8.00, "Modal": 0.001097 * 3600}


def viable():
    return sorted(n for n, claims in CLAIMS.items() if all(c is True for c in claims))


def reserved_day(ref, vendor):
    """One GPU reserved for 24h: the floor, with no tokens on top."""
    return round(ref.cost_per_day(dataclasses.replace(vendor, min_reserved_minutes_per_day=1440), 0, 0), 2)


def code_view(ref):
    """What the lesson's own VENDORS say: dedicated vendors, compliant ones, reserved days."""
    dedicated = [v for v in ref.VENDORS if v.per_minute is not None]
    compliant = [v.name for v in dedicated if "HIPAA" in v.notes or "SOC2" in v.notes]
    return [v.name for v in dedicated], compliant, {v.name: reserved_day(ref, v) for v in dedicated}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    names, compliant, code_days = code_view(ref)
    list_days = {n: round(LISTED_H100[n] * 24, 2) for n in viable()}
    by_name = {v.name: v for v in ref.VENDORS}
    skill = (parity.lesson_dir(PHASE, LESSON) / "outputs" / "skill-inference-platform-picker.md").read_text()
    return {
        "viable": viable(), "list_days": list_days,
        "list_winner": min(list_days, key=list_days.get),
        "code_dedicated": names, "code_compliant": compliant,
        "code_days": code_days, "code_winner": min(code_days, key=code_days.get),
        "skill_refuses_modal": "picked Modal or Replicate, refuse" in skill and "Suggest Baseten" in skill,
        "markup": {n: round(by_name[n].per_minute * 60 / LISTED_H100[n], 1) for n in ("Baseten", "Modal")},
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: Baseten, Fireworks and Together are viable, and Together wins at $95.76/day",
            result["viable"] == ["Baseten", "Fireworks", "Together"]
            and result["list_winner"] == "Together"
            and result["list_days"] == {"Baseten": 156.0, "Fireworks": 192.0, "Together": 95.76},
            f"viable {result['viable']}; one reserved H100-day at list {result['list_days']}; "
            "Anyscale claims no HIPAA, Modal only on Enterprise and without reserved GPUs",
        ),
        practice.Check(
            "FINDING: the lesson's own data admits one vendor and picks Modal",
            all([result["code_dedicated"] == ["Baseten", "Modal", "Anyscale"],
                 result["code_compliant"] == ["Baseten"], result["code_winner"] == "Modal",
                 result["skill_refuses_modal"]]),
            f"dedicated in VENDORS {result['code_dedicated']}, compliant by notes "
            f"{result['code_compliant']}; reserved day {result['code_days']} -> "
            f"{result['code_winner']}, which the shipped skill refuses for regulated work",
        ),
        practice.Check(
            "FINDING: the code's dedicated prices are 5-7x the list prices",
            result["markup"] == {"Baseten": 5.1, "Modal": 7.3},
            f"code $/hr over listed H100 $/hr: {result['markup']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
