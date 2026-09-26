"""Exercise 3 — the raise and the batch discount bill different products, so batch can raise the bill.

    Fireworks raises prices by $1/hr on your primary model. Model the blended
    cost impact if 40% of your traffic moves to batch tier (50% off).

Reading of the exercise: the "$1/hr" is an hourly GPU price, so the primary
model is taken to run on a Fireworks on-demand H100: $7 -> $8/hr, the
announced change. Batch is "50% off" serverless per-token prices. The day is
the lesson's Scenario B, 100M output tokens. Serverless is priced at the
lesson's $0.90/M and the GPU at the code's 900k tok/min, costed with the
reference `cost_per_day` (a dedicated deployment is a per-minute `Vendor`
with a 1440-minute floor). The one-price reading, where the raise and the
discount hit the same rate, is computed too.

**ANSWER: on one dedicated H100 the raise is +14.3%, and moving 40% to batch
makes it +25%.** The day goes $168 -> $192. The 40% shift adds $18 of batch
tokens (40M x $0.45/M) while the GPU stays reserved all day: $210, against
$168 before the raise. Batch only pays when it lets you release whole GPUs.
At saturation a dedicated H100 costs $8 / 54M tokens = $0.148/M, and batch at
$0.45/M is 3.0x dearer, so batch beats a dedicated GPU only below 32.9%
utilization. On serverless the $1/hr raise does not apply: 40% batch takes
the day from $90 to $72, -20%. The one-price reading gives
0.8 x ($0.90 + $0.0185) / $0.90 = -18.4%, because $1/hr spread over a
saturated GPU's 54M tokens is only $0.0185/M.

**FINDING: the raise is on GPU rental, dated September 1, 2026, not May 1.**
Fireworks' pricing page (fetched 2026-09-26) lists an H100 at $8.00/hr.
usagepricing.com records an announcement dated 2026-08-12 that took
H100/H200 from $7.00 to $8.00 and B200 from $10.00 to $13.00, effective
September 1, 2026, on on-demand GPUs only. The lesson dates it May 1, 2026,
and no source found in this pass supports that date. Fireworks' batch guide
says batch is "50% off" *serverless per-token prices*, the product the raise
does not touch.

**FINDING: the module cannot express either change.** The only mention of
batch is the Fireworks note string "batch tier 50% off", and Fireworks has
no per-minute price. So the solution has to build its own dedicated
Fireworks `Vendor`.

Structure: `day()` prices a mix of dedicated minutes, serverless tokens and
batch tokens through the reference `cost_per_day`.
"""

from __future__ import annotations

import dataclasses

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "02-inference-platform-economics"
H100_OLD, H100_NEW = 7.00, 8.00  # $/hr, usagepricing.com + fireworks.ai/pricing, 2026-09-26
TOKENS, SHIFT, BATCH_OFF = 100_000_000, 0.40, 0.50


def dedicated(fw, hourly):
    return dataclasses.replace(fw, per_mtok_output=None, per_minute=hourly / 60,
                               min_reserved_minutes_per_day=1440)


def day(ref, vendor, shift):
    """$/day with `shift` of TOKENS on batch, the rest on `vendor`."""
    fw = ref.VENDORS[0]
    batch = dataclasses.replace(fw, per_mtok_output=fw.per_mtok_output * BATCH_OFF)
    kept = ref.cost_per_day(vendor, int(TOKENS * (1 - shift)), 0)
    return round(kept + ref.cost_per_day(batch, int(TOKENS * shift), 0), 2)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    fw = ref.VENDORS[0]
    old, new = dedicated(fw, H100_OLD), dedicated(fw, H100_NEW)
    per_m_saturated = H100_NEW / (fw.tokens_per_minute * 60 / 1e6)
    delta_per_m = 1.0 / (fw.tokens_per_minute * 60 / 1e6)
    one_price = BATCH_OFF * SHIFT + (1 - SHIFT)
    return {
        "dedicated": (day(ref, old, 0), day(ref, new, 0), day(ref, new, SHIFT)),
        "serverless": (day(ref, fw, 0), day(ref, fw, SHIFT)),
        "per_m_saturated": round(per_m_saturated, 3),
        "batch_rate": fw.per_mtok_output * BATCH_OFF,
        "util_break": round(per_m_saturated / (fw.per_mtok_output * BATCH_OFF), 3),
        "one_price": round(one_price * (fw.per_mtok_output + delta_per_m) / fw.per_mtok_output - 1, 3),
        "delta_per_m": round(delta_per_m, 4),
        "has_batch": [v.name for v in ref.VENDORS if "batch" in v.notes],
        "fw_per_minute": fw.per_minute,
        "doc_may": "May 1, 2026" in parity.doc_text(PHASE, LESSON),
    }


def verify(result):
    base, raised, shifted = result["dedicated"]
    s0, s1 = result["serverless"]
    return [
        practice.Check(
            "ANSWER: on one dedicated H100 the raise is +14.3%, and moving 40% to batch makes it +25%",
            all([(base, raised, shifted) == (168.0, 192.0, 210.0), (s0, s1) == (90.0, 72.0),
                 result["per_m_saturated"] == 0.148, result["util_break"] == 0.329,
                 result["one_price"] == -0.184]),
            f"dedicated ${base} -> ${raised} -> ${shifted} with 40% batch "
            f"({shifted / base - 1:+.1%}); saturated H100 ${result['per_m_saturated']}/M vs "
            f"batch ${result['batch_rate']}/M, batch wins only below {result['util_break']:.1%}; "
            f"serverless ${s0} -> ${s1}; one-price reading {result['one_price']:+.1%} "
            f"(+${result['delta_per_m']}/M)",
        ),
        practice.Check(
            "FINDING: the raise is on GPU rental, dated September 1, 2026, not May 1",
            result["doc_may"] and H100_NEW - H100_OLD == 1.0,
            "the lesson dates the raise May 1, 2026; the published change is H100 $7 -> $8/hr "
            "on on-demand GPUs, announced 2026-08-12, effective 2026-09-01; batch is 50% off "
            "serverless per-token",
        ),
        practice.Check(
            "FINDING: the module cannot express either change",
            result["has_batch"] == ["Fireworks"] and result["fw_per_minute"] is None,
            f"'batch' appears only in the notes of {result['has_batch']}; Fireworks "
            f"per_minute is {result['fw_per_minute']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
