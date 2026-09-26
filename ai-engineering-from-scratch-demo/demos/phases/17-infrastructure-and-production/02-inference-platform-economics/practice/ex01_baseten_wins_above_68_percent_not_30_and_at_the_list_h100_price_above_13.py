"""Exercise 1 — Baseten wins above 68% utilization, not 30%, and at the list H100 price above 13%.

    Run `code/main.py`. At what sustained utilization does Baseten
    (per-minute) beat Fireworks (per-token) for a 70B model on one H100?
    Derive the crossover yourself and compare to the rule of thumb.

Reading of the exercise: "sustained utilization" is the share of a day one
dedicated GPU spends saturated, as `utilization_breakeven()` defines it
(tokens/day = Baseten's 900k tok/min x 1440 min x u). Baseten bills all 1440
minutes; Fireworks bills tokens. The crossover is solved in closed form, then
checked against the reference `cost_per_day`, and re-run at Baseten's
published H100 rate.

**ANSWER: the code crosses at 67.9%, not the lesson's ~30%.** Equating
1440 x p_min with u x 1440 x tpm x p_tok gives u* = p_min / (tpm x p_tok) =
0.55 / (0.9 x 0.90) = 0.679. At 67.9% both cost $792.00/day; at the lesson's
30% Fireworks is $349.92 and Baseten $792.00, 2.26x dearer. The printed table
jumps from 50% to 75%, so it brackets the crossover without showing it. The
module's own closing line says "~60-70%", which contradicts the lesson text's
"~30%", and the shipped skill `outputs/skill-inference-platform-picker.md`
repeats the 30%.

**FINDING: at Baseten's published H100 rate the crossover is 13.4%.** The
code charges $0.55/min ($33/hr). baseten.co/pricing (fetched 2026-09-26)
lists an H100 at $0.10833/min ($6.50/hr), 5.1x less. With every other number
unchanged, u* = 0.10833 / 0.81 = 13.4%. Neither the code nor the rule of thumb
lands on 30%. For 30% to hold at the code's prices, one H100 would need
2.04M tok/min. u* scales as 1/tpm, so the 900k tok/min assumption (15k tok/s
from a 70B model) moves the answer as much as the prices do.

**FINDING: "one H100" is not in the model, and a 16-bit 70B does not fit
on one.** `Vendor` has no GPU or precision field. 70B parameters at 2 bytes
are 140 GB against an H100's 80 GB, and at FP8 they are 70 GB, leaving 10 GB
for KV cache.

**FINDING: the code's Modal beats Fireworks from 2.8% utilization.** Modal
is billed only for saturated minutes above a 60-minute floor, which is scale
to zero with perfect packing. It passes Fireworks at 32M tokens/day, 2.8% of
its own day, and it is the cheapest vendor in Scenario B ($60 against $88).
The rule of thumb names Modal and Baseten together, and the model prices
them on opposite assumptions.

Structure: `crossover()` is the closed form; `cost_at()` evaluates the
reference `cost_per_day` at a utilization; `BASETEN_H100` is the fetched rate.
"""

from __future__ import annotations

import dataclasses

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "02-inference-platform-economics"
BASETEN_H100 = 0.10833  # $/min, baseten.co/pricing, fetched 2026-09-26
H100_GB, PARAMS_B = 80, 70
DAY = 1440


def load_ref():
    return parity.load_reference(PHASE, LESSON, "main")


def vendors(ref):
    return {v.name: v for v in ref.VENDORS}


def crossover(per_minute, tokens_per_minute, per_mtok):
    """Utilization at which a 24h per-minute GPU equals per-token billing."""
    return per_minute / (tokens_per_minute / 1e6 * per_mtok)


def cost_at(ref, vendor, util, tokens_per_minute):
    return round(ref.cost_per_day(vendor, int(tokens_per_minute * DAY * util), 0), 2)


def solve():
    ref = load_ref()
    v = vendors(ref)
    fw, bt, md = v["Fireworks"], v["Baseten"], v["Modal"]
    tpm = bt.tokens_per_minute
    u_code = crossover(bt.per_minute, tpm, fw.per_mtok_output)
    listed = dataclasses.replace(bt, per_minute=BASETEN_H100)
    modal_tokens = md.min_reserved_minutes_per_day * md.per_minute / (fw.per_mtok_output / 1e6)
    scen_b = {n: ref.cost_per_day(x, 100_000_000, 500_000) for n, x in v.items()}
    return {
        "u_code": round(u_code, 4),
        "at_cross": (cost_at(ref, fw, u_code, tpm), cost_at(ref, bt, u_code, tpm)),
        "at_30": (cost_at(ref, fw, 0.3, tpm), cost_at(ref, bt, 0.3, tpm)),
        "u_list": round(crossover(BASETEN_H100, tpm, fw.per_mtok_output), 4),
        "at_list": (cost_at(ref, fw, 0.134, tpm), cost_at(ref, listed, 0.134, tpm)),
        "ratio": round(bt.per_minute / BASETEN_H100, 1),
        "tpm_for_30": round(bt.per_minute / (0.3 * fw.per_mtok_output) * 1e6),
        "fields": [f.name for f in dataclasses.fields(ref.Vendor)],
        "modal_util": round(modal_tokens / (md.tokens_per_minute * DAY), 4),
        "cheapest_b": min(scen_b, key=scen_b.get),
        "doc_30": "above ~30% sustained utilization" in parity.doc_text(PHASE, LESSON),
    }


def verify(result):
    fw30, bt30 = result["at_30"]
    return [
        practice.Check(
            "ANSWER: the code crosses at 67.9%, not the lesson's ~30%",
            all([result["u_code"] == 0.679, abs(result["at_cross"][0] - result["at_cross"][1]) < 1,
                 bt30 > 2 * fw30, result["doc_30"]]),
            f"u* = 0.55 / (0.9 x 0.90) = {result['u_code']}; both cost "
            f"{result['at_cross']} $/day there; at 30% Fireworks ${fw30} vs Baseten ${bt30}",
        ),
        practice.Check(
            "FINDING: at Baseten's published H100 rate the crossover is 13.4%",
            (result["u_list"], result["ratio"], result["tpm_for_30"]) == (0.1337, 5.1, 2_037_037),
            f"$0.55/min is {result['ratio']}x the listed ${BASETEN_H100}/min; u* = "
            f"{result['u_list']} (costs {result['at_list']} at 13.4%); 30% would need "
            f"{result['tpm_for_30']} tok/min on one GPU",
        ),
        practice.Check(
            "FINDING: 'one H100' is not in the model, and a 16-bit 70B does not fit on one",
            not any(k in f for f in result["fields"] for k in ("gpu", "precision"))
            and PARAMS_B * 2 > H100_GB >= PARAMS_B,
            f"Vendor fields {result['fields']}; 70B x 2 bytes = {PARAMS_B * 2} GB > "
            f"{H100_GB} GB, FP8 leaves {H100_GB - PARAMS_B} GB",
        ),
        practice.Check(
            "FINDING: the code's Modal beats Fireworks from 2.8% utilization",
            result["modal_util"] == 0.0278 and result["cheapest_b"] == "Modal",
            f"Modal passes Fireworks at {result['modal_util']:.1%} of its day and is the "
            f"cheapest vendor in Scenario B ({result['cheapest_b']})",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
