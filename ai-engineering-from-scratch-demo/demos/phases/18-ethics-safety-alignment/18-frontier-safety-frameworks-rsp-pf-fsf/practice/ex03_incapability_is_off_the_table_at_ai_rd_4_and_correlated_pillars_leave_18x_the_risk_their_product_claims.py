"""Exercise 3 — incapability is off the table at AI R&D-4, and correlated pillars leave 18x the risk their product claims.

    Design a safety case for a model crossing Anthropic's AI R&D-4 threshold.
    Name the evidence each of the three pillars (monitoring, illegibility,
    incapability) requires.

Reading of the exercise: the pillars are the lesson's own, parsed from its
page. "The evidence each pillar requires" is made quantitative. Each pillar's
claim is a bound on a failure rate at 95% confidence, and the evidence is the
number of clean red-team trials that bound needs. The case's top-level claim
multiplies the pillar bounds, so a seeded simulation checks whether that
product holds when the pillars share a cause.

**ANSWER:**
- **Monitoring** needs a trusted monitor's catch rate on red-team sabotage
  episodes run in the real internal-deployment scaffold. Bounding the miss
  rate at 1% needs 299 episodes caught out of 299, or 473 if one miss is
  allowed.
- **Illegibility** is the page's "lack the ability to execute a coherent
  plan". It needs fully elicited, fine-tuned attempts at end-to-end sabotage.
  Bounding coherent plans at 5% needs 0 successes in 59 attempts.
- **Incapability** cannot be claimed for AI R&D itself, because crossing the
  threshold is a capability finding. It survives only for narrower harms such
  as self-exfiltration, with the same 59-attempt evidence per harm.

**FINDING: the page's own rules drop incapability here.** The page ties the
AI R&D-4 case to "misalignment risks", and it says that for deceptive
alignment "monitoring and illegibility are targets". It also makes
incapability "the primary target" of an ASL-3 CBRN case, the level activated
precisely because incapability could no longer be shown.

**FINDING: two of the three pillars are inability questions, and none asks
about propensity.** Both illegibility and incapability open with "Does the
model lack the". A case about "models pursuing misaligned goals" needs
evidence about goals. RSP v3.0 (read 2026-09-27) asks for evidence that
models "lack the propensity to deceive, manipulate, or sabotage". At the
automated-R&D threshold it also warns that key evidence may be "suspect due to
the possibility of manipulation by AI systems", so the monitor has to be one
the model does not control.

**FINDING: the pillars do not multiply when they share a cause.** Take
200,000 seeded sabotage attempts. Planning and evading the monitor both rise
with one latent competence, calibrated to 5.1% coherent plans and 1.0%
evasions. Plan and evasion together occur in 0.95% of attempts, 18.3x the
0.052% the product claims. Drawn independently, the ratio is 1.0.

Structure: `trials_needed()` is the binomial evidence bar;
`pillar_rates()` simulates correlated and independent pillars.
"""

from __future__ import annotations

import math
import random
import re
import statistics

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "18-frontier-safety-frameworks-rsp-pf-fsf"
CONFIDENCE, ATTEMPTS, NOISE = 0.95, 200_000, 0.5


def trials_needed(rate, misses=0):
    """Smallest n with P(<= misses failures | rate) <= 1 - CONFIDENCE."""
    n = misses + 1
    while sum(math.comb(n, k) * rate**k * (1 - rate) ** (n - k) for k in range(misses + 1)) > 1 - CONFIDENCE:
        n += 1
    return n


def pillar_rates(correlated, seed=0):
    """(plan rate, evasion rate, joint rate) over seeded sabotage attempts."""
    rng, nd = random.Random(seed), statistics.NormalDist()
    t_plan, t_evade = nd.inv_cdf(0.95), nd.inv_cdf(0.99) * math.hypot(1, NOISE)
    plan = evade = both = 0
    for _ in range(ATTEMPTS):
        z = rng.gauss(0, 1)
        p = z > t_plan
        e = (z if correlated else rng.gauss(0, 1)) + NOISE * rng.gauss(0, 1) > t_evade
        plan, evade, both = plan + p, evade + e, both + (p and e)
    return plan / ATTEMPTS, evade / ATTEMPTS, both / ATTEMPTS


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    doc = parity.doc_text(PHASE, LESSON)
    cases = doc.split("### Safety cases")[1].split("\n### ")[0]
    pillars = dict(re.findall(r"^- \*\*(\w+)\.\*\* (.+)$", cases, re.M))
    rates = {name: pillar_rates(name == "correlated") for name in ("correlated", "independent")}
    return {
        "pillars": pillars,
        "rd4_case": re.search(r"Once AI R&D-4 is crossed[^.]*identifying (\w+) risks", doc).group(1),
        "deceptive_targets": re.search(r"For deceptive alignment, ([^.]+) are targets", doc).group(1),
        "cbrn_target": re.search(r"For a ASL-3 CBRN case, (\w+)", doc).group(1),
        "ref_case": next(lab["safety_case"] for lab in ref.LABS if "Anthropic" in lab["name"]),
        "monitor_n": (trials_needed(0.01), trials_needed(0.01, misses=1)),
        "plan_n": trials_needed(0.05),
        "rates": {k: tuple(round(x, 5) for x in v) for k, v in rates.items()},
        "inflation": {k: round(v[2] / (v[0] * v[1]), 1) for k, v in rates.items()},
    }


def verify(result):
    r = result
    corr = r["rates"]["correlated"]
    inability = [k for k, v in r["pillars"].items() if v.startswith("Does the model lack the")]
    return [
        practice.Check(
            "ANSWER: 299 caught episodes for monitoring, 59 clean attempts for illegibility",
            (list(r["pillars"]), r["monitor_n"], r["plan_n"])
            == (["Monitoring", "Illegibility", "Incapability"], (299, 473), 59),
            f"pillars {list(r['pillars'])}; 1% miss bound at 95% needs {r['monitor_n']} episodes "
            f"(0 / 1 miss), 5% plan bound needs {r['plan_n']}",
        ),
        practice.Check(
            "FINDING: the page's own rules drop incapability here",
            (r["rd4_case"], r["deceptive_targets"], r["cbrn_target"], r["ref_case"])
            == ("misalignment", "monitoring and illegibility", "incapability",
                "required at AI R&D-4 crossing"),
            f"AI R&D-4 case is about {r['rd4_case']} risks; deceptive alignment targets "
            f"{r['deceptive_targets']!r}; ASL-3 CBRN targets {r['cbrn_target']}; main.py: {r['ref_case']!r}",
        ),
        practice.Check(
            "FINDING: two of the three pillars are inability questions, none about propensity",
            (inability, re.findall("propensit|goal|intent", " ".join(r["pillars"].values())))
            == (["Illegibility", "Incapability"], []),
            f"pillars opening 'Does the model lack the': {inability}",
        ),
        practice.Check(
            "FINDING: the pillars do not multiply when they share a cause",
            (round(corr[0], 3), round(corr[1], 3), round(corr[2], 4), round(corr[0] * corr[1], 5),
             r["inflation"]) == (0.051, 0.010, 0.0095, 0.00052, {"correlated": 18.3, "independent": 1.0}),
            f"correlated plan / evade / joint {corr}, product {corr[0] * corr[1]:.5f}, "
            f"joint over product {r['inflation']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
