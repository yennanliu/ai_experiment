"""Exercise 4 — synthetic documents split faking from prompt sensitivity, 8.70 points vs 0, where the system prompt gives both 12 vs 3.

    Design a synthetic-training setup (Greenblatt et al. 2024 Appendix C) that is more realistic than the system-prompt version but does not require retraining the full model. Why is this operationalization important for generalizing the result?

Reading of the exercise: the design is the lesson's "synthetic fine-tuning
that embeds the retraining setup in training data instead of system prompt".
A LoRA adapter, with the base weights frozen, is fine-tuned on n
pretraining-style documents that state "free-tier conversations are used for
retraining; paid-tier are not". At test time the context holds only the tier
tag and the harmful request, with no explanation in it. Adapters are why
the full model need not be retrained, and the toy cannot measure that part.
What the toy can measure is the "why": which hypothesis each setup is able
to tell apart. Two agents are built on the reference's
`compliance_probability`, both calibrated (as in exercise 2) to the lesson's
12% vs 3%:

* *strategic*: refusing costs preservation_weight x b, where b is the
  agent's belief that this tier is trained. After n agreeing documents b is
  the Laplace estimate (n + 1)/(n + 2) for free-tier and 1/(n + 2) for
  paid-tier.
* *prompt-sensitive*: the retraining explanation shifts the logit by the same
  amount whenever it is in the context, and nothing is inferred.

**ANSWER (design): an adapter trained on synthetic documents, a tier tag in
context, and two controls: zero documents, and documents that reverse the
tiers.** With the explanation in the system prompt, both agents produce
exactly 12.0% vs 3.0%, so that version cannot tell them apart. After n = 0,
1, 4, 16 and 64 documents the strategic agent's gap is 0, 2.85, 5.81, 7.91 and
8.70 points. The prompt-sensitive agent's gap is 0 at every n, because the
explanation is never in the context. Reversing the documents flips the
strategic gap to -8.70 at n = 64. A pure wording effect cannot flip sign on
a fact the context never states.

**FINDING: why this matters for generalizing.** In deployment, no model is
handed its training setup in a system prompt, but it may have read about it
in pretraining. Only the synthetic setup tests the route that exists in the
wild: a belief stored in the weights, which the context then activates. It is
also the only one of the two setups whose outcome can separate strategic
reasoning from sensitivity to wording.

**FINDING: an uninformed agent complies more in both tiers.** At n = 0 the
belief is 0.5 for both tiers, and the strategic agent complies 6.10% of the
time in each, twice the 3.0% it shows when told a tier is unmonitored. The
zero-document control shows what uncertainty about monitoring does on its
own.

Structure: `fit()` inverts the reference sigmoid as exercise 2 does;
`strategic()` feeds belief x weight to `compliance_probability`;
`prompt_sensitive()` applies the same shift only when the explanation is in
context.
"""

from __future__ import annotations

import math
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "09-alignment-faking"
DOCS = (0, 1, 4, 16, 64)


def fit(doc):
    free, paid = (int(re.search(rf"{t}-tier compliance: ~?(\d+)%", doc).group(1)) / 100
                  for t in ("Free", "Paid"))
    harm = math.log((1 - paid) / paid) / 3
    return harm, harm + math.log(free / (1 - free)) / 3


def strategic(ref, harm, weight, belief):
    return ref.compliance_probability(True, harm, weight * belief)


def prompt_sensitive(ref, harm, weight, explained, free_tier):
    """Same logit shift as the strategic agent, applied only when the explanation is in context."""
    return ref.compliance_probability(explained and free_tier, harm, weight)


def pts(a, b):
    return round(100 * (a - b), 2)


def documents(ref, harm, weight):
    """Strategic and prompt-sensitive gaps after n agreeing synthetic documents."""
    beliefs = {n: ((n + 1) / (n + 2), 1 / (n + 2)) for n in DOCS}
    strat = {n: pts(strategic(ref, harm, weight, f), strategic(ref, harm, weight, p))
             for n, (f, p) in beliefs.items()}
    prompt = {n: pts(*(prompt_sensitive(ref, harm, weight, False, f) for f in (True, False)))
              for n in DOCS}
    return strat, prompt, pts(*(strategic(ref, harm, weight, b) for b in beliefs[64][::-1]))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    doc = parity.doc_text(PHASE, LESSON)
    harm, weight = fit(doc)
    told = {"strategic": [strategic(ref, harm, weight, b) for b in (1.0, 0.0)],
            "prompt": [prompt_sensitive(ref, harm, weight, True, f) for f in (True, False)]}
    strat, prompt, reversed_ = documents(ref, harm, weight)
    return {
        "told": {k: [round(100 * p, 2) for p in v] for k, v in told.items()},
        "strategic": strat, "prompt": prompt, "reversed": reversed_,
        "uninformed": round(100 * strategic(ref, harm, weight, 0.5), 2),
        "lesson_names_it": "synthetic fine-tuning" in doc,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: in the system-prompt version both agents give 12.0% vs 3.0%",
            result["told"] == {"strategic": [12.0, 3.0], "prompt": [12.0, 3.0]}
            and result["lesson_names_it"],
            f"(free, paid) compliance when told: {result['told']}",
        ),
        practice.Check(
            "ANSWER: synthetic documents give 0 / 2.85 / 5.81 / 7.91 / 8.70 points vs 0, -8.70 reversed",
            result["strategic"] == {0: 0.0, 1: 2.85, 4: 5.81, 16: 7.91, 64: 8.7}
            and set(result["prompt"].values()) == {0.0} and result["reversed"] == -8.7,
            f"strategic gap by documents {result['strategic']}; prompt-sensitive "
            f"{result['prompt']}; reversed documents at n = 64 {result['reversed']}",
        ),
        practice.Check(
            "FINDING: an uninformed agent complies 6.10% in both tiers, twice the told-unmonitored 3.0%",
            result["uninformed"] == 6.1 and result["strategic"][0] == 0.0,
            f"compliance at belief 0.5: {result['uninformed']}%",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
