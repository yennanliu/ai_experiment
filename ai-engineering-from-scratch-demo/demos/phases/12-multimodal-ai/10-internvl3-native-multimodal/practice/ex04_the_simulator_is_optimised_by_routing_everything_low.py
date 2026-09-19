"""Exercise 4 — the simulator is optimised by routing everything low.

    ViR routes 60% of traffic to low-resolution encoding. What kinds of queries
    does it misroute (sends to low-res when high-res was needed)? Propose three
    router-failure modes.

Reading of the exercise: the failure modes are named after measuring where the
router's value is concentrated, so the three are ordered by cost rather than by
plausibility -- and the lesson's own `vir_sim` is checked for whether it can
represent a misroute at all, which it cannot, because it has no accuracy term.

**ANSWER: three modes, and the measurement that says which one matters.** The
high-res tier is **20%** of requests and **57.7%** of the tokens, so the saving
and the damage live in the same fifth of the traffic. In order of cost:

1. **A document photographed as a photo.** A receipt held up to a phone camera
   reads as photo QA; the router saves 1,792 tokens and the answer is wrong.
2. **Small text inside a large scene.** A street sign, a price tag, a serial
   number -- the image statistics say landscape and the question is OCR.
3. **A follow-up on an already-routed image.** "What does the label say?"
   arrives after the image was encoded at low resolution. This is the structural
   one: `vir_sim` prices a per-request decision, but an image is encoded once per
   conversation and the decision is not revisable without re-encoding.

**FINDING: the exercise's 60% is not the lesson's 50%.** The shipped tiers are
50 / 30 / 20 and give an average of **710.4** tokens, a **2.88x** saving. At
60 / 30 / 10 the average is **531.2** and the saving **3.86x** -- so the
premise makes the router look **33.8%** better than the lesson's own numbers do.

**FINDING: `vir_sim` cannot represent a misroute.** It takes (tokens, fraction)
pairs and returns a weighted mean; there is no accuracy term anywhere, so every
policy that lowers the average scores better. Its optimum is to send **100%** to
low-res: average 256 tokens, an **8.0x** saving, and a router that has stopped
routing.

**FINDING: and the 2.88x is a choice of baseline.** It is measured against every
request at 2,048, which is not a policy anyone would run. Against a *flat* 576 --
the middle tier for everything -- the routed mix costs **1.23x more**, because
the 20% sent to high-res outweighs the 50% sent to low.

Structure: `tiers` builds the lesson's own `RouterTier` list at a given split,
`share_of_tokens` locates where the bill is, and `DEGENERATE` is the split that
maximises the simulator's own objective.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "10-internvl3-native-multimodal"
TOKENS = (256, 576, 2048)
NAMES = ("low-res photo QA", "medium product shot", "high-res doc + OCR")
SHIPPED = (0.50, 0.30, 0.20)
ASKED = (0.60, 0.30, 0.10)
DEGENERATE = (1.0, 0.0, 0.0)


def tiers(ref, split):
    return [ref.RouterTier(name, tokens, fraction)
            for name, tokens, fraction in zip(NAMES, TOKENS, split)]


def share_of_tokens(split, index=-1):
    total = sum(tokens * fraction for tokens, fraction in zip(TOKENS, split))
    return round(TOKENS[index] * split[index] / total * 100, 1)


ACCURACY_WORDS = ("accuracy", "correct", "quality", "wrong", "misroute", "error")


def accuracy_terms(ref):
    """Any accuracy-like name in vir_sim's source -- the term a misroute would need."""
    source = inspect.getsource(ref.vir_sim).lower()
    return [word for word in ACCURACY_WORDS if word in source]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    shipped = ref.vir_sim(tiers(ref, SHIPPED))
    asked = ref.vir_sim(tiers(ref, ASKED))
    degenerate = ref.vir_sim(tiers(ref, DEGENERATE))
    return {
        "shipped": {"avg": round(shipped["avg_tokens"], 1),
                    "ratio": round(shipped["ratio"], 2)},
        "asked": {"avg": round(asked["avg_tokens"], 1), "ratio": round(asked["ratio"], 2)},
        "degenerate": {"avg": round(degenerate["avg_tokens"], 1),
                       "ratio": round(degenerate["ratio"], 2)},
        "high_request_share": round(SHIPPED[-1] * 100, 1),
        "high_token_share": share_of_tokens(SHIPPED),
        "misroute_cost": TOKENS[-1] - TOKENS[0],
        "premise_gain_pct": round((asked["ratio"] / shipped["ratio"] - 1) * 100, 1),
        "baseline": shipped["baseline"],
        "sim_keys": sorted(shipped),
        "accuracy_terms": accuracy_terms(ref),
        "has_accuracy_term": bool(accuracy_terms(ref)),
        "modes": 3,
    }


def verify(result):
    shipped, asked, degenerate = result["shipped"], result["asked"], result["degenerate"]
    return [
        practice.Check(
            "ANSWER: the saving and the damage live in the same fifth of the traffic",
            all([result["high_request_share"] == 20.0, result["high_token_share"] == 57.7,
                 result["misroute_cost"] == 1792, result["modes"] == 3]),
            f"the high-res tier is {result['high_request_share']}% of requests and "
            f"{result['high_token_share']}% of the tokens, and one misroute saves "
            f"{result['misroute_cost']:,} tokens while returning a wrong answer. The three "
            "modes -- a document photographed as a photo, small text in a large scene, and a "
            "follow-up on an already-routed image -- all land in that fifth",
        ),
        practice.Check(
            "FINDING: the exercise's 60% is not the lesson's 50%",
            all([shipped == {"avg": 710.4, "ratio": 2.88},
                 asked == {"avg": 531.2, "ratio": 3.86},
                 result["premise_gain_pct"] == 33.7]),
            f"the shipped tiers {SHIPPED} give {shipped['avg']} tokens and a "
            f"{shipped['ratio']}x saving; the {ASKED} split the exercise names gives "
            f"{asked['avg']} and {asked['ratio']}x -- {result['premise_gain_pct']}% better "
            "than the lesson's own numbers, before any routing decision is examined",
        ),
        practice.Check(
            "FINDING: vir_sim cannot represent a misroute",
            all([not result["has_accuracy_term"], result["accuracy_terms"] == [],
                 result["sim_keys"] == ["avg_tokens", "baseline", "ratio"],
                 degenerate == {"avg": 256.0, "ratio": 8.0},
                 degenerate["ratio"] > asked["ratio"] > shipped["ratio"]]),
            f"the simulator takes (tokens, fraction) pairs and returns "
            f"{result['sim_keys']} -- a weighted mean, its own maximum and their quotient. "
            f"Searching its source for {list(ACCURACY_WORDS)} finds "
            f"{result['accuracy_terms'] or 'nothing'}, so no policy can be penalised for "
            f"being wrong and every policy that lowers the average scores better. Its "
            f"optimum is {DEGENERATE} -- {degenerate['avg']} tokens, "
            f"{degenerate['ratio']}x -- a router that has stopped routing",
        ),
        practice.Check(
            "FINDING: against a flat medium tier the router is a 1.23x regression",
            all([result["baseline"] == 2048, result["baseline"] == max(TOKENS),
                 round(shipped["avg"] / TOKENS[1], 2) == 1.23]),
            f"the speed-up is computed against {result['baseline']:,} tokens -- every request "
            f"at the highest tier -- which is not a policy anyone would run. Against a flat "
            f"{TOKENS[1]} the same routed mix costs {shipped['avg']} tokens, "
            f"{shipped['avg'] / TOKENS[1]:.2f}x MORE, because the 20% sent to "
            f"{TOKENS[2]:,} outweighs the 50% sent to {TOKENS[0]}. The saving is entirely a "
            "choice of baseline",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
