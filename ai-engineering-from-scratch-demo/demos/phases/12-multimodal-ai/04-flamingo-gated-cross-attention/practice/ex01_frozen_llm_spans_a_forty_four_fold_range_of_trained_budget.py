"""Exercise 1 — "frozen LLM" spans a forty-four-fold range of trained budget.

    Compute Flamingo-9B's visual parameter count: 9B LLM + 1.4B gated
    cross-attention layers + 64M resampler. What fraction of total params is
    trained?

Reading of the exercise: the three numbers are taken at face value and the
fraction is computed from them, then the answer is put next to the other two
adapter designs this phase has already priced -- BLIP-2's bridge (Lesson 12.03)
and LLaVA's projector -- because "what fraction is trained" is only interesting
as a comparison, and the phrase "frozen LLM" is used for all three.

**ANSWER: 1.464B trained of 10.464B total -- 13.99%.** The frozen LLM is 86.01%
of the model.

**FINDING: "frozen LLM" covers a 44x range of trained budget.** LLaVA's 2-layer
projector on a 7B LLM is **0.32%**; BLIP-2's 188M bridge on its 8B stack is
**2.35%**; Flamingo is **13.99%**. All three are described the same way, and the
last trains 65x the parameters of the first.

**FINDING: the resampler everyone names is 0.61% of the model.** Of the 1.464B
trained, the gated cross-attention is **95.6%** and the Perceiver resampler
**4.4%**. The component that gives the architecture its shape is the small one;
the cost is in the layers bolted into the LLM.

**FINDING: one cross-attention block is larger than the whole of BLIP-2's
bridge.** The lesson says the blocks go in every 4 LLM layers, so a 32-layer LLM
takes 8 of them: **175M** each, against the 152.2M that Lesson 12.03's
arithmetic gives for the entire Q-Former. Flamingo pays BLIP-2's whole adapter
eight times over.

Structure: `fraction` is the one-line model, `DESIGNS` holds the three adapters
being compared, and `per_block` divides the cross-attention budget by the
insertion interval.
"""

from __future__ import annotations

from harness import practice

FLAMINGO = {"llm": 9.0e9, "cross_attention": 1.4e9, "resampler": 64e6}
LLM_LAYERS, INSERT_EVERY = 32, 4
QFORMER_TOTAL = 152_229_376          # Lesson 12.03's arithmetic, not the paper's 188M
DESIGNS = {
    "LLaVA 2-layer projector on 7B": (22_552_576, 7.0e9),
    "BLIP-2 bridge on its 8B stack": (188e6, 8.0e9 - 188e6),
    "Flamingo-9B": (FLAMINGO["cross_attention"] + FLAMINGO["resampler"], FLAMINGO["llm"]),
}


def fraction(trained, frozen):
    return round(trained / (trained + frozen) * 100, 2)


def per_block(budget, layers=LLM_LAYERS, every=INSERT_EVERY):
    return budget / (layers // every)


def solve():
    trained = FLAMINGO["cross_attention"] + FLAMINGO["resampler"]
    total = trained + FLAMINGO["llm"]
    shares = {name: fraction(t, f) for name, (t, f) in DESIGNS.items()}
    block = per_block(FLAMINGO["cross_attention"])
    return {
        "trained": trained, "total": total, "share": fraction(trained, FLAMINGO["llm"]),
        "frozen_share": round(FLAMINGO["llm"] / total * 100, 2),
        "shares": shares,
        "spread": round(max(shares.values()) / min(shares.values()), 1),
        "param_ratio": round(trained / DESIGNS["LLaVA 2-layer projector on 7B"][0]),
        "cross_of_trained": round(FLAMINGO["cross_attention"] / trained * 100, 1),
        "resampler_of_trained": round(FLAMINGO["resampler"] / trained * 100, 1),
        "resampler_of_total": round(FLAMINGO["resampler"] / total * 100, 2),
        "blocks": LLM_LAYERS // INSERT_EVERY, "per_block": block,
        "block_vs_qformer": round(block / QFORMER_TOTAL, 2),
    }


def verify(result):
    shares = result["shares"]
    return [
        practice.Check(
            "ANSWER: 1.464B trained of 10.464B total -- 13.99%",
            all([result["trained"] == 1.464e9, result["total"] == 10.464e9,
                 result["share"] == 13.99, result["frozen_share"] == 86.01]),
            f"{FLAMINGO['cross_attention'] / 1e9:g}B of gated cross-attention plus "
            f"{FLAMINGO['resampler'] / 1e6:g}M of resampler is "
            f"{result['trained'] / 1e9:.3f}B trained against a "
            f"{result['total'] / 1e9:.3f}B total -- {result['share']}%, leaving the frozen "
            f"LLM at {result['frozen_share']}%",
        ),
        practice.Check(
            "FINDING: 'frozen LLM' covers a 44x range of trained budget",
            all([shares["LLaVA 2-layer projector on 7B"] == 0.32,
                 shares["BLIP-2 bridge on its 8B stack"] == 2.35,
                 shares["Flamingo-9B"] == 13.99, result["spread"] == 43.7,
                 result["param_ratio"] == 65]),
            f"the same phrase describes {shares} -- a {result['spread']}x range, and "
            f"Flamingo trains {result['param_ratio']}x the parameters of LLaVA's projector. "
            "The phrase names what is held fixed, not what is being learned",
        ),
        practice.Check(
            "FINDING: the resampler everyone names is 0.61% of the model",
            all([result["cross_of_trained"] == 95.6, result["resampler_of_trained"] == 4.4,
                 result["resampler_of_total"] == 0.61]),
            f"of the {result['trained'] / 1e9:.3f}B trained, the gated cross-attention is "
            f"{result['cross_of_trained']}% and the Perceiver resampler "
            f"{result['resampler_of_trained']}% -- {result['resampler_of_total']}% of the "
            "whole model. The component that gives the architecture its shape is the small "
            "one; the cost is in the layers bolted into the LLM",
        ),
        practice.Check(
            "FINDING: one cross-attention block is larger than the whole of BLIP-2's bridge",
            all([result["blocks"] == 8, result["per_block"] == 175e6,
                 result["block_vs_qformer"] == 1.15]),
            f"at one block every {INSERT_EVERY} layers a {LLM_LAYERS}-layer LLM takes "
            f"{result['blocks']} of them, {result['per_block'] / 1e6:g}M each -- "
            f"{result['block_vs_qformer']}x the {QFORMER_TOTAL / 1e6:.1f}M that Lesson 12.03's "
            f"arithmetic gives for the entire Q-Former. Flamingo pays that adapter "
            f"{result['blocks']} times over",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
