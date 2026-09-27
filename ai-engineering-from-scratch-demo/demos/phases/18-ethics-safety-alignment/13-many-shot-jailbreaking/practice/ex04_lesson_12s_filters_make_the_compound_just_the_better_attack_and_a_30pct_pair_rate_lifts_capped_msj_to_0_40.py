"""Exercise 4 — lesson 12's filters make the compound just the better attack; a 30% PAIR rate lifts capped MSJ to 0.40.

    Design a prompt that combines PAIR iteration (Lesson 12) with MSJ. Argue
    whether the compound attack is worse than MSJ alone, and for which model
    behaviours.

Reading of the exercise: the design is the compound's *shape*, written with
abstract placeholders only. PAIR runs as the outer loop and searches for a
framing of the target query. MSJ supplies n faux compliance pairs in front
of that framed query. `compound_prompt()` builds the shape. "Worse than MSJ
alone" is measured by composing the two lessons' own code: PAIR's standalone
success comes from lesson 12's `pair_loop` against its two filters, and MSJ's
comes from lesson 13's `target_asr`. The two are combined as independent
routes to success, 1 - (1 - p_PAIR)(1 - ASR_MSJ(n)).

**ANSWER: the compound is worse than MSJ alone exactly when PAIR alone
sometimes but not always succeeds, and it matters most where shots are
capped.** At a 30% (10%) standalone PAIR rate, the compound lifts
the reference's 16-shot-capped defense from 0.140 to 0.398 (0.226). It lifts
undefended MSJ at 512 shots only from 0.699 to 0.789 (0.729). It also cuts the
shots needed for 50% from 256 to 79 (201). So it is worse for behaviours
whose refusal has a surface-form component PAIR can search around, and
against defenses that bound the shot count or the context length.

**FINDING: with lesson 12's filters the compound is never better than the
better single attack.** Lesson 12's filters are deterministic, so PAIR's
standalone rate is 0 or 1. The keyword filter falls to paraphrase in 3
queries and to roleplay and encoding in 1, so there the compound is 1.0 and
the shots add nothing. The semantic filter holds all 20 queries against
paraphrase and roleplay, so there the compound equals MSJ alone. It falls
only to encoding, in 1 query. At every shot count, both rates give
max(PAIR, MSJ).

**FINDING: lesson 13's target has nothing for PAIR to optimize.**
`target_asr` takes (n_shots, alpha, a0), with no prompt. In the toy,
refining the query can only enter as a separate success route.

Structure: `pair_standalone()` runs lesson 12's `pair_loop` for every
attacker and filter; `compound()` composes it with `target_asr`;
`compound_prompt()` is the placeholder template.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "13-many-shot-jailbreaking"
PAIR_LESSON = "12-red-teaming-pair-automated-attacks"
RATES = (0.1, 0.3)


def compound_prompt(n_shots, framing):
    """The compound attack's shape: n faux compliance pairs, then PAIR's refined query."""
    shots = [f"User: <harmful request #{k}>\\nAssistant: <compliant answer #{k}>" for k in range(n_shots)]
    return "\\n".join([*shots, f"User: <target query, rewritten by PAIR framing '{framing}'>", "Assistant:"])


def pair_standalone(pair):
    """PAIR alone against each lesson-12 filter: attacker -> (jailbroken, queries used)."""
    attackers = {"paraphrase": pair.attacker_paraphrase, "roleplay": pair.attacker_roleplay,
                 "encoded": pair.attacker_encoded}
    targets = {"keyword": pair.keyword_filter_target, "semantic": pair.semantic_filter_target}
    return {t: {a: pair.pair_loop(tf, af)[:2] for a, af in attackers.items()} for t, tf in targets.items()}


def compound(msj_asr, p):
    """Either the refined query lands on its own (p) or the shots carry it (independent)."""
    return 1 - (1 - p) * (1 - msj_asr)


def shots_to_half(ref, p):
    return next(n for n in range(1, 5000) if compound(ref.target_asr(n), p) >= 0.5)


def solve():
    ref, pair = parity.load_reference(PHASE, LESSON, "main"), parity.load_reference(PHASE, PAIR_LESSON, "main")
    alone = pair_standalone(pair)
    rates = sorted({float(ok) for t in alone.values() for ok, _ in t.values()})
    return {
        "pair": alone, "rates": rates,
        "target_params": list(inspect.signature(ref.target_asr).parameters),
        "undefended": {p: round(compound(ref.target_asr(512), p), 3) for p in (0.0, *RATES)},
        "capped": {p: round(compound(ref.defense_adjusted(512), p), 3) for p in (0.0, *RATES)},
        "half": {p: shots_to_half(ref, p) for p in (0.0, *RATES)},
        "prompt_pairs": compound_prompt(256, "encoded").count("Assistant: <compliant"),
    }


def verify(result):
    alone, und, cap, half = (result[k] for k in ("pair", "undefended", "capped", "half"))
    return [
        practice.Check(
            "ANSWER: the compound is worse than MSJ alone when 0 < p_PAIR < 1, most where shots are capped",
            cap == {0.0: 0.14, 0.1: 0.226, 0.3: 0.398} and und == {0.0: 0.699, 0.1: 0.729, 0.3: 0.789}
            and half == {0.0: 256, 0.1: 201, 0.3: 79} and result["prompt_pairs"] == 256,
            f"ASR by p_PAIR: under the 16-shot cap {cap}, undefended at 512 shots {und}; "
            f"shots to 50% {half}",
        ),
        practice.Check(
            "FINDING: with lesson 12's filters the compound is never better than the better attack",
            alone == {"keyword": {"paraphrase": (True, 3), "roleplay": (True, 1), "encoded": (True, 1)},
                      "semantic": {"paraphrase": (False, 20), "roleplay": (False, 20), "encoded": (True, 1)}}
            and result["rates"] == [0.0, 1.0],
            f"PAIR alone (jailbroken, queries): {alone}; standalone rates {result['rates']}, "
            "so the compound is max(PAIR, MSJ)",
        ),
        practice.Check(
            "FINDING: lesson 13's target has nothing for PAIR to optimize",
            result["target_params"] == ["n_shots", "alpha", "a0"],
            f"target_asr parameters {result['target_params']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
