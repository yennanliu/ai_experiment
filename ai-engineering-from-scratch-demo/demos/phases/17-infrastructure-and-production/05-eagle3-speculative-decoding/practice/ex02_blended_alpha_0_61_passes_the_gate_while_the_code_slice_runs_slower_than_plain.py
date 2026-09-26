"""Exercise 2 — blended alpha 0.61 passes the gate while the code slice runs slower than plain.

    Imagine production traffic splits 70% general chat, 30% code. General chat
    hits alpha 0.7 with EAGLE-3 trained on ShareGPT; code hits alpha 0.4. What
    is blended alpha and is spec decode net-positive?

Reading of the exercise: the split is read as a share of generated tokens,
and alpha as the per-position acceptance the lesson's simulator draws. "Net
positive" is asked three ways: the lesson's 0.55 gate, its `expected_speedup`
formula, and the throughput its own simulator implies. That throughput is the
exact expectation behind `simulate_tail` / `plain_tail`, which exercise 1
checks against the simulator at 200,000 tokens. The mix is blended by time,
not by alpha: each slice's tokens take share / speedup of the plain time.

**ANSWER: blended alpha is 0.61, and by the lesson's rules spec decode is
net-positive.** 0.61 clears the 0.55 gate. The formula gives 3.47x at 32
concurrent and 3.12x at 256. It rates even the code slice alone at 2.57x, so
nothing in the lesson would tell you to treat the slices differently.

**FINDING: in the lesson's own simulator the code slice runs slower than
plain.** At 32 concurrent chat runs at 1.56x and code at 0.82x. The mix
blended by time runs at 1.23x; evaluating the single blended alpha 0.61 gives
1.24x, which hides the loser. Routing code to plain decode gives 1.34x, better
than spec on everything. At 256 concurrent code reaches 1.09x, spec on
everything wins (1.63x against 1.57x routed), and the answer flips with load.

**FINDING: the alpha you would log for this mix reads 0.28.** vLLM's
per-request `draft_acceptance_rate` is accepted drafts over proposed drafts.
The lesson's step 3 amounts to the same thing: accepted tokens per request
divided by the draft length. Acceptance stops at the first rejection, so
per-position 0.7 logs as 0.388, 0.4 as 0.132, and the pooled mix as 0.278.
Gated at 0.55, that would switch off a config the formula rates at 3.47x.
The gate and the formula use two different alphas.

Structure: `tokens()` and `sim_speedup()` are the simulator's expectations;
`blend()` adds per-token time across slices.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "05-eagle3-speculative-decoding"
K, EPS, GATE = 5, 0.15, 0.55
MIX = {"chat": (0.7, 0.7), "code": (0.3, 0.4)}  # slice: (token share, alpha)


def tokens(alpha, k=K):
    """Expected tokens per verify when acceptance stops at the first rejection."""
    return sum(alpha**i for i in range(k + 1))


def sim_speedup(alpha, conc, k=K, eps=EPS):
    """plain_tail's mean over simulate_tail's mean, as exact expectations."""
    verify = 8.0 * (1 + eps * (1 + conc / 256))
    return 8.0 * (1 + conc / 512) / ((verify + 8.0 * (1 - alpha**k)) / tokens(alpha, k))


def blend(speed, slices=MIX):
    """Token-weighted throughput: total time is the share-weighted per-token time."""
    return 1 / sum(share / speed(alpha) for share, alpha in slices.values())


def logged_rate(slices=MIX):
    """Pooled accepted / proposed drafts over the mix, as vLLM counts it."""
    accepted = sum(s * (tokens(a) - 1) / tokens(a) for s, a in slices.values())
    proposed = sum(s * K / tokens(a) for s, a in slices.values())
    return accepted / proposed


def at(ref, conc, alpha_blend):
    formula = lambda a: ref.expected_speedup(ref.SpecPoint(a, K, EPS, conc))  # noqa: E731
    sim = lambda a: sim_speedup(a, conc)  # noqa: E731
    return {
        "formula": formula(alpha_blend),
        "formula_code": formula(0.4),
        "chat": sim(0.7),
        "code": sim(0.4),
        "one_alpha": sim(alpha_blend),
        "mix": blend(sim),
        "routed": blend(lambda a: sim(a) if a > 0.5 else 1.0),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    alpha_blend = round(sum(share * alpha for share, alpha in MIX.values()), 4)
    return {
        "alpha": alpha_blend,
        "at": {c: at(ref, c, alpha_blend) for c in (32, 256)},
        "per_slice": {a: (tokens(a) - 1) / K for _, a in MIX.values()},
        "logged": logged_rate(),
    }


def verify(result):
    low, high = result["at"][32], result["at"][256]
    r = {k: round(v, 2) for k, v in low.items()}
    rates = [round(v, 3) for v in result["per_slice"].values()]
    return [
        practice.Check(
            "ANSWER: blended alpha is 0.61, and by the lesson's rules spec decode is net-positive",
            all(
                [
                    result["alpha"] == 0.61 >= GATE,
                    round(low["formula"], 2) == 3.47,
                    round(high["formula"], 2) == 3.12,
                    low["formula_code"] > 2.5,
                ]
            ),
            f"0.7*0.7 + 0.3*0.4 = {result['alpha']} >= {GATE}; formula {low['formula']:.2f}x "
            f"at 32, {high['formula']:.2f}x at 256, code alone {low['formula_code']:.2f}x",
        ),
        practice.Check(
            "FINDING: in the lesson's own simulator the code slice runs slower than plain",
            all(
                [
                    low["code"] < 1 < low["chat"],
                    low["mix"] < low["one_alpha"] < low["routed"],
                    high["code"] > 1,
                    high["routed"] < high["mix"],
                ]
            ),
            f"at 32 concurrent chat {r['chat']}x, code {r['code']}x, mix {r['mix']}x, "
            f"one blended alpha {r['one_alpha']}x, code routed to plain {r['routed']}x; "
            f"at 256 code {high['code']:.2f}x, mix {high['mix']:.2f}x, routed {high['routed']:.2f}x",
        ),
        practice.Check(
            "FINDING: the alpha you would log for this mix reads 0.28",
            all([rates == [0.388, 0.132], round(result["logged"], 3) == 0.278 < GATE]),
            f"accepted / proposed drafts: chat {rates[0]}, code {rates[1]}, pooled "
            f"{result['logged']:.3f} -- under the {GATE} gate the formula passes at 3.47x",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
