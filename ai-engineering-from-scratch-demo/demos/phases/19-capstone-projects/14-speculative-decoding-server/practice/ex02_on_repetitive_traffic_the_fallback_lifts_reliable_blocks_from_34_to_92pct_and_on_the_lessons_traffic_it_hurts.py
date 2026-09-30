"""Exercise 2 -- on repetitive traffic the fallback lifts reliable blocks from 34% to 92%, and on the lesson's own traffic it hurts.

    Implement ngram-fallback: if EAGLE-3 acceptance drops below a threshold, switch to ngram drafts. Report reliability improvement.

Reading of the exercise: "EAGLE-3" is the lesson's `DraftModel`
(alignment 0.9, k = 4), aligned to the target as it was before an upgrade.
At token 2,000 of a 4,000-token stream the target is upgraded (weights times
exp(1.0 * N(0, 1)), as in exercise 1), and the draft goes stale. The ngram
draft is prompt lookup: it copies what followed the latest earlier match of
the last 3, 2 or 1 emitted tokens. The fallback switches for good once the
EAGLE draft's rolling 20-call acceptance falls below 0.5. Everything runs
through the lesson's `speculative_decode` and `TargetModel.verify`. Two
kinds of traffic are used: "chat" is the lesson's own target, and "code" is
the same target sharpened (weights^6) and repeating every 12 positions, the
repetitive text ngram drafts exist for. Reliability is the share of 20-call
blocks after the upgrade that average at least 3.0 tokens per target call.

**ANSWER: on code traffic the fallback raises reliability from 0.343 to
0.923.** Post-upgrade tokens per call rise from 2.85 to 3.75. The switch
happens 90 tokens after the upgrade. Ngram alone gets 3.34 and 0.724, so the
fallback does better than either draft. While healthy it never trips: 0
switches, and 4.72 and 4.76 tokens per call, the same as EAGLE alone on both
traffic kinds.

    after the upgrade        EAGLE only   ngram only   fallback
    code: tok/call           2.85         3.34         3.75
    code: reliable blocks    0.343        0.724        0.923
    chat: tok/call           1.72         1.28         1.28

**FINDING: on the lesson's own traffic the fallback makes things worse.**
On chat it switches 19 tokens after the upgrade and drops post-upgrade
throughput from 1.72 to 1.28 tokens per call. The lesson's target draws each
position from an independent random distribution, so the history has
nothing to copy. On healthy chat, ngram scores 2.13 tokens per call, below a
draft that proposes uniform random tokens (2.16). A threshold on EAGLE
acceptance alone cannot tell "ngram will help" from "ngram is noise". The
fallback also needs ngram's own acceptance on the same traffic.

Structure: `base()` and `target_at()` define the two traffic kinds and the
upgrade; `ngram()` is prompt lookup; `run()` wires a logging verify and the
switching draft around the lesson's scheduler; `summarise()` scores it.
"""

from __future__ import annotations

import math
import random
import statistics
import types

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "14-speculative-decoding-server"
K, N, SWITCH, SIGMA, PERIOD = 4, 4000, 2000, 1.0, 12
W, THRESHOLD, BLOCK, SLO = 20, 0.5, 20, 3.0


def base(ref, s, code):
    """chat = the lesson's target; code = a sharpened target that repeats every PERIOD positions."""
    if not code:
        return ref.softmax_from(s * 7 + 13)
    w = [p**6 for p in ref.softmax_from((s % PERIOD) * 7 + 13)]
    return [x / sum(w) for x in w]


def target_at(ref, s, code, sigma):
    p = base(ref, s, code)
    if s < SWITCH:
        return p
    r = random.Random(((s % PERIOD) if code else s) * 7 + 10013)
    w = [x * math.exp(sigma * r.gauss(0, 1)) for x in p]
    return [x / sum(w) for x in w]


def ngram(hist, k):
    """Prompt lookup: copy what followed the latest earlier match of the last 3, 2 or 1 tokens."""
    for n in (3, 2, 1):
        tail = hist[-n:]
        for i in range(len(hist) - n - 1, -1, -1):
            if hist[i:i + n] == tail:
                out = hist[i + n:i + n + k]
                return (out + [out[-1]] * k)[:k]
    return [hist[-1] if hist else 0] * k


def drafter(ref, st, mode, old):
    """EAGLE (the lesson's DraftModel on the pre-upgrade target), ngram, uniform, or EAGLE with fallback."""
    eagle = ref.DraftModel(alignment=0.9)
    drafts = {"eagle": lambda c, k, rng: eagle.propose(c, k, rng, old),
              "ngram": lambda c, k, rng: ngram(st["hist"], k),
              "uniform": lambda c, k, rng: [rng.randrange(10) for _ in range(k)]}

    def propose(ctx, k, rng, _t):
        recent = [a for _, u, a in st["calls"][-W:] if u == "eagle"]
        if mode == "fallback" and len(recent) == W and sum(recent) / W / K < THRESHOLD and not st["switched"]:
            st["use"], st["switched"] = "ngram", ctx
        return drafts[st["use"]](ctx, k, rng)

    return types.SimpleNamespace(propose=propose)


def run(ref, code, sigma, mode, seed=7):
    target, old = ref.TargetModel(), ref.TargetModel()
    target.distribution, old.distribution = (lambda s: target_at(ref, s, code, sigma)), (lambda s: base(ref, s, code))
    st, verify = {"hist": [], "calls": [], "use": "eagle" if mode == "fallback" else mode, "switched": None}, target.verify

    def logged(tokens, ctx, rng):
        accepted, nxt = verify(tokens, ctx, rng)
        st["hist"] += accepted + [nxt]
        st["calls"].append((ctx, st["use"], len(accepted)))
        return accepted, nxt

    target.verify = logged
    m = ref.speculative_decode(N, K, random.Random(seed), target, drafter(ref, st, mode, old))
    return summarise(m, st)


def summarise(m, st):
    after = [a + 1 for ctx, _, a in st["calls"] if ctx >= SWITCH]
    blocks = [statistics.mean(after[i:i + BLOCK]) for i in range(0, len(after) - BLOCK + 1, BLOCK)]
    return {"tok_call": round(m.tokens_per_target_call(), 2), "after": round(statistics.mean(after), 2),
            "reliable": round(sum(b >= SLO for b in blocks) / len(blocks), 3),
            "switched": None if st["switched"] is None else st["switched"] - SWITCH}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    out = {(t, s, m): run(ref, t == "code", s, m) for t in ("code", "chat") for s in (0.0, SIGMA)
           for m in ("eagle", "ngram", "fallback")}
    return {**out, "uniform": run(ref, False, 0.0, "uniform")}


def verify(result):
    r = result
    row = {k: (v["after"], v["reliable"], v["switched"]) for k, v in r.items() if k != "uniform"}
    return [
        practice.Check(
            "ANSWER: on code traffic the fallback lifts reliable blocks from 0.343 to 0.923",
            (row[("code", SIGMA, "eagle")], row[("code", SIGMA, "ngram")], row[("code", SIGMA, "fallback")])
            == ((2.85, 0.343, None), (3.34, 0.724, None), (3.75, 0.923, 90))
            and (r[("code", 0.0, "fallback")]["tok_call"], r[("chat", 0.0, "fallback")]["tok_call"]) == (4.72, 4.76)
            and all(r[(t, 0.0, "fallback")] == r[(t, 0.0, "eagle")] for t in ("code", "chat")),
            f"code after upgrade (tok/call, reliable, switch at): EAGLE {row[('code', SIGMA, 'eagle')]}, "
            f"ngram {row[('code', SIGMA, 'ngram')]}, fallback {row[('code', SIGMA, 'fallback')]}; "
            "healthy fallback == EAGLE on both traffic kinds",
        ),
        practice.Check(
            "FINDING: on the lesson's own traffic the fallback makes things worse",
            (row[("chat", SIGMA, "eagle")][0], row[("chat", SIGMA, "fallback")], r[("chat", 0.0, "ngram")]["tok_call"],
             r["uniform"]["tok_call"]) == (1.72, (1.28, 0.0, 19), 2.13, 2.16),
            f"chat after upgrade: EAGLE {row[('chat', SIGMA, 'eagle')][0]} vs fallback "
            f"{row[('chat', SIGMA, 'fallback')]}; healthy ngram {r[('chat', 0.0, 'ngram')]['tok_call']} "
            f"vs a uniform-random draft {r['uniform']['tok_call']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
