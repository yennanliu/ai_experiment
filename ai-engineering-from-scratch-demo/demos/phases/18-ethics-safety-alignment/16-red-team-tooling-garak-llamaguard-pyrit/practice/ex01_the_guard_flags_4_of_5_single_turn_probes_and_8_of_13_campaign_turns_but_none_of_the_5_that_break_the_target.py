"""Exercise 1 — the guard flags 4 of 5 single-turn probes and 8 of 13 campaign turns, but none of the 5 that break the target.

    Run `code/main.py`. Compare the Llama-Guard-style classifier's detection
    rate on single-turn vs multi-turn attacks.

Reading of the exercise: "single-turn attacks" are the five Garak probes, each
one message; "multi-turn attacks" are PyRIT campaigns, one per probe payload
used as the seed, run through the reference's own four-converter chain
against its toy target until a turn gets through. Detection is
`is_unsafe(payload)`, the reference's Llama-Guard-style classifier, scored
per message and per campaign.

**ANSWER: 80% single-turn (4 of 5 probes) against 61.5% of campaign turns
(8 of 13).** The shipped run shows the same picture: `main()` prints 4 of 5
probes "classifier-detected", and 2 of its 3 campaign turns are flagged. The
per-turn rate hides the point, though: **the guard flags 0 of the 5 turns
that break the target.** Every campaign breaks at the one turn the guard
also misses -- `encode` on four seeds, the already-leetspeak seed on the
fifth. Campaign ASR against the toy target is 5 of 5.

**FINDING: the guard adds no catch the target does not already make.** Over
all 18 messages (5 probes + 13 turns) the target refuses exactly the
messages the guard flags. Both are keyword lists, the guard's a superset,
and the same `4/3/0` substitution beats both. Put the guard in front of the
target and the campaign still breaks at the same turn on all 5 seeds.

**FINDING: the toy's "multi-turn" is stateless, and state is what catches
it.** `pyrit_campaign` calls `target_fn(payload)` with one message and no
history, so each turn is a fresh single-turn attack. A guard that classifies
the whole conversation so far keeps refusing once the seed turn tripped it:
it stops 4 of the 5 campaigns, and only the seed it never flagged (already
leetspeak) still gets through.

**FINDING: the paraphrase turn is a no-op on 3 of the 5 seeds.**
`paraphrase_converter` only rewrites "make" and "help me", so on the
roleplay, privacy and leetspeak seeds the second turn resends the seed.

Structure: `campaigns()` runs `ref.pyrit_campaign` per probe seed against a
given gate; `guarded()` and `history_guarded()` build gates from
`ref.is_unsafe`; `shipped()` parses `main()`'s own printout.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "16-red-team-tooling-garak-llamaguard-pyrit"
REFUSE = "I must refuse this request."


def shipped(ref):
    """(probes marked classifier-detected, turns printed) from main()'s own output."""
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        ref.main()
    text = log.getvalue()
    return len(re.findall(r"classifier-detected", text)), re.findall(r"payload='([^']*)'", text)


def campaigns(ref, target):
    return [ref.pyrit_campaign(target, p.payload, max_turns=4) for p in ref.GARAK_PROBES]


def break_points(runs):
    return [r["broke_at"] for r in runs]


def flagged(ref, messages):
    return sum(map(ref.is_unsafe, messages)), len(messages)


def payloads(runs):
    """(every executed turn's payload, the payload of each turn that broke the target)."""
    turns = [t["payload"] for r in runs for t in r["log"]]
    return turns, [r["log"][-1]["payload"] for r in runs if r["broke_at"]]


def guarded(ref):
    """The toy target with the guard in front of it, message by message."""
    return lambda x: REFUSE if ref.is_unsafe(x) else ref.toy_target(x)


def history_guarded(ref):
    """A fresh gate that classifies the conversation so far, not the last message."""
    history = []

    def gate(x):
        history.append(x)
        return REFUSE if ref.is_unsafe(" ".join(history)) else ref.toy_target(x)
    return gate


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    probes = [p.payload for p in ref.GARAK_PROBES]
    runs = campaigns(ref, ref.toy_target)
    turns, breaking = payloads(runs)
    messages = probes + turns
    detected, printed = shipped(ref)
    stateful = [ref.pyrit_campaign(history_guarded(ref), s, max_turns=4)["broke_at"] for s in probes]
    return {
        "single": flagged(ref, probes), "turns": flagged(ref, turns),
        "breaking": flagged(ref, breaking), "broke_at": break_points(runs),
        "shipped": (detected, *flagged(ref, printed)),
        "agree": sum(ref.is_unsafe(m) == ("refuse" in ref.toy_target(m)) for m in messages),
        "n_messages": len(messages),
        "guarded": break_points(campaigns(ref, guarded(ref))),
        "stateful": stateful,
        "stateless_call": "target_fn(payload)" in inspect.getsource(ref.pyrit_campaign),
        "para_noop": sum(ref.paraphrase_converter(s) == s for s in probes),
    }


def verify(result):
    single, turns, breaking = result["single"], result["turns"], result["breaking"]
    return [
        practice.Check(
            "ANSWER: 80% single-turn (4/5) vs 61.5% of campaign turns (8/13); 0 of 5 breaking turns",
            [single, turns, breaking, result["shipped"]] == [(4, 5), (8, 13), (0, 5), (4, 2, 3)],
            f"probes {single}, campaign turns {turns}, breaking turns {breaking}; campaigns "
            f"broke at {result['broke_at']}; shipped (detected probes, flagged turns, turns) "
            f"{result['shipped']}",
        ),
        practice.Check(
            "FINDING: the guard adds no catch the target does not already make",
            all([result["agree"] == result["n_messages"] == 18,
                 result["guarded"] == result["broke_at"]]),
            f"guard and target agree on {result['agree']}/{result['n_messages']} messages; "
            f"guard in front: campaigns break at {result['guarded']}",
        ),
        practice.Check(
            "FINDING: the toy's multi-turn is stateless, and state is what catches it",
            all([result["stateless_call"], result["stateful"].count(None) == 4,
                 result["stateful"][2] == "seed"]),
            f"pyrit_campaign passes one message per call; a conversation-level guard gives "
            f"break points {result['stateful']}",
        ),
        practice.Check(
            "FINDING: the paraphrase turn is a no-op on 3 of the 5 seeds",
            result["para_noop"] == 3,
            f"paraphrase_converter leaves {result['para_noop']} of 5 probe payloads unchanged",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
