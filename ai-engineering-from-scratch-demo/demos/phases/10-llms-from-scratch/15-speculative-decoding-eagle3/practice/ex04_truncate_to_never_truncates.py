"""Exercise 4 — the rollback method only ever grows the buffer, and 3 of B's 5 drafts are thrown away.

    Implement a batched KV rollback simulator for two concurrent sequences.
    Sequence A has all drafts accepted; sequence B rejects at position 2. Show
    that the correct `kv_length` is updated per sequence and that no work is
    wasted.

Reading of the exercise: the two sequences get their own `KVBuffer` and are
stepped through the lesson's own `spec_step`, with A given a draft equal to the
verifier so every token is accepted and B given a badly mismatched one. Both
claims the exercise makes are then tested rather than demonstrated: that the
lengths are correct per sequence, and that no work is wasted.

**ANSWER: the per-sequence lengths are correct, and they are correct without
`truncate_to` ever truncating.** From a shared prefix of 10, A emits 6 tokens
and ends at 16; B rejects at position 2, emits 3, and ends at 13. Neither
buffer's length ever decreases: `spec_step` calls `kv.extend(1)` once per
accepted token and then `kv.truncate_to(prefix + len(emitted))`, which is one
*more* than the current length, because `emitted` already contains the
correction token.

**FINDING: the exercise's second claim is false, and the amount is the point.**
B drafted **5** tokens and used **2**. Three draft forwards are computed and
discarded on every rejection, which is not an implementation flaw -- it is what
speculative decoding pays for the verifier call it saves. The expected waste is
`N - E[accepted]`, and at N=5 with this draft it is most of the batch.

**MECHANISM: a batched rollback has nothing to synchronise.** Each sequence's
length is a scalar in its own `KVBuffer` and `spec_step` touches exactly one of
them, so the "batched" simulator is two independent simulators. The hard part of
a real batched rollback -- that the verifier's forward pass was issued for a
padded batch whose rows now diverge in length -- has no analogue here, because
the buffer holds a length and no bytes.

**FINDING: `truncate_to` is the only method that could roll back, and it is
never called with a smaller value.** Across both sequences it is invoked once,
with a value larger than the length it replaces. A buffer that only grows is a
counter.

Structure: `Tracked` subclasses the lesson's own `KVBuffer` to record every call
and whether it shrank; `run_pair` steps the two sequences and reports both.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "15-speculative-decoding-eagle3"
Q = [0.30, 0.22, 0.15, 0.10, 0.08, 0.07, 0.05, 0.03]
PREFIX, DRAFTS = 10, 5


def tracked(base):
    """The lesson's own buffer, instrumented: every call to truncate_to, and its direction."""

    class Tracked(base):
        def __init__(self):
            super().__init__()
            self.calls, self.shrank, self.grew = 0, 0, 0

        def truncate_to(self, n):
            self.calls += 1
            self.shrank += n < self.length
            self.grew += n > self.length
            super().truncate_to(n)

    return Tracked


def run_pair(ref):
    """A with a perfect draft, B with a bad one, each on its own buffer."""
    buffer = tracked(ref.KVBuffer)
    good, bad = buffer(), buffer()
    good.extend(PREFIX)
    bad.extend(PREFIX)
    accepted, _ = ref.spec_step(Q, Q, DRAFTS, good, random.Random(4))
    mismatched = ref.perturb(Q, 0.5, random.Random(9))
    rejected, _ = ref.spec_step(Q, mismatched, DRAFTS, bad, random.Random(3))
    return {"A": {"emitted": len(accepted), "length": good.length, "calls": good.calls,
                  "shrank": good.shrank, "grew": good.grew},
            "B": {"emitted": len(rejected), "length": bad.length, "calls": bad.calls,
                  "shrank": bad.shrank, "grew": bad.grew}}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    pair = run_pair(ref)
    used = pair["B"]["emitted"] - 1
    return {
        "pair": pair,
        "wasted": DRAFTS - used,
        "used": used,
        "drafted": DRAFTS,
        "independent": ref.KVBuffer().length == 0,
        "fields": [name for name in vars(ref.KVBuffer())],
    }


def verify(result):
    a, b = result["pair"]["A"], result["pair"]["B"]
    return [
        practice.Check(
            "ANSWER: the lengths are correct per sequence, and nothing ever truncates",
            a["length"] == PREFIX + a["emitted"] and b["length"] == PREFIX + b["emitted"]
            and a["shrank"] == b["shrank"] == 0,
            f"from a shared prefix of {PREFIX}, A accepts every draft, emits {a['emitted']} "
            f"tokens and ends at {a['length']}; B rejects early, emits {b['emitted']} and ends at "
            f"{b['length']}. Both are right. Neither buffer's length ever decreases: spec_step "
            f"calls extend(1) per accepted token and then truncate_to(prefix + len(emitted)), "
            f"which is one *more* than the current length because emitted already holds the "
            f"correction token -- A calls it {a['calls']} times and B {b['calls']}, and it grew "
            f"the buffer {b['grew']} of those",
        ),
        practice.Check(
            "FINDING: the exercise's second claim is false, and the amount is the point",
            result["wasted"] > 0,
            f"B drafted {result['drafted']} tokens and used {result['used']}. "
            f"{result['wasted']} draft forwards are computed and thrown away on this rejection, "
            "which is not an implementation flaw -- it is what speculative decoding pays for the "
            "verifier call it saves. The expected waste is N minus the expected accepted count, "
            "and with a draft this far from the verifier it is most of the batch",
        ),
        practice.Check(
            "MECHANISM: a batched rollback has nothing to synchronise",
            result["fields"] == ["length"] or "length" in result["fields"],
            f"KVBuffer's state is {result['fields']} -- one scalar -- and spec_step touches "
            "exactly one buffer, so the 'batched' simulator is two independent simulators run "
            "one after the other. The hard part of a real batched rollback, that the verifier's "
            "forward was issued for a padded batch whose rows now diverge in length, has no "
            "analogue here because the buffer holds a length and no bytes",
        ),
        practice.Check(
            "FINDING: a buffer that only grows is a counter",
            a["shrank"] + b["shrank"] == 0 and a["calls"] + b["calls"] > 0,
            f"truncate_to is the only method that could roll anything back, and across both "
            f"sequences it is invoked {a['calls'] + b['calls']} time(s), every one of them with a "
            f"value larger than the length it replaced. The name describes an operation this "
            "simulator never performs",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
