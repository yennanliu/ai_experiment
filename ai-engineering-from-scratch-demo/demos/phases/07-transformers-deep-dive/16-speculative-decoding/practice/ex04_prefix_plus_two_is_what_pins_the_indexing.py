"""Exercise 4 — "prefix + first 2 accepted" is the only thing that pins the index.

    **Hard.** Implement rollback: start with a 10-token prefix KV cache, feed 5
    draft tokens, simulate a rejection at position 3. Verify your cache reads
    correctly match "prefix + first 2 accepted drafts" at the next iteration.

Reading of the exercise: "a rejection at position 3" is ambiguous on its own --
0-indexed it leaves 3 accepted drafts and a cache of 13, 1-indexed it leaves 2
and a cache of 12 -- and the clause that resolves it is the verification target,
"first 2 accepted drafts". So the rollback keeps 2, and the check is not that the
length is right but that every entry is.

**ANSWER: 10 -> 15 -> 12, and the 12 entries are bit-identical to a cache built
from scratch on the same 12 tokens.** Length alone would pass a cache that kept
the right *number* of stale rows; comparing keys and values entry by entry is
what makes the rollback a rollback rather than a resize.

**FINDING: the verifier's own token then makes 13, not 12.** After rejecting
draft 3 the verifier emits a corrected token at that position, so the next
iteration starts from `prefix + accepted + 1 = 13` and the net gain from a
5-token draft with 2 accepted is **3 tokens for one verifier forward**. A
rollback that stops at 12 silently drops the one token the rejection produced.

**FINDING: truncation and rollback agree, which is what makes the cheap
implementation legal.** Deleting the tail of the list gives the same 12 entries
as replaying the 12 tokens, because a causal KV cache entry depends only on its
own position and the ones before it. That is the property speculative decoding
leans on; it is why the 3 rejected entries can be thrown away instead of
recomputed, and it would not hold for a bidirectional cache.

**CONTROL: without the rollback the next step attends 15 positions.** Three of
them are keys and values for tokens the verifier rejected. Nothing raises an
error -- the shapes are all fine -- so the failure of a missing rollback is
silent and shows up as quality, which is the reason the exercise asks you to
check the contents.

Structure: `Cache` is the minimal append/rollback pair; `build` replays a token
range from scratch; `run` performs the draft-and-reject cycle.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "16-speculative-decoding"
PREFIX, DRAFTS, REJECT_AT = 10, 5, 3


class Cache:
    """Append-only KV store with a rollback, which is all a draft loop needs."""

    def __init__(self):
        self.keys, self.values = [], []

    def append(self, key, value):
        self.keys.append(key)
        self.values.append(value)

    def rollback(self, length):
        del self.keys[length:]
        del self.values[length:]

    def entries(self):
        return list(zip(self.keys, self.values))


def kv(token):
    """A stand-in projection: deterministic in the token, as a real one is in its prefix."""
    return [token, token + 0.5], [2 * token, 2 * token + 1]


def build(count):
    """A cache replayed from scratch over the first `count` tokens."""
    fresh = Cache()
    for token in range(count):
        fresh.append(*kv(token))
    return fresh


def run(prefix=PREFIX, drafts=DRAFTS, reject_at=REJECT_AT):
    """Fill the prefix, append drafts, roll back to the accepted ones. Returns snapshots."""
    cache, accepted = build(prefix), reject_at - 1
    for token in range(prefix, prefix + drafts):
        cache.append(*kv(token))
    filled = len(cache.keys)
    cache.rollback(prefix + accepted)
    rolled = cache.entries()
    cache.append(*kv(10_000))
    return {"prefix": prefix, "filled": filled, "rolled": rolled,
            "after_correction": len(cache.keys), "accepted": accepted}


def solve():
    parity.load_reference(PHASE, LESSON, "main")
    state = run()
    stale = Cache()
    for token in range(PREFIX + DRAFTS):
        stale.append(*kv(token))
    return {
        **state, "matches": state["rolled"] == build(PREFIX + state["accepted"]).entries(),
        "kept": len(state["rolled"]),
        "zero_indexed": PREFIX + REJECT_AT, "one_indexed": PREFIX + REJECT_AT - 1,
        "truncated": stale.entries()[:PREFIX + state["accepted"]]
        == build(PREFIX + state["accepted"]).entries(),
        "unrolled": len(stale.keys), "discarded": DRAFTS - state["accepted"],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 10 -> 15 -> 12, and all 12 entries match a cache built from scratch",
            result["matches"] and (result["prefix"], result["filled"], result["kept"])
            == (PREFIX, PREFIX + DRAFTS, PREFIX + REJECT_AT - 1),
            f"the prefix fills {result['prefix']} entries, {DRAFTS} drafts take it to "
            f"{result['filled']}, and rolling back to {result['accepted']} accepted leaves "
            f"{result['kept']}. Every key and value matches a cache replayed over the same "
            "12 tokens -- length alone would also pass a cache holding stale rows",
        ),
        practice.Check(
            "ANSWER: 'first 2 accepted' is what pins the indexing",
            result["one_indexed"] == result["kept"] != result["zero_indexed"],
            f"'a rejection at position {REJECT_AT}' is ambiguous alone: 0-indexed it leaves "
            f"{REJECT_AT} accepted drafts and a cache of {result['zero_indexed']}, 1-indexed it "
            f"leaves {REJECT_AT - 1} and a cache of {result['one_indexed']}. The verification "
            "target the exercise gives is the only thing that resolves it",
        ),
        practice.Check(
            "FINDING: the verifier's own token then makes 13, not 12",
            result["after_correction"] == PREFIX + result["accepted"] + 1,
            f"after rejecting draft {REJECT_AT} the verifier emits a corrected token at that "
            f"position, so the next iteration starts from prefix + accepted + 1 = "
            f"{result['after_correction']}. The net gain from a {DRAFTS}-token draft with "
            f"{result['accepted']} accepted is {result['accepted'] + 1} tokens for one verifier "
            "forward; a rollback that stops at 12 drops the token the rejection produced",
        ),
        practice.Check(
            "FINDING: truncation and rollback agree, which is what makes it legal",
            result["truncated"],
            "deleting the tail of the list gives the same entries as replaying the tokens, "
            "because a causal KV entry depends only on its own position and the ones before it. "
            "That is the property speculative decoding leans on -- the rejected entries can be "
            "thrown away instead of recomputed -- and it would not hold for a bidirectional cache",
        ),
        practice.Check(
            "CONTROL: without the rollback the next step attends 15 positions",
            result["unrolled"] == PREFIX + DRAFTS,
            f"{result['unrolled']} entries, {result['discarded']} of them keys and values for "
            "tokens the verifier rejected. No shape is wrong and nothing raises, so a missing "
            "rollback fails silently and shows up as quality -- which is why the exercise asks "
            "you to check the contents rather than the length",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
