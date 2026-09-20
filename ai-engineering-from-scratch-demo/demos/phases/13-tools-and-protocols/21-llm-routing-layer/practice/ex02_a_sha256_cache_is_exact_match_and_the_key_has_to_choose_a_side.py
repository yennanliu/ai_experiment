"""Exercise 2 — a SHA256 cache is exact-match, and the key has to choose a side.

    Add semantic caching: SHA256 of the prompt is a lookup key; cache hits
    return instantly. Measure cost savings on a repeated call.

Reading of the exercise: the exercise says "semantic" and specifies a hash,
and those are different things -- a digest matches bytes, so two prompts
meaning the same thing miss. That is worth measuring rather than glossing.
Then the key itself turns out to be a decision the exercise does not make:
hashing the prompt alone merges aliases, and hashing before or after
redaction trades a PII leak against a collision.

**ANSWER: the second identical call costs 0 and saves 100% of the repeat.**
Two identical prompts through `smart` cost the provider once: **1** hit,
**1** miss, `cost_usd` **0.0** on the hit, and the cached response is
byte-identical. Over a 5-call workload with 3 repeats the spend falls by the
repeats' share exactly.

**FINDING: it is exact-match, not semantic.** `"summarise the report"` and
`"summarise the report "` -- one trailing space -- hash differently and both
call the provider, as do two paraphrases. Of **4** prompts a human would
call one question, the cache serves **1** and pays for **3**. "Semantic"
needs an embedding; SHA256 gives byte equality with a nicer name.

**FINDING: the prompt alone is the wrong key, because the alias picks the
model.** The same text through `smart` and `fast` must not share an entry --
keyed on the prompt only, the second alias is served `openai/gpt-4o`'s answer
while believing it called `openai/gpt-4o-mini`. Keying on `(alias, digest)`
gives **2** entries and the right model each time.

**FINDING: hashing before or after redaction is a real choice, and both
sides cost something.** The raw prompt as key is not one-way over a space
this small -- enumerating **1000** candidate SSNs recovers the exact prompt
from its digest;
the redacted prompt as key makes two customers' distinct prompts collide on
`[REDACTED]` and serves one the other's answer. This gateway redacts before
routing, so the redacted string is what is available -- and the collision is
the price.

Structure: `Gateway` wraps the lesson's `route` with a cache whose key
function is a parameter, so each keying choice is the same code.
"""

from __future__ import annotations

import contextlib
import hashlib
import io

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "21-llm-routing-layer"
PROMPT = "summarise the incident report"
VARIANTS = [PROMPT, PROMPT + " ", "give me a summary of the incident report",
            PROMPT.upper()]
SSN_A = "customer 123-45-6789 wants a refund"
SSN_B = "customer 987-65-4321 wants a refund"


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def by_prompt(alias, text):
    return digest(text)


def recovered(target, candidates):
    """A digest over a small space is a lookup, not a one-way function."""
    return next((c for c in candidates if digest(c) == target), None)


def by_alias_and_prompt(alias, text):
    return f"{alias}:{digest(text)}"


class Gateway:
    """The lesson's route with a cache in front, keyed by a supplied function."""

    def __init__(self, ref, key=by_alias_and_prompt):
        self.ref, self.key, self.store = ref, key, {}
        self.hits, self.misses, self.spend = 0, 0, 0.0

    def ask(self, alias, text):
        cache_key = self.key(alias, text)
        if cache_key in self.store:
            self.hits += 1
            return self.store[cache_key], 0.0
        self.misses += 1
        with contextlib.redirect_stdout(io.StringIO()):
            inv = self.ref.route(alias, [{"role": "user", "content": text}])
        self.store[cache_key] = inv
        self.spend += inv.cost_usd
        return inv, inv.cost_usd


def uncached_spend(ref, calls):
    with contextlib.redirect_stdout(io.StringIO()):
        return sum(ref.route(a, [{"role": "user", "content": t}]).cost_usd
                   for a, t in calls)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    gateway = Gateway(ref)
    first, first_cost = gateway.ask("smart", PROMPT)
    second, second_cost = gateway.ask("smart", PROMPT)

    workload = [("smart", PROMPT)] * 4 + [("smart", "a different question")]
    batch = Gateway(ref)
    for alias, text in workload:
        batch.ask(alias, text)
    variants = Gateway(ref)
    for text in VARIANTS:
        variants.ask("smart", text)
    naive = Gateway(ref, key=by_prompt)
    naive.ask("smart", PROMPT)
    leaked, _ = naive.ask("fast", PROMPT)
    keyed = Gateway(ref)
    keyed.ask("smart", PROMPT)
    correct, _ = keyed.ask("fast", PROMPT)
    redacted_a = ref.redact_pii(SSN_A)[0]
    redacted_b = ref.redact_pii(SSN_B)[0]
    space = [f"customer {n:03d}-45-6789 wants a refund" for n in range(1000)]
    return {
        "first_cost_positive": first_cost > 0, "second_cost": second_cost,
        "identical": first.response == second.response,
        "hits": gateway.hits, "misses": gateway.misses,
        "batch_spend": round(batch.spend, 10),
        "uncached": round(uncached_spend(ref, workload), 10),
        "batch_hits": batch.hits, "batch_misses": batch.misses,
        "variant_calls": variants.misses, "variant_hits": variants.hits,
        "variants": len(VARIANTS),
        "leaked_model": leaked.chosen_model, "correct_model": correct.chosen_model,
        "naive_entries": len(naive.store), "keyed_entries": len(keyed.store),
        "recovered": recovered(by_prompt("smart", SSN_A), space), "space": len(space),
        "redacted_collide": digest(redacted_a) == digest(redacted_b),
        "raw_distinct": digest(SSN_A) != digest(SSN_B)}


def verify(result):
    return [
        practice.Check(
            "ANSWER: the repeat costs 0 and the saving is the repeats' share",
            all([result["first_cost_positive"], result["second_cost"] == 0.0,
                 result["identical"], result["hits"] == 1, result["misses"] == 1,
                 result["batch_hits"] == 3, result["batch_misses"] == 2,
                 result["batch_spend"] < result["uncached"]]),
            f"the second identical call costs {result['second_cost']} and returns the "
            f"byte-identical response. Over a 5-call workload the cache takes "
            f"{result['batch_hits']} hits and {result['batch_misses']} misses, spending "
            f"{result['batch_spend']} against {result['uncached']}",
        ),
        practice.Check(
            "FINDING: it is exact-match, not semantic",
            all([result["variant_calls"] == 4, result["variant_hits"] == 0,
                 result["variants"] == 4]),
            f"of {result['variants']} prompts a human would call one question -- the text, "
            f"the text with a trailing space, a paraphrase and the same text uppercased -- "
            f"the cache serves {result['variant_hits']} and pays for "
            f"{result['variant_calls']}. Semantic needs an embedding; SHA256 gives byte "
            "equality with a nicer name",
        ),
        practice.Check(
            "FINDING: the prompt alone is the wrong key, because the alias picks the model",
            all([result["leaked_model"] == "openai/gpt-4o",
                 result["correct_model"] == "openai/gpt-4o-mini",
                 result["naive_entries"] == 1, result["keyed_entries"] == 2]),
            f"keyed on the prompt only, the fast alias is served {result['leaked_model']}'s "
            f"answer from {result['naive_entries']} entry; keyed on (alias, digest) it gets "
            f"{result['correct_model']} from {result['keyed_entries']}. The alias is what "
            "chooses the chain, so it belongs in the key",
        ),
        practice.Check(
            "FINDING: hashing before or after redaction is a real choice",
            all([result["recovered"] == SSN_A, result["raw_distinct"],
                 result["redacted_collide"]]),
            f"the raw prompt as key is not one-way over a small space: enumerating "
            f"{result['space']} candidate SSNs recovers {result['recovered']!r} from the "
            "digest alone. The redacted prompt as key "
            "makes two customers' distinct prompts collide on [REDACTED] and serves one the "
            "other's answer. This gateway redacts before routing, so the redacted string is "
            "what is available -- and the collision is the price",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
