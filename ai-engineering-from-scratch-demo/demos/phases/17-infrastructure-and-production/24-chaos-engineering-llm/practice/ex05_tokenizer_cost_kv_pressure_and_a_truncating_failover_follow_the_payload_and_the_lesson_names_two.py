"""Exercise 5 — tokenizer cost, KV pressure and a truncating failover follow the payload, and the lesson names two.

    Name three LLM-specific failure modes that generic network-chaos cannot
    reproduce.

Reading of the exercise: generic network chaos -- delay, drop, corrupt,
duplicate, partition -- acts on packets and never reads what they carry. A
failure mode it cannot reproduce is one whose trigger is the *content* or the
*model*, so each of the three is shown as a small model in which every
network fault leaves the failure absent and one payload change produces it.

**ANSWER: a tokenizer stall, KV preemptions whose cost scales with context,
and a failover that silently truncates the context.** (1) A toy BPE that
merges one pair per scan costs 2,031 pair-steps on 1 KB of prose and 515,775
on 1 KB with no whitespace, 254x; delayed, 1%-corrupted or truncated prose
costs at most 2,094. (2) 32 requests of 1,024 prompt tokens decoding 512 on a
4,096-block KV pool preempt 0 times, as delayed or with 10% dropped; at 8,000
prompt tokens the same 32 requests preempt 5 times and re-prefill 40,704
tokens -- 8,141 per preemption, because recompute re-reads the victim's whole
context. (3) Failing over from a 16K to a 4K model drops 8 of 10 retrieved
1,400-token chunks; every answer returns 200, and the lesson's gate reads 0x
burn and completes.

**FINDING: network chaos reaches KV preemption only by multiplying requests,
which a different guard stops.** Duplicating the 1,024-token requests -- a
retry storm -- preempts 21 times at 1,247 re-prefilled tokens each. A cap of
32 running requests takes that to 0 and leaves the 8,000-token run at 5: the
length-driven case needs a token-budget admission rule.

**FINDING: the lesson's five LLM experiments hold two LLM-specific
mechanisms.** Network failure ("cut connectivity") and provider outage ("100%
429") are generic network and HTTP faults; memory overload ("KV cache
preemption storm") and KV eviction storm ("saturating vLLM block budget") are
one mechanism. And `code/main.py` runs none of them: its malformed prompt is a
fixed `induced_error_rate` of 0.04.

Structure: `bpe_steps()`, `kv_storm()` and `truncated()` are the three
models; `NET` holds the network faults applied to each.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "24-chaos-engineering-llm"
PROSE = ("the retriever returns five chunks and the model answers from them " * 16)[:1024]
POISON = "a" * 1024
BLOCK, POOL, OUT = 16, 4096, 512
DOC_PHRASES = ("cut connectivity", "100% 429", "KV cache preemption storm", "saturating vLLM block budget")
KV_RUNS = {"base": (1024, 32, None), "drop 10%": (1024, 29, None), "duplicate": (1024, 64, None),
           "long": (8000, 32, None), "dup capped": (1024, 64, 32), "long capped": (8000, 32, 32)}


def corrupt(text, rng=None):
    rng = rng or random.Random(0)
    return "".join(rng.choice("xyz ") if rng.random() < 0.01 else c for c in text)


NET = {"delay": str, "corrupt 1%": corrupt, "truncate": lambda t: t[: len(t) // 2]}


def bpe_steps(text, max_len=8):
    """Pair-steps of a greedy BPE that merges one pair per full scan, per word."""
    steps = 0
    for seq in (list(word) for word in text.split()):
        while len(seq) > 1:
            steps += len(seq) - 1
            fits = [i for i in range(len(seq) - 1) if len(seq[i]) + len(seq[i + 1]) <= max_len]
            if not fits:
                break
            i = fits[0]
            seq[i : i + 2] = [seq[i] + seq[i + 1]]
    return steps


def blocks(reqs):
    return sum(-(-(prompt + made + 1) // BLOCK) for prompt, made in reqs)


def admit(waiting, running, max_running):
    """FCFS admission while the next request's blocks fit and the cap allows."""
    cap = max_running or float("inf")
    while waiting and len(running) < cap and blocks(running + waiting[:1]) <= POOL:
        running.append(waiting.pop(0))


def kv_storm(prompts, max_running=None):
    """(preemptions, re-prefilled tokens); on overflow the newest request is recomputed."""
    waiting, running, done, preempted, refill = [[p, 0] for p in prompts], [], 0, 0, 0
    while done < len(prompts):
        admit(waiting, running, max_running)
        while blocks(running) > POOL:
            victim = running.pop()
            preempted, refill = preempted + 1, refill + victim[0] + victim[1]
            waiting.insert(0, victim)
        for req in running:
            req[1] += 1
        done += sum(made >= OUT for _, made in running)
        running = [r for r in running if r[1] < OUT]
    return preempted, refill


def truncated(chunks=10, chunk_tokens=1400, header=800, window=4096):
    """Fraction of retrieved chunks that no longer fit the fallback window."""
    return sum(header + (i + 1) * chunk_tokens > window for i in range(chunks)) / chunks


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    gate = ref.run_experiment(ref.Experiment("failover to 4K model", 5, 0.0, 0.3))
    return {
        "bpe": {"prose": bpe_steps(PROSE), "poison": bpe_steps(POISON),
                **{name: bpe_steps(f(PROSE)) for name, f in NET.items()}},
        "kv": {name: kv_storm([length] * n, cap) for name, (length, n, cap) in KV_RUNS.items()},
        "loss": truncated(), "fits_16k": truncated(window=16384),
        "gate": (gate["status"], gate["burn_rate_x"]),
        "doc": [s in parity.doc_text(PHASE, LESSON, "en") for s in DOC_PHRASES],
        "malformed_rate": ref.EXPERIMENTS[2].induced_error_rate,
    }


def verify(result):
    bpe, kv = result["bpe"], result["kv"]
    net_max = max(bpe[n] for n in NET)
    return [
        practice.Check(
            "ANSWER: a tokenizer stall, KV preemptions whose cost scales with context, "
            "and a failover that silently truncates the context",
            all([bpe["poison"] // bpe["prose"] == 253, net_max < 1.05 * bpe["prose"],
                 kv["base"] == kv["drop 10%"] == (0, 0), kv["long"] == (5, 40704),
                 result["loss"] == 0.8, result["fits_16k"] == 0,
                 result["gate"] == ("COMPLETED", 0.0)]),
            f"BPE pair-steps {bpe}; KV (preemptions, re-prefill) {kv}; failover loses "
            f"{result['loss']:.0%} of chunks, gate {result['gate']}",
        ),
        practice.Check(
            "FINDING: network chaos reaches KV preemption only by multiplying requests, "
            "which a different guard stops",
            all([kv["duplicate"] == (21, 26192), kv["dup capped"] == (0, 0),
                 kv["long capped"] == kv["long"]]),
            f"duplicated {kv['duplicate']}, capped at 32 running {kv['dup capped']}; "
            f"long-context capped {kv['long capped']}",
        ),
        practice.Check(
            "FINDING: the lesson's five LLM experiments hold two LLM-specific mechanisms",
            all([*result["doc"], result["malformed_rate"] == 0.04]),
            "docs/en.md phrases found: network failure, provider outage, memory overload, "
            f"KV storm = {result['doc']}; main.py's malformed prompt is a fixed "
            f"{result['malformed_rate']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
