"""Exercise 5 — chunking cannot raise throughput on an additive cost, and on a roofline it lifts it 17%.

    Explain in one paragraph why chunked prefill helps P99 ITL but not
    throughput in isolation. Where does the throughput win come from in
    practice?

Reading of the exercise: the paragraph is the answer below. Both halves of it
are measured. "In isolation" is the lesson's toy, where a forward pass costs
the sum of its tokens' costs. "In practice" swaps in a roofline step cost:
a step costs the larger of reading the weights once and computing its tokens.
The same reference workload is then run under three policies. "Separate" runs
prefill-only steps first, the way a scheduler without chunking does. "Whole"
is the reference's mix of whole prompts and decodes. "Budget B" is chunked
decode plus prefill, up to B tokens per step.

**ANSWER: chunking changes the order of work, not the amount, so on its own
it cannot raise throughput.** Splitting a prompt into slices leaves every
prefill token in the total, and each slice pays one more fixed per-step cost.
What changes is how long a step lasts. A decode token no longer waits behind a
whole 8192-token prefill, only behind one slice, and that is P99 ITL. The
toy shows it exactly: `simulate_continuous` gives 886 tok/s unchunked and 885
chunked, while P99 ITL falls from 219.3 ms to 66.1 ms. In practice the
throughput win comes from the step's cost not being additive. A decode step
is memory-bound: for 70B FP8 on an H100 it reads 72.67 GB of weights in 21.7
ms, whatever the batch. A prefill slice added to that step uses compute the
step was leaving idle, so the prefill rides along for free until the step
reaches the ridge point.

**FINDING: on a roofline step cost, a 320-token budget lifts 248 to 291
tok/s.** Separate prefill steps give 248 tok/s and the
reference's whole-prompt mix gives 250. Budget 320 gives 291, 17% more, and
P99 ITL drops from 241.2 ms to 23.0 ms. On the additive toy cost the same
policies all land between 878 and 886 tok/s.

**FINDING: the best budget sits at the ridge point, about 304 tokens.**
H100 SXM spec peaks are assumed here: 3.35 TB/s of HBM and 1979 dense FP8
TFLOPS. At 0.0713 ms per token, a step's compute matches its 21.7 ms weight
read at 304 tokens. Budgets give 128 -> 246, 256 -> 286, 320 -> 291, 512 ->
273, 2048 -> 254 tok/s. Below the ridge, steps are memory-bound and wasted.
Above it, prefill finishes early and leaves decode-only steps idle. So the
budget that is best for throughput depends on the ridge point, a property of
the hardware, and not on free KV blocks.

Structure: `simulate(policy, cost)` is a continuous loop with the reference
admission rule on the reference `Request`s. Only the step's work and its
price vary.
"""

import collections

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "04-vllm-serving-internals"
WEIGHT_READ_S = 72.67e9 / 3.35e12  # FP8 checkpoint bytes / H100 SXM HBM3 bandwidth
COMPUTE_S = 2 * 70.56e9 / 1979e12  # 2 FLOPs per parameter per token / dense FP8 peak
POLICIES = ("separate", "whole", 128, 256, 320, 512, 2048)


def pick(running, policy):  # -> [(request, prefill tokens)]; 0 tokens means one decode
    pending = [(r, r.prompt_len - r.prefilled) for r in running if r.in_prefill]
    if policy == "separate" and pending:
        return pending
    return fill(running, policy if isinstance(policy, int) else float("inf"))


def fill(running, left):  # decodes and prefill slices, in running order, up to the budget
    batch = []
    for r in running:
        if left <= 0:
            break
        take = min(r.prompt_len - r.prefilled, left) if r.in_prefill else 0
        batch.append((r, take))
        left -= take or 1
    return batch


def admit(ref, waiting, running, used, now):  # the reference's rule: whole reservation fits
    while (waiting and waiting[0].arrived_at <= now
           and used + waiting[0].blocks_needed() <= ref.KV_BLOCKS_AVAILABLE):
        used += waiting[0].blocks_needed()
        running.append(waiting.popleft())
    return used


def step(ref, running, now, policy, cost):  # one priced iteration -> new time
    batch = pick(running, policy)
    for r, take in batch:
        r.prefilled += take
    decoded = [r for r, take in batch if take == 0]
    now += cost(ref, sum(t for _, t in batch), len(decoded))
    for r in decoded:
        r.itl_samples += [now - r.last_token_at] if r.last_token_at else []
        r.generated, r.last_token_at = r.generated + 1, now
    return now


def summary(reqs, end):  # (tok/s, P99 ITL ms), with the reference's report() percentile
    itls = sorted(x for r in reqs for x in r.itl_samples)
    p99 = itls[int(0.99 * len(itls)) - 1]
    return round(sum(r.generated for r in reqs) / end), round(p99 * 1e3, 1)


def simulate(ref, policy, cost):
    reqs = ref.make_workload()  # fresh Requests on every call, already in arrival order
    waiting, running, used, now = collections.deque(reqs), [], 0, 0.0
    while waiting or running:
        used = admit(ref, waiting, running, used, now)
        if not running:
            now = waiting[0].arrived_at
            continue
        now = step(ref, running, now, policy, cost)
        used -= sum(r.blocks_needed() for r in running if r.done)
        running[:] = [r for r in running if not r.done]
    return summary(reqs, now)


def additive(ref, prefill, decode):
    return (prefill * ref.PREFILL_LATENCY_PER_TOKEN + decode * ref.FORWARD_LATENCY_PER_TOKEN
            + ref.BATCH_OVERHEAD)


def roofline(ref, prefill, decode):
    return ref.BATCH_OVERHEAD + max(WEIGHT_READ_S, (prefill + decode) * COMPUTE_S)


def reference_toy(ref, chunked):
    reqs = ref.make_workload()
    return summary(reqs, ref.simulate_continuous(reqs, chunked=chunked))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    return {"toy": {c: reference_toy(ref, c) for c in (False, True)},
            "additive": {p: simulate(ref, p, additive) for p in POLICIES},
            "roofline": {p: simulate(ref, p, roofline) for p in POLICIES},
            "ridge": round(WEIGHT_READ_S / COMPUTE_S)}


def verify(result):
    toy, add, roof = result["toy"], result["additive"], result["roofline"]
    add_tps = [v[0] for v in add.values()]
    budgets = {p: roof[p][0] for p in POLICIES if isinstance(p, int)}
    return [
        practice.Check(
            "ANSWER: chunking changes the order of work, not the amount, so on its own it "
            "cannot raise throughput",
            toy == {False: (886, 219.3), True: (885, 66.1)},
            f"simulate_continuous (tok/s, P99 ITL ms): unchunked {toy[False]}, chunked {toy[True]}",
        ),
        practice.Check(
            "FINDING: on a roofline step cost, a 320-token budget lifts 248 to 291 tok/s",
            all([roof["separate"] == (248, 241.2), roof["whole"][0] == 250,
                 roof[320] == (291, 23.0), min(add_tps) == 878, max(add_tps) == 886]),
            f"roofline {roof}; the same policies on the additive toy cost give {add_tps} tok/s",
        ),
        practice.Check(
            "FINDING: the best budget sits at the ridge point, about 304 tokens",
            result["ridge"] == 304 and budgets == {128: 246, 256: 286, 320: 291, 512: 273, 2048: 254},
            f"weight read / per-token compute = {result['ridge']} tokens; tok/s by budget {budgets}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
