"""Exercise 5 -- TensorRT-LLM's 3.1x lead at temperature 1 is greedy text, and from batch 128 it loses to vLLM.

    Benchmark TensorRT-LLM speculative decoding on the same H100 hardware. Report where it wins vs vLLM.

Reading of the exercise: there is no H100 here, so this is the scaled-down
benchmark D11 asks for. It runs the parts of the comparison that are
algorithmic and does not guess at the part that is kernel speed. Each
engine's documented verify rule replaces the lesson's `TargetModel.verify`
and runs through the lesson's `speculative_decode`: the lesson's `DraftModel`
(alignment 0.9), k = 1..4, 4,000 tokens. The sources, read 2026-09-29, are:

- TensorRT-LLM: "Currently, only greedy sampling is supported for speculative
  decoding" and "There is currently no way to dynamically disable
  speculation, thus speed ups are only observable at low batch sizes"
  (https://nvidia.github.io/TensorRT-LLM/1.2.0rc6/features/speculative-decoding.html).
- vLLM: rejection sampling, "validated to be lossless"
  (https://docs.vllm.ai/en/latest/features/speculative_decoding/), plus
  dynamic speculative decoding, which sets k per batch-size range
  (https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/).

A step's cost comes from an H100 SXM roofline: 3.35 TB/s, and 1,979 dense
FP8 TFLOPS (https://www.nvidia.com/en-us/data-center/h100/ lists 3,958
with sparsity). The model is Llama 3.3 70B in FP8 (72.66 GB, exercise 4), so
a step is max(weights / bandwidth, 2 x params x tokens / FLOPS). The real
commands to run on an H100 are in `COMMANDS`.

**ANSWER: nowhere that the algorithm can show. TensorRT-LLM's wins have to
come from kernel time, which only the H100 run in `COMMANDS` can measure.**
At temperature 0, vLLM's rule is the same exact-argmax match, so the two tie
at every k. At temperature 1 the lesson samples. There, TensorRT-LLM emits
4.28 tokens per target call at k = 4 against vLLM's 1.38 (3.1x), but it
emits greedy text: the first emitted token is 0.813 in total variation away
from the distribution asked for. vLLM's rule is 0.007 away over 20,000
trials, which is sampling noise. On the roofline both engines give 4.28x up
to batch 32. Past that, speculating at a fixed k = 4 loses to choosing k per
batch:

    batch                     1-32   64     128    256    512
    TensorRT-LLM, fixed k=4   4.28   4.07   2.04   1.02   0.86
    vLLM, best k per batch    4.28   4.07   2.21   1.14   1.00
    vLLM's chosen k           4      4      2      1      0

At batch 512, fixed speculation is 14% slower than none at all. These are
upper bounds, and a real engine's lower efficiency moves the crossover to a
smaller batch.

**FINDING: the lesson's own verify rule is not lossless either.** It
accepts any token within half of the max. So the first emitted token is
0.736 in total variation away from the target, nearly as far as greedy, and
it scores 4.756 tokens per call at k = 4. The lossless rule gets 1.378 on the
same draft and target. The lesson's printed speedups come from not sampling
the target.

**FINDING: the lesson's "EAGLE-3 in vLLM 0.7" predates EAGLE-3.** vLLM v0.7.0
was released on 2025-01-27
(https://github.com/vllm-project/vllm/releases/tag/v0.7.0). The EAGLE-3
paper the lesson cites, arXiv:2503.01840, was first submitted on 2025-03-03
(https://arxiv.org/abs/2503.01840).

Structure: `greedy_verify()` and `lossless_verify()` are the two engines'
rules; `tokens_per_call()` runs one through the lesson's scheduler;
`emitted_tv()` gives the mean total variation from the target over 10,000
positions (greedy emits the argmax, so its TV is 1 - max p; `lesson_emitted()`
is the exact first-token distribution under the lesson's rule);
`lossless_check()` measures vLLM's rule empirically at one position;
`speedups()` is the roofline.
"""

from __future__ import annotations

import random
import re
import statistics

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "14-speculative-decoding-server"
ALIGN, N, KMAX = 0.9, 4000, 4
# H100 SXM: 3.35 TB/s HBM3, 1,979 dense FP8 TFLOPS (3,958 is with sparsity); Llama 3.3 70B FP8
BW, FLOPS, WEIGHT_B, PARAMS = 3.35e12, 1.979e15, 72.66e9, 70.55e9
BATCHES = (1, 8, 32, 64, 128, 256, 512)
MODEL = "meta-llama/Llama-3.3-70B-Instruct"
COMMANDS = (f"trtllm-bench --model {MODEL} throughput --dataset sharegpt.jsonl --extra_llm_api_options eagle3.yaml",
            f"vllm serve {MODEL} --speculative-config '{{\"method\": \"eagle3\", \"num_speculative_tokens\": 4}}'",
            f"vllm bench serve --model {MODEL} --dataset-name sharegpt")


def argmax(p):
    return max(range(len(p)), key=p.__getitem__)


def greedy_verify(target, ref):
    """TensorRT-LLM: 'only greedy sampling is supported'; accept up to the first argmax mismatch."""
    def verify(tokens, ctx, rng):
        n = next((i for i, t in enumerate(tokens) if t != argmax(target.distribution(ctx + i))), len(tokens))
        return tokens[:n], argmax(target.distribution(ctx + n))
    return verify


def lossless_verify(target, ref):
    """vLLM: accept x with min(1, p/q), else resample max(p - q, 0); q is the lesson's draft as a distribution."""
    def verify(tokens, ctx, rng):
        for pos, tok in enumerate(tokens):
            p = target.distribution(ctx + pos)
            q = [(1 - ALIGN) * x + ALIGN * (i == argmax(p)) for i, x in enumerate(p)]
            if rng.random() >= min(1.0, p[tok] / q[tok]):
                resid = [max(a - b, 0.0) for a, b in zip(p, q)]
                return tokens[:pos], ref.sample([r / sum(resid) for r in resid], rng)
        return list(tokens), ref.sample(target.distribution(ctx + len(tokens)), rng)
    return verify


def tokens_per_call(ref, rule, k):
    target = ref.TargetModel()
    rules = {"greedy": greedy_verify, "lossless": lossless_verify, "lesson": lambda t, _r: t.verify}
    target.verify = rules[rule](target, ref)
    return round(ref.speculative_decode(N, k, random.Random(7), target, ref.DraftModel(alignment=ALIGN))
                 .tokens_per_target_call(), 3)


def lesson_emitted(p):
    m = argmax(p)
    miss = (1 - ALIGN) * sum(x for x in p if x < 0.5 * p[m])
    return [ALIGN * (i == m) + (1 - ALIGN) * x * (x >= 0.5 * p[m]) + miss * x for i, x in enumerate(p)]


def emitted_tv(ref):
    dists = [ref.softmax_from(s * 7 + 13) for s in range(1, 10_001)]
    lesson = [0.5 * sum(abs(a - b) for a, b in zip(lesson_emitted(p), p)) for p in dists]
    return {"greedy": round(statistics.mean(1 - max(p) for p in dists), 3), "lesson": round(statistics.mean(lesson), 3)}


def lossless_check(ref, trials=20_000):
    target, rng, counts, draft = ref.TargetModel(), random.Random(3), [0] * 10, ref.DraftModel(alignment=ALIGN)
    verify = lossless_verify(target, ref)
    for _ in range(trials):
        accepted, nxt = verify(draft.propose(1, 1, rng, target), 1, rng)
        counts[(accepted or [nxt])[0]] += 1
    return round(0.5 * sum(abs(c / trials - x) for c, x in zip(counts, target.distribution(1))), 3)


def speedups(tau):
    """Per batch: (fixed k = 4, as TensorRT-LLM cannot switch off; best k in 0..4, as vLLM DSD; that k)."""
    step = lambda b, n: max(WEIGHT_B / BW, 2 * PARAMS * b * n / FLOPS)  # noqa: E731
    rows = {b: {k: tau[k] * step(b, 1) / step(b, k + 1) for k in range(1, KMAX + 1)} for b in BATCHES}
    best = {b: max(ks, key=ks.get) for b, ks in rows.items()}
    return {b: (round(ks[KMAX], 2), round(max(1.0, ks[best[b]]), 2), best[b] if ks[best[b]] > 1 else 0)
            for b, ks in rows.items()}


def solve():
    ref, doc = parity.load_reference(PHASE, LESSON, "main"), parity.doc_text(PHASE, LESSON)
    tau = {r: {k: tokens_per_call(ref, r, k) for k in range(1, KMAX + 1)} for r in ("greedy", "lossless", "lesson")}
    return {"tau": tau, "tv": emitted_tv(ref),
            "lossless_tv": lossless_check(ref), "roofline": speedups(tau["greedy"]), "commands": COMMANDS,
            "claims": (bool(re.search(r"EAGLE-3 in vLLM 0\.7", doc)), re.search(r"arXiv:(\d{4})\.\d+", doc).group(1))}


def verify(result):
    r, tau, tv = result, result["tau"], result["tv"]
    return [
        practice.Check(
            "ANSWER: a 3.1x tokens-per-call lead that is greedy text, and a loss to per-batch k past batch 64",
            (tau["greedy"], tau["lossless"][KMAX], round(tau["greedy"][KMAX] / tau["lossless"][KMAX], 1), tv["greedy"],
             r["lossless_tv"]) == ({1: 1.916, 2: 2.784, 3: 3.543, 4: 4.283}, 1.378, 3.1, 0.813, 0.007)
            and r["roofline"] == {1: (4.28, 4.28, 4), 8: (4.28, 4.28, 4), 32: (4.28, 4.28, 4), 64: (4.07, 4.07, 4),
                                  128: (2.04, 2.21, 2), 256: (1.02, 1.14, 1), 512: (0.86, 1.0, 0)},
            f"tok/call greedy {tau['greedy']} vs lossless {tau['lossless']}; TV greedy {tv['greedy']}, "
            f"lossless {r['lossless_tv']}; batch -> (fixed k=4, best k, k) {r['roofline']}; H100 run: {r['commands']}",
        ),
        practice.Check(
            "FINDING: the lesson's own verify rule is not lossless either",
            (tv["lesson"], tau["lesson"][KMAX]) == (0.736, 4.756),
            f"lesson rule TV {tv['lesson']}, {tau['lesson'][KMAX]} tok/call at k=4 vs lossless {tau['lossless'][KMAX]}",
        ),
        practice.Check(
            "FINDING: the lesson's 'EAGLE-3 in vLLM 0.7' predates EAGLE-3",
            r["claims"] == (True, "2503"),
            f"doc says 'EAGLE-3 in vLLM 0.7': {r['claims'][0]}; EAGLE-3 is arXiv {r['claims'][1]}, v0.7.0 is 2025-01-27",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
