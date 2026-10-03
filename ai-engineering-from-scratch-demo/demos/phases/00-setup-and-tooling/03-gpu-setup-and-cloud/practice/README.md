<!-- generated:start -->
# 00-setup-and-tooling / 03-gpu-setup-and-cloud

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/00-setup-and-tooling/03-gpu-setup-and-cloud/) · upstream spec
`phases/00-setup-and-tooling/03-gpu-setup-and-cloud/docs/en.md`

```bash
uv run demo practice run 03-gpu-setup-and-cloud --ex 1
uv run demo explain 03-gpu-setup-and-cloud --ex 1
uv run pytest demos/phases/00-setup-and-tooling/03-gpu-setup-and-cloud
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run the benchmark above and compare CPU vs GPU times | code | T0 | `ex01_one_cold_timed_matmul_and_a_slower_gpu_prints_0x.py` |
| 2 | If you don't have a GPU, run it on Google Colab and compare | code | T0 | `ex02_on_a_colab_t4_the_benchmark_never_touches_the_tensor_cores.py` |
| 3 | Check how much GPU memory you have and estimate the largest model you can fit (rule of thumb:… | code | T0 | `ex03_rounding_promises_an_8b_model_a_t4_cannot_hold.py` |
<!-- generated:end -->

## Answers

The lesson's code is one function, `gpu_check.check_gpu`: it imports torch,
prints the device, times one matmul on each side and estimates the largest
fp16 model. This runner has neither torch nor a GPU, so every exercise drives
that function against a fake `torch` (and, where timing matters, a scripted
clock). The fakes decide what the hardware *says*; everything printed and
every number compared is the lesson's own code. All three are **T0**.

### 1 — one cold timed matmul each way, and a slower GPU prints "0x"

**ANSWER: on a CPU-only machine the script prints no time at all** — it
returns at `cuda.is_available()` before the benchmark, CPU half included. With
a GPU it times one 4000x4000 fp32 matmul per device (128 GFLOP), the GPU window
`sync, time, matmul, sync, time` with the host-to-device copies outside it;
2.000 s against 0.020 s prints `Speedup: 100x`.

| scripted GPU time (CPU = 2.000 s) | printed |
|---|---|
| 0.020 s | `Speedup: 100x` |
| 5.000 s (2.5x slower) | `Speedup: 0x` |
| 0 ticks | `ZeroDivisionError` |

**FINDING: both timings are cold** — 0 CUDA matmuls run before the timed one,
2 matmuls in total, `time.time` read 4 times: no warm-up, no repeat, wall clock.

**FINDING: the doc's benchmark is not the script's.** `docs/en.md` uses
`size = 5000` (250 GFLOP), `gpu_check.py` 4000 (128 GFLOP): 1.95x less work.

### 2 — on a Colab T4 the benchmark never touches the Tensor Cores

The fake reports a Colab T4 (`Tesla T4`, 15102 MiB, cc 7.5); the GPU matmul is
scripted at the T4's fp32 floor from NVIDIA's datasheet.

**ANSWER:** the script prints `Tesla T4`, `15.8 GB`, cc `7.5`, `~8B`. The
128 GFLOP matmul cannot beat **15.8 ms** at 8.1 TFLOPS fp32 (a placeholder CPU
second prints `Speedup: 63x`).

**FINDING: it is an fp32 benchmark on a card without TF32.** Both `randn`
calls pass no dtype, and TF32 needs cc 8.0. At 65 TFLOPS fp16 the Tensor-Core
floor is **1.97 ms, 8.0x lower** — the Key Terms' "4-8x faster" hardware sits
idle, so the Colab comparison understates the GPU by up to that much.

**CONTROL:** on Colab's default CPU runtime the script prints "use Google
Colab (free)" and runs nothing; switching the runtime to T4 is required.

### 3 — the estimate rounds up, promising an 8B model a T4 cannot hold

Here the script prints `PyTorch not installed`, so the host has no answer.
On a fake `torch` per card:

| card | memory printed | `bytes / 2e9` | printed | fits? |
|---|---:|---:|---:|---|
| Tesla T4 | 15.8 GB | 7.92 | ~8B | Llama-3-8B needs 16.06 GB: **no** |
| RTX 4090 | 25.8 GB | 12.88 | ~13B | Llama-2-13B needs 26.03 GB: **no** |
| A100 40GB | 42.5 GB | 21.25 | ~21B | yes |
| H100 80GB | 85.5 GB | 42.76 | ~43B | a 43B model needs 86 GB: **no** |

**ANSWER:** the printed figure is exactly `round(total_memory / 2e9)`.

**FINDING: rounding to nearest overstates the fit on 3 of 4 cards**, before
counting the CUDA context, activations or KV cache. Only a rounded-*down*
figure is safe.

**FINDING: 2 bytes/param is inference weights.** Mixed-precision Adam holds
16 bytes/param (ZeRO), so the T4 trains at most **0.99B — 8.1x less** than it
prints.
