<!-- generated:start -->
# 17-infrastructure-and-production / 12-edge-inference

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/17-infrastructure-and-production/12-edge-inference/) · upstream spec
`phases/17-infrastructure-and-production/12-edge-inference/docs/en.md`

```bash
uv run demo practice run 12-edge-inference --ex 1
uv run demo explain 12-edge-inference --ex 1
uv run pytest demos/phases/17-infrastructure-and-production/12-edge-inference
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. For a 7B model in Q4 on a Snapdragon 8 Gen 3 (~77 GB/s bandwidth), comput… | code | T0 | `ex01_the_ceiling_is_22_toks_so_6_to_8_is_a_third_of_it_and_the_code_has_one_row_above_its_own_ceiling.py` |
| 2 | WebGPU on Android requires Chrome v121+. Design a fallback for older browsers — server-side v… | code | T0 | `ex02_a_chrome_version_check_breaks_three_of_nine_clients_and_the_same_api_still_needs_a_model_id_map.py` |
| 3 | Your iOS app needs 4K-context streaming. Which model/format combination lets you stay under 4… | code | T0 | `ex03_llama_32_3b_at_int4_fits_in_23_gb_and_no_8b_format_fits_4_gb_with_its_4k_cache.py` |
| 4 | Jetson AGX Orin runs gpt-oss-20b at 40 tok/s. Jetson Nano fits only a 3B. If your product tar… | code | T0 | `ex04_unify_the_api_not_the_model_because_gpt_oss_20b_is_138_gb_and_the_nano_runs_an_8b.py` |
| 5 | Argue whether "WebLLM is production-ready in 2026." Cite the coverage, performance, and the F… | code | T0 | `ex05_webllm_is_ready_on_the_desktop_and_a_progressive_enhancement_on_the_phone.py` |
<!-- generated:end -->

## Answers

### 1 — the ceiling is 22 tok/s, so 6-8 is a third of it, and the code has one row above its own ceiling

**The ceiling is 22.0 tok/s, so 6-8 tok/s is 27-36% of it. The runtime is not
efficient.** The reference `ceiling()` on its own 77 GB/s Snapdragon 8 Gen 3
target, fed the lesson's "3.5 GB" for a 7B Q4 model, gives 45 ms a token. The
verdict holds at any reasonable size:

| 7B Q4 sized as | GB | ceiling (tok/s) | 6-8 tok/s is |
|---|---:|---:|---:|
| the lesson's 4.0 bits a weight | 3.50 | 22.0 | 27-36% |
| Q4_0's 4.5 bits a weight | 3.94 | 19.6 | 31-41% |
| the code's 8B file | 4.70 | 16.4 | 37-49% |

The missing throughput is not compute. 2 x 7e9 FLOPs a token at the 45 TOPS
the lesson quotes for Hexagon is a 3214 tok/s ceiling, 146x above the
bandwidth ceiling. The gap is in the runtime.

Three things in the lesson's numbers do not hold up:

- **One row is above its own ceiling.** Jetson AGX Orin is listed at 45 tok/s
  against a 43.6 tok/s ceiling, 103%, which a bandwidth-bound decode cannot
  reach. `efficiency()` prints it without comment. Every other row is 24-98%.
- **The iPhone gets two speeds.** The Problem section says 3 tok/s on an
  iPhone 16 Pro, and the code says 8 on the A18. Against the 12.8 tok/s
  ceiling those are 24% and 63%.
- **BF16 is 2.7 GB too big.** The quantization table scales from the Q4 size
  (18.8 = 4 x 4.7). Llama 3.1 8B has 8.03B parameters, so BF16 is 16.06 GB
  and its iPhone ceiling is 3.7 tok/s, not 3.2.

### 2 — a Chrome version check breaks three of nine clients, and the same API still needs a model-id map

**Route on capability, not on version, and fall back per request.** The page
asks for a WebGPU adapter. It checks the model's `required_features` and its
`vram_required_MB` against the device's budget. If any check fails, or if
engine load throws, it sends the same chat-completions body to a server; the
real server is `mlc_llm serve`. Over nine labelled client profiles, capability
detection sends 3 local and 6 to the server, and every route loads.

**The exercise's own rule breaks three of the nine.** Three clients pass
"Android Chrome >= 121" and then fail to load:

- Chrome 121 on Android 11. Chrome 121 enables WebGPU only on Android 12+
  with Qualcomm or ARM GPUs.
- A Mistral q4f16 model on a GPU without `shader-f16`.
- An 8B model (5001 MB in WebLLM) on a 4 GB budget.

The same rule sends Safari iOS 26, which does ship WebGPU, to the server. A
catch on engine load rescues the three broken clients, but only after each has
paid for a failed model download.

**"The same API" still needs a map.** The local and server bodies differ only
in `model`. WebLLM names the quantized build
(`Llama-3.2-3B-Instruct-q4f16_1-MLC`), and a server names the model. The
reference code has no routing or fallback. Its only browser fact is the note
string "mobile browser Chrome 121+".

Sources checked 2026-09-26: Chrome's "New in WebGPU 121"; the gpuweb wiki's
Implementation Status page (Firefox Android behind a flag, Safari iOS 26);
WebLLM v0.2.84 `src/config.ts`. The client profiles' memory budgets and f16
support are assumptions.

### 3 — Llama 3.2 3B at INT4 fits in 2.3 GB, and no 8B format fits 4 GB with its 4K cache

**Use Llama 3.2 3B with Core ML INT4 weights and an FP16 KV cache: 2.28 GB,
leaving 1.72 GB.** The weights are 1.81 GB and a 4096-token cache is 0.47 GB
(112 KiB a token). WebLLM lists the same model's q4f16 build at 2263.69 MB
for a 4K context, within 1%. The grid covers four models, three weight
formats and two KV formats, plus the code's 3.6 GB Q3 8B:

| model | format | FP16 KV | INT8 KV |
|---|---|---:|---:|
| Llama 3.2 1B | INT4 | 0.83 | 0.77 |
| Llama 3.2 3B | INT4 | **2.28** | 2.04 |
| Llama 3.2 3B | INT8 | 3.68 | 3.45 |
| Phi-3.5-mini | INT4 | 3.76 | 2.95 |
| Llama 3.1 8B | code's Q3 | 4.14 | 3.87 |
| Llama 3.1 8B | INT4 | 5.05 | 4.79 |

13 of the 26 combinations fit in 4 GB. With an FP16 cache, only the 1B and
this 3B leave more than 1 GB for the app.

- **The only 8B that fits is Q3 weights with an INT8 cache, and it leaves
  0.13 GB.** INT4 weights alone are 4.52 GB.
- **Phi-3.5-mini's cache is 3.4x Llama 3.2 3B's, at a similar size.** It has
  32 KV heads and no grouped-query attention, so it needs 384 KiB a token:
  1.61 GB at 4K, 43% of its footprint. WebLLM's 4K and 1K builds of it differ
  by exactly 1152.00 MB, which is 3072 tokens x 384 KiB.
- **The lesson's 32K trap undercounts the cache by half.** Llama 3.1 8B at
  32K needs 4.29 GB of FP16 KV, not "2 GB"; 2.15 GB is the INT8 figure.
- **The code's ceiling ignores the cache.** `ceiling()` divides bandwidth by
  the weights alone. Reading the full 4K cache adds 26% per token, which
  lowers the A18's ceiling from 33.2 to 26.4 tok/s.

### 4 — unify the API, not the model, because gpt-oss-20b is 13.8 GB and the Nano runs an 8B

**Unify on one OpenAI-compatible API and one engine build, and choose the
model per device from a manifest.** gpt-oss-20b's checkpoint is 13.76 GB (the
three safetensors files on its Hugging Face repo), so it does not fit the Orin
Nano Super's 8 GB. A 3B at INT4 fits both devices. Its ceiling is 56.5 tok/s
on the Nano (102 GB/s) and 113.5 on AGX Orin (205).

That leaves two options, and both keep the same endpoint:

- **One model everywhere.** Run the 3B on both devices and evaluate it once.
- **The largest fit per device.** Run gpt-oss-20b on AGX Orin and an 8B INT4
  on the Nano, and evaluate twice.

Either way, what the two devices share is the endpoint, the container and the
manifest.

**The reference math cannot produce the lesson's 40 tok/s. The MoE active
bytes can.** `ceiling()` over the whole checkpoint gives 14.9 tok/s on AGX
Orin. I counted what one token actually reads, from config.json: attention,
4 of 32 experts and the lm_head. That comes to 3.60B parameters and 3.70 GB,
for a 55.4 tok/s ceiling, so 40 tok/s is 72% of it. The same count gives
20.9B parameters and 13.74 GB in total, which matches the model card's "21B
parameters with 3.6B active" and the file sizes.

**The Nano is not limited to a 3B.** NVIDIA's Orin Nano Super blog, with INT4
weights on the MLC API, measures:

| model | tok/s | share of ceiling |
|---|---:|---:|
| Llama 3.1 8B | 19.14 | 85% of 22.6 |
| Llama 3.2 3B | 43.07 | 76% of 56.5 |
| Gemma 2 9B | 9.21 | — |

**The two models share no tokenizer, so the layer that unifies them is the
chat API, not the prompt.** gpt-oss has a 201,088-token vocabulary and the
harmony format. Llama 3.2 has 128,256 tokens and its own template. Token
budgets, prompt caching and stop sequences therefore have to live per model,
behind `/v1/chat/completions`.

### 5 — WebLLM is ready on the desktop and a progressive enhancement on the phone

**WebLLM is production-ready on the desktop. On the phone it is ready only as
a progressive enhancement over a server fallback.** The case rests on three
things the exercise asks for:

- **Performance on the desktop.** On an M3 Max, WebLLM decodes Llama 3.1 8B
  Q4 at 41 tok/s. That is 74.5% of the 55 tok/s native runtime, inside the
  lesson's "70-80% of native".
- **Coverage on mobile.** The lesson's ~70-75% coverage leaves 25-30% of
  mobile sessions with no WebGPU at all.
- **The Firefox Android gap.** On 2026-09-26, the gpuweb wiki's Implementation
  Status page lists Firefox Android as "behind a flag" and says "Mozilla
  expects to do work on Android in 2026". Safari iOS ships WebGPU in 26.
  Firefox Android is part of the uncovered share, but the lesson gives no
  per-browser split, so it cannot say how large a part.

A launch that depends on the local path fails those sessions. A launch that
routes them to the same API on a server (Exercise 2) does not.

**On the phone, the browser is not the bottleneck.** The code gives the
Pixel 9 (a Tensor G4 phone) the Snapdragon 8 Gen 3's 77 GB/s. In the browser
it runs at 6 tok/s against the native row's 7. That is 86% of native, a
smaller browser penalty than the desktop's 74.5%. The phone is slow because
of bandwidth and the runtime: 6 tok/s is 37% of its 16.4 ceiling and a
seventh of the desktop's rate.

The memory numbers point the same way. The 8B q4f16 build needs 5001 MB in
WebLLM, so a phone gets the 3B (2264 MB) or the 1B (879 MB), not the model the
desktop figure measures. The lesson's "17.6k GitHub stars" is also dated: the
repo shows 19,192 today.

**The code's efficiency column is not the browser penalty.** The row's note
says "browser penalty ~25%", and 41 / 55 is indeed 25% below native. But
`efficiency()` prints 48% for that row, because it divides by the bandwidth
ceiling. That figure folds in the native runtime's own shortfall (native M3
Max runs at 65% of its ceiling).
