<!-- generated:start -->
# 17-infrastructure-and-production / 11-multi-region-kv-locality

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/17-infrastructure-and-production/11-multi-region-kv-locality/) · upstream spec
`phases/17-infrastructure-and-production/11-multi-region-kv-locality/docs/en.md`

```bash
uv run demo practice run 11-multi-region-kv-locality --ex 1
uv run demo explain 11-multi-region-kv-locality --ex 1
uv run pytest demos/phases/17-infrastructure-and-production/11-multi-region-kv-locality
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. At what prompt length does cross-region routing beat local-only routing,… | code | T0 | `ex01_cross_region_wins_above_214_tokens_per_request_and_the_code_has_no_prompt_length.py` |
| 2 | Your cache hit rate drops from 70% to 12%. Diagnose three possible causes and the observables… | code | T0 | `ex02_evictions_rise_under_every_cause_so_only_prefix_spread_and_hash_churn_tell_them_apart.py` |
| 3 | Design a DR manifest for a 70B AWQ-quantized model served in vLLM with 5 LoRA adapters. List… | code | T0 | `ex03_the_awq_scales_ride_inside_the_shards_and_the_files_a_weights_backup_drops_let_the_replica_start_wrong.py` |
| 4 | Argue whether Bedrock cross-region inference is "enough" for a fintech with strict TTFT SLOs.… | explain | T0 | prose, below |
| 5 | A Paris-origin request matches a prefix in us-east-1. Do you route it? Write the policy. | code | T0 | `ex05_no_and_the_reference_global_router_sends_eu_requests_to_the_us_that_a_zone_filter_keeps_home.py` |
<!-- generated:end -->

## Answers

The lesson's `code/main.py` has no prompt length, no residency field and no DR
model, so exercises 1, 2 and 5 extend it from outside and run its own
`simulate()`. The shipped cache evicts with `set.pop()`, whose order follows
string hashes, so `main.py` prints different hit rates in different processes.
Every run below swaps in a FIFO cache of the same 12 slots so the numbers repeat.
External sources were fetched on 2026-09-26: the AWS Bedrock user guide, the
`hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4` repository, vLLM `main`,
and the European Commission's EU-US data transfers page.

### 1 — cross-region wins above 214 tokens per request, and the code has no prompt length

TTFT is made linear in prompt length and calibrated at the lesson's 2K point:
a miss is 800 ms × L/2048. Every RTT is set to 75 ms.

| hit model | per request, remote hit beats local miss | fleet, GLOBAL beats REGIONAL |
|---|---:|---:|
| scaled (80 ms × L/2048) | L > 213.3 → **214 tokens** | **234 tokens** |
| fixed 80 ms | L > 396.8 → **397 tokens** | **416 tokens** |

The fleet crossover is higher than the per-request one. The reason is that a
remote hit never warms the local cache. At 214 tokens GLOBAL sends 588 requests
across regions and averages 63.1 ms, against 59.4 ms for REGIONAL. Above the
crossover GLOBAL keeps winning, which was checked at 512, 1K, 2K and 8K.

- **The shipped code cannot ask the question.** `Request` has no length.
  TTFT is the two constants, so the worst remote hit (80 + 130 = 210 ms)
  beats an 800 ms miss for any prompt.
- **The lesson's APAC example contradicts its own numbers.** It says 800 →
  80 ms of savings is "dwarfed by 440 ms round-trip". But 720 ms beats 440,
  and 440 double-counts a figure that is already a round trip. Break-even is
  626 tokens at 220 ms and 1252 at 440, both under the 2K prompt in the
  example. The skill file's ">8K tokens" is 38x too high.
- **With a working local tie-breaker, cross-region buys 0.09 ms.** The
  shipped tie-breaker is dead (exercise 2). Pin each prefix to a fixed local
  replica instead, and REGIONAL hits 88.0% at 166.4 ms with no cross-region
  calls, which beats the shipped GLOBAL's 233.6 ms. GLOBAL on the same
  affinity gets 166.3 ms and makes 648 cross-region calls to do it. (This
  check lives in exercise 2's file, next to the tie-breaker it fixes.)

### 2 — evictions rise under every cause, so only prefix spread and hash churn tell them apart

The baseline is the reference GLOBAL router at 10 slots per replica, which
hits 70.7%.

| cause | hit | replicas per prefix | distinct hashes / request | hashes seen once | evictions / request |
|---|---:|---:|---:|---:|---:|
| healthy | 70.7% | 2.8 | 0.04 | 0% | 0.26 |
| router blind (KV events lost, round-robin) | 23.8% | **10.4** | 0.04 | 0% | 0.64 |
| KV capacity cut (10 → 2 slots) | 14.5% | 3.0 | 0.04 | 0% | **0.85** |
| unstable prefix (55% carry a per-request token) | 12.4% | 1.1 | **0.60** | **93%** | 0.85 |

What confirms each cause in production:

- **Router went blind.** Prefix-to-replica spread jumps, KV-event lag
  grows, and the router's index shrinks.
- **Capacity cut.** vLLM's GPU block count falls and evictions and
  preemptions rise, while the prefix mix stays the same. Look for a recent
  `--max-model-len`, `--gpu-memory-utilization` or model change.
- **Unstable prefix.** Distinct prefix hashes approach the request count.
  Diff the first N tokens of two requests: a timestamp, request ID or user
  ID has moved to the front.

Evictions rise under all three, so they are not a diagnosis.

Three more things turned up in the simulator:

- **Blind routing floors near slots ÷ working set.** That is 10/40, measured
  at 23.8%, so routing alone cannot reach 12%. Round-robin with 5 slots
  reaches 13.1%.
- **The tie-breaker never breaks a tie.** `queue_depth` goes up by 1 with
  probability 0.4 and down by 1 every request, clamped at 0, so it is 0 on
  all 1000 requests. Every miss goes to replica 0, and 3 of the 12 replicas
  ever cache anything.
- **The shipped hit rates are not reproducible.** `PYTHONHASHSEED` 0–5 gives
  GLOBAL 86.2–89.2%. Relabelling the prefixes under 6 permutations
  reproduces the spread in-process, at 85.3–88.8%.

### 3 — the AWQ scales ride inside the shards, and the files a weights backup drops let the replica start wrong

The manifest for a 70B AWQ model (the repo named above) plus 5 LoRA adapters,
with assumed ranks of 8, 16, 16, 32 and 64:

- **Model repository, all 19 files.** `model-0000{1..9}-of-00009.safetensors`,
  `model.safetensors.index.json` and `config.json`, which holds
  `quantization_config` (4-bit, group 128, GEMM, zero point) and
  `rope_scaling` (llama3, factor 8). Then `generation_config.json`,
  `tokenizer.json`, `tokenizer_config.json` (which carries the chat
  template), `special_tokens_map.json`, `LICENSE`, `USE_POLICY.md`,
  `README.md` and `.gitattributes`.
- **Adapters, 11 entries.** `adapters/<name>/adapter_config.json` and
  `adapter_model.safetensors` for each of the 5, plus a registry of name,
  path, base model and rank.
- **Engine config, 8 settings.** `--quantization awq_marlin`,
  `--enable-lora`, `--max-loras 5`, `--max-lora-rank 64`,
  `--max-cpu-loras 5`, the 5 `--lora-modules` pairs, TP and
  `--max-model-len`, GPU memory utilization and the served model name.
- **Deployment, 9 entries.** The image digest with vLLM pinned, the
  dependency lockfile, the Dockerfile, the K8s Deployment and Service, the
  autoscaler config, the router's prefix-hash config, secret references, a
  sha256 of every file above, and the restore runbook.

That is 47 entries. A `*.safetensors` + index backup restores 10 of them.
Of the 37 it misses:

- 3 stop the replica starting: `config.json` and the two tokenizer files.
- 10 take adapters down.
- 10 change what a replica that starts actually serves.
- 14 are needed to rebuild the stack or to keep an exact, licensed copy.

- **The quantization data cannot be forgotten separately.** The index maps
  560 `scales` and 560 `qzeros` tensors into the 9 shards (39.77 GB), and
  there is no `quantize_config.json`. The lesson's "AWQ scales" are already
  inside the weights. What gets lost is the small JSON next to them.
- **A restore that starts can still be wrong.** Without
  `generation_config.json`, vLLM samples at its defaults, temperature 1.0 and
  top_p 1.0, instead of 0.6 and 0.9. Without the engine config, `max_loras`
  is 1 and `max_lora_rank` is 16, so the 5 adapters share one slot and the
  rank-32 and rank-64 adapters do not load. A drill that only checks "the
  replica is up" passes all of it.
- **"Three-file minimum" means three categories.** The same manifest is 47
  entries. The lesson's 32% statistic does not appear in its TianPan or
  BentoML (now Modular handbook) further reading, and I could not verify it.

### 4 — Bedrock CRI picks the region itself, and says peak routing costs cache writes

*Draws on "Commercial "cross-region inference" does not help here".*

**Not enough on its own.** For a fintech with strict TTFT SLOs it is an
availability and quota layer, and should run on a geographic profile. These
are the behaviours in the Bedrock user guide that decide it:

1. **Bedrock chooses the region, not you.** "Amazon Bedrock automatically
   selects a commercial AWS Region within that geography to process your
   inference request." The request carries no latency target and no region
   pin, so a router cannot aim at a warm cache. Exercise 1's 80-vs-800 ms is
   exactly what gets left to chance.
2. **Rerouting under load costs cache hits.** The prompt caching page says
   CRI "automatically selects the optimal AWS Region within your geography …
   At times of high demand, these optimizations may lead to increased cache
   writes". Implicit caching "is best effort". So cache misses rise at peak,
   which is when the SLO is tested.
3. **No reserved capacity.** "Inference profiles currently don't support
   Provisioned Throughput." The one Bedrock product that reserves capacity
   cannot sit behind CRI.
4. **The fast path degrades silently.** Latency-optimized inference is a
   preview. It is offered through CRI only for a few models and regions, and
   past its quota "we will attempt to serve the request with Standard
   latency".
5. **You can audit it, but only after the fact.** CloudTrail in the source
   region records `additionalEventData.inferenceRegion` for every request.
   Use it to compute TTFT by serving region.
6. **Residency depends on the profile.** A geographic profile stays inside
   US, EU or APAC, but "input prompts and output results might move outside
   of your source Region", and abuse-detection storage sits in the
   destination region. A global profile is about 10% cheaper and routes
   worldwide, which a regulated fintech should not accept. There is "no
   additional routing cost", and pricing follows the source region.

**The architecture that meets the SLO.** Keep CRI with a geographic profile
as the overflow layer. Put an application router in front that you own:
reserved capacity in the home region, and cache-aware routing over
self-hosted replicas where TTFT is contractual. Alert on
`cacheReadInputTokens` per request and TTFT by `inferenceRegion`.

The lesson's claim that CRI routes "during capacity pressure" is not how the
user guide describes it. The guide describes region selection on every
request. I did not check the launch blog for a source-region preference.

### 5 — no, and the reference GLOBAL router sends EU requests to the US that a zone filter keeps home

**No.** The policy:

1. The zone comes from the tenant's data agreement, not the IP.
2. Only replicas inside the zone are candidates. This is a hard filter
   applied before scoring, with no TTFT override.
3. Inside the zone, take min(prefill + RTT). If no replica there has the
   prefix, prefill locally.
4. For a shared prefix with no personal data, such as a system prompt or a
   tool schema, copy the prefix *text* into the zone and warm it there.
   Never move the request.
5. Log the serving region on every request and alarm on any out-of-zone
   serve. Fail over inside the zone, or fail closed.

The latency case pulls the other way. From us-east-1 the Paris request costs
80 + 75 = 155 ms, against an 800 ms local miss.

- **The reference router crosses the border.** GLOBAL serves 175 of its 312
  EU-origin requests in the US, and 191 US requests in eu-west-1.
- **The zone filter stops it, at a cost the simulator exaggerates.** With
  the filter in front of the same rule, out-of-zone serves drop to 0. EU mean
  TTFT goes from 264.2 to 576.2 ms, because eu-west-1 has one caching
  replica.
- **With a working tie-breaker, residency is cheap.** Pin prefixes to
  eu-west-1's 4 replicas and it hits 87.2% at 172.3 ms. That is faster than
  the router that broke residency.
- **The reference has nothing to filter on.** `Request` has no tenant, zone
  or data class, and all 40 prefixes are requested from all three regions.
- **The lesson overstates the law.** It says the route has "violated GDPR
  regardless", about "EU customer PHI". PHI is a HIPAA term. GDPR Chapter V
  allows transfers under an adequacy decision, and the Commission adopted
  one for the EU-US Data Privacy Framework on 10 July 2023. The binding rule
  is usually the tenant's residency contract, and that contract is what the
  filter enforces.
