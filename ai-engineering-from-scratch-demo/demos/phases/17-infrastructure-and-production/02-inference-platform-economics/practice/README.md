<!-- generated:start -->
# 17-infrastructure-and-production / 02-inference-platform-economics

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/17-infrastructure-and-production/02-inference-platform-economics/) · upstream spec
`phases/17-infrastructure-and-production/02-inference-platform-economics/docs/en.md`

```bash
uv run demo practice run 02-inference-platform-economics --ex 1
uv run demo explain 02-inference-platform-economics --ex 1
uv run pytest demos/phases/17-infrastructure-and-production/02-inference-platform-economics
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. At what sustained utilization does Baseten (per-minute) beat Fireworks (p… | code | T0 | `ex01_baseten_wins_above_68_percent_not_30_and_at_the_list_h100_price_above_13.py` |
| 2 | Your product serves image generation plus chat plus speech-to-text. Pick platforms for each m… | code | T0 | `ex02_route_by_modality_behind_one_gateway_because_chat_on_the_multimodal_vendor_costs_34x.py` |
| 3 | Fireworks raises prices by $1/hr on your primary model. Model the blended cost impact if 40%… | code | T0 | `ex03_the_raise_and_the_batch_discount_bill_different_products_so_batch_can_raise_the_bill.py` |
| 4 | A regulated customer requires SOC 2 Type II + HIPAA + dedicated GPUs. Which three platforms a… | code | T0 | `ex04_baseten_fireworks_and_together_qualify_and_together_wins_while_the_code_picks_modal.py` |
| 5 | Compare cost per 1,000 predictions for Llama 3.1 70B on Fireworks serverless, Together on-dem… | code | T0 | `ex05_together_is_cheapest_at_both_volumes_and_only_baseten_changes_by_1000x.py` |
<!-- generated:end -->

## Answers

Vendor pages were fetched on 2026-09-26: the Baseten, Fireworks, Together,
Modal and Replicate pricing pages, Fireworks' batch guide and compliance post,
Together's SOC 2 post, Anyscale's certifications page, and Replicate's Whisper
model page. The Fireworks price-change history comes from usagepricing.com.
Every exercise runs the lesson's own `code/main.py` comparator. The fetched
prices are held as constants, so the checks stay offline.

### 1 — Baseten wins above 68% utilization, not 30%, and at the list H100 price above 13%

**The code crosses at 67.9%.** Baseten bills all 1440 minutes and Fireworks
bills tokens, so u* = p_min / (tpm × p_tok) = 0.55 / (0.9 × 0.90) = 0.679.
At that point both cost $792.00/day.

| utilization | Fireworks $/day | Baseten $/day |
|---:|---:|---:|
| 30% (the lesson's rule) | 349.92 | 792.00 |
| 67.9% (crossover) | 792.00 | 792.00 |
| 100% | 1166.40 | 792.00 |

The printed table jumps from 50% to 75%, so it never shows the crossover.
The module's closing line says "~60-70%". The lesson text and the shipped
skill both say "~30%".

**At Baseten's published H100 rate the crossover is 13.4%.** The code
charges $0.55/min, which is $33/hr. Baseten lists an H100 at $0.10833/min
($6.50/hr), 5.1x less. For 30% to hold at the code's prices, one GPU would
need 2.04M tok/min. u* scales as 1/throughput, so the assumed 900k tok/min
matters as much as the prices do.

**"One H100" is not in the model.** `Vendor` has no GPU or precision field.
A 70B model at 2 bytes per parameter is 140 GB against the H100's 80 GB. At
FP8 it fits, with 10 GB left for KV cache.

**The code's Modal beats Fireworks from 2.8% utilization.** Modal is billed
only for saturated minutes above a 60-minute floor. That makes it the
cheapest vendor in Scenario B, at $60 against $88. The rule of thumb groups
Modal with Baseten, but the model prices the two on opposite assumptions.

### 2 — route by modality behind one gateway, because chat on the multimodal vendor costs 34x

| modality | platform | billing unit | $/day for the workload |
|---|---|---|---:|
| chat (10,000 chats, 2M tokens) | Together (fallback Fireworks, $1.80) | per token | 1.76 |
| images (1,000) | Replicate | per prediction | 6.00 |
| speech-to-text (1,000) | Replicate `openai/whisper` (Groq when real-time) | GPU-seconds, ~$0.0027/run | 2.70 |

The pattern is Lesson 19's **AI gateway**: "one process with one API
(typically OpenAI-compatible) that fans out to providers". Here it routes by
modality and normalizes every bill to a cost per request.

**Chat on the multimodal vendor costs 34x.** At the code's $0.006 per
prediction, the same chats cost $60.00 on Replicate against $1.76 on
Together. The gateway lets image traffic stay on Replicate without dragging
chat along.

**The comparator can price only the chat leg.** All 6 `VENDORS` serve a
Llama 70B, `Vendor` has no modality field, and `cost_per_day` takes only
tokens and predictions. Replicate's price is flat, so a 10-token call costs
the same as a 1M-token one. The module has no image price, so the image leg
uses that same $0.006.

### 3 — the raise and the batch discount bill different products, so batch can raise the bill

The $1/hr is an hourly GPU price and batch is 50% off *serverless per-token*
prices. The two changes land on different products. At the lesson's Scenario
B volume of 100M tokens/day:

| deployment | before | after raise | + 40% to batch |
|---|---:|---:|---:|
| one dedicated Fireworks H100 ($7 → $8/hr) | $168 | $192 (+14.3%) | $210 (+25%) |
| all serverless ($0.90/M) | $90 | $90 (raise does not apply) | $72 (−20%) |

- **Batch only saves money if it frees whole GPUs.** The reserved GPU stays
  billed all day, so the batch tokens are added on top.
- **Batch is dearer than a busy dedicated GPU.** At saturation the H100 costs
  $8 / 54M tokens = $0.148/M. Batch at $0.45/M is 3x that, so it beats the
  GPU only below 32.9% utilization.
- **The one-price reading gives −18.4%.** If the raise and the discount hit
  the same rate, $1/hr is only $0.0185/M on a saturated GPU.

**The raise took effect September 1, not May 1.** Fireworks' pricing page
shows an H100 at $8.00/hr. usagepricing.com records the change as
H100/H200 $7 → $8 and B200 $10 → $13, announced 2026-08-12 and effective
2026-09-01, on on-demand GPUs only. The lesson dates it May 1, 2026, and
nothing found supports that date. The module cannot express either change:
"batch" appears only in a notes string, and Fireworks has no per-minute price.

### 4 — Baseten, Fireworks and Together qualify, and Together wins, while the code picks Modal

| vendor | SOC 2 Type II | HIPAA | reserved GPU | list H100 $/day |
|---|---|---|---|---:|
| **Together** | yes | yes | yes | **95.76** |
| Baseten | yes | yes | yes | 156.00 |
| Fireworks | yes | yes | yes | 192.00 |
| Modal | yes | Enterprise plan only | no, serverless | — |
| Anyscale | yes | not claimed | yes | — |

**The lesson's own data gives a different answer.** In `VENDORS` the
dedicated vendors are Baseten, Modal and Anyscale, and only Baseten's notes
mention SOC 2 or HIPAA. A reserved day there costs:

- Modal $691.20, the code's FinOps winner
- Baseten $792.00
- Anyscale $864.00

The shipped skill refuses Modal for regulated work and says "Suggest
Baseten", so the code and the skill disagree. The code's hourly rates are
also 5.1x Baseten's list price and 7.3x Modal's, and `Vendor` has no GPU type.

### 5 — Together is cheapest at both volumes, and only Baseten changes, by 1000x

A prediction is 200 output tokens, the Scenario A ratio.

| $ per 1,000 predictions | 10/day | 10,000/day |
|---|---:|---:|
| **Together** | **0.176** | **0.176** |
| Fireworks | 0.18 | 0.18 |
| Replicate | 6.00 | 6.00 |
| Baseten dedicated | 79,200 | 79.20 |

**The same vendor wins at both volumes.** Per-token and per-prediction
prices are linear, so their cost per 1,000 does not change with volume. Only
the reserved GPU amortizes. Baseten passes Together at 4.5M predictions/day,
or 886k/day at its listed H100 rate.

**"50-70% cheaper than Replicate" holds only for predictions of 2,045-3,409
tokens.** Together ties Replicate's flat $0.006 at 6,818 tokens, and at 200
tokens Together is 97.1% cheaper. The ranking depends on prediction length,
which the exercise does not give.

**Nothing in the module is Llama 3.1.** The models are "Llama 70B",
"Custom Llama 70B" and "Llama 70B RayTurbo". Replicate's pricing page bills
its example LLM, deepseek-r1, per token, and no Llama 3.1 70B price was
found, so the $0.006 per prediction is the lesson's figure and was not
verified.
