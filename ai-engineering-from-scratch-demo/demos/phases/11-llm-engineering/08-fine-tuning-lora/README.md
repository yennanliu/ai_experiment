# LoRA & QLoRA — the lesson's `LoRALayer` vs `peft`

**Lesson:** [11-llm-engineering / 08-fine-tuning-lora](https://yennj12.js.org/ai-engineering-from-scratch/lesson.html?path=phases%2F11-llm-engineering%2F08-fine-tuning-lora)
**Tier:** T1 (torch-CPU, no download) · **Install:** `uv sync --extra llm`

## What it proves

The lesson's hand-written `LoRALayer` and `peft`'s `lora.Linear` are the same
computation. Checked across four rank/alpha pairs: identical forward pass,
identical trainable-parameter counts (18,512 of 417,882 — 4.43%), and identical
weights after merging the adapter back into the base layer.

It also names one place the lesson simplifies, rather than glossing it: the
lesson's `quantize_to_nf4` is block-wise **symmetric int4-range** quantisation,
not the NF4 codebook `bitsandbytes` implements. The round-trip error that costs
is measured and printed — 10.35% of weight RMS — not asserted away.

## Run

```bash
uv run demo run phases/11-llm-engineering/08-fine-tuning-lora
uv run python run.py --explain     # works with nothing installed
```

## Expected output

```
  ok   fresh adapter changes nothing    n=160     max|d|=0.00e+00  atol=0e+00
  ok   forward, r=8 alpha=16            n=160     max|d|=5.96e-08  atol=1e-06
  ok   merged weights, layer 0          n=131072  max|d|=0.00e+00  atol=1e-06
  ...
18,512 trainable of 417,882 parameters (4.43%) -- the whole point of LoRA.
```
