# Multi-Head Attention — pure stdlib vs `torch.nn.MultiheadAttention`

**Lesson:** [07-transformers-deep-dive / 03-multi-head-attention](https://yennj12.js.org/ai-engineering-from-scratch/lesson.html?path=phases%2F07-transformers-deep-dive%2F03-multi-head-attention)
**Tier:** T1 (torch-CPU, no download) · **Install:** `uv sync --extra llm`

## What it proves

The lesson's file opens with *"No numpy, no torch. A tiny Matrix class carries
the ops we need."* This demo loads the lesson's weights into
`torch.nn.MultiheadAttention` and compares **outputs and per-head attention
weights** — the per-head check matters, because a botched head split still
produces a plausible-looking output.

In float64 the largest disagreement is `2.8e-16`. Switching torch to float32
moves it to `1.6e-07`: that gap is precision, not a different algorithm.

Also covered: one head against `F.scaled_dot_product_attention`, and the
grouped-query variant against `enable_gqa=True`.

## Run

```bash
uv run demo run phases/07-transformers-deep-dive/03-multi-head-attention
uv run python run.py --explain     # works with nothing installed
```

## Expected output

```
  ok   MHA output vs torch (float64)   n=48  max|d|=2.78e-16  atol=1e-12
  ok   head 0 attention weights        n=36  max|d|=1.11e-16  atol=1e-12
  ...
precision, not algorithm:  float32 deviates by 1.55e-07 absolute
KV cache elements:  MHA 96  vs  GQA 48  (2x smaller)
```
