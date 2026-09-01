# lora_exp — the simplest possible LoRA fine-tune

A LoRA fine-tune of **distilgpt2** (82M params) that runs on a laptop in
about 10 seconds and trains **0.18%** of the model's parameters.

Follows [Fine-Tuning with LoRA & QLoRA](https://yennj12.js.org/ai-engineering-from-scratch/lesson.html?path=phases/11-llm-engineering/08-fine-tuning-lora)
(*AI Engineering from Scratch*, phase 11 lesson 8). The lesson builds LoRA by
hand on a toy MLP; this project does the same thing on a real Hugging Face
model, once with PEFT and once from scratch, so you can see they match.

## Setup

```bash
cd hugging_face_exp/lora_exp
uv sync
```

`uv` reads `.python-version` (3.12) and installs torch, transformers and peft
into `.venv/`. First run also downloads distilgpt2 (~350MB) from the Hub.

## Run

```bash
uv run python -m src.train_lora        # the actual fine-tune
uv run python -m src.lora_from_scratch # the same math in ~40 lines, no PEFT
```

## What `train_lora.py` does

Teaches the model one narrow behaviour — answer `Q:` prompts with a one-line
`Fact:` statement about LoRA — from 8 hand-written examples in `src/data.py`.

| Step | What you see |
|---|---|
| Before | `Q: What is LoRA?` → *"LoRA is a very simple, simple, simple, simple…"* |
| Adapter attached | `trainable params: 147,456 \|\| all params: 82,060,032 \|\| trainable%: 0.1797` |
| Training | loss 5.16 → 0.51 over 60 epochs, ~7s on Apple MPS |
| After | `Q: What is LoRA?` → *"Fact: LoRA freezes the base weights and trains two small matrices A and B."* |
| Held-out prompt | `Q: What does a low-rank matrix do?` → *"Fact: low-rank sets the weights of the edges…"* — nonsense content, but the **style transferred**, which is what 8 examples can buy you |
| Saved | `out/adapter/` — **584 KB**, vs 313 MB for the base model |
| Reload + merge | adapter reloaded onto a fresh base gives identical output; `merge_and_unload()` folds it into the weights so inference costs nothing extra |

### The config

```python
LoraConfig(
    r=8,                        # bottleneck width -> how many params you train
    lora_alpha=16,              # scaling = alpha / r = 2.0
    lora_dropout=0.05,
    target_modules=["c_attn"],  # GPT-2's fused q/k/v projection
    task_type="CAUSAL_LM",
)
```

`c_attn` is GPT-2's single fused projection for query, key and value — the
equivalent of targeting `q_proj + k_proj + v_proj` on a Llama-style model. The
lesson's table puts that at "best for attention"; `q_proj + v_proj` only is the
usual cheaper default.

`r=8` is the lesson's recommendation for single-domain Q&A. Try `r=2` and
`r=32` to watch the parameter count and the loss curve move.

The learning rate is `2e-3` — roughly 100x what you would use for full
fine-tuning. That is normal for LoRA: you are training a small, freshly
initialised set of matrices, not nudging pretrained weights.

## What `lora_from_scratch.py` does

The lesson's `LoRALayer` / `LinearWithLoRA` / `inject_lora`, applied to
distilgpt2 instead of a toy `nn.Sequential`:

```
base model:  81,912,576 params, 81,912,576 trainable (100.00%)
after LoRA:  82,060,032 params, 147,456 trainable (0.18%)
injected into 6 layers: transformer.h.0.attn.c_attn ... transformer.h.5.attn.c_attn
```

Same 147,456 trainable parameters PEFT reports — because it is the same
arithmetic: `y = W₀x + (x @ A @ B) · (α/r)`, with `W₀` frozen and `B`
zero-initialised so an untrained adapter is exactly a no-op.

One wrinkle worth knowing: GPT-2 uses `transformers.pytorch_utils.Conv1D`
rather than `nn.Linear`, and its weight is stored `[in, out]` instead of
`[out, in]`. `shape_of()` handles both. PEFT hits the same thing and prints a
`fan_in_fan_out` warning while auto-correcting for it — harmless.

## Not covered here

QLoRA (4-bit base via `bitsandbytes`) is in the lesson but needs CUDA, so it is
left out. On a Mac the base model already fits in memory; quantization buys you
nothing here.

## Layout

```
lora_exp/
├── pyproject.toml         # torch, transformers, peft
├── .python-version        # 3.12
└── src/
    ├── data.py            # 8 training examples + 2 held-out prompts
    ├── train_lora.py      # PEFT LoRA: train, save, reload, merge
    └── lora_from_scratch.py
```
