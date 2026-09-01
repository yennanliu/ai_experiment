"""Simplest LoRA fine-tune: distilgpt2 + Hugging Face PEFT.

Runs on a laptop CPU (or Apple MPS) in about a minute. What it shows:

  1. how many parameters LoRA actually trains (spoiler: ~0.1%)
  2. the model's output before and after training
  3. an adapter saved to disk that is a few hundred KB, not 300MB

Run:  uv run python -m src.train_lora
"""

import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import EVAL_PROMPTS, EXAMPLES

MODEL_NAME = "distilgpt2"  # 82M params, the smallest GPT-2 that writes real text
ADAPTER_DIR = Path(__file__).resolve().parent.parent / "out" / "adapter"

EPOCHS = 60
LR = 2e-3  # LoRA tolerates a much higher LR than full fine-tuning
MAX_LEN = 64


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def generate(model, tokenizer, prompt: str, device: torch.device) -> str:
    batch = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **batch,
            max_new_tokens=24,
            do_sample=False,  # greedy, so before/after is a fair comparison
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text[len(prompt):].strip().replace("\n", " ")


def main() -> None:
    torch.manual_seed(0)
    device = pick_device()
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 ships without a pad token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

    print("\n--- before training ---")
    model.eval()
    for prompt in EVAL_PROMPTS:
        print(f"  {prompt.splitlines()[0]}\n    -> {generate(model, tokenizer, prompt, device)}")

    # --- attach the adapter -------------------------------------------------
    # c_attn is GPT-2's fused q/k/v projection, the equivalent of targeting
    # q_proj + k_proj + v_proj on a Llama-style model.
    config = LoraConfig(
        r=8,
        lora_alpha=16,  # scaling = alpha / r = 2.0
        lora_dropout=0.05,
        target_modules=["c_attn"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    print("\n--- parameter budget ---")
    model.print_trainable_parameters()

    # --- train --------------------------------------------------------------
    batch = tokenizer(
        EXAMPLES,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    ).to(device)
    # Labels = inputs for causal LM; -100 masks the padding out of the loss.
    labels = batch["input_ids"].masked_fill(batch["attention_mask"] == 0, -100)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )

    print("\n--- training ---")
    model.train()
    started = time.time()
    for epoch in range(EPOCHS):
        loss = model(**batch, labels=labels).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch + 1:>2}/{EPOCHS}  loss {loss.item():.4f}")
    print(f"  done in {time.time() - started:.1f}s")

    print("\n--- after training ---")
    model.eval()
    for prompt in EVAL_PROMPTS:
        print(f"  {prompt.splitlines()[0]}\n    -> {generate(model, tokenizer, prompt, device)}")

    # --- save just the adapter ---------------------------------------------
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)
    adapter_bytes = sum(f.stat().st_size for f in ADAPTER_DIR.rglob("*") if f.is_file())
    base_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print("\n--- adapter on disk ---")
    print(f"  {ADAPTER_DIR}")
    print(f"  adapter: {adapter_bytes / 1024:.0f} KB")
    print(f"  base model in memory: {base_bytes / 1024 / 1024:.0f} MB")
    print(f"  adapter is {100 * adapter_bytes / base_bytes:.2f}% of the base model")

    # --- reload onto a fresh base, then merge -------------------------------
    fresh = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    reloaded = PeftModel.from_pretrained(fresh, ADAPTER_DIR).eval()
    prompt = EVAL_PROMPTS[0]
    print("\n--- reloaded adapter ---")
    print(f"    -> {generate(reloaded, tokenizer, prompt, device)}")

    merged = reloaded.merge_and_unload().eval()
    print("--- merged into the base weights (no adapter left at inference) ---")
    print(f"    -> {generate(merged, tokenizer, prompt, device)}")


if __name__ == "__main__":
    main()
