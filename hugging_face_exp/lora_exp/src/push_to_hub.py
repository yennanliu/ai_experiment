"""Push the trained LoRA adapter to the Hugging Face Hub.

Uploads only `out/adapter/` — the adapter weights and config, a few hundred KB.
The base model (distilgpt2) is not re-uploaded; the adapter config points at it
by name, so `PeftModel.from_pretrained` pulls the base from the Hub itself.

Usage:
    hf auth login                                    # once, needs a WRITE token
    uv run python -m src.push_to_hub <user>/<repo>   # public
    uv run python -m src.push_to_hub <user>/<repo> --private
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, whoami

ADAPTER_DIR = Path(__file__).resolve().parent.parent / "out" / "adapter"

CARD = """---
base_model: distilgpt2
library_name: peft
license: apache-2.0
pipeline_tag: text-generation
tags:
- lora
- peft
- educational
- base_model:adapter:distilgpt2
---

# distilgpt2 LoRA — "Fact:" answer style

A deliberately tiny LoRA adapter, trained as a **teaching example**. It teaches
`distilgpt2` one narrow behaviour: answer a `Q:` prompt with a single-line
statement that begins with `Fact:`.

Trained on **8 hand-written examples** in about **7 seconds** on an Apple MPS
laptop. It is not useful for anything real — it is useful for seeing what the
moving parts of LoRA actually do.

## What it does

| Prompt | Base distilgpt2 | With this adapter |
|---|---|---|
| `Q: What is LoRA?\\nA:` | *"LoRA is a very simple, simple, simple, simple…"* | *"Fact: LoRA freezes the base weights and trains two small matrices A and B."* |
| `Q: What does a low-rank matrix do?\\nA:` *(held out)* | *"It's a matrix that is a matrix that is a matrix…"* | *"Fact: low-rank sets the weights of the edges, q_proj and v_proj first."* |

The first answer is **memorised verbatim** from the training set. The second
prompt was never trained on: the `Fact:` style transferred, but the content is
wrong — low-rank matrices do not "set the weights of the edges". That gap is
the honest lesson of an 8-example fine-tune. **LoRA transferred style, not
knowledge.**

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("distilgpt2")
base = AutoModelForCausalLM.from_pretrained("distilgpt2")
model = PeftModel.from_pretrained(base, "{repo_id}").eval()

inputs = tok("Q: What is LoRA?\\nA:", return_tensors="pt")
print(tok.decode(model.generate(**inputs, max_new_tokens=24,
                                do_sample=False,
                                pad_token_id=tok.eos_token_id)[0]))
```

Call `model.merge_and_unload()` to fold the adapter into the base weights for
zero inference overhead.

## Training

| | |
|---|---|
| Base model | `distilgpt2` (82M params) |
| Trainable params | **147,456 (0.18%)** |
| Adapter size | ~584 KB, vs 313 MB for the base |
| Config | `r=8`, `lora_alpha=16` (scaling 2.0), `lora_dropout=0.05` |
| Target modules | `c_attn` — GPT-2's fused q/k/v projection, in all 6 blocks |
| Optimiser | AdamW, lr `2e-3`, 60 full-batch epochs |
| Loss | 5.16 -> 0.51 |
| Hardware | Apple Silicon (MPS), ~7 s |
| Seed | `torch.manual_seed(0)` — reproducible |

The learning rate is ~100x a typical full fine-tuning LR. That is normal for
LoRA: you are training freshly initialised matrices, not nudging pretrained
weights.

## Limitations

Everything about this model is a limitation. It was overfit on purpose to 8
sentences about LoRA, so it will state confident falsehoods on any prompt
outside that set, and it inherits all of distilgpt2's biases underneath. Use it
to learn how adapters work, not for generation.

## Source

Training code and a from-scratch (no-PEFT) reimplementation of the same
arithmetic: see the repository this adapter was trained from. Follows the
*Fine-Tuning with LoRA & QLoRA* lesson from
[AI Engineering from Scratch](https://yennj12.js.org/ai-engineering-from-scratch/lesson.html?path=phases/11-llm-engineering/08-fine-tuning-lora).
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id", help="target repo, e.g. yourname/distilgpt2-lora-fact")
    parser.add_argument("--private", action="store_true", help="create the repo private")
    parser.add_argument("--dry-run", action="store_true", help="write the card, upload nothing")
    args = parser.parse_args()

    if not (ADAPTER_DIR / "adapter_model.safetensors").exists():
        raise SystemExit(
            f"No adapter at {ADAPTER_DIR}.\n"
            "Train one first:  uv run python -m src.train_lora"
        )

    # Overwrite PEFT's auto-generated stub card with a real one.
    (ADAPTER_DIR / "README.md").write_text(CARD.format(repo_id=args.repo_id))

    files = sorted(f.name for f in ADAPTER_DIR.iterdir() if f.is_file())
    size_kb = sum(f.stat().st_size for f in ADAPTER_DIR.iterdir() if f.is_file()) / 1024
    cfg = json.loads((ADAPTER_DIR / "adapter_config.json").read_text())

    print(f"adapter: {ADAPTER_DIR}")
    print(f"  files: {', '.join(files)}  ({size_kb:.0f} KB)")
    print(f"  base: {cfg['base_model_name_or_path']}  r={cfg['r']}  "
          f"alpha={cfg['lora_alpha']}  targets={cfg['target_modules']}")

    if args.dry_run:
        print("\ndry run: card written, nothing uploaded")
        return

    api = HfApi()
    print(f"\nauthenticated as: {whoami()['name']}")
    print(f"target: {args.repo_id}  ({'private' if args.private else 'PUBLIC'})")

    api.create_repo(args.repo_id, repo_type="model",
                    private=args.private, exist_ok=True)
    url = api.upload_folder(
        folder_path=str(ADAPTER_DIR),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Add distilgpt2 LoRA adapter (r=8, c_attn, 147k params)",
    )
    print(f"\nuploaded: {url}")
    print(f"model page: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
