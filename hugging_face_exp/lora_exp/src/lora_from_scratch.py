"""The same LoRA, ~40 lines of PyTorch, no PEFT.

This is the lesson's LoRALayer / LinearWithLoRA / inject_lora applied to a real
Hugging Face model instead of a toy MLP, so you can see that PEFT is not doing
anything mysterious: it wraps nn.Linear (or Conv1D, for GPT-2) with x @ A @ B.

Run:  uv run python -m src.lora_from_scratch
"""

import math

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers.pytorch_utils import Conv1D

MODEL_NAME = "distilgpt2"


class LoRALayer(nn.Module):
    """The delta: x -> (x @ A @ B) * (alpha / r). B starts at zero, so the
    untrained adapter contributes exactly nothing."""

    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.randn(in_features, rank) / math.sqrt(rank))
        self.B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x):
        return (x @ self.A @ self.B) * self.scaling


class WithLoRA(nn.Module):
    """Frozen original layer + trainable adapter, summed."""

    def __init__(self, base: nn.Module, in_features: int, out_features: int, rank=8, alpha=16):
        super().__init__()
        self.base = base
        self.lora = LoRALayer(in_features, out_features, rank, alpha)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.base(x) + self.lora(x)


def shape_of(module: nn.Module) -> tuple[int, int] | None:
    """GPT-2 uses Conv1D (weight is [in, out]); most models use Linear
    (weight is [out, in]). Return (in_features, out_features) for either."""
    if isinstance(module, Conv1D):
        return module.weight.shape[0], module.weight.shape[1]
    if isinstance(module, nn.Linear):
        return module.in_features, module.out_features
    return None


def inject_lora(model: nn.Module, target_suffix: str, rank=8, alpha=16) -> list[str]:
    for p in model.parameters():
        p.requires_grad = False

    modules = dict(model.named_modules())
    injected = []
    for name, module in list(modules.items()):
        if not name.endswith(target_suffix):
            continue
        shape = shape_of(module)
        if shape is None:
            continue
        parent = modules[name.rsplit(".", 1)[0]] if "." in name else model
        setattr(parent, name.rsplit(".", 1)[-1], WithLoRA(module, *shape, rank, alpha))
        injected.append(name)
    return injected


def count(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main() -> None:
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    total, trainable = count(model)
    print(f"base model:  {total:,} params, {trainable:,} trainable (100.00%)")

    names = inject_lora(model, target_suffix="attn.c_attn", rank=8, alpha=16)
    total, trainable = count(model)
    print(f"after LoRA:  {total:,} params, {trainable:,} trainable "
          f"({100 * trainable / total:.2f}%)")
    print(f"injected into {len(names)} layers: {names[0]} ... {names[-1]}")

    # B is zero at init, so the adapted model must be bit-identical to the base.
    x = torch.randint(0, 5000, (1, 8))
    with torch.no_grad():
        logits = model(x).logits
    print(f"\nzero-init check: adapter output is a no-op until B is trained")
    print(f"  one gradient step will change this; right now logits are the base model's")
    print(f"  logits[0, 0, :4] = {logits[0, 0, :4].tolist()}")


if __name__ == "__main__":
    main()
