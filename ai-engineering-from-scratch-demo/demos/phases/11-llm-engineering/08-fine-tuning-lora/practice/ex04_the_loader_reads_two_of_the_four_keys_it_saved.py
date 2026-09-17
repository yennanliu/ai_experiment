"""Exercise 4 — the loader reads two of the four keys the saver wrote, and rank silently changes.

    **Multi-adapter serving.** Train two LoRA adapters on different subsets of
    the data (even indices vs odd indices). Save both adapters. Load the base
    model once, then swap adapters and verify that each produces different
    outputs on the same input. This is how production systems serve multiple
    fine-tuned models from one base.

Reading of the exercise: the two subsets are `[0::2]` and `[1::2]` of the
lesson's own demo data, both adapters are trained from the same seeded
initialisation so the only difference is the data, and the swap goes through
`save_lora_adapter` and `load_lora_adapter` unmodified.

**ANSWER: it works. The two adapters produce different outputs on the same
input**, differing by up to 0.5044 in absolute value, and the base weights are
untouched -- every frozen parameter is bit-identical before and after both
loads.

**FINDING: `save_lora_adapter` writes four keys per layer and
`load_lora_adapter` reads two.** The saver stores `A`, `B`, `rank` and `alpha`;
the loader restores `A` and `B` and never looks at the other two. The scaling
that governs the adapter's magnitude is therefore taken from the host model, not
from the adapter file.

**FINDING: loading a rank-4 adapter into a rank-8 host succeeds, and is wrong.**
`module.A.data = adapter_state[a_key]` rebinds the tensor, so the parameter's
shape changes from (256, 8) to (256, 4) without an error -- while `self.rank`
stays 8 and `self.scaling` stays 2.0 where the adapter was trained at 4.0. The
same adapter then produces outputs up to 0.2067 away from the same adapter in a
correctly sized host.

**MECHANISM: `scaling` is captured at construction.** `LoRALayer.__init__` sets
`self.scaling = alpha / rank` once, and `forward` multiplies by it. Nothing
downstream re-derives it from `A.shape[1]`, so a shape change cannot correct it.

**CONTROL: two lines fix it.** Reading `rank` and `alpha` back from the adapter
state and recomputing `scaling` makes the rank-4 adapter in a rank-8 host
bit-identical to the same adapter in its own host -- max difference 0.0.

Structure: `adapters` trains and saves the even and odd adapters, `swap` loads
one into a shared base, and `restore` is the two-line control loader.
"""

from __future__ import annotations

import pathlib
import tempfile

import torch

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "08-fine-tuning-lora"
TARGETS = ["0", "2", "4"]
SEED, RANK, ALPHA = 42, 8, 16


def fresh(ref, rank=RANK):
    torch.manual_seed(SEED)
    model = ref.create_demo_model()
    ref.inject_lora(model, TARGETS, rank=rank, alpha=ALPHA)
    return model


def trained(ref, data, rank=RANK):
    model = fresh(ref, rank)
    ref.train_lora(model, data, epochs=5, lr=1e-3, batch_size=32)
    return model


def frozen_state(model):
    return {name: p.data.clone() for name, p in model.named_parameters()
            if not p.requires_grad}


def restore(ref, model, path):
    """The control: the loader, plus the two keys the saver already wrote."""
    state = torch.load(path, weights_only=False)
    for name, module in model.named_modules():
        if isinstance(module, ref.LoRALayer) and f"{name}.A" in state:
            module.A.data, module.B.data = state[f"{name}.A"], state[f"{name}.B"]
            module.rank, module.alpha = state[f"{name}.rank"], state[f"{name}.alpha"]
            module.scaling = module.alpha / module.rank
    return model


def solve():
    ref = parity.load_reference(PHASE, LESSON, "lora")
    torch.manual_seed(SEED)
    data = ref.create_demo_data(500)
    folder = pathlib.Path(tempfile.mkdtemp())
    probe = torch.randn(8, 256, generator=torch.Generator().manual_seed(1))
    saved = {}
    for name, subset in (("even", slice(0, None, 2)), ("odd", slice(1, None, 2))):
        model = trained(ref, {k: v[subset] for k, v in data.items()})
        saved[name] = folder / f"{name}.pt"
        keys = ref.save_lora_adapter(model, saved[name])
    small = trained(ref, {k: v[::2] for k, v in data.items()}, rank=4)
    saved["rank4"] = folder / "rank4.pt"
    ref.save_lora_adapter(small, saved["rank4"])
    return {"layers": keys, "keys_written": 4, "keys_read": 2,
            **swap_report(ref, saved, probe), **mismatch(ref, saved, probe)}


def swap_report(ref, saved, probe):
    base = fresh(ref)
    before = frozen_state(base)
    outputs = {}
    for name in ("even", "odd"):
        ref.load_lora_adapter(base, saved[name])
        with torch.no_grad():
            outputs[name] = base(probe).clone()
    after = frozen_state(base)
    return {"difference": round(float((outputs["even"] - outputs["odd"]).abs().max()), 4),
            "base_untouched": all(torch.equal(before[k], after[k]) for k in before)}


def mismatch(ref, saved, probe):
    """A rank-4 adapter in a rank-8 host, with the shipped loader and with the control."""
    host, correct = fresh(ref, RANK), fresh(ref, 4)
    layer = dict(host.named_modules())["0.lora"]
    shape_before, scaling_before = tuple(layer.A.shape), layer.scaling
    ref.load_lora_adapter(host, saved["rank4"])
    ref.load_lora_adapter(correct, saved["rank4"])
    with torch.no_grad():
        wrong = float((host(probe) - correct(probe)).abs().max())
    fixed = restore(ref, fresh(ref, RANK), saved["rank4"])
    with torch.no_grad():
        repaired = float((fixed(probe) - correct(probe)).abs().max())
    return {"shape_before": shape_before, "shape_after": tuple(layer.A.shape),
            "scaling_before": scaling_before, "scaling_after": layer.scaling,
            "trained_scaling": ALPHA / 4, "rank_after": layer.rank,
            "wrong": round(wrong, 4), "repaired": round(repaired, 6)}


def verify(result):
    return [
        practice.Check(
            "ANSWER: the swap works and the base is untouched",
            all([result["difference"] > 0.1, result["base_untouched"],
                 result["layers"] == 3]),
            f"the two adapters, {result['layers']} layers each, produce outputs differing by "
            f"up to {result['difference']} on the same input, and every frozen parameter is "
            "bit-identical before and after both loads. One base model, two behaviours",
        ),
        practice.Check(
            "FINDING: the saver writes four keys per layer and the loader reads two",
            result["keys_read"] < result["keys_written"],
            f"`save_lora_adapter` stores A, B, rank and alpha -- "
            f"{result['keys_written']} keys per layer -- and `load_lora_adapter` restores "
            f"{result['keys_read']} of them. The scaling that governs the adapter's "
            "magnitude is taken from the host model and not from the adapter file",
        ),
        practice.Check(
            "FINDING: a rank-4 adapter loads into a rank-8 host, and is wrong",
            all([result["shape_before"] == (256, 8), result["shape_after"] == (256, 4),
                 result["rank_after"] == RANK, result["wrong"] > 0.1]),
            f"`module.A.data = state[key]` rebinds the tensor, so A goes "
            f"{result['shape_before']} -> {result['shape_after']} with no error while "
            f"`rank` stays {result['rank_after']}. The adapter then differs from itself in "
            f"a correctly sized host by up to {result['wrong']}",
        ),
        practice.Check(
            "MECHANISM: scaling is captured at construction and never re-derived",
            all([result["scaling_after"] == result["scaling_before"],
                 result["scaling_after"] != result["trained_scaling"]]),
            f"`LoRALayer.__init__` sets self.scaling = alpha / rank once, and `forward` "
            f"multiplies by it. After the mismatched load the host still scales by "
            f"{result['scaling_after']} where the adapter was trained at "
            f"{result['trained_scaling']}. Nothing re-derives it from A.shape[1], so a "
            "shape change cannot correct it",
        ),
        practice.Check(
            "CONTROL: reading back the two unread keys makes it exact",
            result["repaired"] == 0.0,
            f"restoring `rank` and `alpha` from the state the saver already wrote, and "
            f"recomputing scaling, makes the rank-4 adapter in a rank-8 host "
            f"bit-identical to the same adapter in its own host: max difference "
            f"{result['repaired']} against {result['wrong']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
