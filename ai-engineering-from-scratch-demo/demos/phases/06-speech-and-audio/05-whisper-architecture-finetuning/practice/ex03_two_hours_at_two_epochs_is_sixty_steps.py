"""Exercise 3 — two hours at two epochs is sixty optimizer steps.

    **Hard.** Using HF `datasets`, pick a language Whisper struggles with (e.g.,
    Urdu), fine-tune Medium with LoRA for 2 epochs on 2 hours, and report WER
    delta.

Reading of the exercise: `datasets`, `transformers`, `peft` and `torch` are all
absent, so no fine-tune can run here. The budget can be computed exactly from the
lesson's own `lora_params` and `transformer_params`, and the budget is the
finding -- the run the exercise specifies is **60 optimizer steps**, which is not
a fine-tune of anything.

Whisper's window is 30 s, so 2 hours is **240 examples**. Two epochs is 480
forward passes; at a batch size of 8 that is **60 gradient updates** for a
3.1 M-parameter adapter. Published Whisper LoRA recipes run thousands.

**`lora_params` undercounts by exactly one third.** It charges
`n_layers * 2 * per_block` -- one attention block per encoder layer and one per
decoder layer -- but a Whisper decoder layer carries **two** attention blocks,
self and cross. The right count is three per layer pair, not two:

| Medium, r=16, q_proj+v_proj | |
|---|---:|
| `lora_params` as written | **3.146 M** |
| counting the decoder's cross-attention | **4.719 M** |
| ratio | exactly 3/2 |

**Step 5's headline is true and is not the number.** It says LoRA "reduces
trainable params 100x+". Against Medium's own printed total of 761.1 M, 3.146 M
is **242x** and **0.41%** of the model; with the cross-attention adapters counted
it is 161x. Neither figure appears anywhere in the output, and 100x+ is the only
claim a reader is left with.

What that buys is memory, and the same ratio carries: Adam at fp32 keeps a master
copy plus two moments, 12 bytes per trainable parameter, so a full fine-tune of
Medium needs **9.1 GB** of optimizer state against LoRA's **37.7 MB** -- and the
adapter that ships afterwards is **12.6 MB**.

So the exercise's three numbers pull against each other: 2 hours and 2 epochs is
a sample size chosen for a full fine-tune's *cost*, applied to a method whose
whole point is that it does not have that cost.

Structure: `steps` turns hours and epochs into optimizer updates; `adapters`
counts LoRA parameters both ways; `optimizer_bytes` is the fp32 Adam footprint;
`model_total` re-reads Step 4's own totals.
"""

from __future__ import annotations

import importlib.util

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "05-whisper-architecture-finetuning"
MEDIUM = {"layers": 24, "d_model": 1024, "d_ff": 4096, "heads": 16, "vocab": 51865}
HOURS, EPOCHS, BATCH, WINDOW = 2.0, 2, 8, 30.0
RANK, MODULES = 16, ("q_proj", "v_proj")
ADAM_BYTES, WEIGHT_BYTES = 12, 4
ABSENT = ("datasets", "transformers", "peft", "torch")


def steps(hours, epochs, batch, window=WINDOW):
    """Whisper pads every clip to one 30 s window, so hours convert to examples directly."""
    examples = hours * 3600 / window
    return {"examples": examples, "updates": examples * epochs / batch}


def adapters(ref, config, rank=RANK):
    """LoRA parameters as the lesson counts them, and with the decoder counted whole."""
    written = ref.lora_params(config["layers"], config["d_model"], rank, MODULES)
    per_module = 2 * config["d_model"] * rank
    blocks_per_layer = 3          # encoder self-attn, decoder self-attn, decoder cross-attn
    whole = config["layers"] * blocks_per_layer * len(MODULES) * per_module
    return {"written": written, "whole": whole}


def model_total(ref, config):
    return sum(ref.transformer_params(config["layers"], config["d_model"], config["d_ff"],
                                      config["heads"], config["vocab"]))


def optimizer_bytes(parameters, per_parameter=ADAM_BYTES):
    """fp32 Adam: an fp32 master copy plus first and second moments."""
    return parameters * per_parameter


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    lora = adapters(ref, MEDIUM)
    total = model_total(ref, MEDIUM)
    return {
        "absent": [m for m in ABSENT if importlib.util.find_spec(m) is None],
        "budget": steps(HOURS, EPOCHS, BATCH), "lora": lora, "total": total,
        "reduction": total / lora["written"], "whole_reduction": total / lora["whole"],
        "share": lora["written"] / total,
        "full_state": optimizer_bytes(total), "lora_state": optimizer_bytes(lora["written"]),
        "shipped": lora["written"] * WEIGHT_BYTES,
    }


def verify(result):
    budget, lora = result["budget"], result["lora"]
    return [
        practice.Check(
            "ANSWER: the specified run is 60 optimizer steps",
            budget["updates"] == 60,
            f"Whisper pads every clip to one {WINDOW:.0f} s window, so {HOURS:.0f} hours is "
            f"{budget['examples']:.0f} examples; {EPOCHS} epochs at batch {BATCH} is "
            f"{budget['updates']:.0f} gradient updates for a "
            f"{lora['written'] / 1e6:.3f} M-parameter adapter. That is not a fine-tune",
        ),
        practice.Check(
            "FINDING: `lora_params` undercounts by exactly one third",
            abs(lora["whole"] / lora["written"] - 1.5) < 1e-12,
            f"it charges `n_layers * 2 * per_block`, one attention block per encoder layer and "
            f"one per decoder layer, but a Whisper decoder layer carries two -- self and cross. "
            f"Three per layer pair rather than two: {lora['whole'] / 1e6:.3f} M against the "
            f"printed {lora['written'] / 1e6:.3f} M, a ratio of exactly "
            f"{lora['whole'] / lora['written']:.1f}",
        ),
        practice.Check(
            "FINDING: Step 5 claims '100x+' and never prints the 242x it is",
            result["reduction"] > 200 and result["share"] < 0.005,
            f"against Medium's own printed total of {result['total'] / 1e6:.1f} M, "
            f"{lora['written'] / 1e6:.3f} M is {result['reduction']:.0f}x and "
            f"{result['share'] * 100:.2f}% of the model; with the cross-attention adapters "
            f"counted it is {result['whole_reduction']:.0f}x. Neither number is in the output",
        ),
        practice.Check(
            "MECHANISM: the same ratio is what the method actually buys, in bytes",
            result["full_state"] / result["lora_state"] > 200,
            f"fp32 Adam keeps a master copy and two moments, {ADAM_BYTES} bytes per trainable "
            f"parameter: {result['full_state'] / 1e9:.1f} GB of optimizer state for a full "
            f"fine-tune against {result['lora_state'] / 1e6:.1f} MB for the adapter, which then "
            f"ships as {result['shipped'] / 1e6:.1f} MB of weights",
        ),
        practice.Check(
            "CONTROL: none of the four libraries the exercise names is installed",
            len(result["absent"]) == len(ABSENT),
            f"find_spec is None for {result['absent']}, so the WER delta itself cannot be "
            "measured here. Everything above is arithmetic on the lesson's own "
            "`lora_params` and `transformer_params`, and none of it needs a GPU to be wrong",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
