"""Exercise 3 -- with lesson 35's sampler the random-init and loaded samples share 14 of 16 tokens.

    Plug the loader into the lesson 35 generation function and produce two side by side samples: one from random init, one from the loaded fixture.

Reading of the exercise: the model is lesson 35's own `GPTModel` (loaded from
lesson 35's `code/main.py`, not this lesson's replica), at this lesson's demo
config (vocab 256, d_model 192, 4 layers, dropout 0). This lesson's
`load_safetensors` fills it from the lesson's stub (`make_stub_safetensors`,
seed 42, written to a temp directory; no real weights). Both samples come from
lesson 35's `generate` with its defaults (multinomial, temperature 1) and
`seed=0`, 16 new tokens after the demo prompt [7, 11, 13, 17].

**ANSWER: the loader plugs in unchanged, and the two samples are nearly the
same.** Lesson 35's model has the same parameter names, so the load reports
`loaded=52 missing=0 unexpected=0 shape_mismatch=0` and the head stays tied.
Random init samples [58, 180, 187, 139, 52, 235, 65, 71, 166, ...], and the
loaded fixture samples [58, 180, 187, 139, 52, 235, 65, 71, 16, ...]: 14 of 16
tokens are identical.

**FINDING: the lesson's sanity gate reads this correct load as a failure.**
The lesson says if post-load samples "look like the pre-load samples, the load
did not change the model". Here all 52 tensors landed, yet the samples match
at 14 of 16 positions. Lesson 35 initialises with N(0, 0.02) and zero biases,
the stub draws from the same distribution, and both next-token distributions
are near uniform (5.49 and 5.51 nats of 5.55). With the same seed the sampler
picks nearly the same tokens: total variation between the two distributions
is only 0.17. A second random init (seed 1) also matches at 14 of 16.

**FINDING: the lesson's own demo passes its gate because its replica is not
lesson 35.** The replica `GPTModel` skips lesson 35's `_init_weights`, so its
tied embedding is N(0, 1). Its greedy next-token entropy is about 1e-34 nats,
and it just repeats the last prompt token (17, 17, 17, ...). Any load moves it:
a file holding only `wte.weight` (1 of 52 tensors, `missing=51`) changes the
demo's greedy sample too. The `seed` argument of `quick_generate` does
nothing; it is argmax, identical to lesson 35's `top_k=1`.

Structure: `sample()` is lesson 35's generate plus an entropy readout;
`solve()` runs both lessons' models.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from harness import parity, practice

try:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise practice.Skip(f"needs torch and safetensors: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "37-loading-pretrained-weights"
CFG = dict(vocab_size=256, context_length=64, d_model=192, num_heads=6, num_layers=4)
PROMPT = [[7, 11, 13, 17]]


def sample(l35, model):
    """Lesson 35 generate (seed 0), plus the next-token distribution after the prompt."""
    model.eval()
    with torch.no_grad():
        probs = model(torch.tensor(PROMPT))[0, -1].softmax(-1)
    tokens = l35.generate(model, torch.tensor(PROMPT), max_new_tokens=16, seed=0)[0, 4:].tolist()
    return tokens, probs


def entropy(probs):
    return float(-(probs * probs.clamp_min(1e-45).log()).sum())


def lesson37_gate(ref, path):
    """The lesson's own demo check: does the greedy sample change after a load?"""
    torch.manual_seed(0)
    model = ref.GPTModel(ref.ModelConfig(**CFG))
    with torch.no_grad():
        probs = model(torch.tensor(PROMPT))[0, -1].softmax(-1)
    before = ref.quick_generate(model, torch.tensor(PROMPT), n=8, seed=0)
    report = ref.load_safetensors(model, path, verbose=False)
    after = ref.quick_generate(model, torch.tensor(PROMPT), n=8, seed=123)
    return {"before": before[4:], "changed": before != after, "report": report.summary(),
            "entropy": entropy(probs), "argmax": after == ref.quick_generate(model, torch.tensor(PROMPT), 8)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    l35 = parity.load_reference(PHASE, "35-gpt-model-assembly", "main")
    with tempfile.TemporaryDirectory() as tmp:
        stub, wte = Path(tmp) / "gpt2-stub.safetensors", Path(tmp) / "wte-only.safetensors"
        ref.make_stub_safetensors(stub, ref.ModelConfig(**CFG), seed=42)
        with safe_open(str(stub), framework="pt") as reader:
            save_file({"wte.weight": reader.get_tensor("wte.weight")}, str(wte))
        torch.manual_seed(0)
        model = l35.GPTModel(l35.GPTConfig(**CFG, dropout=0.0))
        init, p_init = sample(l35, model)
        report = ref.load_safetensors(model, stub, verbose=False)
        loaded, p_loaded = sample(l35, model)
        greedy = l35.generate(model, torch.tensor(PROMPT), 8, top_k=1, seed=0)[0].tolist()
        torch.manual_seed(1)
        other, _ = sample(l35, l35.GPTModel(l35.GPTConfig(**CFG, dropout=0.0)))
        gate = lesson37_gate(ref, wte)
        gate["greedy_is_top1"] = greedy == ref.quick_generate(model, torch.tensor(PROMPT), 8)
    return {
        "report": report.summary(), "tied": model.lm_head.weight is model.tok_embed.weight,
        "init": init, "loaded": loaded, "same": sum(a == b for a, b in zip(init, loaded)),
        "same_other": sum(a == b for a, b in zip(init, other)),
        "entropy": (entropy(p_init), entropy(p_loaded)),
        "tv": 0.5 * float((p_init - p_loaded).abs().sum()), "gate": gate,
    }


def verify(result):
    r, g = result, result["gate"]
    return [
        practice.Check(
            "ANSWER: the loader plugs in unchanged, and the two samples are nearly the same",
            all([r["report"] == "loaded=52 missing=0 unexpected=0 shape_mismatch=0", r["tied"],
                 len(r["init"]) == len(r["loaded"]) == 16, r["same"] == 14]),
            f"{r['report']}; random init {r['init']}; loaded {r['loaded']}; {r['same']}/16 identical",
        ),
        practice.Check(
            "FINDING: the lesson's sanity gate reads this correct load as a failure",
            all([5.4 < min(r["entropy"]), max(r["entropy"]) < 5.545, r["tv"] < 0.25, r["same_other"] == 14]),
            f"entropies {r['entropy'][0]:.2f} / {r['entropy'][1]:.2f} nats (max 5.55), TV {r['tv']:.2f}; "
            f"a seed-1 random init matches {r['same_other']}/16",
        ),
        practice.Check(
            "FINDING: the lesson's own demo passes its gate because its replica is not lesson 35",
            all([g["entropy"] < 1e-20, g["before"] == [17] * 8, g["changed"],
                 g["report"] == "loaded=1 missing=51 unexpected=0 shape_mismatch=0",
                 g["argmax"], g["greedy_is_top1"]]),
            f"replica init entropy {g['entropy']:.1e}, sample {g['before']}; wte-only load "
            f"({g['report']}) changes it: {g['changed']}; seed ignored and equal to top_k=1: "
            f"{[g['argmax'], g['greedy_is_top1']]}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
