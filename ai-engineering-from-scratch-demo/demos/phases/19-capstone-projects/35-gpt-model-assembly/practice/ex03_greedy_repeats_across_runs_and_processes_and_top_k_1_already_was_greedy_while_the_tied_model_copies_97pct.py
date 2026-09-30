"""Exercise 3 -- greedy repeats across runs and processes, top_k=1 already was greedy, and the tied model copies 97%.

    Add a `greedy=True` flag to generation that skips sampling and picks argmax. Confirm the sequence is deterministic across runs.

Reading of the exercise: `generate()` below takes the reference signature
plus `greedy`; with `greedy=True` it runs the same sliding-window loop but
appends `argmax` instead of calling `torch.multinomial`, and otherwise it
defers to the lesson's `generate`. The model is the lesson's tiny demo config
(vocab 512, context 64, d_model 64, 4 heads, 2 layers, no dropout), built at
seed 0, with the demo's prompt [1, 2, 3, 4, 5] and 40 new tokens. "Across
runs" is tested two ways: three calls in one process with the global RNG
reseeded to different values in between, and two fresh Python processes.

**ANSWER: greedy decoding is deterministic.** All three in-process runs and
both fresh processes return the same 45 tokens. The sampled path is not: two
unseeded calls with the demo's temperature 0.8 and top_k 20 differ.

**FINDING: the lesson's generate already had greedy mode: top_k=1.** Passing
`top_k=1` to the reference `generate` gives the identical sequence for every
seed tried, because `top_k_filter` leaves one finite logit and multinomial has
nothing to choose. The lesson says "Temperature near zero collapses to
greedy", but `temperature=0` raises ValueError. Near zero it works: at
temperature 0.01, 20 of 20 seeds match greedy, but at 0.1 none do.

**FINDING: the untrained tied model's greedy output is a copy of its last
input token.** The demo prompt gives [1, 2, 3, 4, 5] followed by 5 forty
times. This is weight tying at work. The final hidden state is closest to the
current token's own embedding row, and the head scores tokens by dot product
with those same rows. Over 200 random prompts, greedy's next token equals the
last prompt token for 97.0% of them; untie the head and it is 1.5%.

Structure: `generate()` is the flagged generator; `once()` is what each fresh
process runs (`python ex03_*.py --once`); `copy_rate()` measures copying.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

from harness import parity, practice

try:
    import torch
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "35-gpt-model-assembly"
TINY = {"vocab_size": 512, "context_length": 64, "d_model": 64, "num_heads": 4, "num_layers": 2,
        "dropout": 0.0}
PROMPT, NEW = [[1, 2, 3, 4, 5]], 40


def generate(ref, model, prompt, max_new_tokens, greedy=False, **sampling):
    if not greedy:
        return ref.generate(model, prompt, max_new_tokens, **sampling)
    model.eval()
    tokens = prompt.clone()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(tokens[:, -model.cfg.context_length :])[:, -1]
            tokens = torch.cat([tokens, logits.argmax(-1, keepdim=True)], dim=1)
    return tokens


def tiny(ref, tied=True):
    torch.manual_seed(0)
    return ref.GPTModel(ref.GPTConfig(**TINY, weight_tying=tied)).eval()


def once():
    ref = parity.load_reference(PHASE, LESSON, "main")
    return generate(ref, tiny(ref), torch.tensor(PROMPT), NEW, greedy=True)[0].tolist()


def copy_rate(ref, tied):
    model = tiny(ref, tied)
    prompts = torch.randint(0, 512, (200, 5), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        nxt = model(prompts)[:, -1].argmax(-1)
    return (nxt == prompts[:, -1]).float().mean().item()


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    model, prompt = tiny(ref), torch.tensor(PROMPT)
    runs = []
    for seed in (1, 2, 3):
        torch.manual_seed(seed)
        runs.append(generate(ref, model, prompt, NEW, greedy=True)[0].tolist())
    cmd = [sys.executable, __file__, "--once"]
    root = pathlib.Path(__file__).resolve().parents[5]
    procs = [subprocess.run(cmd, capture_output=True, text=True, cwd=root, check=True).stdout.strip()
             for _ in range(2)]
    sampled = [generate(ref, model, prompt, NEW, temperature=0.8, top_k=20)[0].tolist() for _ in "ab"]
    try:
        ref.generate(model, prompt, 1, temperature=0.0)
        zero = "accepted"
    except ValueError:
        zero = "ValueError"

    def matches(**kw):
        return sum(ref.generate(model, prompt, NEW, seed=s, **kw)[0].tolist() == runs[0] for s in range(20))

    return {
        "runs": runs, "in_process": all(run == runs[0] for run in runs),
        "procs_equal": procs == [str(runs[0])] * 2, "sampled_differ": sampled[0] != sampled[1],
        "top_k1": matches(top_k=1), "t001": matches(temperature=0.01),
        "t01": matches(temperature=0.1), "zero": zero,
        "copy_tied": copy_rate(ref, True), "copy_untied": copy_rate(ref, False),
    }


def verify(result):
    r = result
    seq = r["runs"][0]
    return [
        practice.Check(
            "ANSWER: greedy decoding is deterministic",
            r["in_process"] and r["procs_equal"] and r["sampled_differ"],
            f"3 in-process runs equal: {r['in_process']}; 2 fresh processes equal: "
            f"{r['procs_equal']}; unseeded sampling differs: {r['sampled_differ']}",
        ),
        practice.Check(
            "FINDING: the lesson's generate already had greedy mode: top_k=1",
            r["top_k1"] == 20 and r["t001"] == 20 and r["t01"] == 0 and r["zero"] == "ValueError",
            f"seeds matching greedy out of 20: top_k=1 {r['top_k1']}, T=0.01 {r['t001']}, "
            f"T=0.1 {r['t01']}; temperature=0 -> {r['zero']}",
        ),
        practice.Check(
            "FINDING: the untrained tied model's greedy output is a copy of its last input token",
            seq == [1, 2, 3, 4, 5] + [5] * NEW and r["copy_tied"] > 0.9 and r["copy_untied"] < 0.1,
            f"greedy output {seq[:8]}... ({len(set(seq[5:]))} distinct new token); next == last "
            f"token on 200 prompts: tied {r['copy_tied']:.1%}, untied {r['copy_untied']:.1%}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    if "--once" in sys.argv:
        print(once())
        raise SystemExit(0)
    raise SystemExit(practice.selfcheck(globals()))
