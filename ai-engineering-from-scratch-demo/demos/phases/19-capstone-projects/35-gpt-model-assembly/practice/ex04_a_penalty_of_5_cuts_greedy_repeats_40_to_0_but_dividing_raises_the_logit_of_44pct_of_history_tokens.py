"""Exercise 4 -- a penalty of 5 cuts greedy repeats from 40 to 0, but dividing raises the logit of 44% of history tokens.

    Add a `repetition_penalty` knob that divides the logit of any token in the prompt or generated history by a constant before softmax. Show on a fixed prompt that values above one reduce repeat counts in the output.

Reading of the exercise: `generate()` below is the reference loop with the
knob added. At each step it divides the logit of every token already in the
prompt or output by `repetition_penalty`, exactly as posed, and then
applies temperature and either argmax or multinomial. The model is the
lesson's tiny demo config built at seed 0, and the fixed prompt is the demo's
[1, 2, 3, 4, 5]. The repeat count is the number of the 40 new tokens that
already appear earlier in the sequence. Greedy decoding gives one sequence per
penalty. The sampled result is the mean over a batch of 100 sequences at
temperature 0.1 (generator seed 0). At temperature 1 this untrained model is
nearly uniform over 512 tokens and barely repeats at all.

**ANSWER: yes. Every value above one lowers the repeat count, and 5 removes
it.**

| penalty | 1.0 | 1.2 | 1.5 | 2.0 | 5.0 |
|---|---:|---:|---:|---:|---:|
| greedy repeats / 40 | 40 | 39 | 37 | 15 | 0 |

Sampled at temperature 0.1, the mean falls from 26.7 to 2.1 at penalty 2.
The untrained tied model copies its last token (exercise 3), so without a
penalty greedy output is the prompt's 5 forty times. The penalty is one flat
factor for a token seen once or forty times, so even at 2.0 the output still
holds adjacent pairs.

**FINDING: dividing a negative logit raises it, so the rule as posed rewards
44% of the history.** x / 1.2 is closer to zero than x, and for x < 0 that
means larger. Over the sampled run at penalty 2, 44% of penalised history
logits went up. The rule Hugging Face ships is
`torch.where(score < 0, score * penalty, score / penalty)`
(`RepetitionPenaltyLogitsProcessor`,
https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/generation/logits_process.py,
read 2026-09-29), and under it the share is 0. On the same batch it also
repeats slightly less: 2.00 against 2.06.

Structure: `penalize()` holds both rules; `generate()` is batched, so one call
gives 100 sampled sequences with their repeat counts and the share of
penalised logits that rose.
"""

from __future__ import annotations

from harness import parity, practice

try:
    import torch
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "35-gpt-model-assembly"
TINY = {"vocab_size": 512, "context_length": 64, "d_model": 64, "num_heads": 4, "num_layers": 2,
        "dropout": 0.0}
PROMPT, NEW, BATCH = [1, 2, 3, 4, 5], 40, 100


def penalize(logits, seen, penalty, rule):
    """'divide' is the exercise as posed; 'hf' multiplies negative logits instead."""
    if rule == "divide":
        hit = logits / penalty
    else:
        hit = torch.where(logits < 0, logits * penalty, logits / penalty)
    return torch.where(seen, hit, logits)


def generate(model, prompt, penalty, rule="divide", greedy=True, temperature=1.0):
    tokens, rng, rose, total = prompt.clone(), torch.Generator().manual_seed(0), 0, 0
    with torch.no_grad():
        for _ in range(NEW):
            logits = model(tokens[:, -model.cfg.context_length :])[:, -1]
            seen = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, tokens, True)
            new = penalize(logits, seen, penalty, rule)
            rose, total = rose + ((new > logits) & seen).sum().item(), total + seen.sum().item()
            if greedy:
                nxt = new.argmax(-1, keepdim=True)
            else:
                nxt = torch.multinomial(torch.softmax(new / temperature, -1), 1, generator=rng)
            tokens = torch.cat([tokens, nxt], dim=1)
    return repeats(tokens), rose / total


def repeats(tokens):
    """Mean over the batch of new tokens that already occurred earlier."""
    rows = tokens.tolist()
    start = len(PROMPT)
    return sum(sum(r[i] in r[:i] for i in range(start, len(r))) for r in rows) / len(rows)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.manual_seed(0)
    model = ref.GPTModel(ref.GPTConfig(**TINY)).eval()
    one, batch = torch.tensor([PROMPT]), torch.tensor([PROMPT] * BATCH)
    sampled = {(p, rule): generate(model, batch, p, rule, greedy=False, temperature=0.1)
               for p, rule in [(1.0, "divide"), (2.0, "divide"), (2.0, "hf")]}
    return {
        "greedy": {p: generate(model, one, p)[0] for p in (1.0, 1.2, 1.5, 2.0, 5.0)},
        "sampled": {f"{rule} {p}": [round(v, 3) for v in out] for (p, rule), out in sampled.items()},
    }


def verify(result):
    g, s = result["greedy"], result["sampled"]
    base, div, hf = s["divide 1.0"], s["divide 2.0"], s["hf 2.0"]
    return [
        practice.Check(
            "ANSWER: yes. Every value above one lowers the repeat count, and 5 removes it",
            list(g.values()) == [40, 39, 37, 15, 0] and div[0] < base[0] / 5,
            f"greedy repeats of {NEW} by penalty {g}; sampled T=0.1 mean {base[0]} -> {div[0]} at 2.0",
        ),
        practice.Check(
            "FINDING: dividing a negative logit raises it, so the rule as posed rewards 44% of the history",
            0.40 < div[1] < 0.48 and hf[1] == 0.0 and hf[0] <= div[0],
            f"share of penalised history logits that rose at 2.0: divide {div[1]:.0%}, hf {hf[1]:.0%}; "
            f"mean repeats divide {div[0]}, hf {hf[0]}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
