"""Exercise 5 -- the nucleus mass check holds at 0.9, but "exceeds" is the wrong test and fails at top_p = 1.

    Add `top_p` (nucleus) sampling next to `top_k`. Two-line check that the sum of probabilities of the kept tokens exceeds `top_p`.

Reading of the exercise: `top_p_filter()` keeps the smallest set of
highest-probability tokens whose mass reaches `top_p` and masks the rest to
-inf, mirroring the lesson's `top_k_filter`. `generate()` below is the
reference loop with both filters applied, top-k first, as Hugging Face orders
its warpers. The check runs on the next-token distributions of the lesson's
tiny demo model (seed 0, float64) for 200 random 8-token prompts. The
exercise's "two lines" are the two lines in `mass_check()`.

**ANSWER: the kept mass is at least top_p on all 200 distributions at 0.5, 0.9
and 0.95.** On average the nucleus keeps 224, 446 and 477 of the 512 tokens.
A 20-token generation with top_k = 20 and top_p = 0.9 runs.

**FINDING: "exceeds" is the wrong inequality, and the strict check fails.**
The nucleus is "the smallest set such that" its mass is ">= p" (Holtzman et
al., section 3.1, eq. 2, https://arxiv.org/pdf/1904.09751, read 2026-09-29).
The mass can equal p exactly. Two equal logits at top_p = 0.5 keep one token
of mass exactly 0.5, which does not exceed 0.5. At top_p = 1.0 the check
fails on real model output: every token is kept, and the kept probabilities
sum to just under 1.0 in float64 rounding: the strict check passes on only
70 of the 200 prompts here (the exact count is platform-sensitive, so the
test asserts only that it is below 200).

**FINDING: on this model the two knobs are not comparable.** The untrained
model is nearly uniform, so top_k = 20 keeps only 5.5% of the mass, while
top_p = 0.9 keeps 446 tokens. Chained (top-k, then top-p on the renormalised
20), 18 survive.

**FINDING: the lesson's `top_k_filter` keeps every tie.** It masks
`logits < threshold`, so top_k = 2 over four equal logits keeps all 4. The
lesson's own test is named `test_top_k_filter_keeps_exactly_k`.

Structure: `top_p_filter()`, `mass_check()` (the two lines) and `generate()`
with both knobs.
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


def top_p_filter(logits, top_p):
    probs, order = torch.softmax(logits, -1).sort(-1, descending=True)
    drop = probs.cumsum(-1) - probs >= top_p  # the mass before this token already reached top_p
    return logits.masked_fill(torch.zeros_like(drop).scatter(-1, order, drop), float("-inf"))


def mass_check(logits, filtered, top_p):
    kept = (torch.softmax(logits, -1) * torch.isfinite(filtered)).sum(-1)
    return kept > top_p, kept


def generate(ref, model, prompt, max_new_tokens, top_k=None, top_p=None, seed=0):
    tokens, rng = prompt.clone(), torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = ref.top_k_filter(model(tokens[:, -model.cfg.context_length :])[:, -1], top_k)
            logits = top_p_filter(logits, top_p) if top_p else logits
            nxt = torch.multinomial(torch.softmax(logits, -1), 1, generator=rng)
            tokens = torch.cat([tokens, nxt], dim=1)
    return tokens


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.manual_seed(0)
    model = ref.GPTModel(ref.GPTConfig(**TINY)).double().eval()
    prompts = torch.randint(0, 512, (200, 8), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        logits = model(prompts)[:, -1]
    per_p = {}
    for p in (0.5, 0.9, 0.95, 1.0):
        filtered = top_p_filter(logits, p)
        strict, kept = mass_check(logits, filtered, p)
        per_p[p] = {"strict": int(strict.sum()), "at_least": int((kept >= p).sum()),
                    "size": torch.isfinite(filtered).sum(-1).double().mean().item()}
    two = torch.zeros(1, 2, dtype=torch.float64)
    top_k = ref.top_k_filter(logits, 20)
    return {
        "per_p": per_p,
        "all_kept": all(per_p[p]["at_least"] == per_p[p]["strict"] == 200 for p in (0.5, 0.9, 0.95)),
        "tie_check": mass_check(two, top_p_filter(two, 0.5), 0.5)[0].item(),
        "tie_mass": mass_check(two, top_p_filter(two, 0.5), 0.5)[1].item(),
        "gen_len": generate(ref, model, prompts[:1], 20, top_k=20, top_p=0.9).shape[1],
        "top_k_mass": (torch.softmax(logits, -1) * torch.isfinite(top_k)).sum(-1).mean().item(),
        "chained": torch.isfinite(top_p_filter(top_k, 0.9)).sum(-1).double().mean().item(),
        "ties_kept": int(torch.isfinite(ref.top_k_filter(torch.ones(1, 4), 2)).sum()),
    }


def verify(result):
    r, per = result, result["per_p"]
    sizes = [round(per[p]["size"]) for p in (0.5, 0.9, 0.95)]
    return [
        practice.Check(
            "ANSWER: the kept mass is at least top_p on all 200 distributions at 0.5, 0.9 and 0.95",
            r["all_kept"] and sizes == [224, 446, 477] and r["gen_len"] == 28,
            f"mean nucleus size {sizes} of 512; 20-token top_k+top_p generation length {r['gen_len']}",
        ),
        practice.Check(
            "FINDING: 'exceeds' is the wrong inequality, and the strict check fails",
            r["tie_check"] is False and r["tie_mass"] == 0.5 and per[1.0]["strict"] < 200
            and per[1.0]["size"] == 512,
            f"two equal logits at 0.5: kept mass {r['tie_mass']}, exceeds: {r['tie_check']}; "
            f"top_p=1.0 keeps {per[1.0]['size']:.0f} tokens and passes the strict check on "
            f"{per[1.0]['strict']}/200",
        ),
        practice.Check(
            "FINDING: on this model the two knobs are not comparable",
            round(r["top_k_mass"], 3) == 0.055 and round(r["chained"]) == 18,
            f"top_k=20 keeps {r['top_k_mass']:.1%} of the mass; top_p=0.9 keeps {sizes[1]} tokens; "
            f"top_k=20 then top_p=0.9 keeps {r['chained']:.1f}",
        ),
        practice.Check(
            "FINDING: the lesson's top_k_filter keeps every tie",
            r["ties_kept"] == 4,
            f"top_k=2 over four equal logits keeps {r['ties_kept']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
