"""Exercise 4 — the swap leaves false approval at 13.2% because no model is called, and makes review 2.24x dearer.

    Swap reviewer from GPT-5.4 to Claude Opus 4.7. Measure false-approval rate and token cost delta.

Reading of the exercise: offline, the swap is a scaled-down run over the
lesson's own seams. False approval is measured on 1,000 seeded `run_team`
runs: runs with a planted bug whose first review approves. Cost is the
reviewer's reads (the merged diff, then any revision) and writes, taken from
the board and priced at each model's list price. Opus 4.7 is charged its
tokenizer's ~30% more tokens for the same text.

**ANSWER: false approval stays 41/310 (13.2%); the reviewer's cost rises
from $0.0401 to $0.0900 a run, +$0.0499 (2.24x).** 1.73x of that is price
($5/$25 against $2.50/$15 per MTok) and the rest is the tokenizer.

**FINDING: no model is called, so the swap cannot move the false-approval
rate.** `main.py` names no reviewer model and has no model seam.
`reviewer_check` is an 85% coin, and the reviewer's tokens are the literal
1,800, or 3,300 after a rejection. The stub's 15% false-approve rate
measures 13.2% on these seeds.

**FINDING: the token delta is the tokenizer alone.** The reviewer reads
2,833.9 and writes 2,203.5 tokens a run; Opus 4.7 counts 1,511 more of the
same text, 4.3% of the 34,973-token team run.
"""

from __future__ import annotations

import contextlib
import random

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "10-multi-agent-software-team"
RUNS = 1000
# $/MTok (input, output), read 2026-09-29:
# developers.openai.com/api/docs/pricing (gpt-5.4, standard, <272K context)
# platform.claude.com/docs/en/about-claude/pricing (Claude Opus 4.7)
PRICE = {"GPT-5.4": (2.50, 15.00), "Claude Opus 4.7": (5.00, 25.00)}
# same page: Claude 4.7+ "tokenizer produces approximately 30% more tokens for the same text"
TOKENIZER = {"GPT-5.4": 1.0, "Claude Opus 4.7": 1.3}


@contextlib.contextmanager
def recording(ref):
    boards, real = [], ref.Board

    class Recording(real):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            boards.append(self)

    ref.Board = Recording
    try:
        yield boards
    finally:
        ref.Board = real


def reviewer_io(ref, board):
    """Tokens the reviewer reads (the merged diff, then any revision) and writes."""
    msgs, k = board.messages, ref.MsgKind
    start = next(i for i, m in enumerate(msgs) if m.kind == k.REVIEW_NEEDED)
    read = sum(m.tokens for m in msgs[start:] if m.kind in (k.REVIEW_NEEDED, k.DIFF_READY))
    wrote = sum(m.tokens for m in msgs if m.by == "reviewer")
    return read, wrote


def cost(model, read, wrote):
    (p_in, p_out), scale = PRICE[model], TOKENIZER[model]
    return scale * (read * p_in + wrote * p_out) / 1e6


def run(ref):
    """One seeded batch; the reviewer is whatever `ref.reviewer_check` is."""
    with recording(ref) as boards:
        rs = [ref.run_team(f"issue-{s}", rng=random.Random(s)) for s in range(RUNS)]
    bugged = [any(m.payload.get("has_bug") for m in b.messages if m.kind == ref.MsgKind.DIFF_READY)
              for b in boards]
    fa = sum(r["approved"] and bug for r, bug in zip(rs, bugged))
    return rs, bugged, fa, [reviewer_io(ref, b) for b in boards]


def mean_cost(io, model, scale=1.0):
    return round(sum(cost(model, r, w) for r, w in io) / RUNS / scale, 5)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rs, bugged, fa, io = run(ref)
    source = parity.lesson_dir(PHASE, LESSON).joinpath("code", "main.py").read_text()
    return {
        "bugged": sum(bugged), "false_approvals": fa,
        "reviewer_tokens": sorted({r["tokens_by_role"]["reviewer"] for r in rs}),
        "read": sum(r for r, _ in io) / RUNS, "wrote": sum(w for _, w in io) / RUNS,
        "cost": {m: mean_cost(io, m) for m in PRICE},
        "opus_same_tokenizer": mean_cost(io, "Claude Opus 4.7", TOKENIZER["Claude Opus 4.7"]),
        "team_tokens": sum(r["total_tokens"] for r in rs) / RUNS,
        "model_named_in_code": [n for n in ("GPT", "Opus", "reviewer_model") if n in source],
    }


def verify(result):
    r = result
    gpt, opus = r["cost"]["GPT-5.4"], r["cost"]["Claude Opus 4.7"]
    extra = (TOKENIZER["Claude Opus 4.7"] - 1) * (r["read"] + r["wrote"])
    return [
        practice.Check(
            "ANSWER: false approval stays 41/310 (13.2%); the reviewer's cost goes from $0.0401 to $0.0900 a run",
            (r["bugged"], r["false_approvals"], gpt, opus, r["opus_same_tokenizer"])
            == (310, 41, 0.04014, 0.09003, 0.06926),
            f"false approvals {r['false_approvals']}/{r['bugged']} bugged runs for either model; reviewer $/run "
            f"GPT-5.4 {gpt:.4f}, Opus 4.7 {opus:.4f} (+{opus - gpt:.4f}, {opus / gpt:.2f}x; "
            f"{r['opus_same_tokenizer'] / gpt:.2f}x before the 1.3x tokenizer)",
        ),
        practice.Check(
            "FINDING: no model is called, so the swap cannot move the false-approval rate",
            (r["model_named_in_code"], r["reviewer_tokens"]) == ([], [1800, 3300]),
            f"model names or a model seam in main.py: {r['model_named_in_code']}; reviewer tokens take only "
            f"{r['reviewer_tokens']} (approve / reject-then-approve), whatever the verdict or the model",
        ),
        practice.Check(
            "FINDING: the token delta is the tokenizer, 4.3% of the team's tokens per run",
            (round(r["read"] + r["wrote"], 1), round(extra), round(extra / r["team_tokens"], 3))
            == (5037.4, 1511, 0.043),
            f"reviewer reads {r['read']:,.1f} and writes {r['wrote']:,.1f} tokens a run; Opus 4.7 counts "
            f"+{extra:,.0f} more of the same text, against a {r['team_tokens']:,.0f}-token team run",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
