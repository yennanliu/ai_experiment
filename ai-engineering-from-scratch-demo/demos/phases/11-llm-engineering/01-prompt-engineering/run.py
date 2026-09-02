"""Phase 11 / Lesson 01 -- the lesson's prompt patterns against a real Claude.

The lesson builds a pattern catalogue, a request formatter and a response
scorer, and then feeds them `simulate_llm_call` -- a function that seeds a hash
with the prompt and assembles a plausible-looking answer. So the scores the
lesson prints measure the simulator, not the prompt.

This demo runs the lesson's own `build_prompt` / `format_anthropic_request` /
`score_response` over responses a real model actually produced, replayed from a
committed cassette. Both are scored with identical criteria and printed side by
side, which is the whole argument for cassettes over simulations (D4).

    DEMO_MODE=replay   # default: free, offline, deterministic
    DEMO_MODE=live     # re-records the tape and prints what it cost

One live artefact falls out for free: `format_anthropic_request` sets
`temperature`, which the current models reject outright. `adapt_request` shows
what has to change, which is exactly the kind of thing a simulation can never
tell you.

Run:  uv run demo run phases/11-llm-engineering/01-prompt-engineering
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from harness.cassette import Cassette             # noqa: E402
from harness.cassette import report as cassette_report  # noqa: E402
from harness.explain import explain               # noqa: E402
from harness.parity import load_reference         # noqa: E402
from harness.tiers import mode                    # noqa: E402

LESSON = "phases/11-llm-engineering/01-prompt-engineering/code/prompt_engineering.py"
CASSETTE = Path(__file__).resolve().parent / "cassettes" / "prompt-patterns.json"

MODEL = "claude-opus-5"
MAX_TOKENS = 1024          # these prompts ask for short answers; keeps T2 under $0.02

# Three of the lesson's own patterns, with the lesson's own scoring criteria.
CASES = [
    {
        "pattern": "persona",
        "variables": {
            "role": "a database reliability engineer",
            "experience": "twelve years running Postgres at scale",
            "style": "blunt and concrete",
            "priority": "avoiding downtime over elegance",
            "task": "In under 80 words, say when to add an index and when not to.",
        },
        "criteria": {
            "max_words": 110,
            "required_keywords": ["index", "write"],
            "forbidden_phrases": ["as an ai", "i cannot"],
        },
    },
    {
        "pattern": "chain_of_thought",
        "variables": {
            "problem": "A batch job that took 4 minutes now takes 40. "
                       "Nothing in the code changed. What do you check first, and why?",
        },
        "criteria": {
            "max_words": 400,
            "expected_format": "numbered_list",
            "required_keywords": ["data"],
        },
    },
    {
        "pattern": "template_fill",
        "variables": {
            "text": "Postgres 14.2, 8 vCPU, 32 GB RAM. p99 latency 240ms, "
                    "up from 90ms last week. Connection pool maxes at 100.",
            "template_structure": "Respond with only a JSON object, no prose and no "
                                  "code fence, with keys: version, cpu, ram, p99_ms, "
                                  "regression, pool_max.",
        },
        "criteria": {"expected_format": "json", "max_words": 120},
    },
]


def adapt_request(lesson_request: dict) -> dict:
    """Bring the lesson's request dict up to what the current API accepts.

    The lesson emits `{model, system, messages, temperature, max_tokens}`.
    Current Claude models removed `temperature` -- sending it is a 400 -- and
    the lesson's 2048-token ceiling is more than these prompts need.
    """
    request = dict(lesson_request)
    dropped = request.pop("temperature", None)
    request["model"] = MODEL
    request["max_tokens"] = MAX_TOKENS
    return request, dropped


def score_line(scores: dict) -> str:
    parts = [f"composite {scores['composite_score']:.3f}"]
    if "word_count" in scores:
        parts.append(f"{scores['word_count']} words")
    if "keyword_coverage" in scores:
        parts.append(f"keywords {scores['keyword_coverage']:.0%}")
    if "format_valid" in scores:
        parts.append(f"format {'ok' if scores['format_valid'] else 'INVALID'}")
    if scores.get("forbidden_violations"):
        parts.append(f"violations {scores['forbidden_violations']}")
    return ", ".join(parts)


def main() -> int:
    ref = load_reference(LESSON)
    tape = Cassette.load(CASSETTE)
    run_mode = mode()

    print(f"{len(CASES)} of the lesson's prompt patterns, scored by the lesson's "
          f"own score_response()\n")

    dropped_note = None
    rows = []
    for case in CASES:
        prompt = ref.build_prompt(case["pattern"], case["variables"])
        request, dropped = adapt_request(ref.format_anthropic_request(prompt))
        if dropped is not None:
            dropped_note = dropped

        real = tape.complete(request, run_mode=run_mode)
        simulated = ref.simulate_llm_call("claude-3.5-sonnet", request)

        real_scores = ref.score_response(real.text, case["criteria"])
        sim_scores = ref.score_response(simulated["response"], case["criteria"])
        rows.append((case["pattern"], real_scores, sim_scores))

        print(f"-- {ref.PROMPT_PATTERNS[case['pattern']]['name']} --")
        print(f"   real       {score_line(real_scores)}")
        print(f"   simulated  {score_line(sim_scores)}")
        first_line = real.text.strip().splitlines()[0] if real.text.strip() else ""
        print(f"   said       {first_line[:88]}")
        print()

    real_mean = sum(r[1]["composite_score"] for r in rows) / len(rows)
    sim_mean = sum(r[2]["composite_score"] for r in rows) / len(rows)
    print(f"mean composite score:  real {real_mean:.3f}   simulated {sim_mean:.3f}")
    print("The simulated column is a function of a hash of the prompt. It moves when")
    print("the prompt text changes and not when the prompt gets better, which is why")
    print("this repo records real responses instead of writing more simulators.")

    if dropped_note is not None:
        print(f"\nAPI drift, found by actually calling the API: the lesson's")
        print(f"format_anthropic_request() sets temperature={dropped_note}, which "
              f"{MODEL} rejects\nwith a 400. adapt_request() drops it.")

    cassette_report(tape, run_mode=run_mode)
    tape.save()
    return 0


if __name__ == "__main__":
    if explain(__file__):
        raise SystemExit(0)
    raise SystemExit(main())
