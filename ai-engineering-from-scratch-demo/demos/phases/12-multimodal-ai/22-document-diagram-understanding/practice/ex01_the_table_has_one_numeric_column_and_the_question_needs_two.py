"""Exercise 1 — the table has one numeric column and the question needs two.

    Your project is 10M invoices per day. Which stack minimizes cost-per-page
    without losing accuracy?

Reading of the exercise: the cost side is arithmetic over the lesson's own token
budget at a stated price, and it is done first so that the second clause --
"without losing accuracy" -- can be checked against what the table actually
carries, which is one number per row and no accuracy anywhere.

**ANSWER: the OCR pipeline with LayoutLMv3, at 512 tokens a page.** Ten million
pages a day is **5.12 billion** input tokens, **$1,280** at a stated $0.25 per
million, against **$30,000** for the frontier row -- a **23.4x** span across six
stacks that differ in nothing the table measures except token count.

**FINDING: the table is not monotone in era.** Donut and Nougat, both era 2, cost
**4,096**; the era-3 AnyRes row costs **2,916**, **28.8%** less. "Newer is dearer"
is false in the lesson's own rows, so the era ordering carries no cost
information.

**FINDING: "without losing accuracy" cannot be evaluated here.** Six rows, one
numeric column, zero accuracy figures. The clause that decides the answer is the
one the evidence does not address -- the same shape Lesson 12.07's exercise 1
finds in that lesson's encoder table, one phase over.

**FINDING: and the 512 does not cover the lesson's own three-stream input.**
`layoutlm_input` returns 8 text ids and **256** patch ids for an eight-word page:
**97.0%** of the positions are patches, before a single additional word of text.
Half the row's budget is spent on a 16x16 grid that the row describes as a "small
image".

Structure: `daily_tokens` and `daily_cost` price one row at the stated volume,
`STACKS` transcribes the lesson's table, and `streams` reads what
`layoutlm_input` actually emits.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "22-document-diagram-understanding"
PAGES_PER_DAY = 10_000_000
PRICE_PER_MILLION = 0.25
STACKS = {
    "OCR pipeline + LayoutLMv3": (512, 1),
    "Donut (OCR-free)": (4096, 2),
    "Nougat (paper pages)": (4096, 2),
    "VLM AnyRes 4-tile (LLaVA)": (2916, 3),
    "VLM native 2048 (Qwen2.5-VL)": (8192, 3),
    "VLM native 2576 (Claude 4.7)": (12000, 3),
}


def daily_tokens(per_page, pages=PAGES_PER_DAY):
    return per_page * pages


def daily_cost(per_page, pages=PAGES_PER_DAY, price=PRICE_PER_MILLION):
    return round(per_page * pages / 1e6 * price)


def streams(ref):
    data = ref.layoutlm_input(ref.mock_page())
    return {name: len(values) for name, values in data.items()}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    costs = {name: daily_cost(tokens) for name, (tokens, _) in STACKS.items()}
    cheapest = min(costs, key=costs.get)
    dearest = max(costs, key=costs.get)
    era_two = [tokens for tokens, era in STACKS.values() if era == 2]
    era_three = [tokens for tokens, era in STACKS.values() if era == 3]
    lengths = streams(ref)
    positions = lengths["text_ids"] + lengths["patch_ids"]
    return {
        "costs": costs, "cheapest": cheapest, "dearest": dearest,
        "cheapest_tokens": STACKS[cheapest][0],
        "daily_tokens": daily_tokens(STACKS[cheapest][0]),
        "span": round(STACKS[dearest][0] / STACKS[cheapest][0], 1),
        "era_two_min": min(era_two), "era_three_min": min(era_three),
        "monotone": min(era_three) >= min(era_two),
        "era_gap_pct": round((1 - min(era_three) / min(era_two)) * 100, 1),
        "rows": len(STACKS), "numeric_columns": 1, "accuracy_columns": 0,
        "streams": lengths, "positions": positions,
        "patch_share": round(lengths["patch_ids"] / positions * 100, 1),
        "budget": STACKS[cheapest][0],
        "patches_of_budget": round(lengths["patch_ids"] / STACKS[cheapest][0] * 100),
    }


def verify(result):
    costs, lengths = result["costs"], result["streams"]
    return [
        practice.Check(
            "ANSWER: the OCR pipeline with LayoutLMv3 -- $1,280 a day against $30,000",
            all([result["cheapest"] == "OCR pipeline + LayoutLMv3",
                 result["cheapest_tokens"] == 512,
                 result["daily_tokens"] == 5_120_000_000,
                 costs[result["cheapest"]] == 1280,
                 costs[result["dearest"]] == 30000, result["span"] == 23.4]),
            f"{PAGES_PER_DAY:,} pages at {result['cheapest_tokens']} tokens is "
            f"{result['daily_tokens'] / 1e9:.2f} billion input tokens, "
            f"${costs[result['cheapest']]:,} at a stated ${PRICE_PER_MILLION} per million, "
            f"against ${costs[result['dearest']]:,} for {result['dearest']} -- a "
            f"{result['span']}x span",
        ),
        practice.Check(
            "FINDING: the table is not monotone in era",
            all([result["era_two_min"] == 4096, result["era_three_min"] == 2916,
                 not result["monotone"], result["era_gap_pct"] == 28.8]),
            f"the cheapest era-2 row is {result['era_two_min']:,} tokens and the cheapest "
            f"era-3 row is {result['era_three_min']:,} -- {result['era_gap_pct']}% less. "
            "'Newer is dearer' is false in the lesson's own rows, so the era ordering carries "
            "no cost information",
        ),
        practice.Check(
            "FINDING: 'without losing accuracy' cannot be evaluated here",
            all([result["rows"] == 6, result["numeric_columns"] == 1,
                 result["accuracy_columns"] == 0]),
            f"{result['rows']} rows, {result['numeric_columns']} numeric column and "
            f"{result['accuracy_columns']} accuracy figures. The clause that decides the "
            "answer is the one the evidence does not address -- the same shape Lesson 12.07's "
            "exercise 1 finds in that lesson's encoder table",
        ),
        practice.Check(
            "FINDING: the 512 does not cover the lesson's own three-stream input",
            all([lengths == {"text_ids": 8, "bbox_stream": 8, "patch_ids": 256},
                 result["positions"] == 264, result["patch_share"] == 97.0,
                 result["patches_of_budget"] == 50]),
            f"layoutlm_input returns {lengths} for an eight-word page -- "
            f"{result['positions']} positions, of which {result['patch_share']}% are patches "
            f"and none carry text. The 16x16 grid alone is {result['patches_of_budget']}% of "
            f"the {result['budget']}-token row that describes it as a 'small image'",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
