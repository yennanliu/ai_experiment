"""Exercise 4 -- no model is called, so faithfulness moves 0.000, and serverless Llama costs 14.5% of Sonnet 4.6.

    Swap Claude Sonnet 4.7 for Llama 3.3 70B self-hosted. Measure $/query
    and faithfulness delta.

Reading of the exercise: the lesson's `code/main.py` calls no model.
`chat_turn` builds a `PromptLayout` and then answers with a stub that joins
the first 60 characters of each retrieved chunk. So the swap is measured at
the seam a model would fill. The prompt is the layout's system + policy +
context + question, and the completion is the stub's answer at its real
length, both counted at 4 characters per token. These are priced for 12
golden questions across the three user groups. Prices were read on
2026-09-29: Claude Sonnet 4.6 at $3 / $15 per MTok and Sonnet 5 at $2 / $10
(platform.claude.com/docs/en/about-claude/pricing); Llama 3.3 70B serverless
at $1.04 / $1.04 and an H100 at $5.49 per GPU-hour list (www.together.ai/pricing).
The serverless price stands in for per-token self-hosting. Two H100s,
enough for FP8 weights plus KV cache, give the break-even load: the queries
per hour at which a rented pair costs the same as Sonnet 4.6. Faithfulness is
the share of claims ("anchor -> text") found verbatim in the cited chunk.

**ANSWER: the faithfulness delta is 0.000, 1.000 under either model,
because neither is called.** A query is about 114 prompt tokens and 60
completion tokens. That costs $1.24 per 1,000 queries on Sonnet 4.6, $0.82
on Sonnet 5 and $0.18 on serverless Llama, so Llama is 14.5% of Sonnet 4.6.
Self-hosting on two H100s ($10.98 an hour) matches Sonnet 4.6 only above
8,878 queries an hour, about 2.5 per second, sustained. Prompt caching changes
none of this: every prefix is under Sonnet's 1,024-token minimum.

**FINDING: "Claude Sonnet 4.7" is not a model.** Anthropic's price list
(read 2026-09-29) has Sonnet 4.5, 4.6, 5 and 5.5, and Opus 4.7, but no
Sonnet 4.7. The lesson names it five times, the exercise included.

**FINDING: the stub cuts every chunk at 60 characters, and 4 of the 5 end
mid-word** ("termina", "agreemen", "annua", "porta"). The fifth, MSA, ends
on "within 30" and loses "days per GDPR Article 17". The verbatim-substring
measure still scores every claim 1.000. A word-level check would mark the
four broken claims unsupported, so the faithfulness number depends on the
scorer before any model is involved.
"""

from __future__ import annotations

import re

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "08-production-rag-chatbot"
CHARS_PER_TOKEN, MIN_CACHE_TOKENS = 4, 1024
PRICES = {"claude-sonnet-4-6": (3.00, 15.00), "claude-sonnet-5": (2.00, 10.00), "llama-3.3-70b serverless": (1.04, 1.04)}
H100_HOUR, GPUS = 5.49, 2
AG, CH, CS = ("analyst", "GDPR"), ("counsel", "HIPAA"), ("counsel", "SOC2")
GOLDEN = [(AG, "how many days to delete EU user profiles after termination"),
          (AG, "when must EU user profiles be deleted"), (AG, "deletion deadline for the restricted data category"),
          (AG, "how fast is restricted data deleted after a termination notice"),
          (AG, "how can users request a data export"), (CH, "what happens to PHI when the agreement terminates"),
          (CH, "how many days to return or destroy PHI"), (CH, "can users request data export through the portal"),
          (CS, "how often are privileged users access reviewed"), (CS, "access review cadence for standard users"),
          (CS, "is the access review quarterly or annual"), (AG, "where do users export their data")]


def faithfulness(ref, answer):
    text = {c.anchor(): c.text for c in ref.CORPUS}
    claims = [c.split(" -> ", 1) for c in answer.split(": ", 1)[1].split("; ")]
    return sum(body in text[anchor] for anchor, body in claims) / len(claims)


def turn(ref, who, q):
    reply = ref.chat_turn(q, *who, ref.CORPUS, ref.PromptCache())
    by_anchor = {c.anchor(): f"[{c.anchor()}] {c.text}" for c in ref.CORPUS}
    layout = ref.PromptLayout(ref.SYSTEM_PROMPT, f"role={who[0]} jurisdiction={who[1]}",
                              [by_anchor[a] for a in reply["citations"]], q)
    prompt = "\n".join([layout.system, layout.policy, *layout.context, layout.question])
    prefix = len(prompt) - len(q)
    return {"in": len(prompt) / CHARS_PER_TOKEN, "out": len(reply["answer"]) / CHARS_PER_TOKEN,
            "prefix": prefix / CHARS_PER_TOKEN, "faith": faithfulness(ref, reply["answer"])}


def truncation(ref):
    cut = {c.anchor(): c.text[:60] for c in ref.CORPUS}
    mid_word = sorted(c.anchor() for c in ref.CORPUS if re.match(r"\w\w", c.text[59:61]))
    return {"mid_word": mid_word, "msa_cut": cut["MSA-2024-03-11 s12.4"]}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = [turn(ref, who, q) for who, q in GOLDEN]
    tin, tout = (sum(r[k] for r in rows) / len(rows) for k in ("in", "out"))
    per_1k = {m: round((tin * i + tout * o) / 1e3, 2) for m, (i, o) in PRICES.items()}
    per_query_sonnet = (tin * 3.00 + tout * 15.00) / 1e6
    source = (parity.lesson_dir(PHASE, LESSON) / "code" / "main.py").read_text()
    return {
        "tokens_in": round(tin), "tokens_out": round(tout), "per_1k": per_1k,
        "llama_share": round(per_1k["llama-3.3-70b serverless"] / per_1k["claude-sonnet-4-6"], 3),
        "breakeven_qph": round(H100_HOUR * GPUS / per_query_sonnet), "max_prefix": round(max(r["prefix"] for r in rows)),
        "faith": sorted({r["faith"] for r in rows}),
        "calls_a_model": any(w in source for w in ("import anthropic", "import openai", "requests.post", "urllib")),
        "sonnet_47_mentions": parity.doc_text(PHASE, LESSON).count("Claude Sonnet 4.7"), **truncation(ref),
    }


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: faithfulness delta 0.000 (1.000 under either model; main.py calls no model)",
            r["faith"] == [1.0] and r["calls_a_model"] is False,
            f"faithfulness values over {len(GOLDEN)} golden questions {r['faith']}; model call in main.py: "
            f"{r['calls_a_model']}",
        ),
        practice.Check(
            "ANSWER: $ per 1,000 queries -- Llama serverless is 14.5% of Sonnet 4.6; two H100s break even at 8,878/h",
            (r["tokens_in"], r["tokens_out"]) == (114, 60) and r["per_1k"] == {
                "claude-sonnet-4-6": 1.24, "claude-sonnet-5": 0.82, "llama-3.3-70b serverless": 0.18}
            and r["llama_share"] == 0.145 and r["breakeven_qph"] == 8878 and r["max_prefix"] < MIN_CACHE_TOKENS,
            f"{r['tokens_in']} in / {r['tokens_out']} out tokens; $/1k {r['per_1k']}; Llama/Sonnet 4.6 "
            f"{r['llama_share']}; break-even {r['breakeven_qph']} queries/h; longest prefix {r['max_prefix']} tokens",
        ),
        practice.Check(
            "FINDING: 'Claude Sonnet 4.7' is not a model, and the lesson names it 5 times",
            r["sonnet_47_mentions"] == 5,
            f"{r['sonnet_47_mentions']} mentions; Anthropic's 2026-09-29 list has Sonnet 4.5/4.6/5/5.5, Opus 4.7",
        ),
        practice.Check(
            "FINDING: the stub cuts chunks at 60 chars; 4 of 5 end mid-word and MSA loses its unit",
            len(r["mid_word"]) == 4 and "MSA-2024-03-11 s12.4" not in r["mid_word"] and r["msa_cut"].endswith("within 30"),
            f"mid-word cuts {r['mid_word']}; MSA claim reads '{r['msa_cut']}'",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
