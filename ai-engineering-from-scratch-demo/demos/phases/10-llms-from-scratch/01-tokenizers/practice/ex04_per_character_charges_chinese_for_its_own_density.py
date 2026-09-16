"""Exercise 4 — tokens per character charges Chinese for its own density.

    Build a multilingual tokenizer efficiency benchmark. Take 10 sentences in
    English, Spanish, Chinese, Korean, and Arabic. Tokenize each with tiktoken
    (cl100k_base) and measure the average tokens per character. Quantify the
    "multilingual tax" for each language.

Reading of the exercise: the 10 sentences are the **same** 10 sentences in all
five languages. The exercise does not say parallel, and that is the whole
difficulty -- across unrelated samples, tokens per character measures which
sentences were picked, not which tokenizer was used. Fixing the meaning is the
only way the word "tax" means anything, so the 50 strings below are 10 meanings
written five ways, and every ratio is against the English of the same 10.

**ANSWER: the metric as specified.** Tokens per character under cl100k_base:
English 0.2367, Spanish 0.3063, Arabic 0.7095, Korean 0.9860, Chinese 1.2000 --
a tax of 1.29x, 3.00x, 4.16x and 5.07x.

**FINDING: that ranking is an artifact of script density.** The same 10 meanings
are 283 characters of English and 85 of Chinese, so dividing by characters
charges Chinese 3.3x for carrying more meaning per character. Hold the meaning
fixed and measure what the context window and the invoice actually see --
tokens per sentence -- and the top of the order reverses: Chinese **1.52x** is
the *lowest* non-Latin tax and Arabic **2.22x** the highest. Chinese goes from
worst to best on the same data.

**MECHANISM: two unrelated failures wear the same number.** 41% of the Chinese
tokens and 45% of the Korean ones decode to U+FFFD on their own -- cl100k splits
below the character, so one hanzi costs two or three tokens. **0%** of the
Arabic tokens do. Arabic's tax is the absence of Arabic merges; Chinese's and
Korean's is sub-character splitting. One number per language cannot tell them
apart, and the fixes are different.

**CONTROL: vocabulary size is the fix, and the size of the fix is measurable.**
The same 50 sentences under `o200k_base`, twice the vocabulary: Chinese
1.52x -> 1.01x, Arabic 2.22x -> 1.12x, Korean 2.10x -> 1.42x. The worst tax on
the board falls from 2.22x to 1.42x. That is the lesson's claim about Llama 3's
128K vocabulary, measured rather than asserted -- and it means the exercise
quantifies the tax of the *tokenizer it was told to use*, not of tokenization.

Structure: `PARALLEL` is the fixed-meaning corpus; `measure` returns all four
denominators for one encoding so the metrics can be compared rather than chosen;
`partial` counts tokens that are not a whole character.
"""

from __future__ import annotations

from harness import practice

try:
    import tiktoken
except ImportError as exc:                       # pragma: no cover - env guard
    raise practice.Skip(f"needs tiktoken: uv sync --extra llm ({exc})") from None

ASKED = "cl100k_base"
LARGER = "o200k_base"
PARALLEL = {
    "English": [
        "The cat sleeps on the chair.", "I drink water every morning.",
        "The book is on the table.", "She works at a hospital.",
        "We went to the market yesterday.", "The weather is cold today.",
        "He is reading a new book.", "The children play in the garden.",
        "My friend lives in a small city.", "Learning a language takes time."],
    "Spanish": [
        "El gato duerme en la silla.", "Bebo agua cada mañana.",
        "El libro está sobre la mesa.", "Ella trabaja en un hospital.",
        "Ayer fuimos al mercado.", "Hoy hace frío.",
        "Él está leyendo un libro nuevo.", "Los niños juegan en el jardín.",
        "Mi amigo vive en una ciudad pequeña.", "Aprender un idioma lleva tiempo."],
    "Chinese": [
        "猫在椅子上睡觉。", "我每天早上喝水。", "书在桌子上。", "她在医院工作。",
        "我们昨天去了市场。", "今天天气很冷。", "他在读一本新书。",
        "孩子们在花园里玩。", "我的朋友住在一个小城市。", "学习一门语言需要时间。"],
    "Korean": [
        "고양이가 의자에서 잠을 잔다.", "나는 매일 아침 물을 마신다.",
        "책이 탁자 위에 있다.", "그녀는 병원에서 일한다.",
        "우리는 어제 시장에 갔다.", "오늘 날씨가 춥다.",
        "그는 새 책을 읽고 있다.", "아이들이 정원에서 논다.",
        "내 친구는 작은 도시에 산다.", "언어를 배우는 데는 시간이 걸린다."],
    "Arabic": [
        "القطة تنام على الكرسي.", "أشرب الماء كل صباح.",
        "الكتاب على الطاولة.", "هي تعمل في مستشفى.",
        "ذهبنا إلى السوق أمس.", "الطقس بارد اليوم.",
        "هو يقرأ كتابا جديدا.", "الأطفال يلعبون في الحديقة.",
        "صديقي يعيش في مدينة صغيرة.", "تعلم اللغة يستغرق وقتا."],
}


def measure(encoding, sentences):
    """Every denominator at once — the metrics are compared, not chosen."""
    ids = [t for s in sentences for t in encoding.encode(s)]
    chars = sum(len(s) for s in sentences)
    raw = sum(len(s.encode("utf-8")) for s in sentences)
    return {"tokens": len(ids), "chars": chars, "bytes": raw,
            "per_char": len(ids) / chars, "per_sentence": len(ids) / len(sentences),
            "per_byte": len(ids) / raw, "partial": partial(encoding, ids) / len(ids)}


def partial(encoding, ids):
    """Tokens that are not a whole character — they decode to U+FFFD alone."""
    return sum(1 for t in ids if "�" in encoding.decode([t]))


def arms(name):
    encoding = tiktoken.get_encoding(name)
    rows = {lang: measure(encoding, sents) for lang, sents in PARALLEL.items()}
    base = rows["English"]
    for row in rows.values():
        row["tax_char"] = row["per_char"] / base["per_char"]
        row["tax_sentence"] = row["per_sentence"] / base["per_sentence"]
    return rows


def table(rows, order, fmt):
    """One `lang value` row per language, in `order` — the detail strings' only formatter."""
    return ", ".join(fmt(lang, rows[lang]) for lang in order)


def solve():
    asked, larger = arms(ASKED), arms(LARGER)
    return {
        "asked": asked,
        "larger": larger,
        "by_char": sorted(asked, key=lambda k: asked[k]["tax_char"]),
        "by_sentence": sorted(asked, key=lambda k: asked[k]["tax_sentence"]),
        "no_regression": all(larger[k]["tax_sentence"] <= asked[k]["tax_sentence"]
                             for k in asked),
    }


def verify(result):
    asked, larger = result["asked"], result["larger"]
    by_char, by_sentence = result["by_char"], result["by_sentence"]
    worst = max(asked.values(), key=lambda r: r["tax_sentence"])["tax_sentence"]
    worst_big = max(larger.values(), key=lambda r: r["tax_sentence"])["tax_sentence"]
    return [
        practice.Check(
            f"ANSWER: tokens per character under {ASKED}, English 0.2367 to Chinese 1.2000",
            by_char == ["English", "Spanish", "Arabic", "Korean", "Chinese"]
            and 4.9 < asked["Chinese"]["tax_char"] < 5.2,
            "the metric the exercise asks for, over 10 sentences per language: "
            + table(asked, by_char, lambda k, r: f"{k} {r['per_char']:.4f} "
                    f"({r['tax_char']:.2f}x)"),
        ),
        practice.Check(
            "FINDING: that ranking is script density, not tokenizer quality",
            by_sentence[-1] == "Arabic" and by_sentence[1] == "Spanish"
            and asked["Chinese"]["tax_sentence"] < asked["Korean"]["tax_sentence"]
            < asked["Arabic"]["tax_sentence"],
            f"the same 10 meanings are {asked['English']['chars']} characters of English and "
            f"{asked['Chinese']['chars']} of Chinese, so per-character charges Chinese "
            f"{asked['English']['chars'] / asked['Chinese']['chars']:.1f}x for carrying more "
            "meaning per character. Hold the meaning fixed and count what the context window "
            "and the invoice see -- tokens per sentence -- and the order at the top reverses: "
            + table(asked, by_sentence, lambda k, r: f"{k} {r['tax_sentence']:.2f}x")
            + ". Chinese is the worst language on the exercise's metric and the best "
            "non-Latin one on this one, from the same encodings of the same text",
        ),
        practice.Check(
            "MECHANISM: sub-character splitting and missing merges wear the same number",
            asked["Chinese"]["partial"] > 0.3 and asked["Korean"]["partial"] > 0.3
            and asked["Arabic"]["partial"] == 0.0 == asked["English"]["partial"],
            f"{100 * asked['Chinese']['partial']:.0f}% of the Chinese tokens and "
            f"{100 * asked['Korean']['partial']:.0f}% of the Korean ones decode to U+FFFD on "
            f"their own -- {ASKED} splits below the character, so one hanzi costs two or "
            f"three tokens -- while {100 * asked['Arabic']['partial']:.0f}% of the Arabic "
            f"tokens do, at a higher tax than either. Arabic pays for merges that were never "
            "learned; Chinese and Korean pay for characters that do not survive the "
            "byte-level fallback. One number per language reports both as the same problem",
        ),
        practice.Check(
            f"CONTROL: {LARGER} nearly erases the tax, which is what the lesson claims of 128K",
            worst_big < 0.7 * worst and result["no_regression"],
            "the same 50 sentences at twice the vocabulary: "
            + table(larger, by_sentence[1:], lambda k, r: f"{k} "
                    f"{asked[k]['tax_sentence']:.2f}x -> {r['tax_sentence']:.2f}x")
            + f". The worst tax on the board falls {worst:.2f}x -> {worst_big:.2f}x and no "
            "language gets worse. The lesson's claim that Llama 3 quadrupled its vocabulary "
            "for fairer multilingual compression is measurable, and it means this exercise "
            "quantifies the tax of the tokenizer it was told to use, not of tokenization",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
