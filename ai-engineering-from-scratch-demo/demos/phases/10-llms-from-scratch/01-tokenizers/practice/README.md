<!-- generated:start -->
# 10-llms-from-scratch / 01-tokenizers

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/10-llms-from-scratch/01-tokenizers/) · upstream spec
`phases/10-llms-from-scratch/01-tokenizers/docs/en.md`

```bash
uv run demo practice run 01-tokenizers --ex 1
uv run demo explain 01-tokenizers --ex 1
uv run pytest demos/phases/10-llms-from-scratch/01-tokenizers
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Modify the BPE tokenizer to print the vocabulary at each merge step. Watch how "t" + "h" beco… | code | T0 | `ex01_the_word_is_assembled_from_the_right.py` |
| 2 | Add special tokens (`<pad>`, `<eos>`, `<unk>`) to the BPE tokenizer. Assign them IDs 0, 1, 2… | code | T0 | `ex02_the_shift_is_a_caesar_cipher_and_unk_is_dead.py` |
| 3 | Implement the WordPiece merge criterion (likelihood ratio instead of frequency). Train both B… | code | T0 | `ex03_the_ratio_is_a_rarity_detector.py` |
| 4 | Build a multilingual tokenizer efficiency benchmark. Take 10 sentences in English, Spanish, C… | code | T1 | `ex04_per_character_charges_chinese_for_its_own_density.py` |
| 5 | Train your BPE tokenizer on a larger corpus (download a Wikipedia article). Tune the number o… | code | T1 | `ex05_the_target_is_met_by_memorising_the_text.py` |
<!-- generated:end -->

## Answers

The lesson ships two copies of one `BPETokenizer` — `code/bpe.py` is
`code/main.py`'s, plus a `print` in the training loop — and a 387-byte corpus it
trains on for 50 merges. All five exercises turn out to ask about the same
thing from five directions: what the tokenizer learns when nothing separates a
word from the space beside it, and what each exercise's chosen metric can and
cannot see. Exercise 1 narrates a merge path that never happens. Exercise 2
specifies two changes that each break something the lesson already tests.
Exercise 3 implies the wrong winner. Exercise 4 picks the denominator that
inverts its own ranking. Exercise 5 sets a target that its own measurement makes
unmissable.

Exercises 1–3 are **T0** — the lesson imports nothing but `collections`, and
neither do they. Exercises 4 and 5 are **T1**, needing `tiktoken` from the `llm`
group (`uv sync --extra llm`); `tiktoken` downloads and caches its vocabulary
file on first use.

### 1 — "the" is assembled from the right

The lesson's own `train` on its own corpus, 50 merges, every merge logged:

| # | merge | result |
|---:|---|---|
| 1 | `'e'` + `' '` | `'e '` |
| 2 | `'h'` + `'e '` | `'he '` |
| 11 | `' t'` + `'he '` | `' the '` |

**ANSWER: the narrated path does not occur.** The exercise says to watch `'t'` +
`'h'` become `'th'` and then `'th'` + `'e'` become `'the'`. The word is built
from its ending, in three merges, and what the vocabulary ends up holding is
`' the '` — with a space on either side.

**FINDING: neither `'th'` nor `'the'` is ever a token.** Not at 50 merges, and
not at the 235 where the corpus runs dry. `'t'`+`'h'` is rank 8 of the initial
pairs at 7 occurrences, and merge 2 takes the `'h'` before its turn comes. So
`encode("the")` returns `['t', 'h', 'e']` — three single bytes, no better than
the character-level tokenizer of Step 1. The tokenizer learned the word with its
delimiters attached and cannot tokenize the word without them.

**MECHANISM: capitalisation splits the prefix bond and leaves the suffix bond
whole.**

```text
'h' + 'e'  =  12  =  "the" (7)  +  "The" (5)
't' + 'h'  =   7  =  "the" (7)
```

A word that appears sentence-initially has its first letter split across two
cases and its interior not, so its suffix bond always outranks its prefix bond
and BPE assembles it from the right. That is the general case, not an accident
of this corpus.

**FINDING: without pre-tokenisation BPE learns phrases.** 26 of the 50 merges
contain a space and 7 span a word boundary — `'. The '`, `' sat on'`,
`' sat on the '`, `'ate the '`, `' is the '`. Space is a mergeable byte here, so
nothing stops a token from crossing words. Run to exhaustion, 235 merges
compress all 387 bytes of the corpus into **one token**. That is memorisation,
and it is exactly what Exercise 2's pre-tokenisation step exists to prevent.

**CONTROL: pre-tokenised, `'the'` does form — still not via `'th'`.**
Restricting merges to word interiors gives `'he'` at merge 1, `'at'` at 2 and
`'the'` at 3, and no token containing a space anywhere in the run. The
exercise's path is wrong under both readings of BPE, and wrong the same way each
time: the intermediate token is the suffix.

### 2 — the shift is a Caesar cipher and `<unk>` is dead

**ANSWER: done correctly the shift is a pure relabelling.** Every ID rises by
exactly 3 across all four probes, token counts are unchanged, the roundtrip
holds, and the vocabulary goes 306 → 309. Nothing measurable moves — which is
the point. Special tokens buy addressability, not compression.

**FINDING: done as written it is a Caesar cipher.** "Shift all other tokens
accordingly" names the vocabulary and the merge table. But `encode` also starts
from raw byte values (`list(text.encode("utf-8"))`), and shifting the first two
and not the third means no merge can ever fire — the shifted pair IDs no longer
match unshifted bytes:

```text
"The cat sat on the mat."  ->  23 tokens for 23 bytes   (0 of 50 merges fired)
                           ->  'Qeb\x1d`^q\x1dp^q\x1dlk\x1dqeb\x1dj^q+'
```

Same length, no exception raised, no `<unk>` emitted. The tokenizer has become a
ROT-3 cipher and says nothing about it.

**FINDING: `<unk>` can never be emitted.** The base vocabulary covers all 256
bytes, so 日本語, 🙂, الع and `\x00\x01` all roundtrip exactly and token 2
appears in no encoding of anything. The lesson states this itself two sections
earlier — byte-level BPE "never produces an unknown token". `<pad>` and `<eos>`
are real; `<unk>` is an embedding row that can never receive a gradient, a habit
carried over from word-level vocabularies.

**FINDING: the pre-tokeniser as specified is lossy, and its compression win is
the text it deleted.**

| pre-tokeniser | roundtrip | recovered | tokens/byte |
|---|---|---:|---:|
| none (flat byte BPE) | ✅ | 387 / 387 | 0.5013 |
| `text.split()` — as specified | ❌ | 319 / 387 | 0.4367 |
| space attached to the word — GPT-2's fix | ✅ | 387 / 387 | **0.5323** |

`text.split()` discards all 68 spaces, so the lesson's Step 3 `PASS`/`FAIL` line
goes to `FAIL` on every sentence, and its 0.4367 is scored over a denominator it
no longer reproduces. Attach the space to the word and the roundtrip returns —
and pre-tokenisation is then **6.2% worse** than not pre-tokenising at all,
because word boundaries forbid exactly the phrase tokens Exercise 1 found. It is
a correctness fix bought with compression, not a free win.

### 3 — the likelihood ratio is a rarity detector

Both arms run through the lesson's own `_get_pairs` and `_merge_pair`, and the
BPE arm is checked merge-for-merge against the reference's `train`, so only the
scoring function differs. "Linguistically meaningful" is scored mechanically: a
merged token is **reusable** if its stripped form occurs inside more than one
distinct word of the corpus.

| criterion | tokens/byte | hapax merges | reusable | first merges |
|---|---:|---:|---:|---|
| BPE — `count(AB)` | **0.5013** | 0 / 50 | **34 / 50** | `'e '`, `'he '`, `'at'`, `' t'`, `'. '` |
| WordPiece — `count(AB)/(count(A)·count(B))` | 0.8527 | **46 / 50** | 3 / 50 | `'LP'`, `'NLP'`, `'bw'`, `'qu'`, `'ubw'` |
| the same, with `count(AB) ≥ 2` | 0.6176 | 0 / 50 | 30 / 50 | `'iz'`, `'Th'`, `'ok'`, `'niz'`, `'wo'` |

**ANSWER: BPE, decisively, and the exercise implies the opposite.** 34 reusable
merges to 3, and 70% better compression.

**MECHANISM: dividing by `count(A)·count(B)` removes the frequency floor.** The
ratio reaches its maximum of 1.0 when both parts occur exactly once and always
together — a score no frequent pair can match. So the criterion is a rarity
detector: 46 of its 50 merges fire on a pair occurring **once**, mean 1.14
against BPE's 3.86, and the first merge is `'LP'`, from the single occurrence of
"NLP" in the corpus.

**FINDING: the formula is right and the setting is wrong.** Real WordPiece
implementations carry a minimum-count threshold that the lesson's one-line Key
Terms row omits. Restore one and the degeneracy goes: hapax merges 46 → 0,
reusable merges 3 → 30, compression 0.8527 → 0.6176. Still 23% worse than BPE,
which is the honest answer to "which produces more linguistically meaningful
subwords" — the criterion buys a different notion of subword and pays
compression for it.

**FINDING: the merge path Exercise 1 attributes to BPE is this criterion's.** At
a floor of 5:

```text
1: 'T' + 'h'  -> 'Th'       3: 't' + 'h'  -> 'th'
2: 'Th' + 'e' -> 'The'      4: 'th' + 'e' -> 'the'
```

Exactly the sequence the lesson narrates for BPE, which BPE produces at no merge
count on this corpus. `'t'` is rarer than `'e'` or `' '`, so dividing by the
parts is what promotes the t–h bond that raw frequency never reaches.

### 4 — tokens per character charges Chinese for its own density

The 10 sentences are the **same** 10 sentences in all five languages. The
exercise does not say parallel, and that is the whole difficulty: across
unrelated samples, tokens per character measures which sentences were picked.

| language | chars | tokens | **tok/char** | tax | **tok/sentence** | tax | partial-UTF-8 tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| English | 283 | 67 | 0.2367 | 1.00× | 6.70 | 1.00× | 0% |
| Spanish | 271 | 83 | 0.3063 | 1.29× | 8.30 | 1.24× | 0% |
| Arabic | 210 | 149 | 0.7095 | 3.00× | 14.90 | **2.22×** | **0%** |
| Korean | 143 | 141 | 0.9860 | 4.16× | 14.10 | 2.10× | 45% |
| Chinese | 85 | 102 | **1.2000** | **5.07×** | 10.20 | **1.52×** | 41% |

**ANSWER: the metric as specified** is the `tok/char` column — a tax of 1.29×,
3.00×, 4.16× and 5.07×, worst for Chinese.

**FINDING: that ranking is script density, not tokenizer quality.** The same ten
meanings are 283 characters of English and 85 of Chinese, so dividing by
characters charges Chinese 3.3× for carrying more meaning per character. Hold
the meaning fixed and count what the context window and the invoice actually
see — tokens per sentence — and the top of the order reverses. Chinese is the
worst language on the exercise's metric and the best non-Latin one on this one,
from the same encodings of the same text.

**MECHANISM: two unrelated failures wear the same number.** 41% of the Chinese
tokens and 45% of the Korean ones decode to U+FFFD on their own — `cl100k_base`
splits below the character, so one hanzi costs two or three tokens. **0%** of
the Arabic tokens do, at a higher tax than either. Arabic pays for merges that
were never learned; Chinese and Korean pay for characters that do not survive
the byte-level fallback. One number per language reports both as one problem,
and the two have different fixes.

**CONTROL: vocabulary size is the fix, and the size of the fix is measurable.**

| language | `cl100k_base` (~100K) | `o200k_base` (~200K) |
|---|---:|---:|
| Spanish | 1.24× | 1.03× |
| Chinese | 1.52× | **1.01×** |
| Korean | 2.10× | 1.42× |
| Arabic | 2.22× | **1.12×** |

The worst tax on the board falls from 2.22× to 1.42× and no language gets worse.
The lesson's claim that Llama 3 quadrupled its vocabulary for fairer
multilingual compression is measurable rather than asserted — and it means this
exercise quantifies the tax of the tokenizer it was told to use, not of
tokenization.

### 5 — the target is met by memorising the text

"Download a Wikipedia article" is read as *a larger piece of encyclopedic
English prose*, and the one used is the lesson's own `docs/en.md`: 22,839 bytes
and 3,210 words, 59× the demo corpus, already on disk and hash-pinned by the
coverage gate. "On that same text" is read literally, because it is the
load-bearing phrase.

**ANSWER: 1,070 merges.** `cl100k_base` compresses the article to 0.2532 tokens
per byte, so the window closes at 0.2785. 1,070 merges reach 0.2785 — 110.0% of
tiktoken — and 1,069 miss it.

**FINDING: the target is a point on a slide, so there is nothing to tune.**

| merges | on the training text | vs tiktoken |
|---:|---:|---:|
| 1,070 | 0.2785 | 110.0% |
| 2,140 | 0.1924 | 76.0% |
| 4,280 | 0.0987 | 39.0% |

Every count above the answer satisfies the window too, and the curve keeps
improving all the way down to the single token Exercise 1 found at exhaustion.

**FINDING: the curve is monotone because train and test are the same text.**
Hold out the last 20% of the article and the slide stops dead:

| merges | 1,300 | 1,400 | 1,500 | 1,600 | 1,700 | 1,800 | 2,100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| held-out tokens/byte | 0.3483 | 0.3415 | 0.3398 | **0.3360** | 0.3360 | 0.3360 | 0.3360 |

Identical, bit for bit, across 500 additional merges. Every merge past ~1,600
encodes a string that occurs only in the 80% it trained on.

**FINDING: within 10% is unreachable on held-out text at any merge count.** The
floor is 0.3360 against tiktoken's 0.2314 on the same held-out text — **145.2%**
— and no merge count improves on it. The exercise's goal is over-satisfied under
the measurement it specifies and impossible under the honest one, on a corpus
this size. That gap *is* the relationship between corpus size, merge count and
compression quality the exercise says it forces you to understand, and the
metric it names is the one that hides it.
