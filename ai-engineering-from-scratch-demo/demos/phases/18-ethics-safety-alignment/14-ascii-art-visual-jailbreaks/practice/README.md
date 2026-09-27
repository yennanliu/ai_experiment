<!-- generated:start -->
# 18-ethics-safety-alignment / 14-ascii-art-visual-jailbreaks

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/18-ethics-safety-alignment/14-ascii-art-visual-jailbreaks/) · upstream spec
`phases/18-ethics-safety-alignment/14-ascii-art-visual-jailbreaks/docs/en.md`

```bash
uv run demo practice run 14-ascii-art-visual-jailbreaks --ex 1
uv run demo explain 14-ascii-art-visual-jailbreaks --ex 1
uv run pytest demos/phases/18-ethics-safety-alignment/14-ascii-art-visual-jailbreaks
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Verify the cloaked string passes a simple keyword filter. Report the char… | code | T0 | `ex01_one_changed_letter_passes_the_filter_the_cloak_spends_126_edits_and_never_trips_the_ppl_filter.py` |
| 2 | Implement a second encoding: base64 for the same target word. Compare the filter-bypass rate… | code | T0 | `ex02_both_encodings_bypass_5_of_5_base64_recovers_all_5_art_2_but_art_survives_97pct_of_one_cell_flips.py` |
| 3 | Read Jiang et al. 2024 Section 4.3 (five-model results). Propose a reason why Claude's ArtPro… | code | T0 | `ex03_asr_alone_cannot_say_why_a_better_reader_refusing_30pct_after_decoding_ties_a_worse_reader.py` |
| 4 | Design a pre-generation defense that detects ASCII-art-shaped regions in the prompt. Measure… | code | T0 | `ex04_a_few_ink_shape_detector_flags_0_of_150_code_table_math_blocks_where_the_ppl_window_flags_7.py` |
| 5 | StructuralSleight lists 10 encoding structures. Sketch a generalized defense that handles all… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

Exercises 1-4 run or read the lesson's `code/main.py`: its 10-glyph
`render_word`, `cloak_prompt`, substring `keyword_filter`, and
`perplexity_heuristic` with the demo's 0.4 threshold. They use only the
reference's own demo sentence and forbidden list.

### 1 — one changed letter passes the filter; the cloak spends 126 edits and never trips the PPL filter

**The cloaked string passes, and it makes 126 edits where 1 is enough.** The
20-character original is blocked. The 143-character cloak (42 of its
characters are '#') is not blocked. The Levenshtein distance between them is
126. Of the 1,980 single-character substitutions of the original, 392 already
pass the filter: every one that changes a letter of the target word
(positions 16-19). The art is there so the *model* can read the word. The
filter would fall to one character.

**The printed caveat "may trip PPL" never comes true.** The cloak scores
0.343. Across the five forbidden words the highest score is 0.383 (weapon),
so the 0.4 PPL filter blocks 0 of 5.

**The cloak is case-sensitive and the filter is not.** `cloak_prompt`
replaces text with exact case matching, while `keyword_filter` lowercases.
Of the 16 case variants of the target word, the cloak gets past the filter
for only 1, the all-lowercase one.

### 2 — both encodings bypass 5 of 5; base64 recovers all 5, art 2, but art survives 97% of one-cell flips

| | keyword bypass | PPL bypass | exact recovery | survives one flip |
|---|---:|---:|---:|---:|
| ASCII art | 5/5 | 5/5 | 2/5 | 78/80, 116/120 |
| base64 | 5/5 | 5/5 | 5/5 | 15/512, 0/512 |

**Bypass is a tie. Recovery goes base64's way on exactness and art's way on
robustness.** Base64 scores at most 0.083 on the PPL heuristic and inverts
exactly for any word, which also means a decode-then-filter defender catches
every base64 cloak. Art spends 20 cells on each letter, so almost every
single-cell flip still reads correctly. All 6 misses are ties between two
equally near glyphs. Only 15 of base64's 512 single-character substitutions
still decode to the word.

**The reference can only cloak 2 of its own 5 forbidden words.** `GLYPHS`
has 10 letters and lacks C, I, L, R and S. The other three words render with
blank cells, and a nearest-glyph reader turns those blanks into P (HAPK,
MAPWAPE, POPPON).

### 3 — ASR alone cannot say why: a better reader refusing 30% after decoding ties a worse reader

The lesson does not reproduce Section 4.3. Its only percent figure is 75%,
and it lists Claude among the models that "all fail". So this answer builds
a model of the two candidate reasons instead of quoting per-model figures.

| reader | misread p | refuse-after-decoding s | recognised | ASR |
|---|---:|---:|---:|---:|
| weaker, no post-decoding safety | 0.10 | 0 | 58.5% | 58.5% |
| stronger, post-decoding safety | 0.05 | 0.3 | 84.2% | 59.4% |

**Proposed reason: Claude applies its safety to the word it reconstructs.**
The rival reason is that Claude simply reads ASCII art worse. The two give
ASR within 0.9 points of each other, so an ASR ranking cannot choose between
them. A transcription-only (ViTC-style) test can.

With s = 0 the lesson's capability-safety trade-off holds exactly. ASR falls
1.0 / 0.842 / 0.585 / 0.332 / 0.14 / 0.021 as p goes 0 / 0.05 / 0.1 / 0.15 /
0.2 / 0.3. With s = 1 it is 0 at every p. The same substring filter that
never fires on the prompt (0 blocks) works once it runs on the model's own
reading.

### 4 — a few-ink shape detector flags 0 of 150 code/table/math blocks where the PPL window flags 7

The corpus is a fixed, seeded fixture of 200 labelled blocks: 50 each of
code (Python, JSON lines, shell), markdown tables, math notation (update
rules, sums, 3x3 matrices) and ASCII diagrams (truth tables, box charts,
scatter plots).

| detector | shipped | letter ink | mixed ink | code | tables | math | diagrams |
|---|---:|---:|---:|---:|---:|---:|---:|
| PPL > 0.4 on any 5-line window | 2/5 | 0/5 | 2/5 | 7 | 0 | 0 | 26 |
| single-ink run | 5/5 | 0/5 | 0/5 | 0 | 0 | 0 | 16 |
| few-ink run | 5/5 | 4/5 | 5/5 | 0 | 0 | 0 | 35 |

**Detect art by its shape, not its density.** The few-ink rule (4 or more
lines of at least 8 characters, using at most 4 symbols and at least 25%
spaces) has no false positives on code, tables or math. Its 35 diagram hits
are truth tables and scatter plots, which are genuine character pictures.
The density rule flags 7 legitimate blocks (the JSON lines) and still misses
hack, malware and poison, the three words that render with blank cells. The
single-ink rule, also clean on code, tables and math, is evaded as soon as
each glyph is drawn with its own letter, and the few-ink rule misses WEAPON
drawn that way.

**The zero depends on the corpus.** A 4x4 identity matrix is flagged by the
few-ink rule. The fixture's matrices are 3 rows of mixed digits, so prompts
with larger sparse matrices would raise the false-positive rate.

### 5 — StructuralSleight: one canonicalize-then-classify pass, costed per structure

*Explain; draws on the lesson's "StructuralSleight" section.* The lesson
names only five of the ten structures (trees, graphs, nested JSON,
CSV-in-JSON, diff-style code blocks). The skill file adds YAML/CSV and the
text encodings from exercises 1-2 (base64, leet-speak, homoglyphs, ASCII art).
Those are the families this sketch covers.

**The generalized defense decodes the content to plain text before judging
it, instead of listing patterns to match.** For each prompt:

1. **Detect structure** with cheap, linear-time sniffers: JSON/YAML parse
   attempts, edge-list and tree-indent patterns, diff markers, base64-shaped
   tokens, and the few-ink art detector from exercise 4.
2. **Canonicalize** each detected region into plain text in reading order.
   Take leaf *values* rather than keys or node ids, because structural
   tokens interleaved between values would break a naive concatenation.
   Fold homoglyphs through a confusables table and leet through a
   substitution map. Decode base64 and read art regions back into letters.
3. **Classify the canonical text** with the same semantic safety classifier
   used on ordinary prompts, not with a substring list. Exercise 3's model
   shows why: safety applied after decoding is what closes the gap.

**Cost.** Steps 1-2 are linear in prompt length. They amount to a handful of
parser passes per prompt, which is negligible next to generation. The real
cost is step 3. Every canonicalized region is one extra classifier input, so
a prompt with k structured regions costs roughly k + 1 classifier calls
instead of 1. The residual risk is structures the sniffers do not recognize,
which the lesson itself says form a large and growing set.
