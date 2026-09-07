<!-- generated:start -->
# 04-computer-vision / 19-ocr-document-understanding

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/04-computer-vision/19-ocr-document-understanding/) · upstream spec
`phases/04-computer-vision/19-ocr-document-understanding/docs/en.md`

```bash
uv run demo practice run 19-ocr-document-understanding --ex 1
uv run demo explain 19-ocr-document-understanding --ex 1
uv run pytest demos/phases/04-computer-vision/19-ocr-document-understanding
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | (Easy) Train the TinyCRNN on 5-digit random numeric strings for 500 steps. Report CER on a he… | code | T1 | `ex01_digit_cer_renderer_collision.py` |
| 2 | (Medium) Replace greedy decoding with beam search (beam_width=5). Report CER delta. On which… | code | T1 | `ex02_beam_search_cer_delta.py` |
| 3 | (Hard) Use PaddleOCR on a set of 20 receipts, extract line items, and compute F1 against hand… | code | T0 | `ex03_receipt_pair_f1_vs_cer.py` |
<!-- generated:end -->

## Answers

The lesson's CTC machinery is correct — `ctc_loss`, `greedy_ctc_decode` and
`TinyCRNN` all do what they claim, and both training runs below drive the loss
down. What the three exercises find is that **the lesson's data generator cannot represent
the task it is asked to represent**, and that **the two metrics the exercises ask
for measure something narrower than they sound like they do**. `synthetic_line`
picks its ink from `shade = 0.0 if c.isalnum() else 0.5`, so every alnum string of
a given length renders to the same bitmap; and PaddleOCR, pytesseract, easyocr and
transformers are all absent from this repo, so exercise 3's engine half is
recorded as a measurement rather than routed around.

### 1 — the CER is 1.0000, and no training run could have moved it

`ex01_digit_cer_renderer_collision.py` — 143 code lines, 6 checks.

**ANSWER: 500 steps on 5-digit random numeric strings gives held-out CER
1.0000.** Adam at lr=1e-3, batch 8, the lesson's own `TinyCRNN` + `ctc_loss` +
`greedy_ctc_decode`, 50 held-out strings:

| | value |
|---|---:|
| held-out CER | **1.0000** |
| exact matches | **0 / 50** |
| decodes equal to `""` | **50 / 50** |
| distinct outputs over the whole held-out set | **1** |

**FINDING: the renderer is character-blind.** `synthetic_line` keys its shade off
`c.isalnum()`, a function of the character *class*, not the character:

| | value |
|---|---:|
| distinct bitmaps over the 36 alnum symbols of `VOCAB` | **1** |
| shades reachable through `build_batch` | **0.0, 1.0** |
| `build_batch(["12345"])` vs `build_batch(["67890"])`, 2560 pixels | **bit-identical** |
| same pair under one shade per digit | **differ** |

Every 5-digit string is literally the same image, so no optimiser, schedule or
step count could have produced a lower CER.

**MECHANISM: the loss plateau sits 14% above an exact information floor.** A
constant image cannot carry the `5*log(10) = 11.5129` nats a uniform 5-digit target
needs, and `reduction="mean"` divides by target length, so nothing can beat
`log(10) = 2.3026` nats/char over the 20 available time steps:

| | nats/char |
|---|---:|
| step 0 | 11.412 |
| mean of the last 50 steps | **2.6252** |
| information-theoretic floor | **2.3026** |

Adam converged *onto* the floor. The optimiser is fine; the data is empty.

**FINDING: CER 1.0 is worse than guessing.** CTC breaks the tie by putting its
mass on blank, and a decode of `""` costs one deletion per reference character.
Emitting the single best repeated digit instead scores **0.8560** on the same
held-out set — **16.8% better** than the trained model.

**CONTROL: the same loop with a sighted renderer reaches CER 0.0680.**
Over-painting each character cell with `DIGITS.index(c)/10` and changing nothing
else — same model, same loss, same decoder, same 500 steps and seed:

| arm | CER | exact | distinct outputs | plateau |
|---|---:|---:|---:|---:|
| lesson renderer | **1.0000** | 0/50 | 1 | 2.6252 |
| one shade per digit | **0.0680** | **35/50** | 50 | 0.2066 |

**CONTROL: the 0.5 shade branch is dead code.** `synthetic_line(" ")` does paint
`[0.5, 1.0]`, but `build_batch` calls `VOCAB.index(c)` on every character and
`VOCAB` holds only blank plus the 36 alnum symbols, so `build_batch(["a b"])`
raises `ValueError: list.index(x): x not in list`. Through the lesson's own
batcher exactly two shades are reachable.

### 2 — beam search wins on the one line whose image carries information

`ex02_beam_search_cer_delta.py` — 146 code lines, 5 checks.

**ANSWER: beam_width=5 moves CER from 0.6790 to 0.6667, a delta of −0.0123.**
Measured over the lesson's own `main()` run reproduced at its seeds — 200 Adam
steps on its own `abc0..abc9` / `xy01..xy910` strings:

| decoder | CER |
|---|---:|
| `greedy_ctc_decode` | **0.6790** |
| prefix beam, width 5 | **0.6667** |
| delta | **−0.0123** (1.8% relative) |

**ANSWER: it wins on exactly one line — the only one whose image is not a
collision.** 1 of the 20 lines decodes differently:

| reference | greedy | beam-5 |
|---|---|---|
| `xy910` | `x910` | **`xy910`** |

String lengths in the set are `[4, 5]`, and since `synthetic_line` inks every
alnum symbol alike, that single length-5 line is the only image unlike the other
19 — greedy collapses all 20 lines to **2** distinct outputs. Beam search cannot
win where the input carries no information; here it wins on the one line that does.

**MECHANISM: greedy maximises over paths, beam over labelings, and on that line
the two orderings cross.** Read out of the lesson's own `ctc_loss` with its
`reduction="mean"` division undone:

| quantity | probability |
|---|---:|
| P(`xy910`), summed over alignments | **0.052094** |
| P(`x910`), summed over alignments | **0.048992** |
| best single path of the 37²⁰ (spells `x910`) | **0.037392** |

The correct labeling is **1.0633×** more probable, but no individual path spelling
it beats the blank-heavy path that spells `x910`.

**CONTROL: on a 3-step hand case the beam matches the lesson's CTC oracle to
1.0e-07.** A distribution of `(0.45, 0.35, 0.20)` over `(blank, '0', '1')` at each
of three steps:

| | probability |
|---|---:|
| greedy path (blank ×3), `0.45**3` | **0.091125** |
| P(`'0'`) over all 27 paths, from `ctc_loss` | **0.365750** |
| ratio | **4.0137×** |

beam-5 returns exactly that, to `1.0e-07` nats — the float32 residue of
`F.ctc_loss`. That agreement is what validates the decoder used above.

**CONTROL: width 5 is over-provisioned and lossy.**

| beam width | 1 | 2 | 3 | 5 | 10 | 25 |
|---|---:|---:|---:|---:|---:|---:|
| CER | 0.6790 | 0.6790 | **0.6667** | 0.6667 | 0.6667 | 0.6667 |

Width 1 reproduces `greedy_ctc_decode` on **20/20** lines and nothing past 3
changes an output. Pruning also costs mass: beam-5 scores `xy910` at **−3.1919**
against the exact **−2.9547**, accounting for only **78.9%** of the labeling's real
probability — enough to order a 1.0633× gap, bought with **3,552** prefix
expansions against **740** argmax comparisons.

### 3 — the engine is absent, and the metric it would be scored with is nearly binary

`ex03_receipt_pair_f1_vs_cer.py` — 146 code lines, 5 checks. Pure stdlib (T0).

**ANSWER: PaddleOCR is not installed and the 20 receipts do not exist.**
`importlib.import_module` on every candidate:

| module | result |
|---|---|
| `paddleocr` | **ModuleNotFoundError: No module named 'paddleocr'** |
| `pytesseract` | **ModuleNotFoundError: No module named 'pytesseract'** |
| `easyocr` | **ModuleNotFoundError: No module named 'easyocr'** |
| `transformers` | **ModuleNotFoundError: No module named 'transformers'** |

None is in any dependency group. The lesson tree holds **0** image or PDF files
and `code/main.py` mentions an engine or receipt **0** times — it is a CRNN
trainer. So the scorer runs against a hand-labelled **20**-receipt corpus built in
the file, corrupted by the classic confusions `{o:0, l:1, i:1, s:5, b:8, z:2}` at a
50% per-character rate on seed 19.

**ANSWER: the same 20 lines score three very different numbers.**

| metric | value |
|---|---:|
| pair F1 (exact `{item_name, price}` match) | **0.2000** (4/20) |
| CER | **0.1350** (86.5% of characters correct) |
| WER | **0.4000** (3.0× the CER) |

**FINDING: WER is unbounded above 1.0 where F1 is not.** One reference word
`milk` against a hypothesis split into 4 characters costs 4 word edits over a
1-word reference: **WER 4.0000**. Pair F1 cannot leave `[0, 1]` by construction, so
the two are not on comparable scales even in principle.

**MECHANISM: exact-match pair F1 is the AND of its fields, so it equals the weaker
one.**

| field | F1 |
|---|---:|
| price | **1.0000** |
| item name | **0.2000** |
| pair | **0.2000** |

Prices are digits and `.`, and every confusion class maps a letter *to* a digit, so
no price is ever touched and the pair score collapses onto the name. The rate
ladder confirms it discriminates rather than saturating: **30% → 0.4000**, **50% →
0.2000**, **60% → 0.1500**. The hand-worked line makes the disagreement concrete —
`"soda 1.50"` read as `"50da 1.5o"` scores CER **3/9 = 0.3333** but WER **2/2 =
1.0000**.

**CONTROL: the confusion fold is idempotent but not injective — a scorer, never a
post-processor.**

| property | value |
|---|---:|
| `fold(fold(x)) == fold(x)` on all 40 corpus strings | **true** |
| 36 alnum symbols collapse to | **30 classes** |
| ordered symbol pairs made equal | **14** |
| pair F1, fold applied to both sides | **1.0000** |
| pair F1, fold emitted as corrected text | **0.0000** |
| already-correct pairs surviving the one-sided fold | **4 → 0** |

The same function is the right answer as a scoring equivalence and the wrong
answer as an output repair — it destroys even the four pairs that were already
right.

**CONTROL: the distance underneath every rate above is a metric on this corpus.**
Over the 20 item names, **0 of 6,840** ordered triples violate
`edit(a,c) <= edit(a,b) + edit(b,c)`, and every name has `edit(a,a) == 0`. CER and
WER are therefore normalised metrics, which is what makes the gap between them a
real gap rather than a scaling artefact.
