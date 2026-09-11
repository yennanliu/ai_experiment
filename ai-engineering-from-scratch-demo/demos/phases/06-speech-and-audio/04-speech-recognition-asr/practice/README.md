<!-- generated:start -->
# 06-speech-and-audio / 04-speech-recognition-asr

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/06-speech-and-audio/04-speech-recognition-asr/) · upstream spec
`phases/06-speech-and-audio/04-speech-recognition-asr/docs/en.md`

```bash
uv run demo practice run 04-speech-recognition-asr --ex 1
uv run demo explain 04-speech-recognition-asr --ex 1
uv run pytest demos/phases/06-speech-and-audio/04-speech-recognition-asr
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run `code/main.py`. It greedily decodes a hand-crafted CTC output and computes WER agai… | code | T0 | `ex01_the_decode_is_never_scored.py` |
| 2 | Medium. Implement the prefix-tree beam search in Step 2 properly (account for the blank merge… | code | T0 | `ex02_the_lesson_beam_cannot_emit_a_double_letter.py` |
| 3 | Hard. Use `whisper-large-v3-turbo` on [LibriSpeech test-clean](https://www.openslr.org/12). C… | code | T0 | `ex03_a_hundred_utterances_cannot_rank_the_table.py` |
<!-- generated:end -->

## Answers

`main()` prints a decode, a second decode, a corrupted pair of decodes, and five
WER numbers — and never joins any of them up. The three exercises are the three
joins: score the decode, fix the beam, and ask what a hundred utterances can
actually measure.

All three run at **T0** on stdlib alone; the lesson's own code imports only
`math`, `random` and `collections`.

### 1 — the decode is never scored

```python
wer('hello world', ctc_greedy(probs))  ==  0.000
```

That call appears nowhere in `code/main.py`. Step 2 prints `'hello world'`; Step
5 scores five hand-typed strings against `'hello world this is a test'`, a
different reference. The exercise describes a pipeline the lesson does not run.

| | decode | WER vs target |
|---|---|---:|
| `ctc_greedy`, clean frames | `'hello world'` | **0.000** |
| `ctc_beam`, clean frames | `'helo world'` | **0.500** |
| `ctc_greedy`, Step 4's corruption | `'hlhce llolo wolnrld'` | 1.500 |
| `ctc_beam`, Step 4's corruption | `'hlhce lolo wolnrld'` | 1.500 |
| an empty hypothesis | `''` | 1.000 |

**FINDING: on clean frames the beam is worse than greedy.** `build_frame_probs`
puts a blank between every character, so greedy is already exact and a beam can
only tie or lose.

**FINDING: Step 4 is captioned "beam should beat greedy" and the two tie**, at
1.500 each — both worse than saying nothing, because `wer` divides by the
reference length only and is unbounded above.

**MECHANISM: `corrupt` leaves the simplex while the row sums still read 1.0.**
It subtracts up to 0.6 from entries that start at 0.02, so the minimum entry
becomes **−0.58**; it moves mass rather than adding it, so every row still sums
to 1.0. `ctc_beam` clamps each negative to `1e-10` and decodes a vector holding
more than its own mass. `ctc_greedy` takes an argmax and never notices.

**CONTROL: the "hand-crafted" frames are deterministic.** `build_frame_probs`
calls `random.seed(0)` and then draws nothing; re-seeding to 999 reproduces all
44 frames byte for byte, and `one_hot_like`'s `noise` is a constant offset.

### 2 — the lesson's beam cannot emit a double letter

Ten short phrases, **nine of them carrying a doubled letter**, on the frames
`build_frame_probs` builds:

| decoder | exact | CER |
|---|---:|---:|
| `ctc_greedy` | 10/10 | 0.000 |
| `ctc_beam` (the lesson's) | **1/10** | 0.116 |
| prefix beam, `p_blank`/`p_nonblank` | **10/10** | **0.000** |

`'helo world'`, `'cofe cup'`, `'bel tower'`, `'litle dog'`, `'smal gren box'`,
`'leter home'`, `'suny day'`, `'a biter pil'`, `'red balon'` — one letter short,
every time.

**MECHANISM: it cannot represent a repeat at all.** The lesson's update sends the
blank path and the repeat path to the same `seq`, so no beam it holds ever
contains two adjacent equal tokens. Across 30 decodes at three corruption levels,
**zero** outputs carry a doubled character. The fix is one extra state per
prefix: a repeat may extend a prefix only through the blank path.

**FINDING: greedy is already optimal here, so the comparison needs corruption** —
and the lesson's own `corrupt` cannot supply it (Exercise 1). Moving a random
share of each winner's mass to one rival keeps every row a distribution and does
flip argmaxes. Once they flip: greedy **0.224** CER, prefix beam **0.139**,
lesson's beam 0.171.

**CONTROL: at heavy corruption the lesson's beam wins on CER by being shorter.**
Every decoder over-emits — reference 11.1 characters against greedy's 26.1 — so
CER is insertion-driven, and not being able to repeat makes the lesson's output
systematically the shortest (23.6 against the prefix beam's 25.1). Lower score,
no better decoding.

### 3 — a hundred utterances cannot rank the lesson's own table

`whisper`, `torch`, `transformers`, `datasets` and `soundfile` are all absent and
LibriSpeech is not here, so the transcription half cannot run. The comparison half
runs in full, and it decides the exercise. This is the `DESIGN D11` scaled-down
run: 2,620 utterances of 20 words, degraded at each rate in the lesson's own Step
6 table, scored with the lesson's own `wer`, sampled 2,000 times at n=100. The
generator reproduces every published rate to **0.07 pp** over the full corpus, so
everything below is about sample size alone.

| | |
|---|---|
| Whisper-L-v3-turbo, published | **1.58%** |
| the same, over 100 utterances (95% of draws) | **1.10 – 2.15%** |
| width of that interval | **1.05 pp** |
| width of the whole five-model table | **0.52 pp** |

**The entire published table fits inside one sample's confidence interval, twice
over.** Over 2,000 draws the five models come back in the published order **7.4%**
of the time, and the best model is picked **45.1%** of the time — against 20% for
a coin.

How many reference words would it take, from
`n = 2·1.96²·p(1−p)/(p₁−p₂)²`:

| comparison | words needed |
|---|---:|
| best vs worst (1.40 vs 1.92) | **4,638** (~232 utterances) |
| Seamless vs wav2vec (1.70 vs 1.92) | 28,212 |
| Whisper vs Seamless (1.58 vs 1.70) | 86,068 |
| Parakeet vs Canary (1.40 vs 1.48) | **170,382** |

test-clean holds about 52,400 words. **The two best rows of the lesson's own table
cannot be separated on the whole of test-clean** — they need 3.3× the corpus — and
even the widest gap in the table needs more than twice the hundred utterances the
exercise names.

The full run the exercise asks for is `whisper.load_model("large-v3-turbo")` over
`LIBRISPEECH(url="test-clean")` — 346 MB of audio, 809M parameters, roughly two
minutes for 100 utterances on a 24 GB GPU. Worth doing; just not worth reading as
a comparison.
