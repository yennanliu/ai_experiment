<!-- generated:start -->
# 06-speech-and-audio / 10-audio-language-models

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/06-speech-and-audio/10-audio-language-models/) · upstream spec
`phases/06-speech-and-audio/10-audio-language-models/docs/en.md`

```bash
uv run demo practice run 10-audio-language-models --ex 1
uv run demo explain 10-audio-language-models --ex 1
uv run pytest demos/phases/06-speech-and-audio/10-audio-language-models
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run `code/main.py` to see a toy projector pattern + fake LALM routing of (audio-embeddi… | code | T0 | `ex01_the_projected_vectors_are_used_for_their_length.py` |
| 2 | Medium. Score Qwen2.5-Omni-7B on 100 MMAU-Pro speech items. Compare to the paper's reported n… | code | T0 | `ex02_a_hundred_items_cannot_reach_the_published_number.py` |
| 3 | Hard. Build a minimal audio-captioning baseline: BEATs encoder + 2-layer projector + frozen L… | code | T0 | `ex03_neither_end_of_the_projector_fits.py` |
<!-- generated:end -->

## Answers

The lesson's own framing is that every LALM is three components — encoder,
projector, LLM — and that the interesting part is the seam between them. All
three exercises land on a seam that is not joined: the projector's output is
computed and thrown away, the benchmark cannot resolve the table it is read
against, and the projector the doc prints does not fit either component the hard
exercise names.

All three run at **T0**; the lesson's own code imports `math` and `random`.

### 1 — the projected vectors are used for their length

The encoder half is right. `fake_audio_encoder(3.0)` returns **150 frames of
1280** at exactly **50 frames per second** — Whisper-large's encoder rate and
width.

**MECHANISM: `main()` projects 8 of the 150 frames**, 5.33% of the clip, because
the pure-Python matmul runs at roughly **0.22 s per frame** and all 150 would take
about **34 s** against the 1.8 s the demo spends. The printed "(first 8 frames)"
is the whole reason the number is 8.

**FINDING: the 32,768 numbers it computes are used for exactly one thing — `len`.**

```python
interleaved = interleave_with_text(list(range(len(projected))), text_tokens)
```

Every `AUDIO` payload in the sequence is an **`int`** index. The 4096-dimensional
vectors are discarded on the line after they are printed. The one seam the lesson
exists to show — projected audio entering the LLM's embedding space — is the seam
left unconnected.

**FINDING: `interleave_with_text` concatenates.** Across the 13-item sequence
there is **1** transition between kinds, where an interleaving would have 12.

**FINDING: the shipped projector is 23.8% of the one the doc prints.**

| | layers | activation | bias | parameters |
|---|---|---|---|---:|
| doc, Step 2 | 2 | GELU | yes | **22.03 M** |
| `code/main.py` | 1 | ReLU | no | **5.24 M** |

**CONTROL: `projector` reseeds the global RNG.** Its first statement is
`random.seed(1)`, so a caller's own stream does not survive the call.

### 2 — a hundred items cannot reach the published number

`transformers`, `torch`, `datasets`, `torchaudio` and `accelerate` are all absent
and MMAU-Pro is not here. This is the `DESIGN D11` scaled-down run: an 1800-item
pool per model, each reproducing its published rate to within **0.022 pp**,
sampled 4,000 times at n=100.

**ANSWER: 57.4% is not a score 100 items can return.** A hundred items move
accuracy in steps of exactly **1.00 pp**; the nearest returnable scores are 57
and 58.

**ANSWER: the interval is wider than the whole table.**

| | |
|---|---|
| 95% interval at `p = 0.574`, n = 100 | **[47.7, 67.1]** |
| width of that interval | **19.4 pp** |
| span of the whole `overall` column | **7.8 pp** |

The confidence interval is **2.5×** the entire spread it would have to resolve.
Sampling agrees — over 4,000 draws of 100 items:

| | |
|---|---|
| five models in published order | **6.9%** of draws |
| best model picked | **57.6%** of draws |
| Qwen's overall, 95% of draws | **43.0 – 62.0%** (published 52.2) |

**FINDING: what would be enough.** From `n = 2z²p(1−p)/(p₁−p₂)²`:

| comparison | items needed |
|---|---:|
| Qwen2.5-Omni-7B vs Gemini 2.5 Pro (52.2 vs ~60) | **311** — 3× the exercise's 100 |
| Qwen2.5-Omni-7B vs GPT-4o Audio (52.2 vs 52.5) | **212,951** — 118× the benchmark |
| multi-audio 26.5% vs the 25% chance line | **3,326** — 1.8× all of MMAU-Pro |

That last row reaches the doc's own headline claim. "Multi-audio is barely above
random" is almost certainly true — and MMAU-Pro's 1800 items cannot be what
establishes it.

### 3 — neither end of the projector fits

`transformers`, `torch`, `peft`, `datasets` and `soundfile` are all absent, so no
fine-tune runs. The shapes check exactly, and they are what the exercise turns on.

**ANSWER: the doc's projector does not connect the components the exercise
names.** Its Step 2 is `Linear(1280, 4096) → GELU → Linear(4096, 4096)`. BEATs
emits **768** and Llama-3.2-1B's hidden size is **2048** — `audio_dim` is 1.67×
too wide and `llm_dim` is exactly **2×** too wide.

| projector | parameters |
|---|---:|
| the doc's, 1280 → 4096 → 4096 | **22.03 M** |
| the exercise's own parts, 768 → 2048 → 2048 | **5.77 M** |
| `code/main.py`, one layer, no bias | 5.24 M |

**MECHANISM: one mismatch crashes and the other passes silently.** A 768-wide
BEATs frame at the default `audio_dim=1280` raises **`IndexError`** — the loop
indexes `f[j]` past the end of the frame. Told `audio_dim=768` it works and
returns **4096**-wide vectors that a 2048-wide Llama cannot consume, with no
error. The silent one is the one that would reach training.

**FINDING: the comparison has no shared metric.** The exercise fine-tunes on
**AudioCaps** — audio captioning, scored with CIDEr or SPIDEr — and compares the
result to SALMONN on **Clotho-AQA** — audio question answering, scored with
accuracy. Different task, different corpus, different unit; there is no number
both sides produce.

**MECHANISM: what the design buys.** A correctly shaped projector is **0.4670%**
of a frozen 1.24 B-parameter Llama-3.2-1B:

| | fp32 Adam optimizer state |
|---|---:|
| projector only | **69.3 MB** |
| full fine-tune | **14.8 GB** |
