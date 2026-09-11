<!-- generated:start -->
# 06-speech-and-audio / 09-music-generation

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/06-speech-and-audio/09-music-generation/) · upstream spec
`phases/06-speech-and-audio/09-music-generation/docs/en.md`

```bash
uv run demo practice run 09-music-generation --ex 1
uv run demo explain 09-music-generation --ex 1
uv run pytest demos/phases/06-speech-and-audio/09-music-generation
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run `code/main.py`. It produces a "generative" chord progression + drum pattern as ASCI… | code | T0 | `ex01_the_tempo_in_the_prompt_is_never_read.py` |
| 2 | Medium. Install `audiocraft`, generate 10-second clips across 4 genre prompts with MusicGen-s… | code | T0 | `ex02_fad_on_four_clips_measures_four_clips.py` |
| 3 | Hard. Using ACE-Step (or MusicGen-melody), generate three variations of the same tune with di… | code | T0 | `ex03_timbre_is_not_in_the_input_space.py` |
<!-- generated:end -->

## Answers

`code/main.py` calls itself a cartoon, and a cartoon is a fair thing to ship. The
part that is not cartoon is the prompt parser — the only piece of real logic in
the file — and every exercise here ends up at it. Two of the four demo prompts do
not arrive intact, and the one thing the hard exercise asks to vary is not in the
parser's vocabulary at all.

All three run at **T0**.

### 1 — the tempo in the prompt is never read

**ANSWER: "at 128 bpm" and "at 140 bpm" both come out at 120.** `fake_generate`
scans `prompt_lower.split()` for a token ending in `"bpm"` — and splitting on
whitespace puts the number in one token and the unit in another, so the token
that matches is the bare `"bpm"`, `int("")` raises `ValueError`, and the `except`
swallows it. Written `128bpm` with no space it parses to **128**, which is the
tell.

**FINDING: the key detector matches inside ordinary words.** It tests `f" {k}"`
as a substring, so a leading space is its only boundary.

| prompt | key | why |
|---|---|---|
| `rock anthem at 140 bpm` | **A** | the word "**a**t" |
| `an upbeat song` | **A** | "**a**n" |
| `a slow groove` | **G** | "**g**roove" |
| `instrumental piece` | C | nothing matched; C is the default |

**FINDING: the genre is chosen by dictionary order, not by the prompt.** It walks
`pop, ballad, jazz, rock, lofi` and returns the first key appearing anywhere in
the string, so `"lofi rock ballad"` is a **ballad**.

**FINDING: the two tables disagree, so one beat is dead and one is shared.**
`COMMON_PROGRESSIONS` has a `ballad` that `DRUM_PATTERNS` lacks — every ballad
silently gets the pop beat. `DRUM_PATTERNS` has a `trap` that no prompt can
select. Across all 20 reachable pieces there are **4** distinct drum tracks.

**CONTROL: nothing here generates.** `fake_generate(prompt, rng=None)` constructs
`random.Random(0)` and never references it again — 50 seeds, **1** output.

### 2 — FAD on four clips measures the number four

`audiocraft`, `torch`, `torchaudio`, `frechet_audio_distance` and `laion_clap`
are all absent, and `code/main.py` emits ASCII rather than samples. FAD is
arithmetic, though, and the arithmetic settles the exercise.

FAD fits a Gaussian to each set of VGGish embeddings and returns
`‖μ₁−μ₂‖² + tr(Σ₁+Σ₂−2(Σ₁Σ₂)^½)`. VGGish is **128-dimensional**, so at N=4 each
covariance has rank **3** and the matrix square root of their product is
degenerate. Drawing *both* sets from the same distribution — true FAD exactly 0:

| N per set | FAD between two samples of one distribution |
|---:|---:|
| **4** | **269.7** |
| 16 | 186.6 |
| 64 | 115.0 |
| 256 | 33.2 |
| 1024 | 8.2 |
| 4096 | 2.1 |

**ANSWER: a real difference is invisible underneath that bias.** Shifting one
set's mean by 0.1 in every dimension is a true FAD of `128 × 0.01 = 1.28`. At
N=1024 it measures **9.7**, of which **8.2 is bias**; at N=4 the same pair
measures **285.2** — inside the spread of two *identical* distributions.

Published FAD uses thousands of clips for exactly this reason. The exercise also
conflates two designs: four prompts with one clip each is four conditions of size
one, not one set of size four, and FAD is a set statistic with no per-clip value.

### 3 — timbre is not in the input space

`acestep`, `audiocraft`, `torch`, `laion_clap` and `transformers` are all absent
— but the exercise can be answered against `code/main.py`, because the thing it
asks to vary is one the parser does not read.

```text
"warm analog pop in C"  ==  "bright digital pop in C"  ==  "muted felt pop in C"
```

**ANSWER: byte for byte.** Three "variations" render to **1** distinct piece.
`fake_generate` reads a key, a genre, and a bpm that never parses; timbre is not
among them. Three prompts differing only in tempo do the same.

**FINDING: the entire output space is 20 pieces.** Four keys × five genres, tempo
frozen at 120, is everything the model can ever emit — **20 chord sequences and
4 drum tracks**. "Three variations" can only be three of twenty, and only if the
prompts disagree about key or genre.

**FINDING: the alignment a CLAP score would verify is already broken.** The only
prompt-to-output relation here is `prompt → (key, genre)`. It holds on all four
prompts that name a key, and fails on ones that do not — `"rock anthem at 140
bpm"` → **A**, `"an upbeat song"` → **A**, `"a slow groove"` → **G**. A
similarity score computed against those pieces would be scoring the parser, and a
single number with no null distribution could not say so.

**CONTROL: a readable CLAP report is three numbers** — the score, the score
against a *mismatched* prompt, and the spread of both. Here the mismatched score
equals the matched one, because the pieces are identical.
