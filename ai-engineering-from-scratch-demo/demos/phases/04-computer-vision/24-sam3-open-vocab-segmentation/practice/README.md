<!-- generated:start -->
# 04-computer-vision / 24-sam3-open-vocab-segmentation

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/04-computer-vision/24-sam3-open-vocab-segmentation/) · upstream spec
`phases/04-computer-vision/24-sam3-open-vocab-segmentation/docs/en.md`

```bash
uv run demo practice run 24-sam3-open-vocab-segmentation --ex 1
uv run demo explain 24-sam3-open-vocab-segmentation --ex 1
uv run pytest demos/phases/04-computer-vision/24-sam3-open-vocab-segmentation
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | (Easy) Run SAM 3 on 10 images with concept prompts you choose. Compare against SAM 2 + Ground… | code | T0 | `ex01_stub_reads_neither_input.py` |
| 2 | (Medium) Build a "click-to-include / click-to-exclude" UI on top of SAM 3: a text prompt retu… | code | T0 | `ex02_instance_ids_collide.py` |
| 3 | (Hard) Fine-tune SAM 3 on a custom concept set (e.g. 5 types of electronic components) with 2… | code | T0 | `ex03_nothing_to_finetune.py` |
<!-- generated:end -->

## Answers

All three exercises name models that are not installable — `sam2`,
`segment_anything`, `groundingdino` and `transformers` every one raise
`ModuleNotFoundError`. So the lesson's own `StubOpenVocabSeg` is the only
implementation of its own interface, and the honest work is to find out what it
does and does not establish. It turns out to establish nothing about
segmentation, which is worth knowing before building three exercises on top
of it.

These three run at **T0**, so unlike most of phase 04 they execute in CI rather
than skipping.

### 1 — The stub reads neither of its inputs

| varied | distinct mask pairs |
|---|---:|
| 6 unrelated concepts (incl. `""`, a 26-char phrase) | **1** |
| 10 independently drawn random frames | **1** |
| frame halved in both dimensions | 2 (shape is the only input) |

**ANSWER: it cannot miss a concept, because it never reads one.** The `concept`
field is a pure echo of the argument, and the scores are the constants **0.89**
and **0.74**.

**MECHANISM: the masks are two hardcoded rectangles** in fractions of h and w —
rows 0.3–0.8 × cols 0.2–0.5, and rows 0.25–0.75 × cols 0.55–0.85. Rebuilding
them from those fractions reproduces both decoded masks **exactly**. That is the
whole model.

**CONTROL: recall is 1.000 and precision is meaningless — the same statement.** A
ground-truth object planted clear of both rectangles scores IoU **0.0** against
each, with **0 of 2** overlapping it at all.

**CONTROL: the returned pair has IoU exactly 0.0** — columns 0.2–0.5 and
0.55–0.85 cannot touch — so any NMS, de-duplication or instance-merging step
built on this fixture is untested by it.

### 2 — What identifies an instance

**FINDING: `instance_id` alone cannot address a click.** `run_multi_concept`
calls `detect` once per concept and each call numbers from zero:

| utterance | detections | ids | distinct ids |
|---|---:|---|---:|
| `cat and dog and bird` | 6 | `[0, 1, 0, 1, 0, 1]` | **2** |

Clicking "instance 0" selects **3** objects, one per concept. The composite key
`(concept, instance_id)` is unique across all six, and that is what makes the
click addressable.

**MECHANISM: `split_concepts` cuts inside ordinary noun phrases**, because the
separators are replaced textually before any splitting:

| prompt | result |
|---|---|
| `salt and pepper shaker` | **`['salt', 'pepper shaker']`** |
| `AT&T logo` | **`['AT', 'T logo']`** |
| `black-and-white cat` | `['black-and-white cat']` |
| `gin and tonic; ice` | `['gin', 'tonic', 'ice']` |

The hyphenated phrase survives because the rule matches `' and '` *with spaces* —
so which prompts break is a matter of typography.

**CONTROL: an empty utterance is a concept.** `split_concepts('')` returns `['']`
rather than `[]`, so the pipeline queries the empty concept and gets **2**
anonymous detections back for a prompt the user never typed.

**CONTROL: the JSON the exercise asks for does not round-trip.** `box` leaves as a
`tuple` and returns as a `list`, and the dataclass's generated `__eq__` compares
by type — so the restored detections compare **unequal** to the originals.

### 3 — There is nothing to fine-tune

`StubOpenVocabSeg` carries **0** attributes of instance state, and two separately
constructed copies agree byte for byte. So "fine-tuned against zero-shot" has one
arm and an improvement of exactly **0.000** by construction, not by measurement.

On the 5 × 20 = **100** labelled frames the exercise specifies:

| arm | mean mask IoU | range |
|---|---:|---|
| zero-shot stub | **0.2052** | 0.0000 – 0.4809 |
| oracle (returns the label) | **1.0000** | — |

**FINDING: all 100 predictions are the same two masks.** So that IoU spread is
produced entirely by where the labels landed — the prediction never varies, and
the metric is measuring the dataset.

**MECHANISM: the score is a rectangle-overlap identity.** Place a label exactly on
the rectangle `detect()` hardcodes and the *same* model scores IoU **1.0**.
Nothing about the model changed between 0.2052 and 1.0; only the label moved.

**CONTROL: the encoding assumes the shape it is given.** `rle_encode` compresses
these rectangular labels **27×** (187 bytes against 5,120 pixels) and expands a
dithered mask of identical size to **1.98× larger than raw**. It is a compression
scheme with a shape assumption inside it, and these labels happen to satisfy it.

### A note on file lengths

The three files run 126 / 118 / 124 lines of code, two of them just over D14's
120-line target and all well clear of the 150-line ceiling.
