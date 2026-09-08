<!-- generated:start -->
# 04-computer-vision / 16-vision-pipeline-capstone

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/04-computer-vision/16-vision-pipeline-capstone/) · upstream spec
`phases/04-computer-vision/16-vision-pipeline-capstone/docs/en.md`

```bash
uv run demo practice run 16-vision-pipeline-capstone --ex 1
uv run demo explain 16-vision-pipeline-capstone --ex 1
uv run pytest demos/phases/04-computer-vision/16-vision-pipeline-capstone
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | (Easy) Run the pipeline on 10 images from any open dataset. Report the average time per stage… | code | T1 | `ex01_stage_time_budget.py` |
| 2 | (Medium) Add a mask output field to `Detection` and encode it as RLE. Verify the JSON stays u… | code | T1 | `ex02_mask_rle_payload_size.py` |
| 3 | (Hard) Add a micro-batcher in front of the classifier: collect crops for up to 10 ms, classif… | code | T1 | `ex03_microbatch_window_yield.py` |
<!-- generated:end -->

## Answers

This is the capstone, so the lesson ships a whole service rather than one
algorithm — and all three exercises turn out to ask about a number the lesson's
own code cannot make interesting. Exercise 1 asks for a distribution that is a
point mass, exercise 2 asks you to add a field that is already there, and
exercise 3 asks for a throughput gain at a load twenty times too light to
produce one. In each case the reportable finding is **which variable actually
moves**, so that is what the three solutions measure. Wall-clock figures below
come from one run on a 2-thread CPU and move a few percent between runs; every
byte count, run count and batch histogram is exact and reproducible.

**1 — the stage budget is real; the detection-count distribution is not.**

Ten frames, 200×260 to 360×460, cut from the two photographs
`sklearn.datasets.load_sample_images` keeps on disk (nothing downloaded), each
walked through the four stages twenty times:

| stage | median | share |
|---|---:|---:|
| preprocess | 0.0629 ms | 31% |
| detect | 0.0072 ms | 4% |
| **crop + resize** | **0.0994 ms** | **49%** |
| classify | 0.0348 ms | 17% |
| total | 0.2043 ms | |

**ANSWER: every image returns exactly 3 detections.** `StubDetector.forward`
builds its three boxes out of `H, W = img.shape[-2:]` and never touches a pixel,
so the requested distribution is `{3: 10}` on any ten images whatsoever. The
count that does move is the *classification* count, and only through
`min_crop=16`:

| square frame | 16 | 32 | 50 | 64 | 96 | 128 |
|---|---:|---:|---:|---:|---:|---:|
| classifications | 0 | 0 | 1 | 3 | 3 | 3 |

That is geometry, not content — below 50 px all three boxes shrink under the
16-pixel floor.

**MECHANISM: the largest stage has no bucket.** Crop-and-resize is inlined in
both `run()` and `benchmark()`, so it is billed to whichever bucket surrounds it,
and the lesson puts it inside `classify`. The three-bucket report therefore
prints `classify` at **0.1342 ms**, **3.9×** the model's real **0.0348 ms**, and
the true hotspot never appears. The isolated loop used for the timing above is
verified to be the lesson's: it returns exactly as many crops as `pipe.run()`
turned into `Classification` records on **10/10** frames.

**CONTROL: that half of the budget is thrown away one layer later.**
`StubClassifier` opens with `AdaptiveAvgPool2d(1)`, so a crop is three channel
means before `Linear(3, 10)` sees it. The same crop scored at 4, 8, 16, 32, 64,
128 and 256 pixels square moves the logits by at most **3.2e-02** and picks class
**7** every time — a 4×4 resize, **256× fewer pixels** than the pipeline's 64×64,
is indistinguishable at the output.

**CONTROL: the `p95` column the lesson prints is the maximum.** `benchmark`
indexes the sorted list at `int(len(times) * 0.95)`: at `num_runs=10` that is
index 9 of 9, at 20 it is index 19 of 19, and only at 100 does it become index 95
of 99. At the default it reports the slowest sample.

**2 — the payload budget is a mask-complexity budget, not an object-count budget.**

`code/main.py` already declares `mask_rle: Optional[str] = None` on `Detection`,
and `VisionPipeline.run()` never writes it — the untouched pipeline emits **3/3**
detections with `"mask_rle": null` in a **613-byte** payload. What the exercise
is really missing is the producer, so the solution supplies a COCO-style
column-major run-length codec and fills the lesson's own model with it. Ten
objects, each a copy of `StubDetector`'s first box (0.3 W × 0.5 H) at seeded
offsets, serialised through `PipelineResult.model_dump_json()`, against a
1,000,000-byte budget:

| frame | solid | ellipse | dithered |
|---|---:|---:|---:|
| 400×600 (the lesson's demo size) | 15,271 B | 15,071 B | 367,188 B |
| 720×1280 | 31,609 B | 31,509 B | **1,398,888 B** |
| 1080×1920 | 46,986 B | 47,006 B | **3,131,666 B** |

**ANSWER: at the lesson's own demo size the clause holds with 65× headroom.**
**FINDING: it holds because of the geometry, not the object count.** Solid and
dithered masks cover *identical pixels* and differ **24×** in bytes at 400×600;
scale the frame and the same ten objects go **1.4× over budget** at 720p and
**3.1× over** at 1080p, while ellipses stay at 3-5% throughout.

**MECHANISM: an RLE counts boundary crossings, not covered pixels.** The scan is
column-major, so a solid box crosses the boundary exactly twice per column it
occupies, plus once to close the image — `2·columns + 1` exactly:

| frame | box columns | runs |
|---|---:|---:|
| 400×600 | 180 | **361** |
| 720×1280 | 384 | **769** |
| 1080×1920 | 576 | **1153** |

The 400×600 box covers **36,000** pixels and not one of them enters the count.
Dithering that same box costs **18,039** runs instead of 361.

**CONTROL: the codec is lossless, so the bytes are honest.** All 30 masks at
400×600 survive encode → `model_dump_json` → `model_validate_json` → decode
bit-exact. **CONTROL: the box contract changes dtype at the boundary.** A box
built from the ints `(60, 40, 240, 240)` parses back as
`(60.0, 40.0, 240.0, 240.0)`, because the annotation is
`Tuple[float, float, float, float]` — pixel indices leave the service as floats.

**3 — the micro-batcher works; 5 requests per second cannot fill it.**

20,000 seeded exponential arrivals at the exercise's own 5 req/s, through its own
10 ms window, form 19,077 batches:

| batch size | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| batches | 18,180 | 872 | 24 | 1 |

**ANSWER: mean batch 1.0484 — 95.3% of batches hold a single request.** Priced
with measured `pipe.classify` calls that is 0.0387 → 0.0371 ms per request,
**1.043× on the classify stage and 1.004× end to end**, bought with **9.77 ms of
mean added wait** — **26×** the 0.3764 ms the whole pipeline takes.

**FINDING: the batcher is not what fails.** One `pipe.classify` call over three
crops per request:

| requests per call | 1 | 2 | 3 | 4 | 5 | 10 |
|---|---:|---:|---:|---:|---:|---:|
| ms per request | 0.0387 | 0.0215 | 0.0171 | 0.0140 | 0.0119 | **0.0076** |

Ten requests in one call cost **5.1× less per request** than ten separate calls.
The design is sound; the arrival rate never hands it a second request.

**MECHANISM: a fixed window opened by the first arrival collects `1 + rate ×
window`.** Each batch is its opener plus the Poisson(0.05) arrivals that land
inside the window:

| rate (req/s) | 5 | 50 | 100 | 500 |
|---|---:|---:|---:|---:|
| simulated mean batch | 1.048 | 1.500 | 2.003 | 6.020 |
| `1 + rate × window` | 1.05 | 1.50 | 2.00 | 6.00 |

A mean of 2 needs **100 req/s — 20× the load the exercise names**. Widening the
window is the only other lever, and it spends latency 1:1.

**CONTROL: Amdahl caps the whole idea.** `pipe.run` takes 0.3764 ms of which
`classify` is 0.0387 ms, **10%**; even the saturated 10-request batch leaves
**1.090× end to end**. The crop-and-resize loop feeding it costs **0.1329 ms**,
**3.4×** the classifier call, and runs once per request however the classifier is
invoked — this design batches the cheap stage and leaves the expensive one alone.

**CONTROL: there is no GPU call, and nothing to saturate.** `VisionPipeline`
defaults to `device="cpu"` and `torch.cuda.is_available()` is `False`, so
`benchmark`'s `sync()` is a no-op and the measured gain is CPU BLAS amortisation
rather than kernel-launch amortisation. One request costs 0.3764 ms, a capacity
of **2,657 req/s**, so 5 req/s is **0.19% utilisation**. Batching is a saturation
tool handed an idle server.
