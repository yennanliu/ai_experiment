# ai-engineering-from-scratch-demo — Design

A companion repo of **runnable, verified solutions to every exercise** in
[`ai-engineering-from-scratch`](https://yennj12.js.org/ai-engineering-from-scratch/index.html)
(local: `../ai-engineering-from-scratch`).

Status: **design settled, nothing built in this branch yet.** M0 was built and
verified once in a prior spike (2026-09-02) — the harness, four gates and four
demos across three tiers all ran; that run is kept as evidence in §9, but none
of its code is here. **Revised 2026-09-03**: exercise solutions are the repo's
unit of value and parity demos are optional (§2, D9). **2026-09-04**: all four
open questions closed (§8), and the build sequence moved out to
[`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md) — this file is design only.

Bare `§N` refers to a section of this file; `PLAN §N` to one of
`IMPLEMENTATION_PLAN.md`; `D<n>` to a decision in §3.

Every number in §1 and §5 was derived by a census script that reads the reference
checkout, and none is hand-maintained. That script lands as `scripts/census.py` in
M0.5 (PLAN §4), so the figures below stay re-checkable rather than becoming folklore.

---


---

## 1. What the reference repo actually is
Measured, not assumed (`phases/` tree, 2026-09-03):

| Fact | Number |
|---|---|
| Phases | 20 |
| Lessons (`phases/*/*/`) | **511** |
| Lessons with a `## Exercises` list | **474** |
| **Total exercises** | **2,090** — 3 to 8 per lesson, mode 5 |
| Lessons with a `## Practice Lab` instead | 5 (38 requirements between them) |
| Lessons with no practice content at all | 33 |
| `docs/zh.md` carrying a `## 練習` block | 470 — line-parallel with `en.md` |
| Lessons with `code/main.py` | 443 (503 have some `.py` under `code/`) |
| Lessons with code | 505 / 511 (the 6 without are Phase 00 shell/editor/docker) |
| Python files (excl. `__pycache__`) | 634 |
| …with a `__main__` block | 607 |
| Test files | 94 (~18% of lessons) |
| TypeScript / Julia / Rust files | 129 / 20 / 10 |
| `notebook/` directories | 295 |
| **Actual `.ipynb` files** | **0** — every one holds only `.gitkeep` |
| Python files importing `torch` | **90** (58 of them in Phase 19 capstones) |
| Python files touching a real LLM SDK (`anthropic`/`openai`) | **20** |
| Per-lesson dependency manifests | 0 (one root `requirements.txt`) |

Two clusters of rows matter, and they point at two different gaps.

**The exercise rows.** 2,090 problems are posed and **not one is answered.** The
lesson ends, the exercises are listed, and there is nowhere to check your work.
This is the gap this repo now exists to close (§2).

**The zero-dependency rows.** Phase 07 Lesson 03 "Multi-Head Attention" opens:

> ```
> """Multi-head attention from scratch in pure stdlib.
> No numpy, no torch. A tiny Matrix class carries the ops we need.
> ```

That is a **deliberate and good** pedagogical choice, stated in `AGENTS.md`: code
must be self-terminating, exit 0, and never hang on a missing API key. So the
curriculum is hand-rolled, deterministic and simulated end to end — which leaves
a second, smaller gap that the parity demos address (§2, D5).

### The exercises are already a testable specification

This is the structural fact the whole design rests on. From
`11-llm-engineering/12-guardrails`:

> - "Test on 50 hand-written prompts and measure precision/recall."
> - "Test with 20 encoded versions of 'ignore previous instructions.'"
> - "Test with a burst of 15 requests in 30 seconds."
> - "…flag any response sentence with <20% overlap as potentially hallucinated."

Acceptance criteria, fixture sizes and thresholds, written into the prompt.
Nothing has to be invented to know whether a solution is correct. That is what
makes 2,090 solutions tractable where hand-designed parity demos were not: the
spec is free, and the gate can be mechanical (§6). Parity demos have to invent
their own spec, and only ~200 lessons support one at all (D5).

### Exercise mix

Classified by keyword by the census script — estimates, which the scaffold
proposes and a human confirms per phase batch (§6.4). One bucket per exercise,
assigned in the priority order shown, so the five sum to 2,090:

| Bucket | Count | Share | Ships as |
|---|---:|---:|---|
| prose only, no code possible | 127 | 6% | an answer in `practice/README.md` |
| needs GPU / heavy training | 119 | 6% | `exNN_*.py` at T3, scaled down (D11) |
| needs a real provider call | 59 | 3% | `exNN_*.py` at T2 + a cassette (D4) |
| patches the lesson's own `code/*.py` | 158 | 8% | `exNN_*.py` importing, never forking, that file |
| plain runnable solution | 1,627 | 78% | `exNN_*.py` |

Buckets are not the same axis as D11's `kind`: everything but the first 127 is
`kind: code`, and the tier is what differs. The buckets exist to size the work —
178 exercises need a key or a GPU, 127 need a source to cite, and the remaining
1,785 need nothing but a laptop.


---

## 2. The gap this repo fills
**The reference repo asks 2,090 questions and answers none of them.** A learner
finishing `11-llm-engineering/12-guardrails` is told to build a LlamaGuard-style
13-category classifier, an encoding-evasion detector across six encodings, a
sliding-window rate limiter, a RAG hallucination detector, and a 100-prompt
red-team suite — with no reference answer, no fixture, and no way to score an
attempt.

So the unit of value is a **solution**:

> For exercise *N* of lesson *L*, this repo ships the clean, simple, runnable
> answer, verified against the acceptance criterion the exercise itself states —
> and the same test grades the learner's own attempt (D13).

That last clause is what keeps 2,090 solutions from being 2,090 spoilers.

### Parity is one flavour of solution, not the flagship

The earlier framing of this repo made parity assertions the headline: the lesson
hand-rolls *X*, the demo runs the production equivalent side by side. That is
still valuable and M0 proved it works — `07/03-multi-head-attention` agrees with
`torch.nn.MultiheadAttention` to **2.8e-16** in float64. But it answers a
question the learner did not ask, and it requires inventing the comparison for
every lesson.

It survives in two forms. Some lessons keep an optional parity demo where the
comparison is genuinely striking (Phases 01–03, 07, 10 — roughly 200 lessons).
And some exercises *are* a parity check — "read the Qwen 2.5 72B config from
HuggingFace, compute total parameters from scratch, compare to the HF-reported
value and identify where any delta comes from" — so their solutions call
`harness/parity.py` directly. The M0 investment is reused, not retired.


---

## 3. Core design decisions
### D1 — Path-identical mirroring

```text
demos/phases/11-llm-engineering/12-guardrails/
```

Byte-for-byte the same relative path as the reference lesson. Consequences: a
lesson URL maps to a demo directory by string substitution, the site can
deep-link, and coverage is a `diff` of two directory listings — not a
hand-maintained table that rots.

### D2 — Cost/hardware tiers

Phase 01 runs in 40 ms on a laptop and Phase 08 wants an A100. One repo cannot
pretend those are the same thing, so every artifact declares a tier:

| Tier | Means | Budget | Network | CI |
|---|---|---|---|---|
| **T0** `cpu-instant` | stdlib / numpy / sklearn | < 10 s | none | every push |
| **T1** `cpu-heavy` | torch-CPU, small HF checkpoint (< 500 MB) | < 5 min | model download, cached | nightly |
| **T2** `api` | real provider call (OpenAI / ChatGPT, Q3) | < 20 s, < $0.02 | provider | every push **in replay mode** (D4) |
| **T3** `gpu` | needs CUDA / bf16 / ≥ 16 GB VRAM | < 30 min | yes | GPU runner available (Q4), tagged only |

A T3 artifact on a Mac must exit 0 with a printed explanation of what it *would*
have done and what to rent — never a stack trace. That rule is inherited from the
reference repo's "no hangs on missing API keys" and generalized. D11 tightens it
for exercises: an explanation alone is not enough, a scaled-down run is required.

### D3 — One manifest per artifact

Two manifest kinds, both machine-readable, both read by the runner, the CI matrix,
the dependency groups and the coverage report.

`practice.yaml` — one per lesson, the primary artifact (schema in D12).

`demo.yaml` — one per *optional* parity demo:

```yaml
lesson: phases/11-llm-engineering/08-fine-tuning-lora
title: Fine-Tuning with LoRA & QLoRA
tier: T1
entrypoint: run.py
runtime_seconds: 180
needs_env: []
deps_group: llm
proves: >
  The from-scratch LoRALayer in the reference lesson is numerically
  equivalent to peft.LoraConfig at the same rank and alpha.
parity_with: phases/11-llm-engineering/08-fine-tuning-lora/code/lora.py
reference_doc: phases/11-llm-engineering/08-fine-tuning-lora/docs/en.md
```

### D4 — Cassettes, not simulations

The reference repo fakes model responses by hand. This repo **records real ones
once** into `cassettes/*.json` and replays them by default:

```text
DEMO_MODE=replay   # default. Deterministic, free, offline, CI-safe.
DEMO_MODE=live     # hits the provider, re-records, prints token cost.
```

Strictly better than a hand-written simulation on three axes: it is real model
output, it costs the learner nothing, and CI can assert on it. Cassettes are
committed, redacted of keys, and carry the model ID and date they were recorded so
staleness is visible. 59 exercises need this.

### D5 — Parity assertions, where the comparison earns its place

Wherever the reference lesson hand-rolls something the ecosystem also provides,
the artifact can assert they match:

```python
mine   = reference_mha(x, heads=4)             # imported from the lesson's code/
theirs = torch.nn.MultiheadAttention(...)(x)   # the real thing
assert_close(mine, theirs, atol=1e-5)
```

This proves the learner's toy implementation *was* the real implementation, and it
turns a claim in prose into a green test. Roughly 200 lessons support it (all of
Phases 01–03, 07, most of 10). Per D9 it is now optional per lesson, and its
machinery (`harness/parity.py`) is called from exercise solutions too.

### D6 — Uniform per-artifact contract

Every lesson directory is a **container** with two halves. The `practice/` half is
the deliverable and is always present; the parity demo is optional **as a unit** —
either all four of its files exist or none do, because `audit_demos.py` requires a
manifest, a README and ≥3 tests of any demo that exists (§7):

```text
demos/phases/<phase>/<lesson>/
├── demo.yaml            # ┐ the parity demo (D5, D9) — optional, but
├── run.py               # │ all-or-nothing: its entrypoint; --explain
├── README.md            # │ prints concept + link
├── tests/test_*.py      # ┘ ≥3, per audit_demos.py
└── practice/            # the deliverable (D10) — always present
    ├── practice.yaml
    ├── README.md
    ├── exNN_<slug>.py
    ├── fixtures/
    └── tests/test_practice.py
```

Rules for both halves: fixed seed; exit 0 or a clear tiered skip; output is a
small metric table or a before/after diff, never a log wall; `--explain` works
with zero deps installed. Uniformity is the only reason one runner can drive 511
directories.

### D7 — Notebooks are generated, never hand-written

The reference repo's 295 empty `notebook/` dirs are a promise nobody kept, and
hand-maintaining notebooks alongside scripts guarantees drift. Use `jupytext` to
derive `notebook.ipynb` from `run.py` / `exNN_*.py` at build time. One source of
truth, and the empty-directory failure mode cannot recur.

### D8 — Dependency groups per phase cluster, not one requirements.txt

`uv` with extras, so a learner on Phase 01 installs numpy and nothing else:

```bash
uv sync --extra math      # 01-03
uv sync --extra vision    # 04, 12
uv sync --extra audio     # 06
uv sync --extra llm       # 07, 10, 11
uv sync --extra agents    # 13-16
uv sync --extra infra     # 17
```

`scripts/check_deps.py` asserts every artifact's imports are covered by its
declared `deps_group` — otherwise groups rot into a second monolith.

### D9 — Solutions are the unit of value

The primary deliverable is 2,090 exercise solutions plus 5 labs, not the ~200 parity
demos. Rationale, in the order it matters:

1. **The spec is free.** The exercise states the task, the fixture size and the
   threshold (§1). A parity demo has to invent all three.
2. **It answers what the learner is actually stuck on.** They hit the exercise
   list; they never wondered whether the lesson matched torch.
3. **One commitment, not two.** 2,090 solutions is already the larger promise;
   layering ~200 invented demos on top adds cost for less return.

Parity keeps its machinery and its ~200 optional demos (D5). A lesson may ship
`practice/` alone, both halves, or — for the 33 lessons with no exercises —
neither.

### D10 — One file per exercise, index-identical

`ex<NN>_<slug>.py`, zero-padded to match the numbered list in `docs/en.md`.
Exercise 3 on the lesson page → `ex03_*.py`, by string substitution: the same
property D1 gives the lesson path, one level deeper. Each file is independently
runnable (`uv run python ex03_sliding_window_rate_limit.py`, per §8 Q2) and
independently graded.

The full shape, for the lesson this design was written against:

```text
demos/phases/11-llm-engineering/12-guardrails/practice/
├── practice.yaml
├── README.md                            # bilingual exercise list; prose answers
├── ex01_safety_category_classifier.py   # 13 MLCommons categories, P/R on 50 prompts
├── ex02_encoding_evasion_detector.py    # base64/ROT13/hex/leet/zero-width/morse
├── ex03_sliding_window_rate_limit.py    # 10 req/min sliding, retry-after
├── ex04_rag_hallucination_detector.py   # sentence overlap < 20% -> flagged
├── ex05_red_team_suite.py               # 100 attacks / 5 categories, per-cat rates
├── fixtures/
│   ├── safety_prompts.json              # the 50 labelled prompts ex01 asks for
│   ├── encoded_injections.json          # the 20 encodings ex02 asks for
│   ├── rag_pairs.json                   # the 10 response/source pairs ex04 asks for
│   └── red_team_100.json                # the 100 attacks ex05 asks for
└── tests/test_practice.py               # one test function per exercise
```

Rejected: renaming the tree to `lessons/` with `parity/` and `practice/` as
siblings. Cleaner on paper, but it churns every harness path and all four M0 demos
to buy a directory name. One tree, one path map, `demo.yaml` optional instead.

### D11 — Three kinds; tier stays orthogonal

`kind` is the only new axis. Cost and hardware stay on D2's tiers.

| `kind` | When | Ships |
|---|---|---|
| `code` | the exercise asks for something runnable — **1,963**, i.e. every non-prose exercise, at whatever tier | `exNN_<slug>.py` + a test |
| `explain` | prose only — "read Section 3 … explain in three sentences" (127) | a sourced answer in `practice/README.md`, no file |
| `lab` | the lesson has `## Practice Lab`: **one** deliverable with N requirements, not N exercises (5 lessons, 38 requirements) | `lab_<slug>.py` + one test per requirement |

`lab` exists because `13/28-mcp-tool-contracts-and-content` does not list five
problems — it says "extend the contract lab with a `search_evidence` tool" and
then lists nine requirements of that one tool. Treating those as nine exercises
would produce nine fragments of one program. (`13/10-mcp-resources-and-prompts`
has both headings, so it ships a lab *and* five exercise files.)

**T3 exercises ship a scaled-down runnable, not just an explanation.** "Swap the
backing model to Qwen3-Coder-30B on vLLM, compare pass@1 and $-per-task" becomes
the same comparison code over a 20-item fixture against cassettes, plus the real
command and its cost printed. D2 permits explain-and-skip; for exercises that is
not enough, because the exercise asked for a measurement and a skip measures
nothing. A T3 solution that only skips fails the audit (§7).

### D12 — The exercise text is the spec: stored verbatim, and bilingually

```yaml
lesson: phases/11-llm-engineering/12-guardrails
reference_doc: phases/11-llm-engineering/12-guardrails/docs/en.md
exercises_sha256: <hash of the Exercises SECTION only>   # spec-level drift, D15
deps_group: llm
exercises:
  - index: 1
    file: ex01_safety_category_classifier.py
    kind: code
    tier: T0
    entry: classify                    # the graded symbol, D13
    text_en: >
      Build a LlamaGuard-style classifier. Create a keyword + regex classifier
      that maps inputs and outputs to 13 safety categories (from the MLCommons
      AI Safety taxonomy: violent crimes, non-violent crimes, …). Return the
      category code and confidence. Test on 50 hand-written prompts and measure
      precision/recall.
    text_zh: >
      做一個 LlamaGuard 風格的分類器。建一個關鍵字 + 正規表達式分類器，把輸入與
      輸出映射到 13 個安全類別（取自 MLCommons AI Safety 分類法：暴力犯罪、
      非暴力犯罪、…）。回傳類別代碼與信心值。在 50 個手寫提示詞上測試，量測
      precision/recall。
    verifies: macro precision >= 0.80 and recall >= 0.75 over fixtures/safety_prompts.json
    runtime_seconds: 2
    fixtures: [fixtures/safety_prompts.json]
    uses_reference: [detect_injection, detect_pii]   # symbols from the lesson's code/
```

`text_zh` is optional — 4 of the 474 lessons lack a translated exercise block.
Solution code and docstrings are English, by code convention; the generated
`README.md` carries both languages, so a solution is findable from the zh lesson
page, which is the one actually being read.

Storing the text verbatim rather than a summary is what makes D15's drift check
meaningful and what lets the generator (§6) be handed a spec rather than a gist.

### D13 — Every solution doubles as an auto-grader

`practice.yaml` declares `entry:` — the symbol the test calls.
`tests/test_practice.py` imports it through `harness/practice.py`, which honours
an override:

```bash
pytest                                    # grades the shipped solution
PRACTICE_IMPL=./my_attempt.py pytest      # grades YOUR attempt, same fixture, same threshold
```

This is the difference between publishing 2,090 answer keys and publishing 2,090
graders, and it costs one indirection. It is also the containment for the obvious
objection to this whole repo — that handing out solutions removes the reason to do
the exercise. A learner can now attempt the exercise, score it against the same
50-prompt fixture the solution is held to, and read the answer afterwards.

### D14 — "Clean and simple" gets a mechanical ceiling

Clean, simple, easy-to-understand code is the stated requirement, and across
2,090 generated files it will not survive on good intentions.
`scripts/audit_practice.py` enforces:

- **≤ 120 lines** of code per file excluding the docstring; hard fail over 150
- one file per exercise; **no imports between exercise files** — only stdlib, the
  declared `deps_group`, `harness/`, and the lesson's own `code/`
- the module docstring must carry, in this order: the exercise text verbatim; a
  **"Reading of the exercise:"** line; the approach in ≤ 3 sentences; the expected
  output
- `ruff format` clean; `ruff check` with `C901` max-complexity **8**
- functions over classes unless the exercise asks for state
- output is a small table or a handful of numbers, never a log wall;
  `--explain` works with zero deps (both from D6)
- fixtures are committed JSON, ≤ 50 KB, **labelled**, and generated by a
  `make_fixture()` in the same file — never an opaque blob

The **"Reading of the exercise:"** line deserves its own note. Many exercises are
genuinely ambiguous — *"Plot the loss of a tiny one-layer model on a synthetic
copy task. Do more heads help, plateau, or hurt?"* has no single right answer and
several defensible setups. Forcing the chosen interpretation into the docstring
makes the one failure mode a generator cannot be gated against — quietly answering
a different question — visible in a diff, and it is the specific thing phase-batch
review reads (§6.4).

A solution that cannot fit in 120 lines is nearly always a misread exercise, not a
hard exercise. The ceiling is a correctness signal as much as a style one.

### D15 — Exercise-level coverage and spec drift

`demo coverage --practice` generates a 2,090-row table from the tree:

```text
✅ verified   ⬚ unbuilt   📝 prose answered   ⏭ tier-skipped   ⚠ spec drifted
```

`exercises_sha256` hashes the exercise section alone, which catches what a
whole-document hash cannot resolve: the lesson body changed but the exercises did
not (solutions still valid), or the body is untouched and an exercise was reworded
(every solution under it is now suspect). Since the exercise *is* the spec, drift
there invalidates the answer — so it is tracked separately from
`reference_doc_sha256`.


---

## 4. Repo layout
```text
ai-engineering-from-scratch-demo/
├── DESIGN.md
├── README.md                  # coverage table, generated
├── pyproject.toml             # uv, extras per D8
├── demos/phases/…             # path-identical mirror (D1), practice/ inside (D10)
├── harness/
│   ├── runner.py              # run one artifact / a phase / a tier
│   ├── practice.py            # practice.yaml loading + the PRACTICE_IMPL shim (D13)
│   ├── cassette.py            # record & replay (D4)
│   ├── parity.py              # assert_close + import-from-reference helper (D5)
│   ├── manifest.py            # demo.yaml + practice.yaml schemas
│   └── tiers.py               # capability probe: cuda? key? net?
├── scripts/
│   ├── census.py              # re-derives every number in §1 and §5
│   ├── coverage.py            # reference tree vs demo tree -> README + badges
│   ├── check_deps.py
│   ├── audit_demos.py         # every demo has manifest, README, >=3 tests
│   ├── audit_practice.py      # D14's ceilings; every doc exercise has an entry
│   ├── scaffold.py            # generate a demo skeleton from a lesson path
│   ├── scaffold_practice.py   # parse the exercise block -> practice.yaml + stubs
│   └── notebooks.py           # generate .ipynb from run.py / exNN (D7)
├── cassettes/                 # shared fixtures
└── (CI) .github/workflows/
    ├── t0.yml                 # every push
    ├── t1.yml                 # nightly, model cache
    └── t2-live.yml            # weekly, real keys, re-record cassettes, cost report
```

This lives inside `ai_experiment` and stays there (Q1). GitHub only reads
workflows from the *repository* root, so the three files sit at
`../.github/workflows/aiefs-demo-*.yml` with a `paths:` filter and
`working-directory:` set — that indirection is permanent, not a stopgap.

Runner UX:

```bash
demo practice run  phases/11-llm-engineering/12-guardrails        # all 5
demo practice run  phases/11-llm-engineering/12-guardrails --ex 3 # just exercise 3
demo practice verify --phase 11 --tier T0
demo practice list --unbuilt --phase 11
demo run phases/11-llm-engineering/08-fine-tuning-lora            # the parity demo
demo verify --tier T0                                             # what CI runs
demo coverage --practice                                          # honest, per-exercise
```


---

## 5. Measured exercise load per phase
`labs` are counted separately: one deliverable each, not N exercises (D11).

| Phase | Lessons | Exercises | code | patch | T2 | T3 | prose | labs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 00 setup-and-tooling | 12 | 45 | 38 | 0 | 3 | 4 | 0 | 0 |
| 01 math-foundations | 22 | 92 | 91 | 0 | 0 | 0 | 1 | 0 |
| 02 ml-fundamentals | 18 | 78 | 71 | 0 | 0 | 1 | 6 | 0 |
| 03 deep-learning-core | 13 | 63 | 56 | 0 | 0 | 5 | 2 | 0 |
| 04 computer-vision | 28 | 84 | 61 | 0 | 0 | 21 | 2 | 0 |
| 05 nlp-foundations-to-advanced | 29 | 87 | 74 | 5 | 1 | 7 | 0 | 0 |
| 06 speech-and-audio | 17 | 51 | 32 | 16 | 0 | 3 | 0 | 0 |
| 07 transformers-deep-dive | 16 | 52 | 33 | 12 | 1 | 5 | 1 | 0 |
| 08 generative-ai | 15 | 47 | 29 | 14 | 0 | 4 | 0 | 0 |
| 09 reinforcement-learning | 12 | 36 | 32 | 3 | 0 | 1 | 0 | 0 |
| 10 llms-from-scratch | 24 | 116 | 85 | 7 | 0 | 15 | 9 | 0 |
| **11 llm-engineering** | **17** | **79** | **72** | **0** | **6** | **1** | **0** | **0** |
| 12 multimodal-ai | 25 | 125 | 101 | 2 | 1 | 5 | 16 | 0 |
| 13 tools-and-protocols | 31 | 148 | 128 | 10 | 8 | 0 | 2 | **5** |
| 14 agent-engineering | 42 | 212 | 195 | 0 | 9 | 0 | 8 | 0 |
| 15 autonomous-systems | 22 | 110 | 60 | 20 | 6 | 2 | 22 | 0 |
| 16 multi-agent-and-swarms | 25 | 123 | 86 | 23 | 4 | 0 | 10 | 0 |
| 17 infrastructure-and-production | 28 | 140 | 63 | 21 | 7 | 24 | 25 | 0 |
| 18 ethics-safety-alignment | 30 | 150 | 93 | 25 | 7 | 2 | 23 | 0 |
| 19 capstone-projects | 85 | 252 | 227 | 0 | 6 | 19 | 0 | 0 |
| **Total** | **511** | **2,090** | **1,627** | **158** | **59** | **119** | **127** | **5** |

Reading this table:

- **Phases 14, 19, 18, 13, 17** carry 900 exercises between them — 43% of the work
  sits in five phases.
- **Phase 01 is the cleanest batch in the repo**: 92 exercises, 91 of them plain
  T0 code, one prose. No keys, no GPU, no cassettes.
- **Phases 17, 15, 18 are the prose-heavy ones** (25, 22, 23) — governance,
  operations and alignment lessons ask you to argue rather than to run. Their
  effective code load is much lighter than the raw count suggests.
- **Phase 04 and 17 hold half of all T3 work** (21 and 24 of 119) — diffusion,
  NeRF, k8s. These are where D11's scaled-down rule does the most work.
- **Phase 19's 252 exercises span only 53 of its 85 lessons**; 32 capstones have no
  exercise block. Solve what exists; do not invent capstone demos.

The 33 lessons with no practice content (`01/15-statistics-for-ml`, two Phase 13
MCP lessons, 30 Phase 19 capstones) get no `practice/` directory. Coverage marks
them `n/a` rather than `⬚`, so they never read as unfinished work.


---

## 6. How 2,090 solutions actually get written

1. **`scaffold_practice.py <lesson>`** parses `## Exercises` from `docs/en.md` and
   `## 練習` from `docs/zh.md` (and `## Practice Lab`, as one `lab` entry), folds
   wrapped lines back together, splits the numbered list, and emits `practice.yaml`
   with verbatim bilingual text plus a stub per exercise with the docstring
   pre-filled. It proposes `kind`, `tier` and `deps_group` by keyword — the same
   classifier `census.py` uses, so §5's estimates and the scaffold never disagree.
2. **One agent invocation per lesson** (3–8 exercises), context capped at that
   lesson's `docs/en.md`, its `code/*`, and the exercise text. No repo-wide
   context, so cost is flat per lesson: ~474 invocations, not 2,090.
3. **The gate is mechanical, and kind-aware (D11).** For `code` and `lab`:
   `audit_practice.py` + every `exNN` actually running + `tests/test_practice.py`
   passing + inside the tier budget. Any of: over length, complexity above 8, a
   missing `verifies` threshold, a surviving scaffold `TODO`, an unlabelled
   fixture, a T3 solution that only skips, or a non-zero exit = rejected.

   `explain` items (127) ship prose in `practice/README.md` and no file, so
   neither "runs" nor `verifies` applies to them. Their equivalent gate is
   **a resolvable citation**: every answer names the lesson section it draws on,
   `audit_practice.py` checks that anchor exists in `docs/en.md`, and an answer
   with no citation or a dead one is rejected. Both paths run with no human in
   the loop.
4. **Human review is per phase batch**, and reads three things: the coverage table;
   every **"Reading of the exercise:"** line, which is where a generator can
   quietly answer a different question and no gate will catch it; and the
   `kind`/`tier` classification the scaffold guessed.

Realistic throughput: one phase per session for T0 phases, slower for
cassette-heavy and prose-heavy phases where each item needs a live recording pass
or a real source to cite.


---

## 7. Risks and how each is contained

| Solutions spoil the exercise | D13 — every solution is also a grader for the learner's own attempt; `practice/README.md` leads with the exercise, not the answer |
| Generator answers a *different* question | Mandatory "Reading of the exercise:" docstring line (D14); the one thing phase-batch review must read, since no gate can catch it |
| 2,090 files of plausible-looking slop | D14's mechanical ceilings — 120 lines, complexity 8, one file, no cross-imports. Needing more than 120 lines usually means the exercise was misread |
| Upstream rewords an exercise | `exercises_sha256` hashes the exercise block alone (D15); affected rows flag `⚠ spec drifted` |
| Fixtures become unlabelled magic blobs | Committed labelled JSON, ≤ 50 KB, generated by an in-file `make_fixture()` (D14) |
| T3 exercises nobody can run become dead weight | D11 — a scaled-down runnable plus the real command and cost; a T3 solution that only skips fails the audit |
| Prose answers become unsourced hand-waving | An `explain` answer must cite the section, figure or table it rests on; the 127 of them are the phase-batch reviewer's other job |
| Reference repo moves; solutions rot | `practice.yaml` pins `reference_doc` and both hashes; `coverage.py` flags drift |
| Cassettes go stale as models ship | Each cassette records model ID + date; weekly `t2-live.yml` re-records and diffs; semantic drift is a finding, not a failure |
| A half-finished repo with a dishonest README | Coverage is generated from the tree, never hand-written; unbuilt exercises show as `⬚`, the 33 exercise-free lessons as `n/a` |
| API costs during authoring | T2 authoring is the only live-key path; one recording per exercise, cost printed and logged per run |
| Scope creep into rewriting the curriculum | Hard rule: a solution answers the exercise as posed. It may not re-teach the lesson, and it may not improve the exercise — a bad exercise is an upstream issue to file, not a thing to silently fix |


---

## 8. Settled questions

1. **Separate GitHub repo, or a directory inside `ai_experiment`?**
   **Settled: a directory inside `ai_experiment`.** One repo, no split. The cost
   is confined to CI (§4): the three workflows live at the *repository* root as
   `../.github/workflows/aiefs-demo-*.yml`, each carrying a `paths:` filter and
   `working-directory:`, because GitHub only reads workflows from the root.
   `harness/parity.py` already searches every ancestor for the reference checkout
   rather than a fixed sibling, so nothing else changes.
2. **Language scope.** **Settled: Python-only, `uv`-managed.** `uv` is the sole
   toolchain — `uv sync --extra <group>` for dependencies (§4), `uv run demo ...`
   for every command, and a committed `uv.lock` so a solution that verifies here
   verifies anywhere. No pip, no conda, no bare `python`. The reference repo's 129
   TS files cluster in Phases 13–14; revisit at M3, where a TS solution alongside
   the Python one is cheap because the exercise text is shared.
3. **Provider.** **Settled: OpenAI (ChatGPT) for T2**, via the `openai` SDK.
   Pin the exact model and `max_tokens` when the first cassette is recorded, sized
   to hold the $0.02/artifact budget. `Cassette` stores a `provider` field, so a
   second provider stays additive rather than a rewrite — but v1 records one
   provider so cassettes remain comparable.
4. **GPU budget.** **Settled: both CPU and GPU are available.** T3's 119
   exercises can therefore be verified at full scale, not just described. D11's
   scaled-down runnable is still required for every T3 artifact — it is what keeps
   a CPU-only contributor unblocked — but it is now the fallback path rather than
   the only path, and the README should say which of the two a given result came
   from.

---


---

## 9. What the prior spike proved (2026-09-02)

to be buildable to this design, and the two findings at the end of this section
are the kind of thing only a real run surfaces. `uv run demo verify` was green
there: 6 pass, 1 skip (the T2 demo, which had no recorded tape).

**Harness** — `harness/` is stdlib-only, so `demo coverage`, `demo list` and
`run.py --explain` work on a bare Python with nothing installed:

| Module | Does |
|---|---|
| `manifest.py` | `demo.yaml` schema + a strict YAML subset parser (a full parser would be a dependency the zero-dep rule cannot afford) |
| `tiers.py` | capability probe; turns "no GPU" into a skip with a remedy, never a stack trace |
| `parity.py` | `load_reference` (imports the lesson's own code, never copies it) + `assert_close` reporting measured deviation |
| `cassette.py` | record/replay, provenance, cost, redaction at the write boundary |
| `coverage.py` | reference tree vs demo tree, plus doc-hash drift detection |
| `runner.py` | `demo list / run / verify / coverage` |
| `explain.py` | `--explain` with zero deps; lesson URL derived from the mirrored path |

**Gates** — all mechanical, no human in the loop: `audit_demos.py` (manifest,
README, ≥3 tests, no surviving scaffold `TODO`, tier budget ceiling),
`check_deps.py` (imports vs declared `deps_group`), `coverage.py --check`
(README table regenerated, never hand-written), `demo verify` (it runs, tests
pass, inside the declared budget). Plus `scaffold.py` and `notebooks.py` (D7).

**Four demos, the four shapes M0 had to prove:**

| Demo | Tier | Result |
|---|---|---|
| `01/02-vectors-matrices-operations` | T0 | 11 parity checks vs numpy, worst deviation **1.8e-15** |
| `07/03-multi-head-attention` | T1 | output *and per-head weights* vs `torch.nn.MultiheadAttention` at **2.8e-16** in float64; GQA vs `enable_gqa=True`; float32 costs 1.6e-07 |
| `11/08-fine-tuning-lora` | T1 | forward / parameter counts / merged weights vs `peft` across four rank-alpha pairs |
| `11/01-prompt-engineering` | T2 | skips cleanly with a record command — **no cassette recorded yet** (this environment has no API key) |

55 harness tests plus 40 demo tests (36 run today; 4 wait on the cassette).

**Two findings the design did not anticipate**, both from actually running things:

- The lesson's `format_anthropic_request()` sets `temperature`, which current
  Claude models reject with a 400. A simulation can never surface that; the T2
  demo does, and `adapt_request()` shows the fix. This is evidence for D4
  beyond cost and determinism.
- The lesson's `quantize_to_nf4` is block-wise *symmetric int4-range*
  quantisation, not the NF4 codebook. So the LoRA demo reports it as a measured
  divergence (10.35% of weight RMS) rather than forcing a parity claim — the
  "where the two diverge" half of §2, which turns out to carry real weight.

**Open for M1:** record `cassettes/prompt-patterns.json` once with a live key
(`DEMO_MODE=live`), which closes the last M0 shape and unskips 4 tests.
