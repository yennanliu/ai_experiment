# ai-engineering-from-scratch-demo — Design

A companion repo of **runnable, verified example code for every lesson** in
[`ai-engineering-from-scratch`](https://yennj12.js.org/ai-engineering-from-scratch/index.html)
(local: `../ai-engineering-from-scratch`).

Status: **M0 built and verified** (2026-09-02). The harness, the four gates and
four demos across three tiers all run; see §10. M1 (Phase 11) is next.
Date: 2026-09-01, revised 2026-09-02.

---

## 1. What the reference repo actually is

Measured, not assumed (`phases/` tree, 2026-09-01):

| Fact | Number |
|---|---|
| Phases | 20 |
| Lessons (`phases/*/*/`) | **511** |
| Lessons with code | 505 / 511 (the 6 without are Phase 00 shell/editor/docker lessons) |
| Python files (excl. `__pycache__`) | 634 |
| …with a `__main__` block | 607 |
| Test files | 94 (~18% of lessons) |
| TypeScript / Julia / Rust files | 129 / 20 / 10 |
| `notebook/` directories | 295 |
| **Actual `.ipynb` files** | **0** — every one holds only `.gitkeep` |
| Python files importing `torch` | **90** (58 of them in Phase 19 capstones) |
| Python files touching a real LLM SDK (`anthropic`/`openai`) | **20** |
| Per-lesson dependency manifests | 0 (one root `requirements.txt`) |

The last three rows are the important ones. Phase 07 Lesson 03
"Multi-Head Attention" opens with:

> ```
> """Multi-head attention from scratch in pure stdlib.
> No numpy, no torch. A tiny Matrix class carries the ops we need.
> ```

That is a **deliberate and good** pedagogical choice, stated in `AGENTS.md`:
code must be self-terminating, exit 0, and never hang on a missing API key. So
the whole curriculum is zero-dependency, deterministic, hand-rolled, and
simulated end to end.

## 2. The gap this repo fills

The reference repo teaches the **mechanism**. It never shows the learner the
**real toolchain doing the same thing**. Concretely, after 511 lessons a learner
has written attention with a hand-built `Matrix` class but has never:

- called `torch.nn.MultiheadAttention` and compared the numbers to their own,
- watched a real HF checkpoint load, tokenize, and generate,
- seen a real `anthropic.messages.create` response object,
- run anything in a notebook (295 empty `notebook/` dirs),
- installed only what one lesson needs (root `requirements.txt` pulls
  `torch` + `librosa` + `transformers` to run lesson 01).

**So this repo is not a rewrite and not a fork.** Its unit of value is the
*delta*:

> For lesson *L*, the reference repo builds *X* from scratch. This repo runs the
> production equivalent of *X*, side by side, and asserts where the two agree
> and where they diverge.

That framing is what keeps 511 examples from being 511 redundant restatements.

## 3. Core design decisions

### D1 — Path-identical mirroring

```
demos/phases/11-llm-engineering/08-fine-tuning-lora/
```

Byte-for-byte the same relative path as the reference lesson. Consequences: a
lesson URL maps to a demo directory by string substitution, the site can
deep-link, and coverage is a `diff` of two directory listings — not a
hand-maintained table that rots.

### D2 — Cost/hardware tiers (the decision that makes 511 tractable)

The real blocker to "example code for every lesson" is that Phase 01 runs in
40ms on a laptop and Phase 08 wants an A100. One repo cannot pretend those are
the same thing, so every demo declares a tier:

| Tier | Means | Budget | Network | CI |
|---|---|---|---|---|
| **T0** `cpu-instant` | stdlib / numpy / sklearn | < 10 s | none | every push |
| **T1** `cpu-heavy` | torch-CPU, small HF checkpoint (< 500 MB) | < 5 min | model download, cached | nightly |
| **T2** `api` | real provider call (Claude etc.) | < 20 s, < $0.02 | provider | every push **in replay mode** (see D4) |
| **T3** `gpu` | needs CUDA / bf16 / ≥ 16 GB VRAM | < 30 min | yes | manual / rented runner, tagged only |

A T3 demo on a Mac must exit 0 with a printed explanation of what it *would*
have done and what to rent — never a stack trace. That rule is inherited from
the reference repo's "no hangs on missing API keys" and generalized.

### D3 — One manifest per demo

`demo.yaml`, machine-readable, is what the runner, the CI matrix, the dependency
groups, and the coverage report all read:

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

`parity_with` is optional and is the highest-value field in the file — see D5.

### D4 — Cassettes, not simulations

The reference repo fakes model responses by hand. This repo **records real ones
once** into `cassettes/*.json` and replays them by default:

```
DEMO_MODE=replay   # default. Deterministic, free, offline, CI-safe.
DEMO_MODE=live     # hits the provider, re-records, prints token cost.
```

This is strictly better than a hand-written simulation on three axes: it is real
model output, it costs the learner nothing, and CI can assert on it. Cassettes
are committed, redacted of keys, and carry the model ID + date they were recorded
so staleness is visible.

### D5 — Parity assertions are the flagship artifact

Wherever the reference lesson hand-rolls something the ecosystem also provides,
the demo asserts they match:

```python
# phases/07-transformers-deep-dive/03-multi-head-attention/run.py
mine  = reference_mha(x, heads=4)              # imported from the lesson's code/
theirs = torch.nn.MultiheadAttention(...)(x)   # the real thing
assert_close(mine, theirs, atol=1e-5)
```

This is the single most convincing thing the repo can produce: it proves the
learner's toy implementation *was* the real implementation, and it turns a claim
in prose into a green test. Roughly 180–220 of the 511 lessons support a parity
check (all of Phases 01–03, 07, most of 10, parts of 02/04/05).

Where parity is impossible (governance lessons, protocol lessons, capstones), the
demo instead ships a **scenario runner** — same convention the reference repo's
`AGENTS.md` already mandates for conceptual lessons, so this is consistent rather
than novel.

### D6 — Uniform per-demo contract

Every demo, in every language, in every tier:

```
demos/phases/<phase>/<lesson>/
├── demo.yaml
├── README.md          # 20 lines max: what it proves, how to run, expected output
├── run.py             # entrypoint; --explain prints concept + lesson link
├── tests/test_*.py    # >= 3 assertions on the lesson's *claim*, not on plumbing
├── cassettes/         # T2 only
└── notebook.ipynb     # optional; generated from run.py via jupytext, not hand-authored
```

Rules: fixed seed; exit 0 or a clear tiered skip; output is a small metric table
or before/after diff, never a log wall; `--explain` works with zero deps
installed. Uniformity is the only reason one runner can drive 511 directories.

### D7 — Notebooks are generated, never hand-written

The reference repo's 295 empty `notebook/` dirs are a promise nobody kept, and
hand-maintaining 511 notebooks alongside 511 scripts guarantees drift. Use
`jupytext` to derive `notebook.ipynb` from `run.py` at build time. One source of
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

`scripts/check_deps.py` asserts every demo's imports are covered by its declared
`deps_group` — otherwise groups rot into a second monolith.

## 4. Repo layout

```
ai-engineering-from-scratch-demo/
├── DESIGN.md
├── README.md                  # coverage table, generated
├── pyproject.toml             # uv, extras per D8
├── demos/phases/…             # path-identical mirror (D1)
├── harness/
│   ├── runner.py              # run one demo / a phase / a tier
│   ├── cassette.py            # record & replay (D4)
│   ├── parity.py              # assert_close + import-from-reference helper (D5)
│   └── tiers.py               # capability probe: cuda? key? net?
├── scripts/
│   ├── coverage.py            # diff reference tree vs demos tree -> README + badges
│   ├── check_deps.py
│   ├── audit_demos.py         # every demo has manifest, README, >=3 tests
│   └── scaffold.py            # generate a demo skeleton from a lesson path
├── cassettes/                 # shared fixtures
└── (CI) .github/workflows/
    ├── t0.yml                 # every push
    ├── t1.yml                 # nightly, model cache
    └── t2-live.yml            # weekly, real keys, re-record cassettes, cost report
```

While this lives inside `ai_experiment`, GitHub only reads workflows from the
*repository* root, so the three files sit at `../.github/workflows/aiefs-demo-*.yml`
with a `paths:` filter and `working-directory:` set. Splitting the repo out
(open question 1) moves them back to the layout above and drops both.

Runner UX:

```bash
demo run phases/11-llm-engineering/08-fine-tuning-lora
demo run --phase 11 --tier T0        # everything free and instant in phase 11
demo verify --tier T0                # what CI runs
demo coverage                        # honest completion table
```

## 5. Tier and effort estimate per phase

| Phase | Lessons | Dominant tier | Parity checks viable | Notes |
|---|---:|---|---|---|
| 00 setup-and-tooling | 12 | T0 | — | 6 lessons are shell/editor; ship a verify script, not a demo |
| 01 math-foundations | 22 | T0 | **high** | hand-rolled vs numpy/scipy — cleanest parity wins in the repo |
| 02 ml-fundamentals | 18 | T0 | **high** | vs scikit-learn |
| 03 deep-learning-core | 13 | T0→T1 | **high** | own autograd vs `torch.autograd` |
| 04 computer-vision | 28 | T1/T3 | medium | 24 lessons already use torch; SD/NeRF/3DGS are T3 |
| 05 nlp-foundations | 29 | T0/T1 | medium | vs gensim, HF tokenizers |
| 06 speech-and-audio | 17 | T1/T3 | low | Whisper-tiny keeps most at T1 |
| 07 transformers-deep-dive | 16 | T1 | **high** | flagship phase — pure-stdlib attention vs torch |
| 08 generative-ai | 15 | T3 | low | diffusion; mostly rented-GPU tier |
| 09 reinforcement-learning | 12 | T0/T1 | medium | vs gymnasium / SB3 |
| 10 llms-from-scratch | 24 | T1/T3 | **high** | own GPT vs HF `GPT2LMHeadModel` |
| 11 llm-engineering | 17 | T2 | low | cassettes carry it; **start here** |
| 12 multimodal-ai | 25 | T2/T3 | low | CLIP is T1, VLMs T2 |
| 13 tools-and-protocols | 31 | T2 | low | real MCP server/client over stdio — highly demoable |
| 14 agent-engineering | 42 | T2 | low | largest phase; cassette-heavy |
| 15 autonomous-systems | 22 | T2 | low | scenario runners |
| 16 multi-agent-and-swarms | 25 | T2 | low | cassettes + deterministic scheduler |
| 17 infrastructure-and-production | 28 | T1/T2 | — | docker/k8s; verify-script style |
| 18 ethics-safety-alignment | 30 | T0/T2 | — | policy scorers, per the reference repo's own convention |
| 19 capstone-projects | 85 | mixed | — | **do not write 85 demos** — see below |

**Phase 19 gets different treatment.** 85 capstones, 58 already torch-based, each
a multi-file project. Writing 85 demos is waste. Instead: cluster them into ~10
themes, ship one end-to-end showcase per theme, and for the rest ship a
`smoke.py` that imports the capstone's entrypoint and asserts it constructs.
That converts 85 line items into 10 real deliverables + 75 cheap guards.

Effective new-authoring load: **~426 lessons + 10 showcases**, of which ~200
carry a parity check.

## 6. Build order

Sequenced by return, not by phase number.

- **M0 — Harness (1 week).** `harness/`, `demo.yaml` schema, runner, cassette
  record/replay, tier probe, T0 CI, `scaffold.py`, `coverage.py`. Prove it on
  **three** demos across three tiers: `01/02-vectors-matrices` (T0),
  `07/03-multi-head-attention` (T1 + parity), `11/08-fine-tuning-lora` (T1) and
  `11/01-prompt-engineering` (T2 + cassette). If the harness can't carry those
  four shapes it can't carry 511.
- **M1 — Phase 11 (17).** Where your own progress currently sits
  (`progress.txt` → `11/08-fine-tuning-lora`). Immediately useful to you, and it
  shakes out the cassette design under real API conditions.
- **M2 — Phases 13 + 14 (73).** Tools/protocols and agent engineering: adjacent
  to your existing `mcp/`, `agent_sysem/`, `orchestration_agents/` work, and the
  most demo-able material in the curriculum (a real MCP server over stdio is a
  far better artifact than a simulated one).
- **M3 — Phases 01–03 + 07 (69).** The parity showcase. Cheap, fast, entirely
  T0/T1, and produces the repo's most persuasive tests.
- **M4 — Phase 10 (24).** own-GPT-vs-HF parity; the natural sequel to M3.
- **M5 — Long tail (Phases 04–06, 08, 09, 12, 15–18).** Batch by dependency
  group so one env install serves a whole batch.
- **M6 — Phase 19 clusters (10 showcases + 75 smoke tests).**

## 7. How 426 demos actually get written

Manually authoring 426 demos is not credible; a generation pipeline is, provided
the gate is execution rather than review-by-eyeball.

1. `scaffold.py <lesson-path>` reads the lesson's `docs/en.md`, its `code/*.py`,
   and `quiz.json`, and emits a filled `demo.yaml` + stub `run.py` + test stub.
2. A per-lesson agent prompt (one template, phase-specific preamble) fills the
   stub. Input is capped at the lesson's own doc + code — no repo-wide context,
   so cost stays flat per lesson.
3. **The gate is mechanical:** `audit_demos.py` + actually running the demo +
   running its tests. Non-zero exit, missing manifest field, fewer than 3 tests,
   or a runtime over the declared budget = rejected, no human in the loop.
4. Human review is per **phase batch**, reading the coverage report and spot
   checking parity assertions — the place where judgment is actually needed.

Realistic throughput: one phase per session for T0/T1 phases, slower for
cassette-heavy T2 phases where each demo needs one live recording pass.

## 8. Risks and how each is contained

| Risk | Containment |
|---|---|
| Reference repo moves; demos rot | `demo.yaml` pins `reference_doc`; `coverage.py` flags lessons whose doc hash changed since the demo was written |
| Cassettes go stale as models ship | Each cassette records model ID + date; weekly `t2-live.yml` re-records and diffs; a semantic drift is a finding, not a failure |
| 511 half-finished demos, dishonest README | Coverage is generated from the tree, never hand-written; unbuilt lessons show as `⬚`, matching the reference repo's own ROADMAP glyphs |
| T3/GPU demos nobody can run | Tiered skip must print the concrete rent-a-GPU command and expected cost; a T3 demo that only crashes fails `audit_demos.py` |
| API costs during authoring | T2 authoring is the only live-key path, one recording per demo, cost printed and logged per run |
| Scope creep into rewriting the curriculum | Hard rule: a demo may not re-teach. If `run.py` needs more than 20 lines of README to justify, it belongs upstream as a lesson edit instead |

## 9. Open questions for you

(M0 proceeded on a stated default for each; none of these are settled.)

1. **Separate GitHub repo, or a directory inside `ai_experiment`?** The design
   assumes it can stand alone (its own CI, its own `pyproject.toml`), which
   argues for a separate repo with the reference repo as a sibling checkout.
   *M0 assumed: a directory, for now.* The one place it hurts is CI (see §4),
   and `harness/parity.py` now searches every ancestor for the reference
   checkout rather than a fixed sibling, so either answer works unchanged.
2. **Language scope.** Reference has 129 TS files. Python-only for v1, TS demos
   only for Phases 13–14 where the ecosystem is TS-first?
3. **Provider.** Claude-only for T2 (cheapest to keep coherent), or
   multi-provider so the cassette layer proves portability?
   *M0 assumed: Claude-only,* `claude-opus-5`, at `max_tokens: 1024` to hold the
   $0.02/demo budget. `Cassette` stores a `provider` field, so a second provider
   is additive rather than a rewrite.
4. **GPU budget.** Is there any rented-GPU allowance? If not, T3 (~40 lessons,
   mostly Phases 08/12 and parts of 04) ships as explain-and-skip only, and the
   README should say so plainly.


---

## 10. M0 — what shipped (2026-09-02)

Built and verified against the real reference checkout. `uv run demo verify`
is green: 6 pass, 1 skip (the T2 demo, which has no recorded tape yet).

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
