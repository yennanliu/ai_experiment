# ai-engineering-from-scratch-demo — Implementation Plan

The build sequence for [`DESIGN.md`](./DESIGN.md). That file settles *what* this
repo is and why each decision was made; this one settles *when* each piece gets
built, what "done" means for it, and what it costs. Refs prefixed `DESIGN` point
there — `DESIGN §5`, `DESIGN D8`; bare `§N` is internal to this file.

**Status (2026-09-04): nothing built yet.** M0 was built and verified once in a
prior spike (`DESIGN §9`), but none of that code is in this branch, so §3 rebuilds
it. All four questions that were open through that spike are now closed
(`DESIGN §8`), so nothing below is blocked on a decision.

---

## 1. Build order and why

Sequenced by return, not by phase number. This section is the *rationale*; §3–§5
carry the items and the effort.

| M | Scope | Why here |
|---|---|---|
| **M0** | Harness | Nothing else can be verified until the runner, the tiers and the gates exist. Proven buildable by the prior spike (`DESIGN §9`) |
| **M0.5** | Practice harness + one golden reference | The conventions get tested against 5 real exercises before they are applied to 2,090. A stop-and-revise gate, not a formality |
| **M1** | Phase 11 | Where `progress.txt` sits, so it is immediately useful. Its 6 T2 exercises are the only thing that forces the cassette design under real conditions |
| **M2** | Phases 01, 02, 03, 07 | All T0/T1 — no keys, no GPU, 1 prose item across 92 in Phase 01. The volume stress-test of the generation pipeline (`DESIGN §6`), and where reused parity assertions land hardest |
| **M3** | Phases 13 + 14 | Adjacent to the existing `mcp/`, `agent_sysem/` and `orchestration_agents/` work in `ai_experiment`; a real MCP server over stdio beats a simulated one. The 5 labs land here, so `kind: lab` gets exercised |
| **M4** | Phase 10 | own-GPT-vs-HF parity, the natural sequel to M2 |
| **M5** | Long tail: 00, 04, 05, 06, 08, 09, 12, 15, 16, 17, 18 | Batched by `deps_group` so one env install serves a whole batch. The three prose-heavy phases sit here, so a large slice is written answers rather than code |
| **M6** | Phase 19 | Largest single phase; deliberately last, once the pipeline is boring |

---

## 2. The effort model

Every number in §3–§5 is an estimate, not a measurement. Unlike `DESIGN §1` and
`DESIGN §5`, none of it comes from a census script, and this section exists so
that stays obvious.

Two inputs. First, the prior spike's actual cost for M0-scale work (`DESIGN §9`).
Second, a stated throughput assumption for solution volume:

| Stage | Throughput | Why |
|---|---|---|
| through M1 | **~13 exercises/day** | the generation pipeline (`DESIGN §6`) is still being tuned |
| M2–M4 | **~30/day** | pipeline hardened, all code exercises |
| M5–M6 | **~40/day** | prose answers and repeated `deps_group` batches dominate |

One person, working days. That assumption is the only knob: move it and re-derive
§5's totals rather than editing them.

---

## 3. M0 — Harness rebuild · ~4 days

Rebuilt from scratch, but against a design the prior spike already validated
(`DESIGN §9`), so this is construction rather than discovery.

| # | Item | Done when |
|---|---|---|
| 1 | `pyproject.toml` + `uv.lock`; the six `--extra` groups of `DESIGN D8` | `uv sync --extra math` resolves offline on a clean machine |
| 2 | `harness/manifest.py` — `demo.yaml` schema + strict YAML subset parser | round-trips every field in `DESIGN D3`; rejects an unknown key loudly |
| 3 | `harness/tiers.py` — capability probe | "no GPU" / "no key" becomes a skip **with a remedy string**, never a stack trace |
| 4 | `harness/parity.py` — `load_reference` + `assert_close` | imports the lesson's own module, never copies it; reports measured deviation |
| 5 | `harness/cassette.py` — record/replay, provenance, cost, redaction | a live record then a replay produce byte-identical output; no key survives the write boundary |
| 6 | `harness/coverage.py` — reference tree vs demo tree + doc-hash drift | `--check` exits non-zero on drift |
| 7 | `harness/runner.py` + `explain.py` — `demo list/run/verify/coverage`, `--explain` | all four work on bare Python with **zero** deps installed |
| 8 | Gates: `audit_demos.py`, `check_deps.py`, `coverage.py --check` | each fails a deliberately broken fixture |
| 9 | CI: `../.github/workflows/aiefs-demo-{t0,t1,t2-live}.yml` with `paths:` + `working-directory:` (`DESIGN §8` Q1) | T0 green on push |
| 10 | Two seed demos, one T0 + one T1 | `uv run demo verify --tier T0` green |

**Exit gate:** `uv run demo verify` green, *and* every gate demonstrated failing on
a broken fixture — a gate never seen to fail is not a gate.

**Risk:** item 2. A strict YAML subset parser is the one place the zero-dep rule
buys trouble; budget a day for it alone and keep the subset genuinely small.

---

## 4. M0.5 — Practice harness + golden reference · ~3 days

| # | Item | Done when |
|---|---|---|
| 1 | `practice.yaml` schema in `manifest.py` (`DESIGN D12`, verbatim bilingual text) | parses `11/12-guardrails`' 5 exercises |
| 2 | `harness/practice.py` — the `PRACTICE_IMPL` grading shim (`DESIGN D13`) | one solution grades itself, and a deliberately wrong variant fails |
| 3 | `demo practice run / verify / list` subcommands | `--ex 3` runs exactly one |
| 4 | `scripts/scaffold_practice.py` — parse `## Exercises` + `## 練習` → stubs | round-trips a lesson with wrapped lines and a `## Practice Lab` |
| 5 | `scripts/audit_practice.py` — `DESIGN D14`'s mechanical ceilings | fails a solution over the ceiling |
| 6 | `coverage.py --practice`; `scripts/census.py` | census reproduces every number in `DESIGN §1` and `DESIGN §5` |
| 7 | **`11/12-guardrails` end to end** — the 5 solutions of §4.1 | all 5 green at T0, no key, no GPU |

**Exit gate:** the golden reference is green *and* re-scaffolding it from scratch
reproduces the same file set. If the conventions cannot carry these five, they
cannot carry 2,090 — stop and revise rather than proceed.

**Risk:** §4.1's Exercise 5, below.

### 4.1 The golden reference, scoped

`11/12-guardrails` is the lesson the design was written against, so it is the one
to build first. It is a good choice on the merits: 5 exercises, all `code`, all
T0, no prose items, no API key, and each one states its own fixture size and
threshold.

| Ex | Solution | Fixture | `verifies` |
|---|---|---|---|
| 1 | `ex01_safety_category_classifier.py` | 50 labelled prompts | macro P/R over 13 MLCommons categories |
| 2 | `ex02_encoding_evasion_detector.py` | 20 encodings of one payload | all 20 decoded and flagged; 6 encodings covered |
| 3 | `ex03_sliding_window_rate_limit.py` | none (generated burst) | 15 req / 30 s → 10 allowed, 5 blocked, retry-after correct |
| 4 | `ex04_rag_hallucination_detector.py` | 10 response/source pairs | every <20%-overlap sentence flagged, no false positives on the paired set |
| 5 | `ex05_red_team_suite.py` | 100 attacks / 5 categories | per-category detection rate; the weakest category identified and 3 rules added |

The lesson's `code/guardrails.py` already exports what these build on —
`detect_injection`, `detect_pii`, `classify_topic`, `check_relevance`,
`check_system_prompt_leak`, `GuardrailPipeline`, `GuardrailMonitor` — so
`uses_reference` is populated from real symbols and no solution needs to fork the
lesson's code. That was verified against the reference checkout before this
section was written, which is why M0.5 is scoped as three days rather than
discovered as three weeks.

Exercise 5 is the one to watch: "identify which category has the lowest detection
rate and write 3 additional rules to improve it" makes the *solution* depend on
its own measurement, so its `verifies` cannot name a fixed category. Grading is
therefore undefined until the contract below is written down — and it must be
settled **before** the fixture is scaffolded, or the fixture gets tuned until the
answer is whatever the author wanted. One contract, shared verbatim by
`verifies`, `tests/test_practice.py` and `PRACTICE_IMPL`:

| Term | Definition |
|---|---|
| Baseline | the lesson's own `GuardrailPipeline` at its shipped rule set, unmodified |
| Metric | per-category **recall** over the 100-attack / 5-category fixture — an attack counts as detected if any rule fires |
| Weakest category | the strictly lowest baseline recall. If two tie, the fixture is wrong and must be regenerated: the exercise presumes a unique answer |
| Required improvement | recall in *that* category rises by **≥ 0.20 absolute**, and no other category's recall falls |
| "3 additional rules" | exactly 3 new entries appear in the pipeline's rule registry between baseline and post state, counted by registry diff, not by lines added |

The solution reports which category it picked; the test asserts the *delta*, so a
correct solution stays correct if the fixture is later regenerated with a
different weakest category.

---

## 5. M1–M6 — Solution volume

| M | Lessons | Exercises | Effort | Milestone-specific items |
|---|---|---|---|---|
| **M1** — Phase 11 | 17 | 79 | **~6 d** | The 6 T2 exercises force the cassette design under real load: record `cassettes/prompt-patterns.json` against OpenAI (`DESIGN §8` Q3) and pin the model + `max_tokens` to the $0.02/artifact budget |
| **M2** — Phases 01, 02, 03, 07 | 69 | 285 | **~10 d** | No keys, no GPU. First run of the pipeline at volume |
| **M3** — Phases 13 + 14 | 73 | 360 + 5 labs | **~13 d** | First `kind: lab` entries; a real MCP server over stdio |
| **M4** — Phase 10 | 24 | 116 | **~4 d** | own-GPT-vs-HF parity |
| **M5** — long tail (11 phases) | — | 998 | **~25 d** | One env install per `deps_group` batch. Phases 04 and 17 carry most of the T3 work — now verifiable at full scale (`DESIGN §8` Q4) |
| **M6** — Phase 19 | 53 | 252 | **~7 d** | — |
| **Total** |  | **2,090** | **~65 d** | plus 7 d for M0 + M0.5 → **~72 working days, ≈ 14–15 weeks** |

Every milestone shares one exit gate: `uv run demo verify` green for its phases,
`coverage.py --check --practice` showing no spec drift, and `audit_practice.py`
clean. A milestone with a red gate is not done, whatever its exercise count.

---

## 6. Sequencing notes

- **§3 and §4 are strictly serial.** After M0.5, *solution work* parallelises by
  phase: `DESIGN D1`'s path-identical mirroring means two phases never touch each
  other's `demos/phases/**` files. Six things are shared, though, and are not
  covered by that argument — `harness/`, `scripts/`, `pyproject.toml` +
  `uv.lock`, the generated root `README.md`, `cassettes/`, and the workflows.
  Rule: a phase branch may not edit any of the six. A harness or dependency change
  a phase needs lands first, on its own, and the phase branches rebase onto it;
  `README.md` and `uv.lock` are regenerated on `main` after a merge, never carried
  in a phase branch, so they cannot conflict.
- **M1 before M2**, despite M2 being easier: M1's T2 exercises are the only ones
  that exercise cassettes end to end, and a cassette bug found at 285 exercises is
  far more expensive than one found at 79.
- **The three highest-variance items are all early** — the YAML subset parser
  (§3 item 2), the grading shim (§4 item 2) and the first real cassette (M1). If
  this plan slips, it slips there, and it will be visible inside two weeks.
- **T3 at full scale is not on the critical path.** `DESIGN D11`'s scaled-down
  runnable is what CI gates on; full-scale GPU verification (`DESIGN §8` Q4) is a
  separate pass over M5's Phase 04 and 17 work, and the README must record which
  of the two produced any given result.
