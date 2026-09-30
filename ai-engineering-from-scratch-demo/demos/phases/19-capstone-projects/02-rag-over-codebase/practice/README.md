<!-- generated:start -->
# 19-capstone-projects / 02-rag-over-codebase

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/02-rag-over-codebase/) · upstream spec
`phases/19-capstone-projects/02-rag-over-codebase/docs/en.md`

```bash
uv run demo practice run 02-rag-over-codebase --ex 1
uv run demo explain 02-rag-over-codebase --ex 1
uv run pytest demos/phases/19-capstone-projects/02-rag-over-codebase
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Swap Voyage-code-3 for nomic-embed-code self-hosted. Measure the MRR@10 delta. Report whether… | code | T0 | `ex01_a_4x_weaker_embedder_loses_0_138_dense_mrr_and_the_reranker_closes_the_gap_by_ignoring_retrieval.py` |
| 2 | Inject 20% generated code (LLM-produced boilerplate) into the corpus and re-evaluate. Observe… | code | T0 | `ex02_generated_wrappers_at_20pct_steal_the_top_hit_in_984_of_1820_placements_and_halving_their_fused_score_stops_all.py` |
| 3 | Benchmark Qdrant hybrid search vs pgvector + pgvectorscale at your corpus size. Report p99 at… | code | T0 | `ex03_both_backend_shapes_are_brute_force_scans_and_after_the_reranker_they_score_the_same_0_917.py` |
| 4 | Add a sampling-based drift check: weekly, rerun the 100-question eval. Alert on MRR@10 drop >… | code | T0 | `ex04_unchanged_code_scores_0_808_to_0_862_mrr_across_26_fresh_processes_because_fake_embed_hashes_with_a_salt.py` |
| 5 | Extend to cross-language symbol resolution: a Python function that calls a Go service over gR… | code | T0 | `ex05_the_proto_contract_links_a_python_grpc_call_to_its_go_handler_which_hybrid_retrieval_ranks_outside_the_top_5.py` |
<!-- generated:end -->

## Answers

Every exercise runs the lesson's `code/main.py`: a hashed bag-of-words
`fake_embed` standing in for the dense model, a field-weighted `BM25Index`,
`rrf` with k=60, and a `rerank` stub that adds 0.9 per query word shared with
the symbol and 0.1 per word shared with the summary. `fake_embed` calls
Python's builtin `hash`, which is salted per process. So every exercise except
ex04 swaps in a keyed blake2b to make its numbers reproducible. ex04 measures
the salt itself. External facts (Qdrant hybrid-queries docs, pgvectorscale
README) were read on 2026-09-29.

### 1 — a 4x weaker embedder loses 0.138 dense MRR@10, and the reranker closes the gap by ignoring retrieval

**The MRR@10 delta is 0.138 dense-only, 0.017 after fusion, and -0.009 with
reranking, so yes, the gap closes.** Neither model can run offline, so the
swap goes into the lesson's dense slot: `fake_embed` at 64 dimensions
("hosted") against 16 dimensions ("self-hosted"). Each is averaged over 20
hash keys, on 12 labelled questions over the 6-chunk corpus.

| stage | 64-dim | 16-dim | gap |
|---|---:|---:|---:|
| dense only | 0.785 | 0.647 | 0.138 |
| dense + BM25 (RRF) | 0.902 | 0.885 | 0.017 |
| + rerank | 0.931 | 0.940 | -0.009 |

The gap closes because the reranker overrides retrieval. Fused scores span
only 0.018, while one shared summary word is worth 0.1. A 4-dimension
embedder (dense 0.558) reranks to 0.960, and reranking the whole corpus with
no retrieval at all scores 0.958. Retrieval filters nothing here either:
dense search asks for k=10 on a 6-chunk corpus, so all 6 chunks reach the
reranker on 12/12 questions.

### 2 — 20% generated wrappers steal the top hit in 984 of 1,820 placements, and halving their fused score stops every steal

**Poisoning shows up as stolen top hits, and a `generated` flag removes all
of them.** The corpus has 16 real chunks plus 4 generated `<symbol>Wrapper`
chunks, so 20% is generated. Each wrapper restates its real chunk's summary.
All C(16,4) = 1,820 placements were run.

| | steals (top-1 taken by a wrapper) | placements with a steal | mean MRR@10 |
|---|---:|---:|---:|
| clean corpus | — | — | 0.861 |
| poisoned | 1,210 | 984 (54%) | 0.852 (worst 0.757) |
| flag, 0.5x fused score before rerank | 0 | 0 | 0.880 |
| flag, 0x final score after rerank | 0 | 0 | 0.880 |

Every steal is a rerank tie broken by the RRF prior, which is why halving the
prior is enough. The mean hides this. With the wrappers present but zeroed,
one question climbs from rank 2 to rank 1 in 809 placements, because the
extra documents shift BM25 and the dense top-10. That side effect (+0.019) is
twice the size of the poisoning drop (-0.009), so on this eval the steal count
is the signal, not MRR.

### 3 — both backend shapes are brute-force scans, and after the reranker they score the same 0.917

**At batch size 1, p99 is under 1 ms on the 6-chunk corpus for both shapes
(about 0.04 ms), and about 8 ms at 3,000 chunks.** No server can run in the
offline gate, so the two backends' query shapes run on the lesson's engine:

- **Qdrant hybrid:** dense + sparse fused with RRF at Qdrant's default k = 2
  over zero-based ranks.
- **pgvector + pgvectorscale:** dense only, since pgvectorscale ships
  StreamingDiskANN and no BM25.

Wall-clock varies by machine, so the check asserts properties. The
deterministic part is the work per query: N cosines, plus N BM25 term-table
scans per matched query term. Neither shape has an ANN index, and ANN is what
actually separates Qdrant (HNSW) from pgvectorscale (DiskANN).

| shape | MRR@10 fused | MRR@10 after rerank |
|---|---:|---:|
| lesson hybrid, k=60 | 0.875 | 0.917 |
| Qdrant hybrid, k=2 | 0.875 | 0.917 |
| pgvector, dense only | 0.771 | 0.917 |

The index build is quadratic. `BM25Index.add` re-sums every stored length to
update `avgdl`, so N chunks cost N(N+1)/2 sums: 4,501,500 at 3,000 chunks.
At the lesson's 25.7 lines per chunk, a 2M-LOC fleet is 77,922 chunks and
3.04e9 sums, and it pays that again on every re-index.

### 4 — unchanged code scores 0.808 to 0.862 MRR@10 across 26 fresh processes, because `fake_embed` hashes with a per-process salt

**`drift_alert` (relative drop > 5% week over week) fires on a real
regression and stays quiet on unchanged weeks.** The eval has 100 questions,
sampled with a seed from the 6-chunk corpus. Losing every chunk summary takes
MRR@10 from 0.838 to 0.707 (-15.6%), and the alert fires. Each week is a
fresh process (PYTHONHASHSEED = week). Over 26 such weeks with no code
change, 0 of 25 week-over-week steps alert.

But the "unchanged" system is a different embedder every week. The builtin
`hash` inside `fake_embed` is salted per process, so the same code scores
anywhere from 0.808 to 0.862, a spread of 6.2% of the best week. That is
larger than the alert threshold. A baseline frozen on a lucky week alerts on
4 of the 325 (baseline week, later week) pairs with nothing changed. The
TypeScript port hashes with FNV-1a and is stable. Sampling noise also crosses
the line: 1,000 bootstrap resamples of one week's 100 answers alert 38 times
under the relative reading and 18 times under the absolute (0.05-point) one.

### 5 — the proto contract links a Python gRPC call to its Go handler, which hybrid retrieval ranks outside the top 5

**The symbol graph links `cancel_stale_uploads` -> `uploaderServer.Abort`
-> `AbortMultipartOnFail`.** That is Python, then gRPC, then Go, then Go. The
lesson's code has no symbol graph, so one is built with regexes standing in
for tree-sitter queries. It resolves a call in three steps:

1. The Python `UploaderStub(...).Abort(...)` gives the service and method.
2. The `.proto` confirms that `Uploader.Abort` exists.
3. `pb.RegisterUploaderServer(srv, &uploaderServer{})` maps the Go receiver
   to its service.

Two hops from the top hit recover both server-side chunks.

Retrieval alone does not get there. On "what runs on the server when
cancel_stale_uploads aborts an upload", `answer` ranks the Python caller
first. Its top 5 hold only `AbortMultipartOnFail`, matched on the word
"abort", and not the Go handler that actually runs. Linking by method name
alone is wrong half the time: a look-alike `Exporter.Abort` rpc gives 2 edges,
one to the export job. Qualifying by service gives 1 edge, the right one.
