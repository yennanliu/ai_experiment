<!-- generated:start -->
# 19-capstone-projects / 42-large-corpus-downloader

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/42-large-corpus-downloader/) · upstream spec
`phases/19-capstone-projects/42-large-corpus-downloader/docs/en.md`

```bash
uv run demo practice run 42-large-corpus-downloader --ex 1
uv run demo explain 42-large-corpus-downloader --ex 1
uv run pytest demos/phases/19-capstone-projects/42-large-corpus-downloader
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a `--shingle-width` flag and measure how the dedup verdict changes at widths 3, 5, 9. Def… | code | T1 | `ex01_width_5_makes_20_dedup_errors_to_60_at_width_3_and_the_lsh_threshold_is_0_42_not_0_8.py` |
| 2 | Add gzip support next to zstd by sniffing the magic bytes. The downloader should not require… | code | T1 | `ex02_the_downloader_pulls_a_whole_gzip_shard_before_failing_and_a_4_byte_sniff_handles_all_6_encodings.py` |
| 3 | Add a `--resume-only` mode that refuses to start a fresh download if no checkpoint is found.… | code | T1 | `ex03_resume_only_refuses_the_4_of_6_cache_states_where_the_downloader_restarts_from_byte_zero.py` |
| 4 | Move the LSH index to a shelf or sqlite file and measure throughput vs the in-memory variant. | code | T1 | `ex04_an_sqlite_lsh_index_doubles_index_time_but_adds_3pct_end_to_end_because_minhash_dominates.py` |
| 5 | Add a manifest sha256 check on startup. The downloader should fail closed if the manifest on… | code | T1 | `ex05_forging_manifest_and_lock_passes_the_check_and_a_rerun_one_second_later_changes_the_hash.py` |
<!-- generated:end -->

## Answers

Every exercise runs the lesson's `code/main.py` as a library: its
`StreamingDownloader`, `ZstdDocIterator`, `MinHasher`, `LSHIndex`, `Dedup` and
`ManifestWriter`, unchanged. Downloads are served from local files or an
in-process HTTP server; nothing touches the network. All five are tier T1
under `deps_group: agents`, because the lesson imports `zstandard` at module
scope, and in this repo `zstandard` is installed only as a transitive
dependency of the `agents` extra. Magic-byte facts were read on 2026-09-29
from RFC 8878 (https://datatracker.ietf.org/doc/html/rfc8878) and RFC 1952
(https://datatracker.ietf.org/doc/html/rfc1952).

### 1 — width 5 makes 20 dedup errors, against 60 at width 3 and 44 at width 9

**Default to width 5, the module's `DEFAULT_SHINGLE_WIDTH`.** The flag
builds the reference `MinHasher`, and the full pipeline runs over a labelled
fixture of 480 documents. There are 300 originals, 120 near-duplicates, and
60 distinct form-like records that share one template.

| width | footer | header | 3 edits | tracking param | templated flagged | errors |
|---|---:|---:|---:|---:|---:|---:|
| 3 | 30/30 | 30/30 | 30/30 | 27/30 | 57/60 | 60 |
| 5 | 30/30 | 30/30 | 28/30 | 13/30 | 1/60 | 20 |
| 9 | 30/30 | 30/30 | 16/30 | 0/30 | 0/60 | 44 |

Short shingles fit between a template's slots, so width 3 calls different
records duplicates. Long shingles are each broken by any single edit, so
width 9 misses edits and tracking parameters. Tracking parameters need URL
normalisation, not a different width.

**The lesson's own demo runs width 3, and that changes its verdicts.** It
flags 7 of 15 documents at width 3 and 5 of 15 at widths 5 and 9. The two
extra flags are paraphrase pairs at exact Jaccard 0.364 and 0.5.

**`k = 128, b = 32, r = 4` is a 0.42 threshold, not the 0.8 the doc
states.** The S-curve midpoint is (1/32)^(1/4) = 0.420, which the
`LSHIndex` docstring itself says. A pair at s = 0.5 collides with
probability 0.873. `Dedup` drops a document on its first band hit and never
checks `jaccard_estimate`. At width 5, 71 of the 102 flagged documents sit
below Jaccard 0.8 with their keeper; the lowest is 0.182.

### 2 — the downloader pulls a whole gzip shard before failing, and a 4-byte sniff handles all six encodings

**Peek at the first 4 bytes.** `28 b5 2f fd`, or a skippable frame
`5? 2a 4d 18`, means zstd; `1f 8b` means gzip; anything else is refused.
Both `download()` and `ZstdDocIterator` go through the module-level name
`zstd`, so the sniffing decompressor goes in there. `ShardPlan` does not
change.

| encoding | shipped | with sniffing |
|---|---|---|
| zstd / 2 frames / skippable frame first | 200 docs | 200 docs |
| gzip / 2-member gzip | `ZstdError` | 200 docs |
| plain JSONL | `ZstdError` | refused, `magic bytes 7b226964` |

**The shipped downloader looks at the codec only after the whole shard is on
disk.** The gzip shard is complete, 1,364 of 1,364 bytes, when
`zstd decompress error: Unknown frame descriptor` is raised, and the message
never mentions gzip. The sniff inherits that order, so a bad shard is still
refused only after all its bytes have landed.

### 3 — resume-only refuses the 4 of 6 cache states in which the downloader restarts from byte zero

**Demand the reference's own verified checkpoint, and refuse any request
without `Range` and any reply that is not 206.** Only checking for a
checkpoint file is not enough, because the reference also starts from zero
when a checkpoint fails verification or the server ignores `Range`.

| cache state | shipped downloader | `--resume-only` |
|---|---|---|
| empty | full GET | refused, 0 requests |
| interrupted at 12,288 bytes | `bytes=12288-`, rest only | same, sha256 matches |
| partial with one byte flipped | full GET, no warning | refused, partial kept |
| server ignores `Range` | deletes the partial, second full GET | refused, partial kept |
| shard already complete | `HTTP Error 416` | read as complete |
| `file://` URL | re-reads all bytes | refused, partial kept |

**Rerunning the shipped downloader on a finished shard crashes.** The
checkpoint is never removed, so the rerun asks for `bytes=<size>-` and gets
416. **The demo's `file://` transport cannot resume at all:** the response
has no status, so the 206 check always fails.

### 4 — an sqlite LSH index doubles index time but adds about 3% end to end

**The verdicts are identical to the in-memory index: 1,400 kept and 600
dropped out of 2,000, with the same keepers.** One run:

| variant | index s | end to end s | vs memory |
|---|---:|---:|---:|
| in-memory `LSHIndex` | 0.065 | 2.73 | 1.00x |
| sqlite, one commit | 0.137 | 2.80 | 1.03x |
| sqlite, commit per document | 1.74 | 4.40 | 1.61x |

The 2,000 MinHash signatures take 2.66 s, about 40x the in-memory index.
The index is not the bottleneck. The checks assert only wide ratios: under
1.5x end to end, over 5x for commit-per-document, and over 10x for MinHash.

**The doc's "one extra disk read per document" is 32 for every new
document.** `query` has to try all 32 bands before it can call a document
new. That comes to 45,552 lookups: 32 for each of the 1,400 keepers, plus 752
for the 600 duplicates. **About half the in-memory heap, 46% of 11.4 KB per
kept document, is `_signatures`,** a store that only `jaccard_estimate`
reads, and the pipeline never calls it.

### 5 — forging manifest and lock together passes the check, and a rerun one second later changes the hash

**`startup_check()` passes the intact manifest and refuses an edited
verdict, a reformatted file, a missing lock, a missing manifest and an
unreadable lock.** A cache with neither file counts as a fresh start.

**The lock file is `manifest.json.lock`, not the `manifest.lock` the
exercise names.** `write` builds the name with `with_suffix(".json.lock")`.

**The lock is a checksum, not a seal.** An edit with a recomputed lock
passes, which is exactly the one-file attacker the doc says the hash stops.
A corrupted cached shard also passes until each shard is re-hashed against
its manifest row.

**The hash reproduces only within the same second.** The manifest stores
`generated_at = int(time.time())` and every source URL. The same data one
second later, or served from another directory, gets a new hash, so a hash
"pinned from a commit" moves on every rerun.
