<!-- generated:start -->
# 19-capstone-projects / 43-hdf5-tokenized-corpus

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/43-hdf5-tokenized-corpus/) · upstream spec
`phases/19-capstone-projects/43-hdf5-tokenized-corpus/docs/en.md`

```bash
uv run demo practice run 43-hdf5-tokenized-corpus --ex 1
uv run demo explain 43-hdf5-tokenized-corpus --ex 1
uv run pytest demos/phases/19-capstone-projects/43-hdf5-tokenized-corpus
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a `--compression gzip` flag to the HDF5 writer and measure the throughput cost on the dem… | code | T1 | `ex01_any_hdf5_filter_crashes_the_swmr_writer_on_close_and_gzip_saves_62pct_of_bytes.py` |
| 2 | Add a deterministic seed to the sliding-window dataloader and verify two runs with the same s… | code | T1 | `ex02_the_seed_already_exists_and_matches_10_of_10_batches_but_a_restart_replays_batch_0.py` |
| 3 | Add a `--validate` mode that reads every shard, recomputes the sha256 over its tokens, and co… | code | T1 | `ex03_the_reference_validator_passes_a_shifted_global_start_that_corrupts_49pct_of_windows.py` |
| 4 | Compare the dataloader throughput at chunk sizes equal to, half of, and twice the window size… | code | T1 | `ex04_chunk_equal_to_window_touches_2_chunks_per_read_and_the_corpus_never_leaves_hdf5_cache.py` |
| 5 | Add a `--max-document-tokens` flag that truncates very long documents at write time. Defend t… | code | T1 | `ex05_a_512_token_cap_keeps_23_8pct_of_the_corpus_and_read_time_would_reject_76pct_of_windows.py` |
<!-- generated:end -->

## Answers

Every exercise runs the lesson's `code/main.py` on its shipped demo corpus:
two shards of 6,456 byte-level tokens each (12,912 in all), written in
512-token chunks. Each shard holds three copies of one 2,151-token
document. External facts were read on 2026-09-29 from the h5py docs
(docs.h5py.org/en/stable/swmr.html, docs.h5py.org/en/stable/high/file.html)
and the HDF5 SWMR note
(support.hdfgroup.org/documentation/hdf5/latest/_s_w_m_r_t_n.html). Every file
is T1 because the lesson imports h5py at module scope.

### 1 — any HDF5 filter crashes the reference writer on close; gzip saves 62% of bytes

**Default `none`. gzip saves 62% of the bytes for about 1.1x the write time
and about 1.0x the read time, but only with SWMR switched off.**

| compression | stored bytes | write (best of 7) | 250-batch read |
|---|---:|---:|---:|
| none | 26,624 | ~2.0 ms | 1.00x |
| gzip | 10,130 | ~1.1x | ~1.0x |
| gzip + shuffle | 7,670 | ~1.1x | ~1.0x |

The timings are printed but asserted only loosely (under 3x), because the
whole corpus writes in about 2 ms. The reason for keeping `none` as the
default is the next finding.

**The reference writer cannot finalize a filtered shard.** It enables
`swmr_mode` in `__enter__` and only creates its three attributes in
`__exit__`. The HDF5 SWMR note says the writer "cannot add new objects to the
file". Unfiltered, the attributes happen to fit the dataset's object header.
A filter message uses up that room, so lzf, shuffle, fletcher32 and gzip
each raise `RuntimeError` on close. The shard is then left locked, and a
second failed close in the same process segfaults inside HDF5, so the
solution probes each filter in a child process. The flag therefore writes
without SWMR.

Half of the stored bytes carry no information. The tokens are bytes + 1,
stored as uint16, and the high byte is 0 for 100% of the ASCII demo tokens.
That is why shuffle does better than gzip alone.

### 2 — the seed already exists, and two seed-7 runs match on 10 of 10 batches

**Nothing needs adding.** `SlidingWindowDataloader(seed=0)` already draws its
starts from a private `random.Random`. Two seed-7 loaders give identical
inputs and targets on 10 of 10 batches. Their checksums equal all 10 that
`run_demo()` prints (e2747ebe, 14bc5794, fe65c4c3, ...). The batches are the
same again when the shards are rechunked to 64. Seed 8 matches 0 of 10.

**Reproducible is not resumable.** A restart after batch 5 that rebuilds the
loader with the same seed replays batch 0, and matches 0 of the next 5
batches. Replaying the 20 start draws, or restoring `_random.getstate()`
from the checkpoint, matches 5 of 5.

**One seed across workers duplicates everything.** A second seed-7 worker
repeats 40 of 40 windows. Seeds have to be offset per worker.

### 3 — the reference validator passes a shifted global_start that corrupts 49% of windows

**`--validate SHARDS_JSON` exits 1 on each of three edits and 0 on the
clean and a moved corpus.**

| edit | `--validate` | reference `validate_corpus` | what the store then does |
|---|---|---|---|
| none | 0 | pass | - |
| one token flipped in shard-0001 | 1 | `['shard-0001']` | - |
| index `token_count` 6456 -> 6455 | 1 | pass | raises on 65 of 12,847 windows |
| index `global_start` 6456 -> 6400 | 1 | pass | wrong tokens on 6,336 of 12,848 windows, raises on 176 |
| corpus directory moved | 0 | `FileNotFoundError` | - |

The reference validator checks each file against its own `token_count`
attribute. It never reads the index's count or offsets, so it passes the edit
that silently corrupts half the training windows. `shards.json` stores
absolute paths, so moving the corpus crashes the reference validator. The
mode instead finds each shard next to the index. The two demo shards have
the same sha256 (5c811f78961b...), and no per-shard hash check can see that.

### 4 — chunk equal to the window touches 2 chunks per read; the corpus never leaves HDF5's cache

**Throughput barely moves (32 < 64 < 128, within 2x), and the page-cache
effect cannot show at this size.** The loader reads `window + 1` = 65
tokens from a uniformly random start:

| chunk | chunks per read | bytes per read | tok/s (cache on) | cache off |
|---:|---:|---:|---:|---:|
| 32 (half) | 3.0 | 192 | ~20M | ~1.4x slower |
| 64 (equal) | 2.0, never fewer | 256 | ~23M | ~1.3x slower |
| 128 (twice) | 1.51 | 386 | ~24M | ~1.3x slower |

The whole corpus is 25,824 bytes, and HDF5 2.0's default chunk cache is 8 MiB
per dataset. So every chunk is served from HDF5's own cache after the first
touch. Only turning that cache off (`rdcc_nbytes=0`) costs anything.

**The lesson's advice points at its own warning.** A chunk equal to the
window means every sample touches two chunks, because a read is one token
longer than the window and starts anywhere. Half the window gives about 0.9x
the throughput, not the "halve" the lesson predicts. And `MmapTokenStore`
maps nothing: a chunked dataset's `get_offset()` is None at every chunk size.

### 5 — a 512-token cap keeps 23.8% of the demo corpus; at read time it would reject 76% of windows

**Truncate at write time, and treat the cap as part of the corpus version.**
`--max-document-tokens 512` shrinks the shards from 12,912 to 3,078 tokens,
with all 6 documents kept. The read-time alternative rejects 76.4% of 10,000
uniformly drawn windows, which is 4.2 reads per accepted sample. It also
needs a scan for boundary tokens, because the store keeps no document
offsets. Write-time truncation is irreversible: the shard sha256 changes
(5c811f78961b -> d3102859baae), so a new cap means a new corpus.

Two things to watch. The cap counts bytes, so a Chinese document capped at 512
keeps 170 whole characters and then decodes to U+FFFD. And the lesson
already ships `pack_documents`, which splits long documents instead of
dropping their tails: it keeps all 12,911 tokens in 26 groups. The pipeline
never calls it.
