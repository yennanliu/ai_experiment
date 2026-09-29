<!-- generated:start -->
# 19-capstone-projects / 47-checkpoint-save-resume

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/47-checkpoint-save-resume/) · upstream spec
`phases/19-capstone-projects/47-checkpoint-save-resume/docs/en.md`

```bash
uv run demo practice run 47-checkpoint-save-resume --ex 1
uv run demo explain 47-checkpoint-save-resume --ex 1
uv run pytest demos/phases/19-capstone-projects/47-checkpoint-save-resume
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Replace round-robin sharding with sharding by parameter group (layers ending in `.weight` vs… | code | T1 | `ex01_by_group_one_shard_holds_95pct_of_the_parameters_and_meta_pt_still_outweighs_all_shards_3_to_1.py` |
| 2 | Extend the save loop to keep the last K checkpoints and prune older ones. What is the right K… | code | T1 | `ex02_keeping_k_needs_k_plus_1_on_disk_and_a_kill_mid_save_leaves_a_tmp_no_pruner_deletes.py` |
| 3 | Add a `--ckpt-every-seconds` flag that triggers a save on a wallclock interval, not just step… | code | T1 | `ex03_a_20s_trigger_cuts_worst_case_lost_work_from_124s_to_32s_and_a_scrambled_rng_changes_nothing.py` |
| 4 | Add a checksum verification path that runs at startup, scans every checkpoint in the director… | code | T1 | `ex04_the_scan_flags_4_of_4_corrupt_checkpoints_and_the_lessons_loaders_miss_a_flipped_weight.py` |
| 5 | Implement a `migrate_v1_to_v2` function that adds a new field to the payload and bumps the sc… | code | T1 | `ex05_the_lessons_loader_accepts_any_ckpt_schema_and_a_renamed_field_fails_after_overwriting_the_weights.py` |
<!-- generated:end -->

## Answers

Every exercise runs the lesson's `code/main.py` on CPU: a 16 -> 24 -> 24 -> 4
MLP trained with AdamW and a cosine schedule for 24 steps on seed 11, with
`save_checkpoint` / `load_checkpoint`, the sharded pair, and
`run_resume_demo`. Checkpoints go only to temporary directories. Crashes
are real: child processes killed with `os._exit` between the temp write and
the rename, or a save that raises in the middle of a sharded write. Byte
sizes are from torch 2.14.0 and are checked as ratios, not exact values.

### 1 — by group, one shard holds 95% of the parameters, and meta.pt still outweighs all shards 3 to 1

**Both layouts resume exactly (max loss diff 0.0), so the choice depends on
who reads the files.** By group, the three `.weight` tensors (1,056
parameters) sit in one shard and the three `.bias` tensors (52) in the
other. A reader that wants one group, such as bias-only fine-tuning, a
weights-only quantiser, or per-group weight decay, opens one file. Under
round robin both groups are spread over all 3 files. Round robin suits
parallel reads of the whole model, but it deals out keys, not bytes:

| layout | parameters per shard | largest / smallest | files per group |
|---|---|---:|---:|
| round robin, 3 shards | 600 / 388 / 120 | 5.0x | 3 |
| by group, 2 shards | 1,056 / 52 | 20.3x | 1 |

**`meta.pt` is not "small".** It holds both AdamW moments and the RNG state,
and it is 3.1x all the model shards together under round robin (33,733 vs
10,895 bytes) and 3.6x by group. The RNG state alone is about 19 KB, 2.7x the
model's 7 KB, because the torch CPU state is saved as a Python list of 5,056
ints.

### 2 — keeping K needs K+1 on disk, and a kill mid-save leaves a .tmp no pruner deletes

**K = (disk / checkpoint size) - 1, and K = 1 when the disk holds only two.**
The new file has to land before the oldest can go, so the peak is K+1:

| K | peak files | peak bytes |
|---:|---:|---:|
| 1 | 2 | 79,422 |
| 2 | 3 | 119,101 |
| 3 | 4 | 158,716 |

K = 1 is already crash-safe because the save is write-then-rename. After
three kills mid-save, the one kept checkpoint still resumes with a max loss
diff of 0.0. It cannot protect against a kept file that is silently bad
(exercise 4), so use K = 2 once three checkpoints fit.

**Every kill leaves a full-size orphan.** The lesson's `finally: unlink`
never runs on a kill. Three kills leave 118,845 bytes of
`ckpt-0008.pt.<random>.tmp`, three checkpoints' worth, and a `ckpt-*.pt`
pruner never sees them. A small-disk loop has to sweep `*.tmp` at startup.
The code also never calls `fsync`, although its reading list cites fsync
"for the durability guarantee behind atomic rename".

### 3 — a 20 s trigger cuts worst-case lost work from 124 s to 32 s, and a scrambled RNG changes nothing

**The flag is `--ckpt-every-seconds`, next to `--ckpt-every-steps`, and a
save fires when either interval elapses.** The lesson's own `parse_args`
rejects the flag with exit code 2. The clock is injected so the run is
deterministic: 1 s steps, except steps 11-14, which stall for 30 s each.

| trigger | saves at step | worst-case work lost to a kill |
|---|---|---:|
| every 8 steps | 8, 16, 24 | 124 s |
| every 20 s | 11, 12, 13, 14 | 40 s |
| both | 8, 11, 12, 13, 14, 22 | 32 s |

The clock is only checked between steps, so its bound is the interval plus
one step. Every checkpoint written by the combined trigger, including the
mid-epoch ones, resumes with a max diff of 0.0.

**The lesson's resume test cannot see the RNG bucket.** With
`restore_rng_state` replaced by `torch.manual_seed(999)`,
`run_resume_demo` still reports 0.0 at all 23 interrupt points. Data comes
from a per-epoch generator reseeded to `12345 + epoch`, and there is no
dropout, so nothing reads the global RNG. The doc's "without the RNG state
the resumed loss curve is a different curve" is not tested by this code.

### 4 — the scan flags 4 of 4 corrupt checkpoints, and the lesson's loaders miss a flipped weight

**`scan()` checks each single file against a sha256 sidecar written right
after the save, and each sharded directory against its `index.json`. It
flags exactly the 4 damaged checkpoints out of 7.**

| checkpoint | damage | scan | lesson's loader |
|---|---|---|---|
| ckpt-0004.pt, ckpt-0008.pt | none | ok | loads |
| ckpt-0012.pt | truncated | mismatch | OSError |
| ckpt-0016.pt | 1 weight byte flipped | mismatch | **loads, step 16** |
| sharded-0020 | none | ok | loads |
| sharded-0024 | 1 shard byte flipped | mismatch (shard-001) | AssertionError |
| sharded-torn | crash mid re-save | mismatch (shard-000) | AssertionError |

**The single-file format has no checksum, and `torch.load` does not check the
zip CRC,** so a flipped weight loads cleanly. **The sharded hash checks are
`assert` statements:** under `python -O` the flipped `sharded-0024` loads
(exit 0, step 24), where the doc promises it "fails loudly". **Re-saving
shards over the same directory is not atomic as a whole:** the crash leaves
a new shard 000 beside the old meta, and the good step-20 checkpoint that
was there is gone.

### 5 — the lesson's loader accepts any "ckpt" schema, and a renamed field fails after overwriting the weights

**`migrate_v1_to_v2` adds `loader` (`seed_base` 12345, `batches_per_epoch`),
the data-order contract a resume depends on, and bumps the schema to
`ckpt.v2`.** `load_any` dispatches through a migration table, rejects
unknown schemas, upgrades a v1 file atomically in place, and restores
through the lesson's `load_checkpoint`. The v1 file and the v2 file it
becomes both resume with a max diff of 0.0. Loading with 6 batches per
epoch instead of 5 raises ValueError under v2. The v1 loader accepts the
same mistake silently: it matches for 3 steps and then diverges from step
15 by up to 0.4375 in loss.

**No dispatch exists in the lesson.** The doc says "the loader dispatches",
but `load_checkpoint` only asserts `schema.startswith("ckpt")`. It loads
`ckpt.v9`, `ckptX` and `ckpt-shard.v1`; `load_any` rejects all three. A
`ckpt.v3` that renames `state` raises KeyError after the model's weights
have already been overwritten. **The file also fails torch's default
loader:** `weights_only=True` (the default since torch 2.6) raises
UnpicklingError on the NumPy and Python RNG state. That is why the lesson
passes `weights_only=False`, which runs arbitrary pickle.
