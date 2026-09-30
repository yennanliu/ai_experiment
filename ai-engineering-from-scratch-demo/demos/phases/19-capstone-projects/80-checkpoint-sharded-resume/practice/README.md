<!-- generated:start -->
# 19-capstone-projects / 80-checkpoint-sharded-resume

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/80-checkpoint-sharded-resume/) · upstream spec
`phases/19-capstone-projects/80-checkpoint-sharded-resume/docs/en.md`

```bash
uv run demo practice run 80-checkpoint-sharded-resume --ex 1
uv run demo explain 80-checkpoint-sharded-resume --ex 1
uv run pytest demos/phases/19-capstone-projects/80-checkpoint-sharded-resume
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add async write: kick off the save in a thread and let training continue. Block the next save… | code | T1 | `ex01_an_async_save_must_snapshot_first_or_5_training_steps_land_in_the_step_100_checkpoint.py` |
| 2 | Add a `last_5_steps` rotation: keep the 5 most recent checkpoints, delete the oldest before s… | code | T1 | `ex02_rotate_before_save_needs_keep_last_4_and_5_crashed_saves_rotate_away_all_5_good_checkpoints.py` |
| 3 | Add a CRC-only fast verification path for the inner-loop reload (rotation rolls a checkpoint… | code | T1 | `ex03_crc_catches_2000_of_2000_bit_flips_but_a_4_byte_fixup_forges_it_and_a_crc32_field_crashes_the_loader.py` |
| 4 | Add a cross-world-size load: shard rebalance from N=4 to N=8 by reading the manifest, concate… | code | T1 | `ex04_n4_to_n8_and_back_rewrites_the_original_shards_but_view_shards_each_hold_the_whole_tensor.py` |
| 5 | Add an upload to a fake S3 (a second directory) and write the upload manifest. Defend the two… | code | T1 | `ex05_at_half_the_save_cadence_the_lessons_rotation_deletes_6_of_20_checkpoints_before_upload.py` |
<!-- generated:end -->
## Answers

Every exercise imports the lesson's `code/main.py` and runs its own
`save_sharded`, `load_sharded` and `rotate_checkpoints` on the lesson's 4
demo ranks (`make_demo_state`: 1,024-value param, m and v shards per rank).
Checkpoints go only to temporary directories. Crashes are injected by
wrapping the lesson's `_serialize_state` or by stopping an upload after k
objects. Nothing needs a GPU, and the only timing measured (exercise 3) is
asserted with a wide margin. Byte counts are from torch 2.14.0 and are
checked as ratios.

### 1 — an async save must snapshot first, or 5 training steps land in the step-100 checkpoint

**The saver joins the previous save thread, clones every tensor, and starts
a new thread.** Training ran 5 steps while the first save was in flight, and
the second save waited for it. At most one save was in flight. Both
checkpoints loaded byte-equal to the state at the step they were requested
(param, m and v, all 4 ranks). The first save is gated until the main thread
reaches the second save, so the overlap is forced, not left to timing:

| event log |
|---|
| save100-start, train x5, save105-waiting, save100-done, save105-start, save105-done |

**Without the snapshot the checkpoint is torn, and it still verifies.** Hand
the live dicts to the thread and the step-100 checkpoint holds step-105
parameters on 4 of 4 ranks. Each shard's `step` field reads 105 while the
manifest reads 100. `load_sharded` accepts the checkpoint. The sha256 is
computed over whatever bytes were written, and the loader never compares the
shard step with the manifest step.

### 2 — rotate-before-save needs keep_last=4, and 5 crashed saves rotate away all 5 good checkpoints

**"Keep 5, delete before saving" is `rotate_checkpoints(keep_last=4)` followed
by the save.** Over 8 saves:

| rotate before save | dirs after each save | kept |
|---|---|---|
| keep_last=4 | 1 2 3 4 5 5 5 5 | step_0003 .. step_0007 |
| keep_last=5 | 1 2 3 4 5 6 6 6 | 6 dirs, one over budget |

The lesson's own `main` rotates once, after all 8 saves.

**The rotation counts crashed saves as checkpoints.** A save that dies on the
third rank leaves two `.tmp` shards and no manifest. `rotate_checkpoints`
still counts that directory, and it orders directories by mtime. Start from 5
good checkpoints and let the next 5 saves crash. Each rotation deletes one
more good checkpoint to make room for a broken one, and at the end 0 of the 5
directories load. A rotation that first removes directories without
`manifest.json`, and then counts only complete ones, keeps steps 1-4
loadable after the same 5 crashes.

### 3 — CRC catches 2,000 of 2,000 bit flips, but a 4-byte fix-up forges it, and a crc32 field crashes the loader

**`fast_verify` rereads each shard and checks its `zlib.crc32` and length
against a `crc32.json` sidecar, with no sha256 and no unpickling.** It passes
the clean checkpoint and rejects every accidental damage tried:

| damage to rank0.bin | fast_verify |
|---|---|
| none | passes |
| 2,000 seeded single-bit flips | 2,000 rejected |
| truncated by 1 byte | rejected |
| the lesson's `+ b"corruption"` | rejected |

Over 4 x 4 MB shards, sha256 takes more than 2x as long as CRC32 (asserted),
about 13x on the authoring machine (best of 5).

**The manifest cannot carry the CRC.** Add `crc32` to each shard entry and
`load_sharded` raises `TypeError: ShardEntry.__init__() got an unexpected
keyword argument 'crc32'`, not `CheckpointError`. Bumping `schema_version`
to 2 does not help. The version check runs after `ShardEntry(**s)` has
already crashed, so the schema-version defence cannot fire for any schema
that adds a field. That is why the CRCs go in a sidecar file.

**CRC is an accident check, not a tamper check.** CRC32 is affine over GF(2).
Set one weight to 99.0, solve a 32-bit linear system for the 4 bytes of the
next weight, and the CRC is unchanged (residue 0). `fast_verify` accepts the
edited shard, `torch.load` returns 2 changed values, and only the lesson's
sha256 path rejects it. The fast path is only safe for files the job itself
just wrote. A checkpoint from anywhere else needs the full sha256.

### 4 — N=4 to N=8 and back rewrites the original shards, but view shards each hold the whole tensor

**The rebalance is exact.** The lesson's loader refuses the N=4 checkpoint at
N=8 (`world_size mismatch: manifest=4, expected=8`). `reshard` parses the
manifest, checks that the offsets tile the flat tensor, loads every shard
through `load_sharded` so each sha256 is checked, concatenates param, m and v,
and splits them with `torch.tensor_split`. The state had 3 optimiser steps
first, so m and v are not constants.

| check | result |
|---|---|
| N=8 layout (offset, numel) | (0, 512), (512, 512), ..., (3584, 512) |
| flat param / m / v equal to N=4 | yes |
| equal after one more Adam step at each size | yes, because the update is elementwise |
| N=8 -> N=4 shard sha256 equal to the original files | yes |

**Re-shard with views and every shard holds the whole model.**
`tensor_split` returns views, and `torch.save` writes a view's entire
underlying storage:

| shards | bytes on disk |
|---|---:|
| N=4, original | 56,964 |
| N=8, cloned pieces | 64,776 |
| N=8, views | 408,840 (6.3x) |

A loaded view shard has 512 values over a 4,096-value storage. `load_sharded`
accepts it: the manifest's numel is still 512, and the sha256 covers the bytes
that were written.

**The offsets are never checked.** Set rank 1's `param_shard_offset` to 0, so
two shards claim [0, 1024), and `load_sharded` still accepts the checkpoint.
The manifest has no checksum of its own. A rebalance that trusts the offsets
puts shards in the wrong place. The tiling check in `reshard` rejects this
one (`rank 1 offset 0 != 1024`).

### 5 — at half the save cadence, the lesson's rotation deletes 6 of 20 checkpoints before upload

**The fake S3 is a second directory.** `upload` sends the shards first,
`manifest.json` second, and `upload_manifest.json` last. Each object is
copied to `.tmp` and then renamed, and its sha256 is checked against the
lesson's manifest. The upload manifest records the step, the local path, the
remote path, and each object's bytes and sha256. It is the commit marker:
resume takes the newest remote directory that has one.

**Defence of the two-tier policy.** The local tier is the fast resume for the
common failure, a process restart on a healthy node. The remote tier is the
only copy that survives node loss. After `rmtree` of the whole local tier,
step 100 resumes from S3 byte-equal on all 4 ranks. An upload killed after 2
of 5 objects leaves `rank0.bin` and `rank1.bin` with no marker, so resume
falls back to step 100. The policy holds only while an upload fits in one
save interval. Over 20 saves, with 5 local checkpoints rotated before each
save:

| upload link | rotation | never uploaded | peak local dirs | newest on S3 (local: 19) |
|---|---|---|---:|---:|
| 1 upload per save | lesson's `rotate_checkpoints` | none | 5 | 19 |
| 1 upload per 2 saves | lesson's `rotate_checkpoints` | steps 4, 6, 8, 10, 12, 14 | 5 | 15 |
| 1 upload per 2 saves | skip un-uploaded (pinned) | none | 11 | 9 |

**With a slow link, the lesson's rotation silently loses archive copies.**
`rotate_checkpoints` knows nothing about uploads, so it deletes 6 of 20
checkpoints before they reach S3, and nothing reports it.

**Pinning moves the failure onto the local disk.** A rotation that keeps
checkpoints not yet uploaded loses none, but local disk peaks at 11
checkpoints against a budget of 5 and grows by one every two saves. S3 also
falls further behind (newest step 9, backlog 10), because the FIFO uploader
stays on the oldest checkpoints. No rotation rule fixes a link slower than the
save cadence. The save interval has to be at least the upload time.
