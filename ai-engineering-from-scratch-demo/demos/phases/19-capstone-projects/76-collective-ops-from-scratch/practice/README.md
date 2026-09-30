<!-- generated:start -->
# 19-capstone-projects / 76-collective-ops-from-scratch

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/76-collective-ops-from-scratch/) · upstream spec
`phases/19-capstone-projects/76-collective-ops-from-scratch/docs/en.md`

```bash
uv run demo practice run 76-collective-ops-from-scratch --ex 1
uv run demo explain 76-collective-ops-from-scratch --ex 1
uv run pytest demos/phases/19-capstone-projects/76-collective-ops-from-scratch
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a tree allreduce variant and switch between ring and tree by message size. Measure the cr… | code | T1 | `ex01_the_tree_wins_2x_below_16kb_on_the_queue_mesh_and_moves_exactly_the_rings_bytes.py` |
| 2 | Add a `recv_timeout_ms` so a stalled rank surfaces a deadline error instead of hanging forever. | code | T1 | `ex02_a_300ms_deadline_catches_a_dead_rank_but_broadcast_from_any_src_but_0_spins_forever.py` |
| 3 | Replace `multiprocessing.Queue` with TCP sockets for the four primitives. Same tests, real wire. | code | T1 | `ex03_the_tcp_mesh_passes_the_lessons_tests_bit_for_bit_and_a_plain_sendall_deadlocks_at_64mb.py` |
| 4 | Add a bandwidth instrumentation hook so the per-rank byte counter logs to JSONL. | code | T1 | `ex04_the_jsonl_log_shows_broadcast_sends_2t_t_0_0_not_t_and_padding_breaks_2t_n_1_over_n.py` |
| 5 | Compare wall-clock time of ring versus tree on 4 ranks for tensors of size 1KB, 1MB, 16MB. De… | code | T1 | `ex05_over_tcp_the_ring_overtakes_the_tree_at_about_1mb_but_on_the_queue_mesh_the_tree_always_wins.py` |
<!-- generated:end -->

## Answers

Every exercise imports the lesson's `code/main.py` and runs its own `Mesh`,
`ring_allreduce`, `broadcast`, `allgather` and `reduce_scatter` on 2-4
fork-context ranks on this machine. Each run has a timeout, and any rank
still alive at the end is killed. Nothing needs a GPU. The gloo baselines
use torch 2.14.0 on loopback. Timings depend on machine load. They are
reported in the check details, and only ratios with at least a 1.1x margin,
in the direction seen on every run, are asserted.

### 1 — the tree wins 2x below 16 KB on the queue mesh and moves exactly the ring's bytes

**The tree allreduce is correct, the switch dispatches by size, and the tree
wins where latency dominates.** The tree is a binomial reduce onto rank 0
followed by the lesson's own `broadcast`. At world sizes 2, 3 and 4, for 64
and 66 floats, it lands within 3.6e-7 of a float64 sum, as the ring does.
`allreduce(mesh, t, threshold_bytes)` moves the tree's bytes below the
threshold and the ring's at or above it. On 4 ranks, median of 7 calls:

| message | ring/tree time |
|---|---:|
| 1 KB | 1.8-2.1 |
| 16 KB | 1.7-2.2 |
| 256 KB | 1.1-1.9 |
| 4 MB | 1.1-1.3 |
| 16 MB | 1.0-1.3 (0.78-1.4 over more runs) |

The ranges are over four runs on this machine.

The advantage fades between 256 KB and 4 MB, so the crossover is a band, not
a point. A 1 MB threshold sits inside it.

**The tree sends as many bytes as the ring. It just puts them on fewer
ranks.** At T = 4096 B the ring sends 6144 B from every rank in 6 messages.
The tree sends 8192/8192/4096/4096 B in 2/2/1/1 messages. Both total 24,576
B, so the doc's "T log2(N)" for the tree describes its busiest rank, not
every rank.

**The queue mesh has no bandwidth term.** A queued torch tensor pickles to a
shared-memory handle of 359-362 bytes at both 1 KB and 16 MB. So the ring's
bandwidth advantage never shows on this transport.

### 2 — a 300 ms deadline catches a dead rank, but broadcast from any src except 0 spins forever

**`with_deadline(mesh, recv_timeout_ms)` makes every ring primitive fail fast
and say where.** In the test, rank 3 of 4 never starts. In allreduce,
allgather and reduce_scatter, all 3 live ranks raise `DeadlineError` after
0.3 s. Broadcast from rank 0 completes, because the dead rank is a leaf.

**Only rank 0's error names the dead rank.** The stall moves one hop per
ring step, so the other two errors blame a healthy neighbour:

| rank | error names | recv # |
|---|---|---:|
| 0 | rank 3 | 1 |
| 1 | rank 0 | 2 |
| 2 | rank 1 | 3 |

To find the culprit, take the lowest recv number, not the first error to
arrive.

**The lesson's own timeout is anonymous and slow.** `Mesh.recv` waits
`RECV_TIMEOUT_S = 30.0` and then raises a bare `queue.Empty` with an empty
message. `run_mesh` waits `get(timeout=60)` for results and
`join(timeout=30)` per process.

**The lesson's `broadcast` never ends for `src` other than 0.** It only
reaches ranks `h + 2^k` above the holders. With a no-op mesh, `src=0`
returns, and `src` = 1, 2 and 3 are all still running after 3 s. There is
no recv in that loop, so no deadline can catch it. Every lesson test uses
`src=0`.

### 3 — the TCP mesh passes the lesson's tests bit for bit, and a plain sendall deadlocks at 64 MB

**The lesson's four primitives run unchanged over loopback TCP and pass every
lesson test.** Each frame is an 8-byte length plus the raw float32 bytes.
With the inputs of `tests/test_collectives.py`, seed for seed, the output is
bit-identical to the lesson's `run_mesh` on all four 4-rank cases. It is
within the tests' tolerances of gloo run in the same ranks, and the 2-rank
sum of ones and twos is 3.0.

**A blocking `sendall` deadlocks the ring.** Each ring step sends to the
next rank, then receives from the previous one.
`multiprocessing.Queue.put` never blocks, because a feeder thread does the
write. A socket's `sendall` blocks once the kernel buffers are full:

| transport | 128 B allreduce | 64 MB allreduce |
|---|---|---|
| plain `sendall` | ok on all 4 ranks | `TimeoutError` on all 4 ranks |
| one feeder thread per peer | ok | ok on all 4 ranks |

**The doc's "more than float32 epsilon, the test fails" would fail the
lesson's own allreduce.** Ring and gloo add in different orders:

| op | max \|tcp - gloo\| |
|---|---:|
| allreduce | 4.8e-7 (4 eps) |
| broadcast | 0 |
| allgather | 0 |
| reduce_scatter | 2.4e-7 (2 eps) |

The tests pass because they use atol 1e-5, about 84 epsilons.

### 4 — the JSONL log shows broadcast sending 2T/T/0/0, not T, and padding breaks 2T(N-1)/N

**`jsonl_hook(mesh, path, op)` writes one line per send, and the per-rank
sums equal the lesson's byte counter.** Five runs write 75 lines, each with
the keys `dst`, `nbytes`, `op`, `rank`, `seq` and `t_ns`.

**The doc's per-rank byte table holds for 2 of the 4 primitives.** Measured
on 4 ranks:

| op | T | bytes per rank | table says |
|---|---:|---|---|
| allreduce | 256 | 384 each | 2T(N-1)/N = 384 |
| reduce_scatter | 1024 | 768 each | T(N-1)/N = 768 |
| broadcast | 256 | 512 / 256 / 0 / 0 | T = 256 |
| allgather | 256 | 768 each | T(N-1)/N = 192 |

In the broadcast, rank 0 sends the whole tensor twice and rank 1 once, and
the leaves send nothing. In the allgather, each rank forwards 3 full
256-byte chunks, not 3/4 of one.

**Padding breaks the allreduce formula.** A 66-float allreduce pads to 68
floats (17 per chunk) and sends 408 B per rank. The formula gives 396, and `main.py`'s
byte check expects 384.

### 5 — over TCP the ring overtakes the tree at about 1 MB, but on the queue mesh the tree always wins

**On a real wire the crossover is at about 1 MB, as the bandwidth-latency
argument predicts.** On 4 ranks, median of 7 interleaved calls, ring/tree
time:

| message | loopback TCP | lesson's queue mesh |
|---|---:|---:|
| 1 KB | 1.5-2.1 (tree wins) | 2.2-2.9 |
| 1 MB | 0.9-1.3 (about even) | 1.5-2.1 |
| 16 MB | 0.7-0.8 (ring wins) | 1.05-1.6 |

The ranges are over four runs on this machine.

The critical path explains this. The ring takes 6 dependent messages of T/4,
so 1.5T crosses in sequence. The tree takes 4 dependent messages of the
whole T, so 4T. At 1 KB the two extra hops cost more than the bytes. At 16
MB the extra 2.5T costs more than the hops.

**On the queue mesh there is no crossover.** A queued tensor is a
shared-memory handle, so the extra bytes never cross anything, and only the
hop count is left.

**gloo is on the other side of both.** At 16 MB gloo's C++ allreduce is
1.6x to 1.9x faster than the Python ring (13-21 ms against 23-39 ms). At
1 KB it is 2x to 4x slower than the Python tree (0.6-0.9 ms), because a fixed per-call
cost dominates small messages. The doc's "ring above ~1 MB, tree below" is
NCCL's rule for real links, and it only shows up here once bytes move.
