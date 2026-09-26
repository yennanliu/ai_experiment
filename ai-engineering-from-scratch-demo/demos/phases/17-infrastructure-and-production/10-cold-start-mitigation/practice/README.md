<!-- generated:start -->
# 17-infrastructure-and-production / 10-cold-start-mitigation

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/17-infrastructure-and-production/10-cold-start-mitigation/) · upstream spec
`phases/17-infrastructure-and-production/10-cold-start-mitigation/docs/en.md`

```bash
uv run demo practice run 10-cold-start-mitigation --ex 1
uv run demo explain 10-cold-start-mitigation --ex 1
uv run pytest demos/phases/17-infrastructure-and-production/10-cold-start-mitigation
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Compute the break-even request rate above which a warm replica is cheaper… | code | T0 | `ex01_a_warm_replica_pays_above_2_7_requests_an_hour_and_the_reference_table_never_reads_the_price.py` |
| 2 | You deploy a 13B model with P99 TTFT SLA of 3s. Pick the minimum mitigation stack (fewest lay… | code | T0 | `ex02_one_layer_suffices_and_the_snapshot_only_passes_because_it_restores_weights_in_zero_seconds.py` |
| 3 | Bottlerocket pre-seeding eliminates image pull but weights still load from snapshot to HBM. C… | code | T0 | `ex03_93_seconds_at_7_gb_s_and_an_ebs_snapshot_cannot_read_at_7_gb_s.py` |
| 4 | Your serverless provider offers GPU snapshots (Modal) and your team refuses because "snapshot… | explain | T0 | prose, below |
| 5 | Design a tiered warm-pool policy: how many warm replicas for paid users, trial users, and bat… | code | T0 | `ex05_paid_needs_three_warm_replicas_trial_rides_on_them_and_batch_needs_none.py` |
<!-- generated:end -->

## Answers

The lesson's `code/main.py` has a 70B phase table, a `total_for_stack` over
three layers, and a printed warm-pool table. Every code exercise runs it. The
AWS EBS pages and Modal's memory-snapshot guide were fetched on 2026-09-26.

### 1 — a warm replica pays above 2.7 requests an hour, and the reference table never reads the price

**Break-even: 2.72 requests per hour.** The inputs are `main()`'s own: a 328 s
cold start and $4.50 per GPU-hour. On top of those, each SLO miss is assumed to
cost $1. Arrivals are Poisson at rate λ, and scale-to-zero has an idle timeout T.

- **Scale-to-zero** loses v·λ·e^(−λT)·(1 + λC) per hour in misses: the request
  that finds no replica, plus every request that arrives while it boots.
- **The warm replica** costs g·e^(−λT)·(1 − λC) more per hour. It pays for the
  time at zero and saves the boot time.

The e^(−λT) cancels, so the idle timeout drops out. Warm wins when
λ·(v(1 + λC) + gC) > g. A seeded 60-day simulation agrees:

| idle timeout | $ saved at 2.5 req/hr | $ saved at 3 req/hr |
|---:|---:|---:|
| 60 s | −536 | +510 |
| 300 s | −413 | +467 |
| 900 s | −265 | +297 |

With a 3 s snapshot cold start the break-even is 4.47/hr. At $0.10 a miss it is
7.75/hr.

**The reference table reads neither the price nor the cold-start time.**
`warm better?` is decided by drops > budget, where drops =
`min(20, max(1, int(24 / rate)))`. `gpu_hourly` and `cold_seconds` are only
printed, and the $3,240 monthly cost is compared with nothing. The column is
identical for ($4.50, 328 s), ($0.85, 30 s) and ($50, 3 s). The first
`cold_starts_per_day` assignment is dead code.

**It also answers the opposite direction.** Its only "yes" is at 1 req/hr, so
it says a warm replica pays *below* about 5 req/hr. The dollar model says it
pays *above* 2.72/hr, because at low traffic the idle GPU costs more than the
few misses it prevents.

### 2 — one layer suffices, and the snapshot only passes because it restores weights in zero seconds

The reference has only `PHASES_70B`. So the 13B path scales image pull, weights
and first forward by 13/70 and runs every subset through `total_for_stack`:

| stack | 13B seconds |
|---|---:|
| none | 117.91 |
| pre-seeded | 84.49 |
| streamer | 110.49 |
| pre-seeded + streamer | 77.06 |
| GPU snapshot (with anything) | 2.59 |
| warm pool (first forward only) | 0.56 |

**The minimum is one layer, and it should be the warm pool.** The snapshot
passes 3 s too, but only because its `weights to HBM` row is 0.0 s at any size.
Charge the restore for 26 GB of bf16 weights at exercise 3's 7 GB/s and the
snapshot path is 6.30 s. Modal's docs promise "3-10x faster" starts, not
zero-time weights.

Two more things the code does:

- **A snapshot hides every other layer.** `total_for_stack` returns the
  snapshot column whenever `gpu_snapshot` is present, so stacking cannot be
  measured.
- **It cannot be asked about 13B at all.** The phases are a module constant and
  the function takes no size, so this solution swaps the constant and restores
  it.

The lesson's "~15s" with mitigations matches none of the 70B totals, which are
328, 288, 148, 108 and 3.0 s.

### 3 — 93 seconds at 7 GB/s, and an EBS snapshot cannot read at 7 GB/s

**93 s.** That is 50 s of node provision, 0 s of image pull, 20 s to read 140 GB
of bf16 weights, 20 s of engine init and 3 s for the first forward. fp8 gives
83 s and int4 78 s. After pre-seeding, node provision is the largest phase.

The reference's pre-seeded path is 148 s, because its weights row is a constant
75 s, which works out to 1.87 GB/s. `Phase` has no bytes or bandwidth field.
Its streamer row, 35 s, is 4.0 GB/s, so a plain read at 7 GB/s beats the
reference's "streamed" load.

**A volume restored from an EBS snapshot does not read at 7 GB/s.** The
lesson's pattern references an EBS snapshot in `EC2NodeClass`. AWS: "For
volumes created from snapshots, the data blocks must be downloaded from Amazon
S3 to the new volume", and the provisioned initialization rate is 100–300
MiB/s. gp3 throughput tops out at 2,000 MiB/s.

| read path | weights | total |
|---|---:|---:|
| local NVMe, 7 GB/s | 20.0 s | 93.0 s |
| gp3 at its 2,000 MiB/s cap | 66.8 s | 139.8 s |
| waiting on 300 MiB/s initialization | 445.0 s | 518.0 s |

The last row is slower than the 328 s raw path it was meant to shorten. Fast
snapshot restore ("fully initialized at creation") removes the S3 wait but not
the gp3 cap. Reaching 7 GB/s means copying the weights onto local instance
NVMe ahead of traffic.

### 4 — the snapshot is taken before the first request, so the PII risk is whatever init puts in memory

*Draws on "Layer 3 — GPU memory snapshots (Modal)".*

**The team's case.** A GPU snapshot serializes process memory: host RAM, and
with GPU snapshots the device state too. The lesson describes it as "weights,
CUDA graphs, KV cache region" taken "after first load". A KV cache written by
real traffic *is* user prompts. So is a tokenizer cache, a logged request
buffer, or an allocator that was never zeroed. A snapshot of that state is a
durable copy of user data, restored into every new replica, and stored
somewhere the team does not operate. Secrets loaded before the snapshot, such
as API keys or database credentials, are captured the same way.

**The other side.** Modal's guide says the snapshot "is created during
deployment and before any requests are processed". What it captures is code in
global scope and in `@modal.enter(snap=True)` methods. By construction, no user
request has reached a process that is snapshotted, so the lesson's "after first
load" with a live KV cache is not how that product works. The realistic risk is
narrower: whatever *initialization* puts in memory. That means warm-up prompts
taken from production logs, credentials read in the snapshot hook, and any
fixture data. GPU memory snapshots are also marked alpha in the guide, which is
a better reason for caution than PII.

**Mitigations, mapped to that risk:**

- **Keep request data out of init.** Warm up with synthetic prompts only. Read
  secrets in a plain `@modal.enter()` method, which the guide says runs after
  restore. Allocate the KV cache pool without writing to it.
- **Ephemeral snapshots.** The guide says a redeploy with new code or
  configuration makes old snapshots obsolete. Redeploy on a schedule so nothing
  captured lives long.
- **Encryption and namespace isolation.** Snapshots are per worker type (2–3
  per GPU type). The fetched guide says nothing about encryption at rest or
  who can read a stored snapshot, so those go to the vendor as questions. They
  are not assumptions.

The decision follows: allow snapshots with a written init contract (no user
data, no secrets before `snap=True` returns), and review that contract like
code.

### 5 — paid needs three warm replicas, trial rides on them, and batch needs none

A warm floor has to carry a tier for one cold start, the 328 s before an
autoscaled replica helps. So each tier is sized on the peak rate it can reach
in that window. In-flight requests are Poisson with mean rate × service time
(M/G/∞). N is the smallest replica count whose slots are exceeded no more often
than the tier's miss budget. Assumed: 8 slots per replica, 6 s per request,
$4.50/hr.

| tier | peak | in flight | budget | warm replicas | overflow |
|---|---:|---:|---:|---:|---:|
| paid (5 tenants × 0.4 req/s) | 2 req/s | 12 | 1% | **3** | 0.07% (2 replicas: 10.13%) |
| trial, alone | 0.3 req/s | 1.8 | 5% | 1 | — |
| trial, pooled onto paid's 3 | 2.3 req/s | 13.8 | 1% | **+0** | 0.42% |
| batch, nightly 20,000 items | — | — | none | **0** | — |

Batch runs 15,000 s on one replica. Its 328 s cold start is 2.2% of the job.
Keeping a replica warm for the other 19.8 hours would cost $89.25 a day to save
$0.41 of boot. **Policy: 3 pooled warm replicas, $9,720 a month.**

**The lesson's per-tenant premium floor doubles that.** One `min_workers` per
paid tenant (2.4 in flight each, 1 replica) plus one for trial is 6 replicas,
$19,440 a month, for the same budgets. The lesson's 3,600 GPU-hours for 5
products is right; at $4.50/hr that is $16,200 a month. `main.py` has no tier,
concurrency or replica count, and prices exactly one replica.
