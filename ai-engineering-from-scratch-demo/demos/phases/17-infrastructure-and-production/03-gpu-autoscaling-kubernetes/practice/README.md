<!-- generated:start -->
# 17-infrastructure-and-production / 03-gpu-autoscaling-kubernetes

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/17-infrastructure-and-production/03-gpu-autoscaling-kubernetes/) · upstream spec
`phases/17-infrastructure-and-production/03-gpu-autoscaling-kubernetes/docs/en.md`

```bash
uv run demo practice run 03-gpu-autoscaling-kubernetes --ex 1
uv run demo explain 03-gpu-autoscaling-kubernetes --ex 1
uv run pytest demos/phases/17-infrastructure-and-production/03-gpu-autoscaling-kubernetes
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Under a bursty workload, how many requests does naive duty-cycle HPA drop… | code | T0 | `ex01_queue_depth_hpa_drops_63_more_than_duty_cycle_because_the_sims_duty_cycle_reads_one_request_as_full.py` |
| 2 | Design a Karpenter NodePool for a cluster serving Llama 3.3 70B FP8 on H100 SXM5. Specify `ca… | code | T0 | `ex02_whenempty_with_1h_still_evicts_every_gpu_node_at_30_days_unless_expireafter_is_never.py` |
| 3 | Your team reports that deployments are stuck in Pending because "GPUs available but pod won't… | code | T0 | `ex03_gpus_available_is_a_cluster_sum_no_scheduler_uses_so_392_of_609_available_states_cannot_place_an_8_gpu_pod.py` |
| 4 | Pick a signal to autoscale disaggregated prefill pods and a different signal for decode pods.… | code | T0 | `ex04_prefill_scales_on_queued_tokens_not_queued_requests_and_decode_on_kv_utilization.py` |
| 5 | Compute the cost of the `WhenEmptyOrUnderutilized` consolidation trap on a 24x7 production se… | code | T0 | `ex05_one_consolidation_event_is_95s_which_is_more_than_a_whole_day_of_a_999_error_budget.py` |
<!-- generated:end -->

## Answers

Sources were fetched on 2026-09-26: the Karpenter NodePool, disruption and
metrics pages, the Kubernetes metrics reference, the KAI Scheduler quickstart,
the NVIDIA Dynamo Planner configuration reference, and Vantage's p5.48xlarge
page. Every exercise also runs the lesson's own `code/main.py`.

### 1 — queue-depth HPA drops 63 more than duty-cycle, because the sim's duty cycle reads one request as full

**None. The lesson's claim is reversed.** On the shipped 743-request workload:

| strategy | dropped | idle GPU-min | ready replicas at t = 600 |
|---|---:|---:|---:|
| DUTY_CYCLE | 1 | 266.25 | 6 |
| QUEUE_DEPTH | 64 | 49.5 | 1 |

Queue-depth HPA also drops the one request duty-cycle drops, and it drops 63
that duty-cycle serves. The script's closing "Read:" line claims the reverse
of the table it just printed.

**Where the difference comes from:**

- **The spike onset.** The sim computes "utilization" as busy/ready just after
  dispatch, so one request on one replica reads as 100%. Duty-cycle has
  already scaled to 6 replicas before the spike, while queue-depth waits for
  more than 5 queued requests. 33 of its 64 drops arrive in 600-703s.
- **The scale-down rule.** Queue-depth removes a replica on every tick with an
  empty queue, and a replacement takes 95s, so it saw-tooths. Duty-cycle only
  scales down below 20%.

Two ablations confirm it. `MIN_WARM_REPLICAS = 6` takes queue-depth to 2
drops. Instant nodes still leave it at 44.

**A real duty cycle would never scale this sim.** A replica is busy 2.4s of
each 15s tick, which is 16% and below the 20% scale-down line. Held at one
replica, duty-cycle drops 511 of 743.

### 2 — WhenEmpty with 1h still evicts every GPU node at 30 days, unless expireAfter is Never

The design is a `karpenter.sh/v1` NodePool:

- `karpenter.sh/capacity-type: on-demand` on `p5.48xlarge` (8x H100 80 GiB, $55.04/h).
- `consolidationPolicy: WhenEmpty` with `consolidateAfter: 1h`, and a budget of 1 node.
- Taint `nvidia.com/gpu=true:NoSchedule`.
- `expireAfter: Never`.

Sizing: 70.6 GB of FP8 weights at 0.9 memory utilization leaves about 20k
tokens of FP16 KV cache at TP=1 and 256k at TP=2. So the plan is TP=2, with 4
replicas per node.

**The lesson says this setting never evicts a running job, and it does.**
`expireAfter` defaults to 720h, and Karpenter lists Expiration as a forceful
method. Disruption budgets cannot rate-limit it, and `do-not-disrupt` does not
exclude nodes from it. That is 12.2 drains per node-year, or 146 on a 12-node
pool.

Karpenter also now has a third policy, `Balanced`. The defaults are
`WhenEmptyOrUnderutilized` with `0s`, so omitting the block gives you the
trap. The lesson's code models no NodePool. The nearest knob, making every
node warm, takes QUEUE_DEPTH drops from 64 to 44.

### 3 — "GPUs available" is a cluster sum no scheduler uses, so 392 of 609 available states cannot place an 8-GPU pod

**Who owns the problem:**

- **kube-scheduler reports it.** It fits pods node by node, so
  `scheduler_pending_pods{queue="unschedulable"}` rises.
- **Karpenter owns the fix.** Which metric moves tells you why it has not
  fixed it yet:
  - `karpenter_nodepools_usage` has reached `karpenter_nodepools_limit`: the NodePool limit is blocking it.
  - `karpenter_cloudprovider_instance_launch_failures_total` is rising: the cloud has no capacity.
  - `karpenter_nodeclaims_created_total` is still moving: a node is on its way.
  - A missing toleration also shows in `karpenter_scheduler_unschedulable_pods_count`.
- **KAI Scheduler owns it only when the pod sets `schedulerName: kai-scheduler`.**
  Those pods never enter kube-scheduler's queue. A Pending pod while that
  metric stays flat means KAI is holding an incomplete gang, and its Events
  say why.

**What "GPUs available" hides.** Enumerating all 729 free-GPU states of
3 x 8-GPU nodes, 609 have 8 or more GPUs free:

| job shape | fits (of 609) | partial starts, no gang (of 729) |
|---|---:|---:|
| 1 pod x 8 GPUs | 217 | 0 |
| 2 pods x 4 GPUs | 473 | 192 |
| 8 pods x 1 GPU | 609 | 119 |

The lesson's `KAI_GANG` strategy does no gang scheduling. `GPU_PER_REPLICA`
is defined once and never read, and setting it to 8 leaves the result at 61
drops.

### 4 — prefill scales on queued tokens, not queued requests, and decode on KV utilization

**Prefill scales on queued prompt tokens, and decode on KV-cache
utilization.** These are the two thresholds in the Dynamo Planner
configuration: `prefill_scale_up_queue_tokens` and `decode_scale_up_kv_rate`.

At 1.5 req/s each signal is blind to the other role's load:

| shape (in / out tokens) | prefill utilization | KV utilization |
|---|---:|---:|
| RAG (8000 / 200) | 75% | 24% |
| reasoning (500 / 4000) | 5% | 146% |

**The prefill signal has to be counted in tokens.** In 4000 seeded arrivals
with mixed prompt lengths, TTFT wait correlates 0.80 with requests ahead and
1.0 with tokens ahead. With exactly 2 requests ahead, the wait ranges from
0.03s to 1.97s.

**Queue depth is too late for decode.** Decode pods queue nothing until KV is
full. On the reasoning shape KV crosses 80% at 0.82 req/s, while a queue first
forms at 1.02 req/s. The lesson's simulator models neither signal: it has no
tokens and no KV.

### 5 — one consolidation event is 95s, which is more than a whole day of a 99.9% error budget

**Each event costs 95s** (NODE_PROVISION_SEC + MODEL_LOAD_SEC), so 60 events
are 5700s a day. The replacement node bills while it loads:

| | per event | per day | per year |
|---|---:|---:|---:|
| idle GPU at $55.04/h | $1.45 | $87.15 | $31,809 |

**Dropped requests depend on node throughput, which the exercise does not
give.** At an illustrative 4 req/s per node on a 12-node pool:

| fleet utilization | requests dropped per event |
|---|---:|
| 100% | 340 (20,400 a day) |
| 95% | 136 |
| 11/12 or below | 0 |

**The SLO cost is the bigger problem.** A 99.9% SLO allows 86.4s a day, less
than a single event. A 99% SLO allows 9 events a day. At 60 events the
service reaches only 93.4%.

The lesson's simulator cannot price any of this. It never evicts a pod, and
its drop rule is a hard-coded 30s wait rather than 10s TTFT, which would give
260 drops per event instead of 340. `do-not-disrupt` or a blocking PDB also
stops graceful consolidation, but neither stops Expiration.
