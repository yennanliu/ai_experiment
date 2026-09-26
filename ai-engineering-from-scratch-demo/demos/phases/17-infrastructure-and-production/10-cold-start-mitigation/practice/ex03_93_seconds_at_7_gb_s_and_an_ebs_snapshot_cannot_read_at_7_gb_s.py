"""Exercise 3 — 93 seconds at 7 GB/s, and an EBS snapshot cannot read at 7 GB/s.

    Bottlerocket pre-seeding eliminates image pull but weights still load from
    snapshot to HBM. Compute wall-clock for a 70B model if the snapshot-backed
    NVMe reads at 7 GB/s.

Reading of the exercise: the path is the reference's `pre_seeded` column
(node provision, no image pull, weights to HBM, engine init, first forward)
with the weights row replaced by bytes / bandwidth. 70B parameters in bf16 are
140 GB (decimal, as the 7 GB/s is); fp8 and int4 are shown alongside. The
disk is taken as the bottleneck, since PCIe to the GPUs is faster than 7 GB/s.

**ANSWER: 93 s -- 50 s node, 20 s of weights, 20 s engine init, 3 s first
forward.** The reference's pre-seeded path is 148 s, because its weights row is
75 s. fp8 (70 GB) gives 83 s and int4 (35 GB) 78 s: after pre-seeding, the
50 s node provision is the largest phase, and no disk speed touches it.

**FINDING: the reference's weights row is a constant that implies 1.87 GB/s.**
`Phase` has no bytes or bandwidth field; 140 GB in 75 s is 1.87 GB/s. Its
streamer row, 35 s, is 4.0 GB/s -- a plain read at 7 GB/s (20 s) beats the
reference's "streamed" load by 15 s.

**FINDING: a volume restored from an EBS snapshot cannot read at 7 GB/s.** The
lesson's pattern references an EBS snapshot in `EC2NodeClass`. AWS: "For volumes
created from snapshots, the data blocks must be downloaded from Amazon S3 to the
new volume", and the provisioned initialization rate is "between 100 and 300
MiB/s"; gp3 throughput is capped at 2,000 MiB/s. At the gp3 cap the read is
66.8 s and the path 139.8 s. If the read waits on initialization at 300 MiB/s it
is 445.0 s and the path 518.0 s -- slower than the raw 328 s path it was meant
to shorten. 7 GB/s is a local-NVMe number, not an EBS one. Fast snapshot restore
("fully initialized at creation") removes the S3 wait but not the gp3 cap; the
7 GB/s case needs the weights copied onto local instance NVMe ahead of traffic.

Structure: `path()` swaps the weights row of the reference's pre-seeded column
for bytes / bandwidth; AWS figures are from docs.aws.amazon.com (EBS
initialization and gp3 pages), fetched 2026-09-26.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "10-cold-start-mitigation"
PARAMS, NVME = 70e9, 7e9  # parameters; bytes per second
BYTES_PER_PARAM = {"bf16": 2, "fp8": 1, "int4": 0.5}
MIB = 2**20
GP3_MAX, INIT_MAX = 2000 * MIB, 300 * MIB  # bytes/s, from the AWS EBS docs


def path(ref, read_bps, bytes_per_param=2):
    """Pre-seeded cold start with weights read at `read_bps`: (read s, total s)."""
    read = PARAMS * bytes_per_param / read_bps
    rest = sum(p.pre_seeded_sec for p in ref.PHASES_70B if p.name != "weights to HBM")
    return round(read, 1), round(rest + read, 1)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    weights = next(p for p in ref.PHASES_70B if p.name == "weights to HBM")
    return {
        "phases": {p.name: p.pre_seeded_sec for p in ref.PHASES_70B},
        "at_7": {k: path(ref, NVME, b) for k, b in BYTES_PER_PARAM.items()},
        "reference": ref.total_for_stack({"pre_seeded"}),
        "raw": ref.total_for_stack(set()),
        "implied_gbps": round(PARAMS * 2 / weights.pre_seeded_sec / 1e9, 2),
        "streamer_gbps": round(PARAMS * 2 / weights.streamer_sec / 1e9, 2),
        "streamer_s": weights.streamer_sec,
        "fields": sorted(vars(weights)),
        "gp3": path(ref, GP3_MAX), "init": path(ref, INIT_MAX),
    }


def verify(result):
    at7, phases = result["at_7"], result["phases"]
    return [
        practice.Check(
            "ANSWER: 93 s -- 50 s node, 20 s of weights, 20 s engine init, 3 s first forward",
            all([at7["bf16"] == (20.0, 93.0), at7["fp8"][1] == 83.0, at7["int4"][1] == 78.0,
                 result["reference"] == 148.0, phases["node provision"] > at7["bf16"][0]]),
            f"at 7 GB/s by precision (read s, total s): {at7}; reference pre-seeded "
            f"{result['reference']}s from phases {phases}",
        ),
        practice.Check(
            "FINDING: the reference's weights row is a constant that implies 1.87 GB/s",
            all([result["implied_gbps"] == 1.87, result["streamer_gbps"] == 4.0,
                 not any("byte" in f or "bw" in f for f in result["fields"]),
                 at7["bf16"][0] < result["streamer_s"]]),
            f"Phase fields {result['fields']}; 140 GB in 75 s is "
            f"{result['implied_gbps']} GB/s, the streamer's {result['streamer_s']}s is "
            f"{result['streamer_gbps']} GB/s",
        ),
        practice.Check(
            "FINDING: a volume restored from an EBS snapshot cannot read at 7 GB/s",
            result["gp3"] == (66.8, 139.8) and result["init"] == (445.0, 518.0)
            and result["init"][1] > result["raw"],
            f"at the gp3 cap of 2000 MiB/s: {result['gp3']}; at the 300 MiB/s "
            f"initialization rate: {result['init']}, against the raw {result['raw']}s",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
