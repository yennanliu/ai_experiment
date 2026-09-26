"""Exercise 4 — one spinning disk reads the KV 42x slower than the lesson's re-prefill, and no 10GbE Ceph cluster can catch it.

    LMCache stores to Ceph on spinning disk. For a 4K-token KV at 70B FP8 (500
    MB), what's the read time vs re-prefill?

Reading of the exercise: 4K is taken as 4096 tokens and the 500 MB as given.
Re-prefill is priced two ways: at the lesson's own rate (40 tokens/ms), and
from FLOPs, 2 x 70e9 x 4096, on one H100 SXM at 50% of its 1979 TFLOPS dense
FP8 peak. The disk side is a stated model, not a measurement of any cluster:
RADOS's default 4 MiB objects, 150 MB/s sustained per 7200 rpm spindle and
8 ms of seek plus rotation per object. Objects are striped evenly over n
OSDs behind one client NIC, and the read takes whichever is slower, the
busiest spindle or the link.

**ANSWER: about 4.3 s from one spindle against 0.10 s to re-prefill, 42x
slower.** 500 MB is 120 objects. At 150 MB/s plus a seek each, one disk takes
4.32 s, and the lesson's prefill rate recomputes 4096 tokens in 102.4 ms. The
lesson's own DRAM-tier price for the same KV (256 blocks x 3.0 ms) is 768 ms,
already 7.5x the prefill.

**FINDING: striping cannot close the gap on 10GbE or 25GbE.** Spread over
12 OSDs the busiest spindle needs 0.36 s, but a 10GbE client link needs
0.40 s for 500 MB whatever the disk count, 3.9x the prefill. 25GbE floors at
0.16 s. Matching 102.4 ms takes 100GbE and at least 60 OSDs, and Ceph's
replication does not help: by default reads go to the primary OSD only.

**FINDING: disk wins only against a much slower prefill than the lesson's.**
40 tokens/ms on 70B is 5.6 PFLOP/s, 2.83 H100s at 100% of dense FP8 peak.
One H100 at 50% MFU needs 0.58 s for the prefill, and 8 OSDs on 10GbE read
the KV in 0.40 s, under it.

**FINDING: 500 MB is not 4K tokens of a 70B model.** With Llama-3-70B's
geometry (80 layers, 8 KV heads, head dim 128) an FP8 KV cache is 160 KiB a
token: 671 MB at 4096 tokens. 500 MB holds 3052. "70B FP8" also usually means
FP8 weights, and a BF16 KV cache doubles the figure to 1.34 GB.

Structure: `ceph_read()` is the striping model, `osds_needed()` inverts it,
and `prefill_s()` is the FLOP count. The lesson module supplies its prefill
rate and block price.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "18-vllm-production-stack-lmcache"
KV_BYTES, TOKENS = 500e6, 4096                 # the exercise's figures
OBJECT = 4 * 2**20                             # RADOS default object size, 4 MiB
HDD_BPS, SEEK_S = 150e6, 0.008                 # assumed: sustained 7200 rpm read, seek + rotation
NETS = {"10GbE": 1.25e9, "25GbE": 3.125e9, "100GbE": 12.5e9}
PARAMS, H100_FP8, MFU = 70e9, 1979e12, 0.5     # dense FP8 peak, H100 SXM spec sheet
LLAMA70B = {"layers": 80, "kv_heads": 8, "head_dim": 128}


def ceph_read(osds, net_bps, size=KV_BYTES):
    """Seconds to read `size` striped over `osds` spindles behind one client link."""
    objects = math.ceil(size / OBJECT)
    per_osd = math.ceil(objects / osds) * (SEEK_S + OBJECT / HDD_BPS)
    return max(per_osd, size / net_bps)


def osds_needed(target_s, net_bps):
    """Fewest spindles that read the KV within target_s, or None if the link cannot."""
    if KV_BYTES / net_bps > target_s:
        return None
    return next(n for n in range(1, 10_000) if ceph_read(n, net_bps) <= target_s)


def prefill_s(gpus=1, mfu=MFU, tokens=TOKENS):
    return 2 * PARAMS * tokens / (gpus * H100_FP8 * mfu)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    lesson_prefill = TOKENS / ref.PREFILL_TOK_PER_MS / 1000
    kv_per_token = 2 * math.prod(LLAMA70B.values())          # K and V, 1 byte each
    return {
        "lesson_prefill": lesson_prefill,
        "dram_load": TOKENS / ref.KV_BLOCK_TOKENS * ref.LMCACHE_TIME_MS_PER_BLOCK / 1000,
        "one_disk": ceph_read(1, NETS["10GbE"]), "objects": math.ceil(KV_BYTES / OBJECT),
        "striped": {n: {k: round(ceph_read(n, bps), 3) for k, bps in NETS.items()}
                    for n in (1, 12, 60, 120)},
        "need_lesson": {k: osds_needed(lesson_prefill, bps) for k, bps in NETS.items()},
        "one_gpu": prefill_s(), "need_gpu": osds_needed(prefill_s(), NETS["10GbE"]),
        "lesson_gpus": 2 * PARAMS * ref.PREFILL_TOK_PER_MS * 1000 / H100_FP8,
        "kv_token": kv_per_token, "kv_4k": kv_per_token * TOKENS,
        "tokens_in_500mb": KV_BYTES / kv_per_token,
    }



def verify(result):
    striped, need = result["striped"], result["need_lesson"]
    return [
        practice.Check(
            "ANSWER: about 4.3 s from one spindle against 0.10 s to re-prefill, 42x slower",
            all([result["objects"] == 120, round(result["one_disk"], 2) == 4.32,
                 result["lesson_prefill"] == 0.1024,
                 round(result["one_disk"] / result["lesson_prefill"]) == 42,
                 result["dram_load"] == 0.768]),
            f"{result['objects']} objects read in {result['one_disk']:.2f} s against a "
            f"{result['lesson_prefill'] * 1000:.1f} ms re-prefill; the lesson's DRAM tier "
            f"prices the same KV at {result['dram_load'] * 1000:.0f} ms",
        ),
        practice.Check(
            "FINDING: striping cannot close the gap on 10GbE or 25GbE",
            striped[12] == {"10GbE": 0.4, "25GbE": 0.36, "100GbE": 0.36}
            and need == {"10GbE": None, "25GbE": None, "100GbE": 60},
            f"read seconds by OSD count and link {striped}; OSDs needed to match the "
            f"lesson's prefill by link {need}",
        ),
        practice.Check(
            "FINDING: disk wins only against a much slower prefill than the lesson's",
            round(result["lesson_gpus"], 2) == 2.83 and round(result["one_gpu"], 2) == 0.58
            and result["need_gpu"] == 8,
            f"the lesson's rate is {result['lesson_gpus']:.2f} H100s at dense FP8 peak; one "
            f"H100 at 50% MFU prefills in {result['one_gpu']:.2f} s, which "
            f"{result['need_gpu']} OSDs on 10GbE beat",
        ),
        practice.Check(
            "FINDING: 500 MB is not 4K tokens of a 70B model",
            result["kv_token"] == 160 * 1024 and result["kv_4k"] == 671_088_640
            and round(result["tokens_in_500mb"]) == 3052,
            f"{result['kv_token']} B/token FP8 -> {result['kv_4k'] / 1e6:.0f} MB at "
            f"{TOKENS} tokens; 500 MB holds {result['tokens_in_500mb']:.0f}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
