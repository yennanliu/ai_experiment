"""Exercise 2 -- 20% generated wrappers steal the top hit in 984 of 1,820 placements; halving their fused score stops every steal.

    Inject 20% generated code (LLM-produced boilerplate) into the corpus and re-evaluate. Observe retrieval poisoning. Add a "generated" flag to the payload and down-weight those hits.

Reading of the exercise: the corpus is the lesson's 6-chunk `SAMPLE_CORPUS`
plus 10 more real chunks from the same four repos, 16 in all. Injecting 20%
means adding 4 generated chunks, so they make up 4 of 20. The generated
code is the kind an assistant emits around real code: a `<symbol>Wrapper`
that forwards to the real symbol, under a "Code generated" header, with the
real summary restated as "helper wrapper that ...". Which 4 real chunks get
a wrapper is not chosen by hand. All C(16,4) = 1,820 placements are run. The
eval is 12 labelled questions over the lesson's chunks, scored as MRR@10 on
the lesson's own `answer` path: dense + BM25 at k=10, `rrf`, then `rerank`
keeping 5. The flag is a `generated` field set on the chunk (the payload).
It is down-weighted in two places: the fused score before `rerank`, or the
final score after it.

**ANSWER: poisoning is real, and the flag removes it.** In 984 of the 1,820
placements (54%), a wrapper takes the top hit for at least one question,
1,210 steals in all. Mean MRR@10 goes from 0.861 clean to 0.852 poisoned,
and the worst placement falls to 0.757. With flagged hits down-weighted
there are 0 steals and MRR@10 is 0.880. Halving the fused score before
`rerank` does it, and so does zeroing the final score after it.

**FINDING: every steal is decided by the fused score.** A wrapper restates
the real summary, so `rerank` gives it the same overlap bonus as the real
chunk. That leaves the RRF prior, which spans at most 0.018, to break the
tie. A 0.5x weight on that prior alone removes all 1,210 steals.

**FINDING: the flagged corpus scores above the clean one, so MRR@10 alone
hides the poisoning.** With the wrappers present and zeroed, "who decides if
a user may act on a resource" moves from rank 2 to rank 1 in 809 placements.
The four extra documents change the BM25 statistics and the dense top-10, and
that reorders a rerank tie. The side effect is +0.019. The poisoning drop is
-0.009. On a 12-question eval the steal count is the signal, not the mean.

Structure: `EXTRA` plus `SAMPLE_CORPUS` are the 16 real chunks, `wrapper` makes the generated
twin of one, and `score` runs the lesson's pipeline for one corpus under every
down-weighting variant.
"""

from __future__ import annotations

import hashlib
import itertools
from collections import Counter

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "02-rag-over-codebase"
QUESTIONS = [
    ("how is S3 multipart abort wired into retry budget", 0), ("where is authorization centralized", 3),
    ("how does rank fusion work", 5), ("AbortMultipartOnFail", 0), ("check_permission", 3), ("abortUpload", 2),
    ("what is the backoff schedule for a bucket", 1), ("where are s3 abort metrics emitted", 2),
    ("OPA policy engine query", 4), ("merge dense and sparse results", 5),
    ("who decides if a user may act on a resource", 3), ("per bucket retry budget config", 1),
]
EXTRA = [  # (repo, path, symbol, body, summary)
    ("uploader", "services/upload.go", "StartUpload", "func StartUpload(ctx, key) (*Upload, error)", "starts an S3 multipart upload and returns its upload id"),
    ("uploader", "services/parts.go", "UploadPart", "func UploadPart(u *Upload, n int, data []byte) error", "uploads one part of a multipart upload with checksum"),
    ("client", "libs/s3client/presign.ts", "presignUrl", "export function presignUrl(bucket, key, ttl)", "creates a presigned GET url for an object with a ttl"),
    ("client", "libs/http/backoff.ts", "withBackoff", "export async function withBackoff(fn, delays)", "generic exponential backoff wrapper for http calls"),
    ("auth", "services/authn/login.py", "login", "def login(username, password): return session.create(...)", "authenticates a user by password and opens a session"),
    ("auth", "libs/tokens/jwt.py", "decode_token", "def decode_token(raw): return jwt.decode(raw, key)", "decodes and verifies a signed JWT access token"),
    ("auth", "services/audit/log.py", "record_decision", "def record_decision(user, resource, allowed)", "writes an authorization decision to the audit log"),
    ("catalog", "services/search/bm25.rs", "bm25_score", "pub fn bm25_score(tf: f32, df: f32, dl: f32) -> f32", "scores one term with BM25 term frequency saturation"),
    ("catalog", "services/search/embed.rs", "embed_query", "pub fn embed_query(q: &str) -> Vec<f32>", "embeds a search query into a dense vector"),
    ("catalog", "services/index/writer.rs", "commit_segment", "pub fn commit_segment(seg: Segment) -> Result<()>", "flushes an index segment to disk and publishes it"),
]
LIFTED = "who decides if a user may act on a resource"
VARIANTS = {"poisoned": ("pre", 1.0), "pre_half": ("pre", 0.5), "post_zero": ("post", 0.0)}


def wrapper(ref, c):
    name = c.path.split("/")[-1].replace(".", "_gen.")
    body = f"// Code generated by an assistant. DO NOT EDIT.\nfunc {c.symbol}Wrapper(args) {{ return {c.symbol}(args) }}"
    g = ref.Chunk(c.repo, f"gen/{name}", 1, 40, f"{c.symbol}Wrapper", body, f"helper wrapper that {c.summary}")
    g.generated = True
    return g


def weigh(hits, w):
    flagged = [(c, s * w if getattr(c, "generated", False) else s) for c, s in hits]
    return sorted(flagged, key=lambda x: -x[1])


def score(ref, chunks, variants):  # {variant: (reciprocal rank per question, top-1 steals)}
    dense, bm25 = ref.DenseIndex(), ref.BM25Index()
    for c in chunks:
        dense.add(c)
        bm25.add(c)
    out = {v: ([], 0) for v in variants}
    for q, gold in QUESTIONS:
        fused, want = ref.rrf(dense.search(q, k=10), bm25.search(q, k=10)), ref.SAMPLE_CORPUS[gold].anchor()
        for v, (where, w) in variants.items():
            rr, steal = rank_one(ref, q, fused, want, where, w)
            out[v] = (out[v][0] + [rr], out[v][1] + steal)
    return out


def rank_one(ref, q, fused, want, where, w):  # (gold reciprocal rank, a wrapper took top-1)
    ranked = ref.rerank(q, weigh(fused, w) if where == "pre" else fused, top_k=len(fused))
    top = [c for c, _ in (weigh(ranked, w) if where == "post" else ranked)][:5]
    anchors = [c.anchor() for c in top]
    return (1 / (anchors.index(want) + 1) if want in anchors else 0.0), getattr(top[0], "generated", False)


def tally(a, rr, steals, clean, n):
    mrr = sum(rr) / len(rr)
    a["mrr"], a["steals"], a["worst"] = a["mrr"] + mrr / n, a["steals"] + steals, min(a["worst"], mrr)
    a["placements_hit"] += steals > 0
    a["lifted"].update(QUESTIONS[i][0] for i, (x, y) in enumerate(zip(rr, clean)) if x > y)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    ref.hash = lambda s: int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")
    real = list(ref.SAMPLE_CORPUS) + [ref.Chunk(r, p, 1, 20, s, b, m) for r, p, s, b, m in EXTRA]
    twins = [wrapper(ref, c) for c in real]
    clean = score(ref, real, {"clean": ("pre", 1.0)})["clean"][0]
    agg = {v: {"mrr": 0.0, "steals": 0, "placements_hit": 0, "worst": 1.0, "lifted": Counter()} for v in VARIANTS}
    combos = list(itertools.combinations(range(len(real)), 4))
    for combo in combos:
        for v, (rr, steals) in score(ref, real + [twins[i] for i in combo], VARIANTS).items():
            tally(agg[v], rr, steals, clean, len(combos))
    for a in agg.values():
        a["mrr"], a["worst"], a["lifted"] = round(a["mrr"], 3), round(a["worst"], 3), dict(a["lifted"])
    return {"placements": len(combos), "share": 4 / (len(real) + 4), "clean": round(sum(clean) / len(clean), 3), **agg}


def verify(result):
    r = result
    p, h, z = r["poisoned"], r["pre_half"], r["post_zero"]
    return [
        practice.Check(
            "ANSWER: 20% wrappers steal the top hit in 984 of 1,820 placements; the flag leaves 0 steals",
            (r["placements"], r["share"], p["placements_hit"], p["steals"], r["clean"], p["mrr"], p["worst"])
            == (1820, 0.2, 984, 1210, 0.861, 0.852, 0.757) and (z["steals"], z["mrr"]) == (0, 0.88),
            f"{r['placements']} placements at {r['share']:.0%}: {p['placements_hit']} with a steal, {p['steals']} "
            f"steals; MRR@10 clean {r['clean']}, poisoned {p['mrr']} (worst {p['worst']}), flagged {z['mrr']}",
        ),
        practice.Check(
            "FINDING: every steal is a rerank tie broken by the fused score",
            (h["steals"], h["mrr"]) == (0, 0.88),
            f"a 0.5x weight on the fused prior alone: {h['steals']} steals, MRR@10 {h['mrr']}",
        ),
        practice.Check(
            "FINDING: the flagged corpus scores above the clean one, so MRR@10 alone hides the poisoning",
            z["lifted"] == {LIFTED: 809} and round(z["mrr"] - r["clean"], 3) == 0.019
            and round(r["clean"] - p["mrr"], 3) == 0.009,
            f"questions ranked higher than on the clean corpus once wrappers are zeroed: {z['lifted']}; "
            f"side effect +{z['mrr'] - r['clean']:.3f} vs poisoning -{r['clean'] - p['mrr']:.3f}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
