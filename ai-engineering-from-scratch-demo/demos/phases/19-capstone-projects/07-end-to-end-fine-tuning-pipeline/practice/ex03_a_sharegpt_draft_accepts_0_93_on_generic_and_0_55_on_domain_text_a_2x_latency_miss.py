"""Exercise 3 -- a draft trained on ShareGPT accepts 0.93 on generic text and 0.55 on domain text.

    Measure EAGLE-3 acceptance rate on domain data vs generic ShareGPT. Report the delta and what it means for latency budgets.

Reading of the exercise: the lesson's serve stage prints a constant 0.74, so
speculative decoding is run for real at toy scale. The target model is a
60-token bigram language with two modes. Generic text (the ShareGPT
stand-in) follows one transition matrix. Domain text is half that matrix
and half a peaked "jargon" matrix. As EAGLE-3 does, the draft is fitted to
the target's own outputs on generic prompts: 50,000 tokens, add-0.1
smoothing. A second draft is refitted on a 50/50 generic and domain mix,
which is the Speculators retraining route. Each step drafts k = 4 tokens.
The target accepts token x with probability min(1, p(x)/q(x)) and resamples
from the residual on a rejection. Acceptance is per-token: tokens accepted
divided by tokens verified, over 4,000 decode steps per corpus. The latency
unit is one target forward pass, and a draft step costs 1/32 of one (one
decoder layer of a 32-layer 8B model).

**ANSWER: acceptance falls from 0.926 on generic text to 0.551 on domain
text, a delta of -0.375.** The mean number of tokens per target pass drops
from 4.32 to 2.11. Per-token latency is 0.261 of plain decoding on generic
text and 0.532 on domain text, so speculation speeds generic text up 3.83x
and domain text 1.88x. A latency budget sized on ShareGPT therefore
underestimates domain decode time by 2.04x. Refitting the draft on the
domain mix lifts domain acceptance to 0.777 and brings the gap down to
1.33x. The exact acceptance, sum(min(p, q)), agrees with the counted rates to
within 0.01.

**FINDING: the lesson's acceptance rate is a constant, not a measurement.**
`stage_serve` returns 0.74 whatever the data: changing the seed and the raw
corpus size leaves the endpoint hash unchanged. The endpoint has no field for
the draft, the data or the quant it was built from.

**FINDING: the lesson's definition of acceptance gives a different number.**
Key Terms defines it as the "fraction of drafted tokens the target model
accepts". All k = 4 tokens are drafted before verification, and on that
definition acceptance is 0.829 on generic text and 0.279 on domain text. The
domain figure is half the per-token rate. A reported "0.74" means nothing
until it says which of the two it is, and at what k.

Structure: `world()` builds the target and the drafts; `speculate()` is the
accept/resample loop; `solve()` measures both corpora and probes the
reference serve stage.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "07-end-to-end-fine-tuning-pipeline"
V, K, DRAFT_COST, STEPS = 60, 4, 1 / 32, 4000
CFG = {"base_model": "llama-3.3-8b", "raw_examples": 300_000, "seed": 7, "dpo_beta": 0.08}


def generate(p, n, r):
    s, cum = np.zeros(n, int), p.cumsum(1)
    for i in range(1, n):
        s[i] = min(np.searchsorted(cum[s[i - 1]], r.random()), V - 1)
    return s


def fit(seqs):
    c = np.full((V, V), 0.1)
    for s in seqs:
        np.add.at(c, (s[:-1], s[1:]), 1)
    return c / c.sum(1, keepdims=True)


def world(r):
    generic = r.dirichlet([0.3] * V, size=V)
    domain = 0.5 * generic + 0.5 * r.dirichlet([0.05] * V, size=V)
    sharegpt = fit([generate(generic, 50_000, r)])
    mixed = fit([generate(generic, 25_000, r), generate(domain, 25_000, r)])
    return generic, domain, sharegpt, mixed


def pick(p, r):
    return min(int(np.searchsorted(p.cumsum(), r.random())), V - 1)


def speculate(target, draft, r):
    tok, accepted, emitted, tried = 0, 0, 0, 0
    for _ in range(STEPS):
        for _ in range(K):
            x, tried = pick(draft[tok], r), tried + 1
            if r.random() < min(1.0, target[tok, x] / draft[tok, x]):
                tok, accepted, emitted = x, accepted + 1, emitted + 1
                continue
            resid = np.maximum(target[tok] - draft[tok], 0)
            tok, emitted = pick(resid / resid.sum(), r), emitted + 1
            break
        else:
            tok, emitted = pick(target[tok], r), emitted + 1
    return accepted / tried, accepted / (STEPS * K), emitted / STEPS


def exact(target, draft, r):
    ctx = generate(target, 20_000, r)[:-1]
    return float(np.minimum(target[ctx], draft[ctx]).sum(1).mean())


def serve_probe(ref):
    out = []
    for cfg in (CFG, {**CFG, "seed": 8, "raw_examples": 1000}):
        with parity.quiet():
            m = ref.run_pipeline(cfg)
        out.append((m.get("endpoint").content_hash(), m.get("dataset").content_hash()))
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    r = np.random.default_rng(7)
    generic, domain, sharegpt, mixed = world(r)
    rows = {}
    for name, target, draft in (("generic", generic, sharegpt), ("domain", domain, sharegpt),
                                ("domain_refit", domain, mixed)):
        acc, block, per_pass = speculate(target, draft, r)
        rows[name] = {"acc": round(acc, 3), "block": round(block, 3), "tokens": round(per_pass, 2),
                      "exact": round(exact(target, draft, r), 3), "latency": round((1 + K * DRAFT_COST) / per_pass, 3)}
    (e1, d1), (e2, d2) = serve_probe(ref)
    with parity.quiet():
        payload = ref.run_pipeline(CFG).get("endpoint").payload
    return {"rows": rows, "same_endpoint": e1 == e2, "data_changed": d1 != d2,
            "lesson_accept": payload["eagle_acceptance"], "fields": sorted(payload)}


def verify(result):
    r, rows = result, result["rows"]
    g, d, f = rows["generic"], rows["domain"], rows["domain_refit"]
    ratio, refit = round(d["latency"] / g["latency"], 2), round(f["latency"] / g["latency"], 2)
    speed = [round(1 / x["latency"], 2) for x in (g, d)]
    return [
        practice.Check(
            "ANSWER: acceptance 0.926 on generic vs 0.551 on domain, and a 2.04x latency underestimate",
            (g["acc"], d["acc"], round(d["acc"] - g["acc"], 3), g["tokens"], d["tokens"], ratio,
             speed, f["acc"], refit) == (0.926, 0.551, -0.375, 4.32, 2.11, 2.04, [3.83, 1.88], 0.777, 1.33)
            and all(abs(x["acc"] - x["exact"]) <= 0.0105 for x in rows.values()),
            "; ".join(f"{k}: accept {x['acc']} (exact {x['exact']}), {x['tokens']} tok/pass, "
                      f"latency {x['latency']}" for k, x in rows.items()) + f"; speedups {speed}",
        ),
        practice.Check(
            "FINDING: the lesson's acceptance rate is a constant, not a measurement",
            (r["lesson_accept"], r["same_endpoint"], r["data_changed"]) == (0.74, True, True)
            and not {"from", "draft", "dataset_hash"} & set(r["fields"]),
            f"eagle_acceptance {r['lesson_accept']}; new data, same endpoint hash: {r['same_endpoint']}; "
            f"endpoint fields {r['fields']}",
        ),
        practice.Check(
            "FINDING: the lesson's definition of acceptance gives a different number",
            (g["block"], d["block"], f["block"]) == (0.829, 0.279, 0.558),
            f"accepted / drafted at k={K}: generic {g['block']}, domain {d['block']}, refit {f['block']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
