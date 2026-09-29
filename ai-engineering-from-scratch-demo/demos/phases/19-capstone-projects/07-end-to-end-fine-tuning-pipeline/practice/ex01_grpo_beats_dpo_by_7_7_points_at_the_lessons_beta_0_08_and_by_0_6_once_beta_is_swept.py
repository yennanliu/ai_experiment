"""Exercise 1 -- GRPO beats DPO by 7.7 points at the lesson's beta 0.08, and by 0.6 once beta is swept.

    Run SFT-only vs SFT+DPO vs SFT+GRPO on the same task-specific benchmark. Report which preference method wins and by how much.

Reading of the exercise: the lesson's `run_pipeline` cannot run this. It has
no GRPO stage, it evaluates only the DPO checkpoint, and its "training" stages
return fixed numbers. So the three recipes are run at toy scale. The task is
4-way classification over 8 features (the benchmark's right answer is the
argmax of a hidden linear map). The policy is a softmax over 4 answers. SFT
trains on 300 demonstrations, 40% of them wrong. DPO then trains on one
preference pair per prompt: the right answer is chosen, and the rejected one
is a wrong answer sampled from the SFT policy. The KL reference is the SFT
policy. GRPO samples 8 answers per prompt and scores each against a
verifiable 0/1 reward, using a group-normalised advantage and a KL term with
beta 0.04. Every recipe gets 300 full-batch steps at lr 0.5. The benchmark is
4,000 held-out prompts, scored by greedy accuracy and averaged over 5 seeds
(asserted to within 0.003, since BLAS rounding differs across platforms).
DPO is run at the lesson's beta of 0.08 and swept, as Build It step 4 says.

**ANSWER: GRPO wins, by 7.7 accuracy points over DPO at the lesson's beta of
0.08.** Benchmark accuracy is SFT-only 0.743, SFT+DPO 0.858 and SFT+GRPO
0.935. Sweeping DPO's beta over {0.08, 0.3, 1, 3} peaks at beta 1, with
accuracy 0.929. That shrinks GRPO's lead to 0.6 points. At a fixed step
budget, beta also scales the DPO gradient, so a small beta trains slowly.

**FINDING: the lesson's pipeline cannot run the ablation it asks for.**
`PIPELINE` has 8 stages and no GRPO stage. Its only eval reads
`dpo_checkpoint`. It ignores `cfg["dpo_beta"]`: at beta 0.08 and 0.5, all 8
artifact hashes come out identical, and the stage always records 0.08.

**FINDING: GRPO's win costs 2,400 times the labels.** GRPO makes 720,000
reward calls (300 prompts x 8 samples x 300 steps). DPO uses 300 preference
labels. A GRPO reward has to be verifiable to be affordable at that volume.

Structure: `task()` builds one seed's data; `sft()`, `dpo()` and `grpo()` are
the three trainers on the same logits; `solve()` runs them and probes the
reference pipeline.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "07-end-to-end-fine-tuning-pipeline"
D, K, N, TEST, STEPS, LR, SEEDS, G, BETAS = 8, 4, 300, 4000, 300, 0.5, 5, 8, (0.08, 0.3, 1.0, 3.0)
TOL = 0.003  # BLAS rounding can flip a sampled answer; 0.003 is 12 of 4,000 prompts
CFG = {"base_model": "llama-3.3-8b", "raw_examples": 300_000, "seed": 7, "dpo_beta": 0.08}


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def task(seed):
    r = np.random.default_rng(seed)
    w, x, xt = r.normal(size=(K, D)), r.normal(size=(N, D)), r.normal(size=(TEST, D))
    y = (x @ w.T).argmax(1)
    noisy = np.where(r.random(N) < 0.6, y, (y + r.integers(1, K, N)) % K)
    return x, y, noisy, xt, (xt @ w.T).argmax(1)


def acc(th, x, y):
    return float(((x @ th.T).argmax(1) == y).mean())


def sample(p, r, g):
    return np.minimum((p.cumsum(1)[:, None, :] < r.random((len(p), g))[..., None]).sum(-1), K - 1)


def sft(x, lab):
    th, onehot = np.zeros((K, D)), np.eye(K)[lab]
    for _ in range(STEPS):
        th -= LR * (softmax(x @ th.T) - onehot).T @ x / len(x)
    return th


def dpo(th0, x, y, beta, r):
    p0 = softmax(x @ th0.T) * (1 - np.eye(K)[y])  # the SFT policy over wrong answers only
    lose = sample(p0 / p0.sum(1, keepdims=True), r, 1)[:, 0]
    diff, th = np.eye(K)[y] - np.eye(K)[lose], th0.copy()  # diff = d(logpi_w - logpi_l)/d logits
    for _ in range(STEPS):
        margin = beta * ((x @ (th - th0).T) * diff).sum(1)
        th += LR * ((beta / (1 + np.exp(margin)))[:, None] * diff).T @ x / N
    return th


def grpo(th0, x, y, r, beta=0.04):
    th, p0 = th0.copy(), softmax(x @ th0.T)
    for _ in range(STEPS):
        p = softmax(x @ th.T)
        a = sample(p, r, G)
        rew = (a == y[:, None]).astype(float)  # the verifiable reward, then the group-normalised advantage
        adv = (rew - rew.mean(1, keepdims=True)) / (rew.std(1, keepdims=True) + 1e-6)
        grad = sum(adv[:, j : j + 1] * (np.eye(K)[a[:, j]] - p) for j in range(G)) / G
        log_ratio = np.log(p + 1e-12) - np.log(p0 + 1e-12)
        grad -= beta * p * (log_ratio - (p * log_ratio).sum(1, keepdims=True))
        th += LR * grad.T @ x / N
    return th


def probe_pipeline(ref):
    def hashes(cfg):
        with parity.quiet():
            m = ref.run_pipeline(cfg)
        return [a.content_hash() for a in m.artifacts.values()], m

    (h1, m), (h2, _) = hashes(CFG), hashes({**CFG, "dpo_beta": 0.5})
    return {"stages": [n for n, _ in ref.PIPELINE], "beta_same": h1 == h2, "n_hash": len(h1), "beta_recorded":
            m.get("dpo_checkpoint").payload["beta"],
            "eval_from_dpo": m.get("eval_report").payload["from"] == m.get("dpo_checkpoint").content_hash()}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = []
    for seed in range(SEEDS):
        x, y, noisy, xt, yt = task(seed)
        th = sft(x, noisy)
        rows.append([acc(th, xt, yt), acc(grpo(th, x, y, np.random.default_rng(100 + seed)), xt, yt)]
                    + [acc(dpo(th, x, y, b, np.random.default_rng(200 + seed)), xt, yt) for b in BETAS])
    m = [round(float(v), 3) for v in np.mean(rows, 0)]
    return {"sft": m[0], "grpo": m[1], "dpo": dict(zip(BETAS, m[2:])), "grpo_calls": N * G * STEPS,
            "dpo_labels": N, **probe_pipeline(ref)}


def verify(r):
    best = max(r["dpo"], key=r["dpo"].get)
    lead, swept = round(r["grpo"] - r["dpo"][0.08], 3), round(r["grpo"] - r["dpo"][best], 3)
    return [
        practice.Check(
            "ANSWER: GRPO wins, by 7.7 points over DPO at beta 0.08 and 0.6 over the best swept beta",
            best == 1.0 and 0 < swept < lead and np.allclose(
                (r["sft"], r["dpo"][0.08], r["grpo"], r["dpo"][best], lead, swept),
                (0.743, 0.858, 0.935, 0.929, 0.077, 0.006), atol=TOL),
            f"SFT {r['sft']}, SFT+DPO {r['dpo'][0.08]} (beta 0.08), SFT+GRPO {r['grpo']}; "
            f"DPO sweep {r['dpo']} peaks at beta {best}",
        ),
        practice.Check(
            "FINDING: the lesson's pipeline cannot run the ablation it asks for",
            ("grpo" not in r["stages"], len(r["stages"]), r["eval_from_dpo"], r["beta_same"],
             r["beta_recorded"], r["n_hash"]) == (True, 8, True, True, 0.08, 8),
            f"stages {r['stages']}; eval reads dpo: {r['eval_from_dpo']}; beta 0.5 same hashes: {r['beta_same']}",
        ),
        practice.Check(
            "FINDING: GRPO's win costs 2,400 times the labels",
            (r["grpo_calls"], r["dpo_labels"], r["grpo_calls"] // r["dpo_labels"]) == (720000, 300, 2400),
            f"{r['grpo_calls']:,} reward calls vs {r['dpo_labels']} preference labels",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
