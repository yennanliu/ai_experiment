"""Exercise 3 — Gao's form fits exercise 1 exactly with beta = 0, and only heavy-tailed error gives beta = 0.876.

    Read Gao et al. Figure 1 (ICML 2023). The paper proposes a functional form
    for the proxy-gold gap. Fit it to your simulated curves from Exercise 1 and
    compare parameters.

Reading of the exercise: the forms fitted are the two Gao et al. give for
gold reward against d = sqrt(KL) -- best-of-N, d(alpha - beta d), and RL,
d(alpha - beta log d) -- and the lesson's own version, which gives proxy and
gold the quadratic form with a shared alpha and beta_gold > beta_proxy. Each
is a two-parameter linear least-squares fit, solved with the reference's own
`solve()`. The curves are exercise 1's three KL-constrained sweeps, the
reference's best-of-N curve, and, for contrast, exercise 2's best-of-N curves
under Gaussian and Student-t(3) proxy error.

**ANSWER: both of Gao's forms fit exercise 1's curves exactly, with beta = 0.**
For proxies fit on 100, 300 and 1000 samples, alpha_gold = 2.239, 2.253 and
2.254, |beta| < 1e-13 in either form, and the residual is below 1e-13. The
form's whole content -- a peak at d* = alpha / (2 beta) -- is absent: d* is
infinite. The lesson's version fails on both of its parameter claims. Proxy
and gold do not share alpha (alpha_proxy = 2.274, 2.325, 2.216), and
beta_gold is not larger than beta_proxy: both are zero.

**FINDING: the reference's best-of-N curve fits only after its x axis is
replaced by Gao's best-of-N KL.** On the printed axis (which starts at 1.920
for n = 1) the fit is alpha = -1.892, beta = -1.292 with an RMS residual of
0.544, a convex curve with a negative slope. On d = sqrt(log n - (n - 1)/n)
it is alpha = 1.997, beta = -0.055, RMS 0.016: nearly a line, and no peak.

**FINDING: the only curve that gives Gao's parameters is exercise 2's
heavy-tailed one.** Best-of-N under Student-t(3) proxy error fits alpha =
2.542, beta = 0.876, a predicted peak at d* = 1.452, which falls between the
measured grid points n = 16 (d = 1.355) and n = 64 (d = 1.782). The same
selection under Gaussian error fits beta = -0.012, with no peak.

Structure: `fit()` builds the 2x2 normal equations for a basis (d, d^2) or
(d, d log d) and hands them to `ref.solve`; curves come from exercise 1's
`proxies()` and `shipped()` and exercise 2's `best_of_n()`.
"""

from __future__ import annotations

import math
import pathlib

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "02-reward-hacking-goodhart"
HERE = pathlib.Path(__file__).resolve().parent
EX02 = practice.load_module(next(HERE.glob("ex02_*.py")))
EX01 = EX02.EX01
BON_NS = (1, 2, 4, 8, 16, 64, 256, 1024)                   # main()'s best-of-N grid


def quad(d):
    return d, d * d


def rl(d):
    return d, (d * math.log(d) if d > 0 else 0.0)


def fit(ref, ds, ys, basis):
    """(alpha, beta, rms) of y = alpha d - beta * basis(d)[1], by least squares."""
    rows = [basis(d) for d in ds]
    gram = [[sum(r[i] * r[j] for r in rows) for j in range(2)] for i in range(2)]
    rhs = [sum(r[i] * y for r, y in zip(rows, ys)) for i in range(2)]
    a, b = ref.solve(gram, rhs)
    rms = math.sqrt(sum((a * r[0] + b * r[1] - y) ** 2 for r, y in zip(rows, ys)) / len(ys))
    return a, -b, rms


def bon_kl(ns):
    """Gao et al.'s best-of-n KL, as sqrt(KL)."""
    return [math.sqrt(math.log(n) - (n - 1) / n) for n in ns]


def r3(t):
    return tuple(round(v, 3) for v in t)


def sweeps(ref):
    """Exercise 1's three sweeps: alphas per form, and the largest |beta| and RMS."""
    fits = []
    for rm in EX01.proxies(ref):
        ds, proxy, gold = zip(*ref.kl_constrained_policy_sweep(rm, EX01.BUDGETS))
        fits.append((fit(ref, ds, gold, quad), fit(ref, ds, gold, rl), fit(ref, ds, proxy, quad)))
    flat = [f for triple in fits for f in triple]
    alphas = [[round(t[k][0], 3) for t in fits] for k in range(3)]
    return {"gold": alphas[0], "rl": alphas[1], "proxy": alphas[2],
            "beta": max(abs(f[1]) for f in flat), "rms": max(f[2] for f in flat)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = EX01.parse(EX01.shipped(ref))[1][-8:]
    gold = [r[2] for r in rows]
    heavy = fit(ref, bon_kl(EX02.NS), EX02.best_of_n(ref, lambda: ref.student_t(3.0)), quad)
    doc = " ".join(parity.doc_text(PHASE, LESSON).split())
    return {
        "sweeps": sweeps(ref),
        "bon_printed": r3(fit(ref, [r[0] for r in rows], gold, quad)),
        "bon_true": r3(fit(ref, bon_kl(BON_NS), gold, quad)),
        "heavy": r3(heavy), "heavy_peak": round(heavy[0] / (2 * heavy[1]), 3),
        "gauss": r3(fit(ref, bon_kl(EX02.NS), EX02.best_of_n(ref, ref.gauss), quad)),
        "grid": [round(d, 3) for d in bon_kl((16, 64))],
        "doc_form": all(s in doc for s in ("R_proxy(d) = alpha * d - beta_proxy * d^2",
                                           "R_gold(d) = alpha * d - beta_gold * d^2",
                                           "beta_gold > beta_proxy")),
    }


def verify(result):
    sw, heavy = result["sweeps"], result["heavy"]
    return [
        practice.Check(
            "ANSWER: both of Gao's forms fit exercise 1 exactly, with beta = 0",
            (sw["gold"], sw["rl"]) == ([2.239, 2.253, 2.254],) * 2
            and max(sw["beta"], sw["rms"]) < 1e-13,
            f"alpha_gold {sw['gold']} (RL form {sw['rl']}); max |beta| {sw['beta']:.1e}; "
            f"max RMS {sw['rms']:.1e}",
        ),
        practice.Check(
            "ANSWER: the lesson's shared-alpha, beta_gold > beta_proxy form fails",
            (result["doc_form"], sw["proxy"]) == (True, [2.274, 2.325, 2.216]),
            f"alpha_proxy {sw['proxy']} vs alpha_gold {sw['gold']}; beta_gold = beta_proxy "
            "= 0 (|beta| < 1e-13)",
        ),
        practice.Check(
            "FINDING: the reference's best-of-N curve fits only on Gao's best-of-N KL",
            (result["bon_printed"], result["bon_true"])
            == ((-1.892, -1.292, 0.544), (1.997, -0.055, 0.016)),
            f"printed axis (alpha, beta, rms) {result['bon_printed']}; Gao's axis "
            f"{result['bon_true']}",
        ),
        practice.Check(
            "FINDING: only exercise 2's heavy-tailed curve gives beta > 0",
            (heavy[:2], result["heavy_peak"], result["grid"], result["gauss"][1])
            == ((2.542, 0.876), 1.452, [1.355, 1.782], -0.012),
            f"Student-t(3) (alpha, beta) {heavy[:2]} -> d* = {result['heavy_peak']} between "
            f"{result['grid']}; Gaussian beta = {result['gauss'][1]}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
