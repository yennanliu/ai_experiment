"""Exercise 5 — direct beats recursive by 29% at five steps, where the doc says use recursive.

    **Multi-step forecasting.** Modify the AR model to predict 5 steps ahead
    instead of 1. Compare two strategies: (a) predict one step, use the
    prediction as input for the next step (recursive), and (b) train separate
    models for each horizon (direct). Which is more accurate?

Reading of the exercise: the series is the lesson's `make_synthetic_series(500)`
with 10 lags. Recursive is the lesson's own `SimpleAR.forecast`; direct fits one
`SimpleAR` per horizon h on the same lag rows with the target shifted h-1 steps.
Both are refitted on all history at every origin from t=250 to t=495, and scored
by MSE at each horizon over those 246 origins.

**ANSWER: direct.** MSE at h=1..5 is 8.12, 11.92, 18.18, 27.74, 39.79
recursive against 8.12, 11.18, 15.71, 22.08, 28.08 direct: 29% lower at h=5 and
19% lower averaged over the five horizons. At h=1 the two are the same model.

**FINDING: this is the case the doc tells you to solve recursively.** It says
"start with recursive for short horizons (1-5 steps)"; on the lesson's own
series, five steps is where recursive loses most. On `make_seasonal_series`
the two tie (6.79 against 6.57 averaged).

**FINDING: the error growth is the model's, not the horizon's.** The generator
is a deterministic curve plus i.i.d. noise of sd 2, so a forecaster that knew
the curve would score 4 at every horizon (measured 3.97 over the same points).
Recursive grows 4.9x from h=1 to h=5: AR(10) is a misspecified model of a sine
plus trend, and feeding its own outputs back compounds the misspecification.

**CONTROL: where AR is the true model, recursive wins.** On an AR(1) process
(phi 0.8) fitted with one lag, recursive scores 2.54 at h=5 against 2.59 direct,
both near the theoretical `sum 0.64^j` = 2.48.

Structure: `errors` returns per-horizon squared errors for both strategies.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "15-time-series"
H, LAGS, SIGMA = 5, 10, 2.0


def direct(ref, history, n_lags):
    X, y = ref.make_lag_features(history, n_lags)
    latest = history[-n_lags:][::-1].reshape(1, -1)
    return np.array([ref.SimpleAR(n_lags).fit(X[:len(X) - h + 1], y[h - 1:]).predict(latest)[0]
                     for h in range(1, H + 1)])


def errors(ref, series, n_lags=LAGS):
    """Mean squared error per horizon, recursive and direct, over every origin."""
    rec, dirc = [], []
    for origin in range(len(series) // 2, len(series) - H + 1):
        history, truth = series[:origin], series[origin:origin + H]
        rec.append(ref.SimpleAR(n_lags).fit_series(history).forecast(history, H) - truth)
        dirc.append(direct(ref, history, n_lags) - truth)
    return (np.array(rec) ** 2).mean(0), (np.array(dirc) ** 2).mean(0)


def ar1(n=500, phi=0.8):
    noise = np.random.RandomState(0).normal(0, 1, n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + noise[t]
    return x


def solve():
    ref = parity.load_reference(PHASE, LESSON, "time_series")
    series = ref.make_synthetic_series(500)
    t = np.arange(500)
    curve = 50 + 0.05 * t + 10 * np.sin(2 * np.pi * t / 30)
    rec, dirc = errors(ref, series)
    seasonal = errors(ref, ref.make_seasonal_series())
    return {
        "rec": rec, "dir": dirc, "seasonal": [float(e.mean()) for e in seasonal],
        "oracle": float(np.mean((series - curve)[250:] ** 2)),
        "ar1": errors(ref, ar1(), n_lags=1), "theory": sum(0.64**j for j in range(H)),
        "doc": "start with recursive for short horizons (1-5 steps)"
        in parity.doc_text(PHASE, LESSON),
    }


def verify(result):
    rec, dirc, (a_rec, a_dir) = result["rec"], result["dir"], result["ar1"]
    fmt = ", ".join
    return [
        practice.Check(
            "ANSWER: direct is more accurate at every horizon past the first",
            abs(rec[0] - dirc[0]) < 1e-9 and all(dirc[1:] < rec[1:]) and dirc[-1] < 0.8 * rec[-1],
            f"recursive {fmt(f'{v:.2f}' for v in rec)} against direct "
            f"{fmt(f'{v:.2f}' for v in dirc)}: {100 * (1 - dirc[-1] / rec[-1]):.0f}% lower at "
            f"h={H}, {100 * (1 - dirc.mean() / rec.mean()):.0f}% averaged",
        ),
        practice.Check(
            "FINDING: the doc recommends recursive for exactly this horizon",
            result["doc"] and abs(result["seasonal"][0] / result["seasonal"][1] - 1) < 0.1,
            "docs/en.md: 'start with recursive for short horizons (1-5 steps)'; on the "
            f"seasonal series the two tie, {result['seasonal'][0]:.2f} against "
            f"{result['seasonal'][1]:.2f}",
        ),
        practice.Check(
            "FINDING: an oracle scores sigma^2 at every horizon; the growth is the model's",
            abs(result["oracle"] - SIGMA**2) < 0.5 and rec[-1] / rec[0] > 4,
            f"oracle MSE {result['oracle']:.2f} (sigma^2 = {SIGMA**2}); recursive grows "
            f"{rec[-1] / rec[0]:.1f}x from h=1 to h={H}",
        ),
        practice.Check(
            "CONTROL: on a true AR(1), recursive edges out direct",
            a_rec[-1] < a_dir[-1] and abs(a_rec[-1] / result["theory"] - 1) < 0.1,
            f"h={H}: recursive {a_rec[-1]:.2f}, direct {a_dir[-1]:.2f}, theory "
            f"{result['theory']:.2f}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
