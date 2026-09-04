"""Exercise 5 — forward diffusion over 100 steps, then a naive reverse.

    **Implement the forward diffusion process.** Start with a 1D signal (e.g., a
    sine wave). Add noise progressively over 100 steps with a linear noise
    schedule. Show how the signal degrades to pure noise. Then implement a simple
    denoiser that reverses the process (even a naive one that just subtracts the
    estimated noise).

Reading of the exercise: two claims fail as specified, and both failures are the
lesson.

"Degrades to pure noise" does not happen in 100 steps. With β from 1e-4 to 0.02,
ᾱ_T = Π(1−β_t) = 0.364, so √ᾱ = 0.60 of the amplitude survives and correlation
with the clean signal is still **+0.50**. That schedule is the standard DDPM one,
designed for 1000 steps; at 1000 it does reach noise (corr +0.03).

And "just subtracts the estimated noise" is where the whole difficulty lives. The
obvious naive move — divide by √ᾱ to undo the scaling — is **not a denoiser**: it
cannot change correlation at all, and it makes RMSE *worse*, 0.841 → 1.340. What
works is a **prior**: keeping the single strongest FFT bin recovers the sine at
RMSE 0.073, because "this is one sinusoid" is real information. That prior is
what a diffusion model's network supplies.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "22-stochastic-processes"
N, SEED = 256, 20260904
BETA_START, BETA_END = 1e-4, 0.02


def solve():
    numpy = parity.try_numpy()
    if numpy is None:
        raise practice.Skip("needs numpy — uv sync --extra math")
    ref = parity.load_reference(PHASE, LESSON, "stochastic")
    clean = numpy.sin(2 * numpy.pi * 3 * numpy.arange(N) / N)

    def rmse(x):
        return float(numpy.sqrt(((x - clean) ** 2).mean()))

    def degrade(steps, beta_end=BETA_END):
        trajectory, betas = ref.diffusion_forward(clean, steps, beta_start=BETA_START,
                                                  beta_end=beta_end, seed=SEED)
        noisy = trajectory[-1]
        bar = float(numpy.prod(1 - betas))
        rescaled = noisy / math.sqrt(bar)
        spectrum = numpy.fft.rfft(rescaled)
        strongest = numpy.argmax(numpy.abs(spectrum))
        kept = numpy.zeros_like(spectrum)
        kept[strongest] = spectrum[strongest]
        low_pass = numpy.fft.irfft(kept, n=N)
        return {"alpha_bar": bar,
                "corr": float(numpy.corrcoef(clean, noisy)[0, 1]),
                "corr_rescaled": float(numpy.corrcoef(clean, rescaled)[0, 1]),
                "corr_low_pass": float(numpy.corrcoef(clean, low_pass)[0, 1]),
                "rmse_noisy": rmse(noisy), "rmse_rescaled": rmse(rescaled),
                "rmse_low_pass": rmse(low_pass), "bin": int(strongest)}

    return {"at_100": degrade(100), "at_1000": degrade(1000),
            "big_beta": degrade(100, beta_end=0.2),
            "clean_rms": float(clean.std())}


def verify(result):
    hundred, thousand, big = result["at_100"], result["at_1000"], result["big_beta"]
    return [
        practice.Check("the forward process degrades the signal",
                       0.4 < hundred["corr"] < 0.9,
                       f"after 100 steps correlation with the clean signal is "
                       f"{hundred['corr']:+.4f}, down from 1.0, at RMSE "
                       f"{hundred['rmse_noisy']:.3f} against a clean RMS of "
                       f"{result['clean_rms']:.3f}"),
        practice.Check("FINDING: 100 steps do NOT reach pure noise",
                       hundred["alpha_bar"] > 0.3 and hundred["corr"] > 0.4,
                       f"ᾱ = {hundred['alpha_bar']:.4f}, so √ᾱ = "
                       f"{math.sqrt(hundred['alpha_bar']):.3f} of the amplitude survives. "
                       f"The β range 1e-4 to 0.02 is the standard DDPM schedule, designed "
                       f"for 1000 steps — the exercise applies it over 100"),
        practice.Check("…and it does reach noise at 1000 steps, or with a bigger β",
                       abs(thousand["corr"]) < 0.1 and abs(big["corr"]) < 0.1,
                       f"1000 steps: ᾱ = {thousand['alpha_bar']:.2g}, corr "
                       f"{thousand['corr']:+.4f}. Same 100 steps with β up to 0.2: "
                       f"ᾱ = {big['alpha_bar']:.2g}, corr {big['corr']:+.4f}"),
        practice.Check("FINDING: dividing by √ᾱ is not a denoiser — it makes RMSE worse",
                       hundred["rmse_rescaled"] > hundred["rmse_noisy"]
                       and abs(hundred["corr_rescaled"] - hundred["corr"]) < 1e-9,
                       f"RMSE {hundred['rmse_noisy']:.3f} -> "
                       f"{hundred['rmse_rescaled']:.3f}, and correlation is *identical* at "
                       f"{hundred['corr_rescaled']:+.4f} because scaling every sample by a "
                       f"constant cannot change it. Undoing the known scale factor "
                       f"amplifies the noise along with the signal"),
        practice.Check("…while a denoiser with a PRIOR works, which is the network's job",
                       hundred["rmse_low_pass"] < hundred["rmse_noisy"] / 5
                       and hundred["corr_low_pass"] > 0.99,
                       f"keeping only the strongest FFT bin (bin {hundred['bin']}) gives "
                       f"RMSE {hundred['rmse_low_pass']:.4f} and correlation "
                       f"{hundred['corr_low_pass']:+.4f} — "
                       f"{hundred['rmse_noisy'] / hundred['rmse_low_pass']:.0f}x better "
                       f"than doing nothing. 'This is one sinusoid' is real information, "
                       f"and it is the only reason the noise can be estimated at all. A "
                       f"diffusion model learns that prior instead of being told it"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
