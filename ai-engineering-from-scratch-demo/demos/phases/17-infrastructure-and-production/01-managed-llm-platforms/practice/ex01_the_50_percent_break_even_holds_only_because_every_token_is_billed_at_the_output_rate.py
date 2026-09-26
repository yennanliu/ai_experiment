"""Exercise 1 — the 50% break-even holds only because every token is billed at the output rate.

    Run `code/main.py`. At what sustained utilization does Azure PTU beat
    on-demand for a 70B class model? Compute the break-even and compare to the
    advertised 40-60% band.

Reading of the exercise: utilization is what `break_even_demo()` sweeps --
tokens a day as a fraction of one PTU's 24-hour capacity -- so the break-even
is the utilization u* at which 24 x ptu_hourly equals the on-demand bill,
u* = ptu_hourly / (ptu_tokens_per_hour x $/token). It is computed in closed
form, read off the reference's printed sweep, and then recomputed at the
token mixes the reference's own `simulate()` uses.

**ANSWER: 50%, dead centre of the band, and PTU first wins at 60%.** One PTU
is $240/day and delivers 48M tokens/day; at $10/M that is $480 at full load,
so u* = 0.5. At 50% the two tie at $240 and the sweep's strict `<` calls it
for on-demand; PTU wins every row from 60% up. PTU saves at most 50% (at
100% utilization) -- the lesson's "up to ~70%" is out of the model's reach.

**FINDING: the 50% exists only because input tokens are billed at the
output rate.** The sweep prices every token at `per_mtok_output`. Blend in
the $2.50 input rate at the mixes `simulate()` itself runs -- 3:1 and 2:1
input:output -- and u* is 114% and 100%: one PTU never strictly beats
on-demand at any utilization it can deliver. The reference contradicts
itself on the spot: at 45M tokens/day (93.75% of one PTU) `simulate()` picks
on-demand ($225 vs $240) while the sweep says PTU wins at 90%.

**FINDING: there is no 70B-class model, and u* moves inversely with the
token price.** The sweep is labelled "GPT-4o class" and `PLATFORMS` has no
70B entry, so the exercise's model cannot be run as asked. What can be
said: the 40-60% band corresponds to a blended rate of $12.50-$8.33/M at
this PTU price, and any 70B-class rate below $5/M puts u* above 100%. The
reference's Bedrock row is worse still: u* = 117% even at the output rate.

Structure: `sweep()` parses the reference's printed break-even table;
`break_even()` is the closed form; `azure_path()` reads `simulate()`.
"""

from __future__ import annotations

import contextlib
import io

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "01-managed-llm-platforms"
MIXES = {"output only": (0, 1), "3:1": (3, 1), "2:1": (2, 1)}


def printed(fn, *args, **kwargs):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        fn(*args, **kwargs)
    return out.getvalue().splitlines()


def sweep(ref):
    """{utilization %: winner} as the reference prints it."""
    rows = [line.split() for line in printed(ref.break_even_demo) if "%  $" in line]
    return {int(r[0].rstrip("%")): r[-1] for r in rows}


def blended(p, mix):
    n_in, n_out = mix
    return (n_in * p.per_mtok_input + n_out * p.per_mtok_output) / (n_in + n_out)


def break_even(p, rate_per_mtok):
    return p.ptu_hourly / (p.ptu_tokens_per_hour / 1e6 * rate_per_mtok)


def azure_path(ref, tokens_in, tokens_out):
    rows = printed(ref.simulate, tokens_in, tokens_out, 100, use_ptu=True)
    line = next(r for r in rows if r.startswith("Azure"))
    return float(line.split("$")[1].split()[0]), line.rsplit("[", 1)[1].rstrip("]")


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    azure, bedrock = ref.PLATFORMS[1], ref.PLATFORMS[0]
    return {
        "sweep": sweep(ref),
        "u": {
            name: round(break_even(azure, blended(azure, m)), 4)
            for name, m in MIXES.items()
        },
        "max_saving": 1
        - 24
        * azure.ptu_hourly
        / (azure.ptu_tokens_per_hour * 24 / 1e6 * azure.per_mtok_output),
        "sim": azure_path(ref, 30_000_000, 15_000_000),
        "sim_util": 45e6 / (azure.ptu_tokens_per_hour * 24),
        "band_rates": tuple(round(azure.ptu_hourly / (2 * u), 2) for u in (0.4, 0.6)),
        "names": [p.name for p in ref.PLATFORMS],
        "sweep_title": any(
            "GPT-4o class" in line for line in printed(ref.break_even_demo)
        ),
        "bedrock_u": round(break_even(bedrock, bedrock.per_mtok_output), 4),
    }


def verify(result):
    s, u = result["sweep"], result["u"]
    return [
        practice.Check(
            "ANSWER: 50%, dead centre of the band, and PTU first wins at 60%",
            all(
                [
                    u["output only"] == 0.5,
                    s[50] == "on-demand",
                    [k for k, w in s.items() if w == "PTU"] == [60, 70, 80, 90, 100],
                    round(result["max_saving"], 2) == 0.5,
                ]
            ),
            f"u* = {u['output only']:.0%}; the sweep ties at 50% and calls it on-demand, "
            f"PTU wins from 60%; best-case saving {result['max_saving']:.0%}, not ~70%",
        ),
        practice.Check(
            "FINDING: the 50% exists only because input tokens are billed at the output rate",
            all(
                [
                    u["3:1"] > 1,
                    u["2:1"] == 1.0,
                    result["sim"] == (225.0, "on-demand"),
                    s[90] == "PTU",
                    result["sim_util"] > 0.9,
                ]
            ),
            f"at the 3:1 and 2:1 mixes simulate() runs, u* = {u['3:1']:.1%} and "
            f"{u['2:1']:.0%}; simulate() at {result['sim_util']:.1%} utilization picks "
            f"on-demand at ${result['sim'][0]:.0f} while the sweep says PTU wins at 90%",
        ),
        practice.Check(
            "FINDING: there is no 70B-class model, and u* moves inversely with the token price",
            all(
                [
                    result["sweep_title"],
                    not any("70" in n for n in result["names"]),
                    result["band_rates"] == (12.5, 8.33),
                    result["bedrock_u"] > 1,
                ]
            ),
            f"platforms {result['names']}, sweep titled GPT-4o class; the 40-60% band "
            f"needs ${result['band_rates'][0]}-${result['band_rates'][1]}/M blended; "
            f"Bedrock's row breaks even at {result['bedrock_u']:.1%}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
