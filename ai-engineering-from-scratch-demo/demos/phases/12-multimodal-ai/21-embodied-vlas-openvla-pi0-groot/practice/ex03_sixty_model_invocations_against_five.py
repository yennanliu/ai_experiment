"""Exercise 3 — sixty model invocations against five.

    π0's flow-matching head denoises in ~5 steps. Compare throughput to
    OpenVLA's autoregressive decode at 4-5 Hz.

Reading of the exercise: the comparison is made per *second of motion produced*
rather than per forward pass, because the two architectures emit different things
per invocation and a raw pass count would make the autoregressive arm look
cheaper. The token figures come from the lesson's own `discretize` and the 4-5 Hz
from its own OpenVLA section.

**ANSWER: 5 invocations against 300, a factor of 60.** One second of 30 Hz,
10-DOF motion is **300** discrete action tokens, each an autoregressive step; the
same second is one 30-step chunk for a flow head, denoised in **5** passes.

**FINDING: at its own reported decode rate OpenVLA takes 7.5 seconds to produce
one second of motion.** Four Hz of 10-token steps is **40** tokens a second, and
the second needs 300 -- a real-time factor of **7.5**, or **6.0** at 5 Hz. It is
not slow at 30 Hz; it cannot run at 30 Hz at all.

**FINDING: the flow head's requirement is 5 passes a second and its headroom is
enormous.** At the same 30-80 tok/s the phase uses for a 7B, a head that emits a
whole chunk per pass needs **5** invocations a second -- **6x to 16x** under
budget. Which is why π0 reports control rates in the tens of Hz and OpenVLA
reports 4-5.

**FINDING: and the advantage is the chunk, not the flow.** The 60x is
`chunk_steps x DOF / denoise_steps` -- it would be 60x for a diffusion head, a
regression head or anything else that emits 30 steps at once. Flow matching
supplies the 5; the 300 comes from emitting one token per DOF per step, and any
architecture that stops doing that gets the same win.

Structure: `discrete_tokens` prices a second of motion through the lesson's own
tokenizer, `flow_passes` counts denoise steps per chunk, and `realtime_factor`
divides a requirement by a decode rate.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "21-embodied-vlas-openvla-pi0-groot"
DOF, CONTROL_HZ, CHUNK_STEPS = 10, 30, 30
DENOISE_STEPS = 5
OPENVLA_HZ = (4, 5)
THROUGHPUTS = (30, 80)


def discrete_tokens(ref, seconds=1.0):
    return len(ref.discretize([0.0] * DOF)) * int(CONTROL_HZ * seconds)


def decode_rate(hertz, dof=DOF):
    return hertz * dof


def realtime_factor(required, rate):
    return round(required / rate, 2)


def flow_passes(seconds=1.0, chunk=CHUNK_STEPS, rate=CONTROL_HZ, steps=DENOISE_STEPS):
    return int(rate * seconds / chunk) * steps


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    tokens = discrete_tokens(ref)
    passes = flow_passes()
    return {
        "tokens_per_second": tokens, "flow_passes": passes,
        "ratio": tokens // passes,
        "decode_rates": {hz: decode_rate(hz) for hz in OPENVLA_HZ},
        "realtime": {hz: realtime_factor(tokens, decode_rate(hz)) for hz in OPENVLA_HZ},
        "headroom": {tp: round(tp / passes, 1) for tp in THROUGHPUTS},
        "chunk": CHUNK_STEPS, "denoise": DENOISE_STEPS, "dof": DOF,
        "formula": CHUNK_STEPS * DOF // DENOISE_STEPS,
        "matches_ratio": CHUNK_STEPS * DOF // DENOISE_STEPS == tokens // passes,
        "target_hz": CONTROL_HZ,
    }


def verify(result):
    realtime, rates = result["realtime"], result["decode_rates"]
    return [
        practice.Check(
            "ANSWER: 5 invocations against 300, a factor of 60",
            all([result["tokens_per_second"] == 300, result["flow_passes"] == 5,
                 result["ratio"] == 60]),
            f"one second of {CONTROL_HZ} Hz, {DOF}-DOF motion is "
            f"{result['tokens_per_second']} discrete action tokens, each an autoregressive "
            f"step; the same second is one {CHUNK_STEPS}-step chunk denoised in "
            f"{result['flow_passes']} passes -- {result['ratio']}x fewer model invocations "
            "for the same motion",
        ),
        practice.Check(
            "FINDING: OpenVLA takes 7.5 seconds to produce one second of motion",
            all([rates == {4: 40, 5: 50}, realtime == {4: 7.5, 5: 6.0}]),
            f"{OPENVLA_HZ[0]} Hz of {DOF}-token steps is {rates[4]} tokens a second, and the "
            f"second needs {result['tokens_per_second']} -- a real-time factor of "
            f"{realtime[4]}, or {realtime[5]} at {OPENVLA_HZ[1]} Hz. It is not slow at "
            f"{result['target_hz']} Hz; it cannot run at {result['target_hz']} Hz at all",
        ),
        practice.Check(
            "FINDING: the flow head needs 5 passes a second and has 6x to 16x of headroom",
            all([result["headroom"] == {30: 6.0, 80: 16.0},
                 min(result["headroom"].values()) > 1]),
            f"at the same {list(THROUGHPUTS)} tok/s this phase uses for a 7B, a head that "
            f"emits a whole chunk per pass needs {result['flow_passes']} invocations a "
            f"second -- {result['headroom']}x under budget. Which is why flow-matching "
            "policies report control rates in the tens of Hz and OpenVLA reports 4-5",
        ),
        practice.Check(
            "FINDING: the advantage is the chunk, not the flow",
            all([result["formula"] == 60, result["matches_ratio"],
                 result["chunk"] == CHUNK_STEPS, result["denoise"] == DENOISE_STEPS]),
            f"the {result['ratio']}x is chunk_steps x DOF / denoise_steps = "
            f"{result['chunk']} x {result['dof']} / {result['denoise']} = "
            f"{result['formula']}. It would be the same for a diffusion head, a regression "
            "head or anything else that emits 30 steps at once -- flow matching supplies the "
            "5, and the 300 comes from emitting one token per DOF per step",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
