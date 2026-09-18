"""Exercise 1 — the bin count does not enter the rate at all.

    A 10-DOF arm at 30 Hz control rate. Discrete-bin tokenization at 256 bins
    emits how many tokens per second? Can a 7B VLM keep up?

Reading of the exercise: the rate comes from the lesson's own `discretize`, run
at several bin counts, because the question names 256 bins as though the number
mattered to the answer and it does not. "Can a 7B keep up" is then checked
against the 30-80 tok/s range this phase has used since Lesson 12.20, and against
the lesson's own reported OpenVLA control rate.

**ANSWER: 300 tokens a second, and no.** Ten DOF is ten tokens a step at any bin
count, and 30 Hz makes 300 a second. At **30-80** tok/s a 7B model is **3.75x to
10x** too slow.

**FINDING: `bins` changes the precision and not the rate.** 2 bins and 1,024 bins
both emit **10** tokens a step. What 256 buys is a round-trip error of
**0.0069** on the lesson's own demo action -- one bin width -- against **1.9**
at two bins and 0.0017 at 1,024. Rate is `DOF x control_rate`; the vocabulary does not appear.

**FINDING: achievable control rate is throughput divided by DOF, and it brackets
the lesson's own figure.** At 30-80 tok/s and 10 DOF that is **3.0 to 8.0 Hz**,
and the lesson reports OpenVLA at **4-5 Hz**. Its architecture section and its
token arithmetic agree, and neither reaches 30.

**FINDING: so DOF is the lever, and it is the one nobody names.** At 80 tok/s a
7-DOF arm runs at **11.4 Hz**, a 10-DOF at 8.0, and a 26-DOF humanoid at
**3.1** -- the identical model is **3.7x** slower on the humanoid, for reasons
that have nothing to do with the humanoid being harder.

Structure: `tokens_per_second` is the rate model, `round_trip` measures what a
bin count buys, and `control_rate` inverts a throughput into a control frequency.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "21-embodied-vlas-openvla-pi0-groot"
DOF, CONTROL_HZ, BINS = 10, 30, 256
BIN_SWEEP = (2, 16, 256, 1024)
THROUGHPUTS = (30, 80)
DOF_SWEEP = (7, 10, 26)
OPENVLA_HZ = (4, 5)
ACTION = (0.1, -0.5, 0.25, -0.75, 0.9, -0.1, 0.0, 0.33, -0.67, 0.5)


def tokens_per_step(ref, dof=DOF, bins=BINS):
    return len(ref.discretize([0.0] * dof, bins))


def tokens_per_second(ref, dof=DOF, rate=CONTROL_HZ, bins=BINS):
    return tokens_per_step(ref, dof, bins) * rate


def round_trip(ref, bins):
    recovered = ref.undiscretize(ref.discretize(list(ACTION), bins), bins)
    return round(max(abs(a - b) for a, b in zip(ACTION, recovered)), 4)


def control_rate(throughput, dof=DOF):
    return round(throughput / dof, 2)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rate = tokens_per_second(ref)
    return {
        "per_step": tokens_per_step(ref), "per_second": rate,
        "by_bins": {bins: tokens_per_step(ref, bins=bins) for bins in BIN_SWEEP},
        "errors": {bins: round_trip(ref, bins) for bins in BIN_SWEEP},
        "rate_independent": len({tokens_per_step(ref, bins=b) for b in BIN_SWEEP}) == 1,
        "shortfall": {tp: round(rate / tp, 2) for tp in THROUGHPUTS},
        "achievable": {tp: control_rate(tp) for tp in THROUGHPUTS},
        "openvla": OPENVLA_HZ,
        "brackets": (control_rate(THROUGHPUTS[0]) <= OPENVLA_HZ[0]
                     and OPENVLA_HZ[1] <= control_rate(THROUGHPUTS[1])),
        "by_dof": {dof: control_rate(THROUGHPUTS[1], dof) for dof in DOF_SWEEP},
        "humanoid_penalty": round(control_rate(THROUGHPUTS[1], DOF_SWEEP[0])
                                  / control_rate(THROUGHPUTS[1], DOF_SWEEP[2]), 1),
        "target": CONTROL_HZ,
    }


def verify(result):
    errors, achievable = result["errors"], result["achievable"]
    return [
        practice.Check(
            "ANSWER: 300 tokens a second, and no -- a 7B is 3.75x to 10x too slow",
            all([result["per_step"] == 10, result["per_second"] == 300,
                 result["shortfall"] == {30: 10.0, 80: 3.75}]),
            f"{DOF} DOF is {result['per_step']} tokens a step and {CONTROL_HZ} Hz makes "
            f"{result['per_second']} a second. Against the {list(THROUGHPUTS)} tok/s range "
            f"this phase has used since Lesson 12.20 that is {result['shortfall']}x too slow",
        ),
        practice.Check(
            "FINDING: bins changes the precision and not the rate",
            all([result["by_bins"] == dict.fromkeys(BIN_SWEEP, 10),
                 result["rate_independent"],
                 errors[256] == 0.0069, errors[2] == 1.9, errors[1024] == 0.0017]),
            f"every bin count in {list(BIN_SWEEP)} emits {result['per_step']} tokens a step. "
            f"What the vocabulary buys is round-trip error -- {errors} on the lesson's own "
            "demo action -- and the rate is DOF x control_rate, in which it does not appear",
        ),
        practice.Check(
            "FINDING: control rate is throughput over DOF, and it brackets the lesson's figure",
            all([achievable == {30: 3.0, 80: 8.0}, result["brackets"],
                 result["openvla"] == (4, 5)]),
            f"at {list(THROUGHPUTS)} tok/s and {DOF} DOF the achievable rate is "
            f"{achievable} Hz, and the lesson reports OpenVLA at {result['openvla'][0]}-"
            f"{result['openvla'][1]} Hz -- inside the bracket. Its architecture section and "
            f"its token arithmetic agree, and neither reaches {result['target']}",
        ),
        practice.Check(
            "FINDING: so DOF is the lever, and it is the one nobody names",
            all([result["by_dof"] == {7: 11.43, 10: 8.0, 26: 3.08},
                 result["humanoid_penalty"] == 3.7]),
            f"at {THROUGHPUTS[1]} tok/s the achievable rate by DOF is {result['by_dof']} Hz, "
            f"so the identical model is {result['humanoid_penalty']}x slower on a 26-DOF "
            "humanoid than on a 7-DOF arm -- for reasons that have nothing to do with the "
            "humanoid being harder",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
