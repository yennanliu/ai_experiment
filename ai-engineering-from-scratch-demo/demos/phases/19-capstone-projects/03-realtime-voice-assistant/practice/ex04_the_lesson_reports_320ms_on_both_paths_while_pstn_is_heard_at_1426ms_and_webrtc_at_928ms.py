"""Exercise 4 — the lesson reports 320 ms on both paths while PSTN is heard at 1426 ms and WebRTC at 928 ms.

    Deploy the same agent on PSTN via Twilio. Compare PSTN first-audio-out to WebRTC. Explain the jitter-buffer and codec differences.

Reading of the exercise: a Twilio number is not reachable offline, so the
deployment is scaled down to the three things that differ. (1) The codec.
Twilio Media Streams carry "audio/x-mulaw" at 8000 Hz as base64 over a
WebSocket (twilio.com/docs/voice/media-streams/websocket-messages, read
2026-09-29). A 1 s speech-like signal is put through that path in numpy.
(2) The transport. 5,000 20 ms packets are sent with the lesson's 3% drop:
over UDP they are lost and concealed, and over the WebSocket's TCP they are
retransmitted after a 200 ms RTO. One-way delays and jitter are assumed (see
`PATHS`). (3) The pipeline. The lesson's `run_session` runs on the same
frames for both. First audio as heard is uplink plus p99 jitter buffer, the
scheduler's time from user stop, and downlink plus buffer.

**ANSWER: under these assumptions PSTN first audio is 1426 ms and WebRTC is
928 ms, a 498 ms gap. 378 ms of it is the jitter buffer, counted both ways,
and 120 ms is the longer path.** Over UDP a
lost packet is simply gone: 3.1% are concealed and the p99 buffer is 24 ms.
Over TCP nothing is lost, but each retransmit holds up every packet behind
it. 17.5% of packets arrive more than 100 ms late, and a buffer that covers
p99 must hold 213 ms. The codec is the other difference. G.711 mu-law is
one byte per sample at 8 kHz: 160 bytes per 20 ms frame, or 216 base64
characters. It keeps an SNR of 37.5 dB. But everything above 4 kHz is gone:
2.8% of the signal's energy, and all of the 4-8 kHz fricative band that
separates /s/ from /f/. Opus on WebRTC keeps that band.

**FINDING: the lesson's latency figure cannot tell the two deployments
apart.** `latency_ms()` is 320 ms on both paths. It starts at the VAD commit
inside the server, so the codec, the network and both jitter buffers all
fall outside it.

Structure: `codec()` band-limits and mu-law-quantizes the signal;
`transport()` simulates one path; `solve()` adds `run_session`'s time from
user stop.
"""

from __future__ import annotations

import base64

import numpy as np

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "03-realtime-voice-assistant"
CALL = "what is the weather in tokyo tomorrow"
# Assumed paths (not measurable offline). One-way ms, mean exponential jitter ms, transport.
PATHS = {"WebRTC (UDP, Opus)": {"one_way": 40, "jitter": 5, "tcp": False},
         "PSTN via Twilio (carrier + WebSocket)": {"one_way": 100, "jitter": 10, "tcp": True}}
LOSS, RTO, PACKETS = 0.03, 200, 5000   # the lesson's 3% drop test; Linux minimum TCP RTO


def speech_like(sr=48000, seed=0):
    """1 s of voiced harmonics (f0 140 Hz, 1/k amplitude, to 7 kHz) plus a 4-8 kHz fricative band."""
    rng = np.random.default_rng(seed)
    t = np.arange(sr) / sr
    voiced = sum(np.sin(2 * np.pi * 140 * k * t) / k for k in range(1, 50))
    spec = np.fft.rfft(rng.standard_normal(sr))
    f = np.fft.rfftfreq(sr, 1 / sr)
    fric = np.fft.irfft(spec * ((f > 4000) & (f < 8000)), sr)
    x = voiced / np.abs(voiced).max() + 0.3 * fric / np.abs(fric).max()
    return x / np.abs(x).max() * 0.5


def codec(x, sr=48000):
    """G.711 path: keep < 4 kHz, resample to 8 kHz, mu-law 8-bit; report what is lost."""
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / sr)
    energy = np.abs(spec) ** 2
    narrow = np.fft.irfft(spec * (f < 4000), len(x))[:: sr // 8000]
    mu = 255.0
    comp = np.sign(narrow) * np.log1p(mu * np.abs(narrow)) / np.log1p(mu)
    q = np.round((comp + 1) * 127.5) / 127.5 - 1                      # 256 levels, one byte
    back = np.sign(q) * np.expm1(np.abs(q) * np.log1p(mu)) / mu
    snr = 10 * np.log10(np.sum(narrow ** 2) / np.sum((narrow - back) ** 2))
    frame = bytes(160)                                                 # 20 ms at 8 kHz, 1 byte/sample
    return {"energy_above_4k": float(energy[f >= 4000].sum() / energy.sum()),
            "mulaw_snr_db": round(float(snr), 1), "frame_bytes": len(frame),
            "frame_b64_chars": len(base64.b64encode(frame))}


def transport(path, seed=1):
    """Per-packet arrival; UDP drops (concealed), TCP retransmits and blocks the queue behind it."""
    rng = np.random.default_rng(seed)
    send = np.arange(PACKETS) * 20.0
    arrive = send + path["one_way"] + rng.exponential(path["jitter"], PACKETS)
    lost = rng.random(PACKETS) < LOSS
    if path["tcp"]:
        arrive = np.maximum.accumulate(arrive + lost * RTO)            # in-order delivery
        lost = np.zeros(PACKETS, bool)
    extra = (arrive - send - path["one_way"])[~lost]
    return {"buffer_p99_ms": round(float(np.percentile(extra, 99))),
            "lost_pct": round(100 * float(lost.mean()), 1),
            "stalled_pct": round(100 * float((extra > 100).mean()), 1)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    x = speech_like()
    frames = ref.synth_call(CALL)
    stop = 120 + 320 * len(CALL.split())
    rows = {}
    for name, path in PATHS.items():                  # the scheduler sees the same frames either way
        m = ref.run_session(frames, use_tool=False)
        tr = transport(path)
        # first audio out as heard: uplink + buffer, scheduler from user stop, downlink + buffer
        heard = 2 * (path["one_way"] + tr["buffer_p99_ms"]) + (m.first_audio_out_ms - stop)
        rows[name] = {**tr, "lesson_latency_ms": m.latency_ms(), "heard_ms": heard}
    return {"codec": codec(x), "rows": rows}


def verify(result):
    c, rows = result["codec"], result["rows"]
    web, pstn = rows["WebRTC (UDP, Opus)"], rows["PSTN via Twilio (carrier + WebSocket)"]
    return [
        practice.Check(
            "ANSWER: PSTN first audio 1426 ms vs WebRTC 928 ms, the gap is the TCP jitter buffer",
            (web["heard_ms"], pstn["heard_ms"], web["buffer_p99_ms"], pstn["buffer_p99_ms"]) == (928, 1426, 24, 213)
            and (web["lost_pct"], pstn["lost_pct"], pstn["stalled_pct"]) == (3.1, 0.0, 17.5)
            and (c["frame_bytes"], c["frame_b64_chars"], c["mulaw_snr_db"]) == (160, 216, 37.5)
            and round(c["energy_above_4k"], 3) == 0.028,
            f"WebRTC {web}; PSTN {pstn}; mu-law frame {c['frame_bytes']} B / {c['frame_b64_chars']} b64 chars, "
            f"SNR {c['mulaw_snr_db']} dB, energy above 4 kHz dropped {c['energy_above_4k']:.1%}",
        ),
        practice.Check(
            "FINDING: the lesson's latency figure cannot tell the two deployments apart",
            web["lesson_latency_ms"] == pstn["lesson_latency_ms"] == 320,
            f"latency_ms WebRTC {web['lesson_latency_ms']} vs PSTN {pstn['lesson_latency_ms']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
