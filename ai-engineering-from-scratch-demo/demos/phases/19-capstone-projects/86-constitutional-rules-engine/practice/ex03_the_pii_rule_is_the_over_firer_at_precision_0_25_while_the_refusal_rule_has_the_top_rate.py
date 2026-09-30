"""Exercise 3 — the PII rule is the over-firer at precision 0.25, while the refusal rule has the top violation rate.

    Add a metrics endpoint that, given a corpus of drafts, returns the per-rule violation rate so the team can see which rule is over-firing.

Reading of the exercise: `POST /metrics` on a stdlib `ThreadingHTTPServer`
(127.0.0.1, ephemeral port, shut down in `finally`) takes
`{"drafts": [{"text": ..., "expected": [...]}]}` and runs the lesson's
shipped `Engine` over it. Per rule it returns applicable, fired, rate
(fired / drafts) and, when drafts carry reviewer labels, tp/fp/fn and
precision. A violation rate alone cannot say "over-firing": a rule that is
often right also fires often. So the endpoint names the rule with the most
false fires, and only when labels are given. The corpus is 20 harmless
drafts: refusals, phrases that look like refusals, contact details,
10-digit IDs, code, citations, an internal name, and clean text.

**ANSWER: the endpoint names `no-pii-in-examples` as the over-firer: it
fires on 4 of 20 drafts (rate 0.20), 3 of them false, precision 0.25.**
The three false fires are an order number, an epoch timestamp and a build
ID, all 10-digit numbers that the phone regex matches. Code, citation,
internal-name and length rules have 0 false fires. Without labels the same
rates come back and no over-firer is named. Malformed JSON gets a 400.

| rule | rate | fired | false | precision |
|---|---:|---:|---:|---:|
| no-empty-refusal | 0.25 | 5 | 2 | 0.60 |
| no-pii-in-examples | 0.20 | 4 | 3 | 0.25 |
| the other three that fire | 0.05 | 1 | 0 | 1.00 |

**FINDING: the highest raw violation rate points at the wrong rule.**
`no-empty-refusal` tops the rate column at 0.25, but 3 of its 5 fires are
real refusals. Its 2 false fires are "I cannot recommend this library
enough" and "I will not bore you with details". Ranking by rate alone
sends the team to tune it first.

**FINDING: `no-pii-in-examples` fires on 4/4 drafts it applies to, and
misses the one real email outside them.** "Reach the author at
dana@acme.test" has no "example" or "sample" word, so the rule is
`not_applicable`. The rule over-fires and under-fires at once.
"""
import json
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "86-constitutional-rules-engine"
REFUSAL, PII = "no-empty-refusal", "no-pii-in-examples"
CODE_DRAFT = "Here is the code:\n```python\ndef add(a, b):\n    return a + b\n```\nLet me know."
CORPUS = [      ("I cannot help with that question.", [REFUSAL]), ("I refuse to answer that.", [REFUSAL]),
    ("Unable to help with this request.", [REFUSAL]), ("Example user: lee@example.com. Look them up.", [PII]),
    ("Reach the author at dana@acme.test for access.", [PII]), (CODE_DRAFT, ["end-with-runnable-or-assumption"]),
    ("According to the docs it works.", ["cite-when-asserting-fact"]),
    ("Use the internal-only adapter for the database call.", ["no-internal-library-leak"]),
] + [(t, []) for t in (
    "I cannot recommend this library enough; it saved us a week.", "I will not share that, but try the docs instead.",
    "I will not bore you with details: run pip install requests.", "Example: order 1234567890 shipped on time.",
    "Sample epoch timestamp: 1727654400.", "For example, build 2024061512 passed all tests.",
    "According to the changelog (v2.1), it works.", "```python\nprint('hi')\n```", "The meeting moved to Thursday.",
    "Here is a haiku about autumn leaves drifting onto a still pond surface.", "Restart the service, then check the logs.",
    "Here are three ways to speed up the build.")]
KEYS = ("applicable", "fired", "tp", "fp", "fn")


def tally(row, status, expected):
    fired = status == "violation"
    hits = (status != "not_applicable", fired) + (() if expected is None else
                                                  (fired and expected, fired and not expected, expected and not fired))
    for key, hit in zip(KEYS, hits):
        row[key] += hit


def metrics(engine, items):
    """items: [{"text": str, "expected": [rule names] (optional)}] -> per-rule counts; no labels, no tp/fp/fn."""
    rows = {r["name"]: dict.fromkeys(KEYS, 0) for r in engine.rules()}
    for item in items:
        labels = item.get("expected")
        for res in engine.evaluate(item["text"]).results:
            tally(rows[res.rule_name], res.status, None if labels is None else res.rule_name in labels)
    for row in rows.values():
        row["rate"] = round(row["fired"] / max(len(items), 1), 4)
        row["precision"] = round(row["tp"] / row["fired"], 4) if row["fired"] else None
    worst = max(rows, key=lambda k: rows[k]["fp"])
    return {"drafts": len(items), "rules": rows, "most_false_fires": worst if rows[worst]["fp"] else None}


def make_handler(engine):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - http.server naming
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                code, payload = (200, metrics(engine, body["drafts"])) if self.path == "/metrics" else (404, {})
            except (ValueError, KeyError, TypeError) as exc:
                code, payload = 400, {"error": f"{type(exc).__name__}: {exc}"}
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        log_message = lambda self, *args: None  # noqa: E731 - keep the grading output quiet
    return Handler


def post(port, body):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", data=body, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.request.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def solve():
    # main.py does `from yaml_subset import load_yaml`; register the lesson's module for that import only
    with mock.patch.dict(sys.modules, {"yaml_subset": parity.load_reference(PHASE, LESSON, "yaml_subset")}):
        main = parity.load_reference(PHASE, LESSON, "main")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(main.Engine()))
    threading.Thread(target=server.serve_forever, daemon=True).start()  # shutdown() waits for it to exit
    try:
        port, labelled = server.server_address[1], [{"text": t, "expected": e} for t, e in CORPUS]
        return {"labelled": post(port, json.dumps({"drafts": labelled}).encode()),
                "unlabelled": post(port, json.dumps({"drafts": [{"text": t} for t, _ in CORPUS]}).encode()),
                "bad": post(port, b"{not json")[0]}
    finally:
        server.shutdown()
        server.server_close()


def verify(result):
    (code, m), (code2, m2) = result["labelled"], result["unlabelled"]
    rows, rates = m["rules"], [{k: v["rate"] for k, v in x["rules"].items()} for x in (m, m2)]
    row = {k: tuple(rows[k][c] for c in KEYS + ("rate", "precision")) for k in rows}
    return [
        practice.Check(
            "ANSWER: POST /metrics over 20 drafts names no-pii-in-examples as over-firing: 4 fires, 3 false, precision 0.25",
            (code, m["drafts"], m["most_false_fires"], row[PII][1:], [v["fp"] for v in rows.values()]) ==
            (200, 20, PII, (4, 1, 3, 1, 0.2, 0.25), [2, 0, 3, 0, 0, 0])
            and (code2, m2["most_false_fires"], rates[0] == rates[1], result["bad"]) == (200, None, True, 400),
            f"(applicable, fired, tp, fp, fn, rate, precision): {row}; bad JSON -> {result['bad']}",
        ),
        practice.Check(
            "FINDING: the highest raw violation rate is no-empty-refusal (0.25, 3 of 5 fires real), not the over-firer",
            (max(rows, key=lambda k: rows[k]["rate"]), row[REFUSAL][1:]) == (REFUSAL, (5, 3, 2, 0, 0.25, 0.6)),
            f"refusal fired/tp/fp/fn/rate/precision {row[REFUSAL][1:]}",
        ),
        practice.Check(
            "FINDING: no-pii-in-examples fires on 4/4 drafts it applies to and misses the 1 email outside one",
            (row[PII][:2], row[PII][4]) == ((4, 4), 1),
            f"applicable/fired {row[PII][:2]}, missed {row[PII][4]}: an email in a draft without example/sample",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
