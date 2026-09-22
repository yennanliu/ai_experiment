"""Exercise 2 — truncating before redacting drops the BEGIN and keeps the key.

    Add a `redaction` step that strips lines matching `^Bearer ` or
    `password=`. Test on a fixture record.

Reading of the exercise: redaction ships, with **5** patterns covering rather
more than the two the exercise names, and `redact` is called on every
capture. So the fixture is the work -- and running one through
`_process_capture` exposes that the order is wrong: it truncates first and
redacts second, which is safe for a single-line secret and unsafe for a
multi-line one.

**ANSWER: the shipped patterns catch 5 of 6 fixture secrets, and the sixth
leaks through truncation.** A Bearer token, a `password=` assignment, an AWS
key, a Slack token and a short PEM block are all replaced. A PEM block whose
`-----BEGIN-----` line falls in the truncated middle keeps **9** lines of
key material and the `-----END-----` marker in the tail, unredacted, because
the anchor the pattern needs was deleted before the pattern ran.

**FINDING: `_process_capture` truncates and then redacts, and says so.** Its
docstring reads "Truncate first, then redact", so this is a decision rather
than an accident -- and it is the wrong one for any pattern that spans
lines. Redacting first costs one pass over the untruncated text and removes
the failure entirely: the same fixture leaks **0** lines.

**FINDING: the redaction counter undercounts by whatever was truncated
away.** Output containing **200** Bearer tokens, of which **165** fall in the
dropped middle, records `redactions={'stdout': 35}`. So the number in the
record is "secrets in the kept excerpt", not "secrets the command emitted",
and a reviewer reading **35** has no way to learn the real figure.

**FINDING: the two patterns the exercise names are the two with the loosest
anchors.** `bearer\\s+` requires whitespace, so `Bearer:ya29.x` is untouched;
the assignment pattern needs `[:=]`, so `password is hunter2` is untouched.
Both survive as plaintext in a record that reports **0** redactions --
**2** of **2** near-misses leak while all **5** canonical shapes are caught.

Structure: `capture()` drives the shipped `_process_capture`; `FIXTURE`
holds the six secrets it is measured on.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "37-runtime-feedback-loops"
SECRETS = (
    ("bearer", "Authorization: Bearer ya29.AbCdEfGhIjKl"),
    ("assignment", "password=hunter2"),
    ("aws", "AKIAIOSFODNN7EXAMPLE"),
    ("slack", "xoxb-1234567890-abcdefghij"),
    ("pem", "-----BEGIN RSA PRIVATE KEY-----\nMIIabc\n-----END RSA PRIVATE KEY-----"),
)
NEAR_MISSES = ("Bearer:ya29.AbCdEfGhIjKl", "password is hunter2")


def capture(ref, text):
    body, cut, hits = ref._process_capture(text)
    return {"body": body, "cut": cut, "hits": hits}


def leaks(body, needle):
    return needle in body


def spanning_pem(ref):
    """A PEM block whose BEGIN line lands in the truncated middle.

    head keeps lines 0-4 and tail keeps the last 30, so with 42 lines the
    dropped window is 5-11 and a BEGIN at line 8 disappears.
    """
    key_lines = [f"MII{n:04d}" for n in range(12)]
    text = "\n".join(
        [f"log line {n}" for n in range(8)]
        + ["-----BEGIN RSA PRIVATE KEY-----"]
        + key_lines
        + ["-----END RSA PRIVATE KEY-----"]
        + [f"trailing {n}" for n in range(20)])
    shipped = capture(ref, text)
    redacted_first, _ = ref.redact(text)
    corrected, _ = ref.deterministic_tail(redacted_first)
    kept = sum(line in shipped["body"] for line in key_lines)
    return {"cut": shipped["cut"], "hits": shipped["hits"],
            "kept_lines": kept, "has_begin": "BEGIN RSA" in shipped["body"],
            "has_end": "END RSA" in shipped["body"],
            "corrected_kept": sum(line in corrected for line in key_lines)}


def undercount(ref):
    """Many secrets, most of them in the part truncation deletes."""
    lines = [f"Authorization: Bearer tok{n:03d}" for n in range(200)]
    text = "\n".join(lines)
    shipped = capture(ref, text)
    _, total = ref.redact(text)
    return {"recorded": shipped["hits"], "actual": total, "cut": shipped["cut"]}


def near_miss(ref):
    rows = {}
    for text in NEAR_MISSES:
        out = capture(ref, text)
        rows[text.split()[0]] = {"hits": out["hits"], "leaked": text in out["body"]}
    return rows


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    caught = {}
    for name, text in SECRETS:
        out = capture(ref, text)
        caught[name] = out["hits"] > 0 and "REDACTED" in out["body"]
    spanning = spanning_pem(ref)
    return {
        "patterns": len(ref.REDACTION_PATTERNS),
        "fixture": len(SECRETS) + 1,
        "caught": sum(caught.values()), "per_secret": caught,
        "spanning": spanning,
        "head": ref.HEAD_LINES, "tail": ref.TAIL_LINES,
        "order_documented": "Truncate first" in (ref._process_capture.__doc__ or ""),
        "undercount": undercount(ref),
        "near_miss": near_miss(ref),
        "near_leaked": sum(row["leaked"] for row in near_miss(ref).values()),
    }


def verify(result):
    span, under, near = result["spanning"], result["undercount"], result["near_miss"]
    return [
        practice.Check(
            "ANSWER: 5 of 6 fixture secrets are caught and the sixth leaks",
            all([result["patterns"] == 5, result["caught"] == 5,
                 result["fixture"] == 6, all(result["per_secret"].values()),
                 span["kept_lines"] == 9, span["has_begin"] is False,
                 span["has_end"] is True]),
            f"the {result['patterns']} shipped patterns replace all "
            f"{result['caught']} single-capture secrets, and a PEM block whose BEGIN "
            f"line falls in the truncated middle keeps {span['kept_lines']} lines of key "
            f"material with the END marker intact ({span['has_end']}) and the anchor gone "
            f"({span['has_begin']})",
        ),
        practice.Check(
            "FINDING: _process_capture truncates and then redacts, and says so",
            all([result["order_documented"] is True,
                 span["corrected_kept"] == 0, span["kept_lines"] == 9,
                 result["head"] == 5, result["tail"] == 30]),
            f"the docstring reads 'Truncate first, then redact', so the order is a "
            f"decision. With head={result['head']} and tail={result['tail']} it keeps "
            f"{span['kept_lines']} key lines; redacting the untruncated text first and "
            f"then tailing keeps {span['corrected_kept']}",
        ),
        practice.Check(
            "FINDING: the redaction counter undercounts what was truncated away",
            all([under["recorded"] == 35, under["actual"] == 200,
                 under["cut"] == 165]),
            f"output containing {under['actual']} Bearer tokens records "
            f"{under['recorded']} redactions, because {under['cut']} lines were dropped "
            "before the patterns ran. The number in the record is secrets in the kept "
            "excerpt, not secrets the command emitted",
        ),
        practice.Check(
            "FINDING: the two patterns the exercise names have the loosest anchors",
            all([result["near_leaked"] == 2, len(near) == 2,
                 all(row["hits"] == 0 for row in near.values())]),
            f"bearer requires whitespace and the assignment pattern requires [:=], so "
            f"{near} -- {result['near_leaked']} of 2 near-misses survive as plaintext in "
            "a record reporting zero redactions, while every canonical shape is caught",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
