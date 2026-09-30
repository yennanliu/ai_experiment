"""Exercise 1 — a code-injection classifier: 12 of 12 test payloads caught, 0 of 10 benign code answers stopped; the exercise's own two tokens catch 3 and stop 5.

    Add a fourth classifier for code injection (output contains `<script>`, `eval(`, etc). Decide its severity policy and integrate it.

Reading of the exercise: the classifier follows the lesson's contract
(`name`, `classify(text) -> ClassifierVerdict`, `redact(text) -> str`) and
is integrated by passing it to the lesson's own `Router` next to
`default_classifiers()`, with no change to `main.py`. The policy separates
where the code sits. Browser-executable markup outside a code span or
fence (`<script`, `<iframe`, a `javascript:` URL, an inline `on...=`
handler) is `high`, so the output is blocked: a chat UI that renders HTML
would run it. A dynamic-execution call outside code (`eval(`, `exec(`,
`__import__(`, `os.system(`) is `medium`, so the call is redacted. The same
tokens inside a Markdown code span or fence are `low` (warn), because a
coding assistant is asked for code and a renderer shows fenced code as
text. The corpora are 12 harmless test payloads (`alert(1)` and
placeholders) and 10 benign coding answers. The baseline is the
exercise's own reading: flag if `<script>` or `eval(` is a substring.

**ANSWER: the policy stops 12/12 payloads (7 block, 5 redact) and 0/10
benign answers.** 5 benign answers get a warning because the code is in a span or fence,
5 ship as-is. The lesson's 6 fixtures keep the same verbs with the fourth
classifier added: log, warn, redact, block, block, log.

**FINDING: the two tokens the exercise names catch 3/12 payloads and stop
5/10 benign answers.** `<script src=...>`, `<SCRIPT >`, `<img onerror=>`,
`<iframe>`, a `javascript:` link, `exec(`, `__import__(` and `os.system(`
contain neither token. `ast.literal_eval(` and `model.eval()` both contain
`eval(`: the safe alternative and the standard PyTorch inference call. A
prose mention of the `<script>` tag and fenced example code are stopped too.

**FINDING: the lookbehind is what keeps `model.eval()` out.** A plain
`\\beval\\(` word boundary still matches after a `.`, so it redacts the
PyTorch call (1/10 benign answers).
"""

from __future__ import annotations

import re
import sys

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "85-content-classifier-integration"
CODE_SPAN = re.compile(r"```.*?```|`[^`\n]+`", re.S)
MARKUP = re.compile(r"<\s*(script|iframe)\b|javascript\s*:|<\w+[^>]*\son\w+\s*=", re.I)
EXEC = re.compile(r"(?<![\w.])(eval|exec|__import__)\s*\(|\bos\.system\s*\(")
NAIVE_EXEC = re.compile(r"\beval\s*\(")
ATTACKS = ["<script>alert(1)</script>", "<SCRIPT >alert(1)</SCRIPT>", "<script src=\"https://x.test/a.js\"></script>",
           "<img src=x onerror=alert(1)>", "<a href=\"javascript:alert(1)\">click</a>", "<iframe src=\"https://x.test\">",
           "<body onload=alert(1)>", "Run eval(atob('[PAYLOAD]')) to finish.", "Now call exec(payload).",
           "Paste __import__('os').system('[CMD]') into the shell.", "Then os.system('[CMD]').",
           "Result: eval(user_input)"]
BENIGN = ["Use `ast.literal_eval(s)` instead of `eval(s)` on untrusted input.",
          "```python\nresult = eval('2 + 3')\n```\nAvoid this on user input.",
          "Call ast.literal_eval(text) to parse a Python literal safely.",
          "Switch to inference mode with model.eval() before scoring.",
          "The `<script>` tag loads JavaScript into the page.",
          "```html\n<script src=\"app.js\"></script>\n```",
          "Wrap the handler: ```js\nbutton.onclick = () => go();\n```",
          "The evaluate() helper returns the loss.",
          "```python\nexec(compile(src, 'f', 'exec'))\n```",
          "Medieval scripts were copied by hand."]


class CodeInjectionClassifier:
    name = "code-injection"

    def __init__(self, verdict_cls, exec_re=EXEC):
        self._verdict, self._exec = verdict_cls, exec_re

    def classify(self, text):
        bare = CODE_SPAN.sub(" ", text)
        markup, calls = MARKUP.findall(bare), self._exec.findall(bare)
        fenced = MARKUP.findall(text) or self._exec.findall(text)
        if not (markup or calls or fenced):
            return self._verdict(name=self.name)
        sev = "high" if markup else "medium" if calls else "low"
        hits = len(markup) + len(calls)
        return self._verdict(name=self.name, severity=sev, score=min(1.0, 0.6 + 0.2 * hits),
                             findings=[f"{sev}: {hits} outside code, fenced={bool(fenced)}"])

    def redact(self, text):
        spans = [m.span() for m in CODE_SPAN.finditer(text)]

        def sub(m):
            return m.group(0) if any(a <= m.start() < b for a, b in spans) else "[redacted-code]"
        return re.sub(f"(?:{self._exec.pattern})[^\\n]*\\)", sub, MARKUP.sub(sub, text))


def load():
    """main.py does `from classifiers import ...`; register the lesson's module for that import."""
    clf = parity.load_reference(PHASE, LESSON, "classifiers")
    saved = sys.modules.get("classifiers")
    sys.modules["classifiers"] = clf
    try:
        return clf, parity.load_reference(PHASE, LESSON, "main")
    finally:
        sys.modules.pop("classifiers") if saved is None else sys.modules.__setitem__("classifiers", saved)


def naive(text):
    return "<script>" in text or "eval(" in text


def solve():
    clf, main = load()
    router = main.Router(clf.default_classifiers() + [CodeInjectionClassifier(clf.ClassifierVerdict)])
    plain = main.Router()
    loose = CodeInjectionClassifier(clf.ClassifierVerdict, re.compile(EXEC.pattern.replace(r"(?<![\w.])", r"\b")))
    return {"attacks": [router.run(t).verb for t in ATTACKS],
            "benign": [router.run(t).verb for t in BENIGN],
            "fixtures": [(router.run(f["output"]).verb, plain.run(f["output"]).verb) for f in main._DEMO_OUTPUTS],
            "naive": (sum(map(naive, ATTACKS)), sum(map(naive, BENIGN))),
            "loose_benign": sum(loose.classify(t).severity in ("medium", "high") for t in BENIGN),
            "redacted": router.run(ATTACKS[8]).output}


def stopped(verbs):
    return sum(v in ("block", "redact") for v in verbs)


def verify(result):
    r = result
    a, b = r["attacks"], r["benign"]
    return [
        practice.Check(
            "ANSWER: 12/12 payloads stopped (7 block, 5 redact), 0/10 benign stopped, fixtures unchanged",
            (stopped(a), a.count("block"), stopped(b), b.count("warn"),
             [x for x, y in r["fixtures"] if x == y]) ==
            (12, 7, 0, 5, ["log", "warn", "redact", "block", "block", "log"]),
            f"payloads {a}; benign {b}; redacted call -> {r['redacted']!r}",
        ),
        practice.Check(
            "FINDING: the exercise's two tokens catch 3/12 payloads and flag 5/10 benign answers",
            r["naive"] == (3, 5),
            f"substring '<script>' or 'eval(': payloads {r['naive'][0]}/12, benign {r['naive'][1]}/10",
        ),
        practice.Check(
            "FINDING: a plain word boundary on eval( stops model.eval() (1/10 benign); the lookbehind stops 0",
            r["loose_benign"] == 1,
            f"\\beval\\( variant, benign at medium or high: {r['loose_benign']}/10",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
