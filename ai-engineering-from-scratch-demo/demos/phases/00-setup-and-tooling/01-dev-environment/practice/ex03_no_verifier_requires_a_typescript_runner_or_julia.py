"""Exercise 3 — no verifier requires a TypeScript runner or Julia.

    Write a "hello world" in all four languages and run each one

Reading of the exercise: the four languages are the ones the lesson names --
Python, TypeScript, Rust and Julia. All four programs are written to a temp
directory; only the Python one is run, because whether `node`, `rustc` or
`julia` exist is a fact about this machine, not about the lesson. What can be
checked deterministically is whether the lesson's own three verifiers would
have told the reader each one is runnable.

**ANSWER: four programs, four commands, one guaranteed to run.** Python runs
here and prints `hello world`. TypeScript needs a TypeScript runner (`npx tsx
hello.ts`), Rust needs `rustc`, Julia needs `julia`.

**FINDING: the lesson never installs a TypeScript runner.** Step 3 installs
Node and pnpm; `tsx` appears **0** times in the lesson text and **0** times in
`verify.py`. Only `verify.ts` probes it, as `npx -y tsx --version` -- and
`-y` makes npx download tsx from the registry without asking, so the probe
installs the thing it is checking for and can only fail offline.

**FINDING: across the three verifiers, Julia and the TypeScript runner are
required 0 times.** Python is required by **3** of 3. Rust (`rustc`) is
required by **1** -- `main.rs` itself, which has to be compiled by `rustc`
before it can run, so that check cannot fail for anyone who built it. A green
result from any of the three says nothing about two of the four programs.

**FINDING: `verify.ts` probes a runtime the lesson never mentions.** It
checks for `deno`, which appears **0** times in the lesson text, and does not
check for `julia` at all.

**CONTROL: the Python program really runs.** Under this interpreter it exits
**0** and prints exactly `hello world`.

Structure: `PROGRAMS` holds the four sources and their run commands;
`required_by` reads which tools each verifier treats as required.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import tempfile

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "01-dev-environment"
PROGRAMS = {
    "python": ("hello.py", 'print("hello world")\n', "python3 hello.py"),
    "typescript": ("hello.ts", 'console.log("hello world");\n', "npx tsx hello.ts"),
    "rust": ("hello.rs", 'fn main() { println!("hello world"); }\n', "rustc hello.rs && ./hello"),
    "julia": ("hello.jl", 'println("hello world")\n', "julia hello.jl"),
}
TOOL = {"python": "python", "typescript": "tsx", "rust": "rustc", "julia": "julia"}


def required_by(ref, code_dir):
    """{verifier: required tool names, lower-cased}."""
    ts = (code_dir / "verify.ts").read_text(encoding="utf-8")
    rs = (code_dir / "main.rs").read_text(encoding="utf-8")
    ts_req = re.findall(r'name: "([^"]+)",\s*required: true', ts)
    return {
        "verify.py": " ".join(ref.ROUTES["beginner"].required),
        "verify.ts": " ".join(ts_req).lower(),
        "main.rs": " ".join(re.findall(r'Check::new\("[^"]+", "([^"]+)", false\)', rs)),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "verify")
    code_dir = parity.lesson_dir(PHASE, LESSON) / "code"
    doc = parity.doc_text(PHASE, LESSON).lower()
    ts = (code_dir / "verify.ts").read_text(encoding="utf-8")
    py = (code_dir / "verify.py").read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as root:
        for name, source, _ in PROGRAMS.values():
            (pathlib.Path(root) / name).write_text(source, encoding="utf-8")
        done = subprocess.run([sys.executable, "hello.py"], cwd=root, capture_output=True,
                              text=True, timeout=30)
        written = sorted(p.name for p in pathlib.Path(root).iterdir())
    req = required_by(ref, code_dir)
    return {
        "written": written, "py_code": done.returncode, "py_out": done.stdout,
        "tsx_doc": doc.count("tsx"), "tsx_py": py.count("tsx"),
        "npx_y": '"npx", ["-y", "tsx"' in ts,
        "required": {lang: sum(TOOL[lang] in tools for tools in req.values())
                     for lang in PROGRAMS},
        "deno_ts": "deno" in ts, "deno_doc": doc.count("deno"), "julia_ts": "julia" in ts,
    }


def verify(result):
    req = result["required"]
    return [
        practice.Check(
            "ANSWER: four programs written, the Python one run",
            len(result["written"]) == 4 and result["py_code"] == 0,
            f"wrote {result['written']}; run commands are "
            + "; ".join(cmd for _, _, cmd in PROGRAMS.values()),
        ),
        practice.Check(
            "FINDING: the lesson installs no TypeScript runner, and verify.ts downloads one",
            result["tsx_doc"] == 0 and result["tsx_py"] == 0 and result["npx_y"],
            f"tsx appears {result['tsx_doc']} times in the lesson and {result['tsx_py']} times "
            "in verify.py; verify.ts probes it as `npx -y tsx --version`, and -y installs tsx "
            "from the registry unprompted, so the probe fetches what it claims to check",
        ),
        practice.Check(
            "FINDING: Julia and the TypeScript runner are required by 0 of 3 verifiers",
            req == {"python": 3, "typescript": 0, "rust": 1, "julia": 0},
            f"required counts across verify.py, verify.ts and main.rs: {req}. The single rustc "
            "requirement is main.rs's own, which needs rustc to be built at all",
        ),
        practice.Check(
            "FINDING: verify.ts checks deno, which the lesson never mentions, and not julia",
            result["deno_ts"] and result["deno_doc"] == 0 and not result["julia_ts"],
            f"deno appears {result['deno_doc']} times in the lesson text; verify.ts probes it "
            "and has no julia probe",
        ),
        practice.Check(
            "CONTROL: the Python hello world prints exactly that",
            result["py_out"] == "hello world\n",
            f"exit {result['py_code']}, stdout {result['py_out']!r}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
