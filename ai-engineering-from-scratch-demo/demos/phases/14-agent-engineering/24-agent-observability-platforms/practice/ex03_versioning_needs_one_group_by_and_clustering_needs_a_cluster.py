"""Exercise 3 — versioning needs one group-by and clustering needs a cluster.

    Compare Langfuse prompt versioning against Phoenix's trace clustering.
    Which tells you what broke faster?

Reading of the exercise: "faster" is measurable if it is read as *how many
traces have to arrive before the culprit is identified*, which is the number
an on-call engineer actually waits for. Both approaches are run against the
same regression -- a prompt edit that raises the failure rate on one topic --
and the count is taken at the moment each first names the cause.

**ANSWER: versioning names `v7` at trace 8; clustering names `refund` at 42.**
On a regression caused by a prompt edit, grouping by `prompt.version` splits
the failure rate **10.0%** against **68.5%** and names the bad release as
soon as both groups hold four traces. On a regression caused by a changed
corpus -- which carries no version difference at all -- grouping by behaviour
names the affected topic at trace **42**, **5.2x** later. "Detected" here
means the verdict holds for the rest of the stream rather than flickering
once, which is the only definition an on-call engineer can act on.

**FINDING: neither works on the shipped span, because there is no version.**
`SpanEvent` has **5** fields and the lesson's own `main()` writes **0**
attributes naming a prompt or a version, so the Langfuse column does not
exist and the bisect is impossible -- which is the lesson's third pitfall,
"prompt versions not tied to traces", present in the emitter.

**FINDING: each tool is blind to the other's regression.** A corpus change
leaves every trace on `v7`, so the version group-by has **1** group at
**14.2%** and no signal at all. A prompt change hits every topic equally, so
the behavioural split never clears the threshold and returns `None` over
**400** traces. Versioning finds **1** of the **2** regressions and
clustering finds the other -- which is why Langfuse and Phoenix are listed
for different needs rather than ranked.

**FINDING: the faster tool is the one whose hypothesis was already written
down.** Versioning enumerates **2** candidates, so it needs only enough
traces to fill both groups -- **8** -- and says nothing when the cause is not
among them. Clustering must discover the partition, which costs **24**
traces and works on causes nobody labelled. Speed and coverage trade
against each other, and the answer to "which tells you faster" is "the one
you already instrumented for".

Structure: `stream()` emits traces for a scenario; `by_version()` and
`by_cluster()` are the two detectors, each returning the trace count at
which it names a cause.
"""

from __future__ import annotations

import ast
import inspect
import random

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "24-agent-observability-platforms"
TOPICS = ("refund", "shipping", "warranty", "returns")
TOTAL, MIN_GROUP = 400, 4
BASE, BAD = 0.10, 0.70


def failure_rate(scenario, version, topic):
    """A prompt edit hurts every topic; a corpus change hurts exactly one."""
    if scenario == "prompt":
        return BAD if version == "v7" else BASE
    return 0.40 if topic == "refund" else 0.0667


def stream(scenario, seed=11):
    rng, rows = random.Random(seed), []
    for index in range(TOTAL):
        version = "v7" if scenario == "corpus" or index % 2 else "v6"
        topic = TOPICS[(index // 2) % 4]
        rows.append({"version": version, "topic": topic,
                     "failed": rng.random() < failure_rate(scenario, version, topic)})
    return rows


def rates(rows, key):
    groups = {}
    for row in rows:
        calls, fails = groups.get(row[key], (0, 0))
        groups[row[key]] = (calls + 1, fails + row["failed"])
    return {name: round(100 * f / c, 1) for name, (c, f) in groups.items() if c}


def leader_of(counts, fails, gap):
    """The leading group, if the split clears `gap` and every group has enough data."""
    if len(counts) < 2 or min(counts.values()) < MIN_GROUP:
        return None
    split = {name: 100 * fails[name] / counts[name] for name in counts}
    leader = max(split, key=split.get)
    return leader if split[leader] - min(split.values()) >= gap else None


def detect(rows, key, gap=20.0):
    """The trace from which the split holds for the rest of the stream.

    An on-call engineer needs a verdict that stops flapping, so a one-window
    excursion is not a detection: the same group has to lead from that trace
    to the end of the data.
    """
    counts, fails, leaders = {}, {}, []
    for row in rows:
        counts[row[key]] = counts.get(row[key], 0) + 1
        fails[row[key]] = fails.get(row[key], 0) + row["failed"]
        leaders.append(leader_of(counts, fails, gap))
    found, run = None, None
    for index in reversed(range(len(leaders))):
        if leaders[index] is None or (run is not None and leaders[index] != run):
            return (found, run) if found else (None, None)
        found, run = index + 1, leaders[index]
    return (found, run) if found else (None, None)


def shipped_attributes(ref):
    """Read the attribute keys straight out of the lesson's own main()."""
    tree = ast.parse(inspect.getsource(ref.main))
    keys = sorted({n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
                   and isinstance(n.value, str) and "." in n.value
                   and n.value.split(".")[0] in ("gen_ai", "error")} | {"tokens"})
    return {"fields": list(ref.SpanEvent.__dataclass_fields__), "attrs": keys,
            "version_attrs": [k for k in keys if "version" in k or "prompt" in k]}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    prompt_rows, corpus_rows = stream("prompt"), stream("corpus")
    version_hit, corpus_cluster = detect(prompt_rows, "version"), detect(
        corpus_rows, "topic")
    return {
        "version_at": version_hit[0], "version_names": version_hit[1],
        "cluster_at": corpus_cluster[0], "cluster_names": corpus_cluster[1],
        "ratio": round(corpus_cluster[0] / version_hit[0], 1),
        "prompt_split": rates(prompt_rows, "version"),
        "prompt_cluster_hit": detect(prompt_rows, "topic")[0],
        "corpus_version_split": rates(corpus_rows, "version"),
        "corpus_topic_split": rates(corpus_rows, "topic"),
        "corpus_version_hit": detect(corpus_rows, "version")[0],
        "versions": len(rates(prompt_rows, "version")), **shipped_attributes(ref)}


def verify(result):
    topics = result["corpus_topic_split"]
    return [
        practice.Check(
            "ANSWER: versioning names v7 at trace 8, clustering names refund at 42",
            all([result["version_at"] == 8, result["version_names"] == "v7",
                 result["cluster_at"] == 42, result["cluster_names"] == "refund",
                 result["ratio"] == 5.2,
                 result["prompt_split"] == {"v6": 10.0, "v7": 68.5}]),
            f"on a prompt regression the version group-by splits {result['prompt_split']} "
            f"and names {result['version_names']!r} at trace {result['version_at']}; on a "
            f"corpus regression behaviour names {result['cluster_names']!r} at "
            f"{result['cluster_at']}, {result['ratio']}x later",
        ),
        practice.Check(
            "FINDING: neither works on the shipped span, because there is no version",
            all([len(result["fields"]) == 5, result["version_attrs"] == [],
                 result["attrs"] == ["error.reason", "gen_ai.output.reference_id",
                                     "gen_ai.provider.name", "gen_ai.tool.name",
                                     "tokens"]]),
            f"SpanEvent carries {result['fields']} and main writes {result['attrs']} -- "
            f"{len(result['version_attrs'])} naming a prompt or a version. The Langfuse "
            "column does not exist, which is the 'prompt versions not tied to traces' "
            "pitfall sitting in the emitter",
        ),
        practice.Check(
            "FINDING: each tool is blind to the other's regression",
            all([result["corpus_version_split"] == {"v7": 14.2},
                 result["corpus_version_hit"] is None,
                 result["prompt_cluster_hit"] is None,
                 topics["refund"] == 37.0,
                 max(v for k, v in topics.items() if k != "refund") < 10.0]),
            f"a corpus change leaves one version group at "
            f"{result['corpus_version_split']}, so the group-by returns "
            f"{result['corpus_version_hit']}; a prompt change hits every topic equally, so "
            f"the behavioural split returns {result['prompt_cluster_hit']}. The corpus "
            f"regression splits {topics}",
        ),
        practice.Check(
            "FINDING: the faster tool is the one whose hypothesis was already written",
            all([result["versions"] == 2, result["version_at"] < result["cluster_at"],
                 result["prompt_cluster_hit"] is None]),
            f"versioning enumerates {result['versions']} candidates, so it needs only "
            f"enough traces to fill both groups -- {result['version_at']} -- and is silent "
            f"otherwise. Clustering discovers the partition at {result['cluster_at']} "
            "traces and works on causes nobody labelled",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
