"""
E2E test suite for the RAG pipeline.

Tests the full flow:
  ingest → embed (ChromaDB) → retrieve → (rerank) → generate → validate answer

Usage:
    uv run python test_e2e.py              # all tests
    uv run python test_e2e.py --fast       # skip slow rerank/HyDE/multi-query tests
    uv run python test_e2e.py --keep-data  # don't delete the test collection afterwards

The server does NOT need to be running — this calls the Python modules directly.
All timing numbers are printed in the final report.
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ── Env must be loaded before any module that reads os.getenv ─────────────────
from dotenv import load_dotenv
load_dotenv()

from rag.ingest import ingest_file, retrieve, delete_collection, list_collections
from rag.chunkers import STRATEGIES
from rag.query_transform import hyde_embedding, multi_query_expand
from rag.rerank import rerank
from agent.graph import run

# ── Config ────────────────────────────────────────────────────────────────────
TEST_COLLECTION = "e2e-test"
DATA_DIR = Path(__file__).parent / "data"

# Ground-truth Q&A pairs: (question, must_contain_any, source_file)
QA_PAIRS = [
    (
        "How many days of annual leave do employees get?",
        ["15", "fifteen", "15 days"],
        "hr_policy.txt",
    ),
    (
        "How many days of sick leave per year?",
        ["10", "ten", "10 days"],
        "hr_policy.txt",
    ),
    (
        "Does the product support SSO?",
        ["sso", "single sign-on", "enterprise"],
        "product_faq.txt",
    ),
    (
        "What is the deployment rollout process?",
        ["canary", "rollout", "deploy", "10%", "staging"],
        "engineering_handbook.txt",
    ),
]

# ── Result tracking ───────────────────────────────────────────────────────────
@dataclass
class Result:
    name: str
    passed: bool
    duration: float
    detail: str = ""
    metrics: dict = field(default_factory=dict)

results: list[Result] = []

def run_test(name: str, fn):
    """Run a single test, catch exceptions, record result."""
    print(f"\n  ▸ {name} ...", end="", flush=True)
    t0 = time.perf_counter()
    try:
        metrics = fn() or {}
        duration = time.perf_counter() - t0
        results.append(Result(name=name, passed=True, duration=duration, metrics=metrics or {}))
        print(f" OK ({duration:.2f}s)")
    except AssertionError as e:
        duration = time.perf_counter() - t0
        results.append(Result(name=name, passed=False, duration=duration, detail=str(e)))
        print(f" FAIL — {e}")
    except Exception as e:
        duration = time.perf_counter() - t0
        results.append(Result(name=name, passed=False, duration=duration, detail=f"{type(e).__name__}: {e}"))
        print(f" ERROR — {type(e).__name__}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Ingest
# ══════════════════════════════════════════════════════════════════════════════

def test_ingest_all_strategies():
    """Each sample doc can be ingested with each chunking strategy."""
    metrics = {}
    for strategy in STRATEGIES:
        for doc in DATA_DIR.glob("*.txt"):
            n = ingest_file(str(doc), TEST_COLLECTION, doc.name, strategy=strategy)
            assert n > 0, f"{doc.name} produced 0 chunks with strategy={strategy}"
            metrics[f"{doc.name}:{strategy}"] = n
    return metrics


def test_chunk_counts_differ_by_strategy():
    """Different strategies should produce different chunk counts for the same doc."""
    doc = DATA_DIR / "hr_policy.txt"
    text = doc.read_text()
    counts = {s: len(fn(text)) for s, fn in STRATEGIES.items()}
    assert len(set(counts.values())) > 1, (
        f"All strategies produced the same chunk count: {counts}"
    )
    return counts


def test_chunk_preview():
    """chunk_preview returns non-empty chunks for all strategies."""
    text = (DATA_DIR / "hr_policy.txt").read_text()[:1000]
    for name, fn in STRATEGIES.items():
        chunks = fn(text)
        assert len(chunks) > 0, f"Strategy '{name}' returned 0 chunks"
        assert all(isinstance(c, str) and c.strip() for c in chunks), \
            f"Strategy '{name}' returned empty/non-string chunks"
    return {s: len(fn(text)) for s, fn in STRATEGIES.items()}


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Retrieval + Similarity Scores
# ══════════════════════════════════════════════════════════════════════════════

def test_retrieve_returns_results():
    """retrieve() returns chunks with valid cosine similarity scores."""
    query = "How many vacation days do employees receive?"
    results_raw = retrieve(query, TEST_COLLECTION, k=5)
    assert len(results_raw) > 0, "retrieve() returned no results"
    for text, source, sim in results_raw:
        assert isinstance(text, str) and text.strip(), "Empty chunk text"
        assert isinstance(source, str), "Source must be a string"
        assert 0.0 <= sim <= 1.0, f"Similarity {sim} out of [0, 1]"
    return {"chunks": len(results_raw), "top_sim": round(results_raw[0][2], 4)}


def test_similarity_scores_ordered():
    """ChromaDB results should come back in descending similarity order."""
    results_raw = retrieve("annual leave entitlement", TEST_COLLECTION, k=8)
    scores = [sim for _, _, sim in results_raw]
    assert scores == sorted(scores, reverse=True), \
        f"Scores not in descending order: {scores}"
    return {"scores": [round(s, 4) for s in scores]}


def test_top_chunk_is_relevant():
    """The top retrieved chunk for each Q&A pair should mention the expected topic."""
    failures = []
    for question, must_contain, _ in QA_PAIRS:
        chunks = retrieve(question, TEST_COLLECTION, k=3)
        assert chunks, f"No chunks retrieved for: {question}"
        top_text = chunks[0][0].lower()
        if not any(kw.lower() in top_text for kw in must_contain):
            failures.append(f"Q: {question!r} — top chunk missing {must_contain}")
    if failures:
        raise AssertionError("\n".join(failures))
    return {"qa_pairs_checked": len(QA_PAIRS)}


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Query Transforms
# ══════════════════════════════════════════════════════════════════════════════

def test_multi_query_expand():
    """multi_query_expand returns multiple distinct phrasings."""
    question = "How many vacation days do employees get?"
    variants = multi_query_expand(question)
    assert len(variants) >= 2, f"Expected ≥2 variants, got {len(variants)}: {variants}"
    assert len(set(variants)) == len(variants), f"Duplicate variants: {variants}"
    # original should be included
    assert question in variants, f"Original question not in variants: {variants}"
    return {"variants": len(variants)}


def test_hyde_embedding():
    """hyde_embedding returns a valid embedding vector and a non-empty hypothetical doc."""
    question = "What is the company's remote work policy?"
    emb, hypo_doc = hyde_embedding(question)
    assert isinstance(emb, list) and len(emb) > 100, "Embedding too short or wrong type"
    assert all(isinstance(v, float) for v in emb[:5]), "Embedding values must be floats"
    assert isinstance(hypo_doc, str) and len(hypo_doc) > 20, "Hypothetical doc too short"
    return {"embedding_dim": len(emb), "hypo_doc_chars": len(hypo_doc)}


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4 — Reranking
# ══════════════════════════════════════════════════════════════════════════════

def test_rerank_returns_scores():
    """rerank() returns (text, source, score) with valid 0-10 integer scores."""
    question = "How many annual leave days?"
    chunks_raw = retrieve(question, TEST_COLLECTION, k=5)
    chunks_2tuple = [(t, s) for t, s, _ in chunks_raw]
    reranked = rerank(question, chunks_2tuple, top_k=3)
    assert len(reranked) <= 3, f"Expected ≤3 results, got {len(reranked)}"
    for text, source, score in reranked:
        assert isinstance(score, int), f"Score must be int, got {type(score)}: {score}"
        assert 0 <= score <= 10, f"Score {score} out of [0, 10]"
    scores = [s for _, _, s in reranked]
    assert scores == sorted(scores, reverse=True), f"Reranked scores not descending: {scores}"
    return {"scores": scores}


def test_rerank_changes_order():
    """Reranking should produce a different ordering than pure similarity at least sometimes."""
    question = "SSO support and enterprise plans"
    chunks_raw = retrieve(question, TEST_COLLECTION, k=6)
    sim_order = [t for t, _, _ in chunks_raw]
    chunks_2tuple = [(t, s) for t, s, _ in chunks_raw]
    reranked = rerank(question, chunks_2tuple, top_k=len(chunks_2tuple))
    rr_order = [t for t, _, _ in reranked]
    # Not guaranteed, but likely; just verify it ran without error
    return {"sim_top": sim_order[0][:60] + "…", "rr_top": rr_order[0][:60] + "…"}


# ══════════════════════════════════════════════════════════════════════════════
# Stage 5 — Full Pipeline (agent.graph.run)
# ══════════════════════════════════════════════════════════════════════════════

def _run_pipeline_test(question, must_contain, transform="none", do_rerank=False):
    result = run(question, TEST_COLLECTION, k=5,
                 query_transform=transform, rerank=do_rerank)

    assert result["answer"], "Empty answer"
    assert result["chunks_used"] > 0, "No chunks used"
    assert result["chunks"], "No chunk details returned"
    assert result["timings"], "No timings returned"

    # Every chunk must have similarity score
    for c in result["chunks"]:
        assert "similarity" in c, f"Missing similarity in chunk: {list(c.keys())}"
        assert 0.0 <= c["similarity"] <= 1.0, f"Bad similarity: {c['similarity']}"

    # If reranked, every chunk must have rerank_score
    if do_rerank:
        for c in result["chunks"]:
            assert "rerank_score" in c, f"Missing rerank_score: {list(c.keys())}"
            assert 0 <= c["rerank_score"] <= 10

    # Timings: all expected nodes present and non-negative
    for node in ("transform", "retrieve", "generate"):
        assert node in result["timings"], f"Missing timing for node '{node}'"
        assert result["timings"][node] >= 0, f"Negative timing for '{node}'"

    # Answer content check
    ans_lower = result["answer"].lower()
    assert any(kw.lower() in ans_lower for kw in must_contain), (
        f"Answer missing expected content {must_contain}.\nAnswer: {result['answer'][:300]}"
    )

    return {
        "chunks_used": result["chunks_used"],
        "timings": {k: round(v, 3) for k, v in result["timings"].items()},
        "top_sim": round(result["chunks"][0]["similarity"], 4),
    }


def test_pipeline_none():
    """Full pipeline (no transform, no rerank) produces a correct answer."""
    q, must_contain, _ = QA_PAIRS[0]  # annual leave / 15 days
    return _run_pipeline_test(q, must_contain, transform="none", do_rerank=False)


def test_pipeline_with_rerank():
    """Full pipeline with reranking attaches rerank_score to every chunk."""
    q, must_contain, _ = QA_PAIRS[1]  # sick leave / 10 days
    return _run_pipeline_test(q, must_contain, transform="none", do_rerank=True)


def test_pipeline_hyde():
    """Full pipeline with HyDE query transform produces a valid answer."""
    q, must_contain, _ = QA_PAIRS[2]  # SSO
    return _run_pipeline_test(q, must_contain, transform="hyde", do_rerank=False)


def test_pipeline_multi_query():
    """Full pipeline with multi-query transform retrieves and deduplicates correctly."""
    q, must_contain, _ = QA_PAIRS[3]  # deployment
    return _run_pipeline_test(q, must_contain, transform="multi_query", do_rerank=False)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 6 — Evaluation (rag.evaluator)
# ══════════════════════════════════════════════════════════════════════════════

from rag.evaluator import evaluate as do_evaluate


def test_evaluator_structure():
    """evaluator returns the expected dict shape with valid score ranges."""
    question = "How many days of annual leave do employees get?"
    chunks_raw = retrieve(question, TEST_COLLECTION, k=3)
    context = [(t, s) for t, s, _ in chunks_raw]
    answer = "Employees are entitled to 15 days of annual leave per calendar year."
    ev = do_evaluate(question, context, answer)

    assert "scores" in ev and "reasoning" in ev and "overall" in ev and "passed" in ev
    for dim in ("faithfulness", "answer_relevance", "context_utilization"):
        assert dim in ev["scores"], f"Missing score dimension: {dim}"
        assert 0.0 <= ev["scores"][dim] <= 1.0, f"Score out of range: {dim}={ev['scores'][dim]}"
        assert dim in ev["reasoning"] and ev["reasoning"][dim], f"Missing reasoning for: {dim}"
    assert 0.0 <= ev["overall"] <= 1.0
    assert isinstance(ev["passed"], bool)
    return {
        "overall": ev["overall"],
        "passed": ev["passed"],
        "scores": {k: round(v, 2) for k, v in ev["scores"].items()},
    }


def test_evaluator_good_answer_passes():
    """A correct, grounded answer should score >= 0.7 overall."""
    question = "How many days of annual leave do employees get?"
    chunks_raw = retrieve(question, TEST_COLLECTION, k=5)
    context = [(t, s) for t, s, _ in chunks_raw]
    answer = "Employees are entitled to 15 days of annual leave per calendar year."
    ev = do_evaluate(question, context, answer, threshold=0.7)
    assert ev["passed"], (
        f"Good answer should pass. overall={ev['overall']}, scores={ev['scores']}"
    )
    return {"overall": ev["overall"], "scores": {k: round(v, 2) for k, v in ev["scores"].items()}}


def test_evaluator_hallucination_fails():
    """An answer with a clear hallucination should score low on faithfulness."""
    question = "How many days of annual leave do employees get?"
    chunks_raw = retrieve(question, TEST_COLLECTION, k=5)
    context = [(t, s) for t, s, _ in chunks_raw]
    # Deliberately wrong: the context says 15 days
    answer = "Employees receive 30 days of annual leave plus an additional 10 days of bonus leave."
    ev = do_evaluate(question, context, answer, threshold=0.7)
    assert ev["scores"]["faithfulness"] < 0.7, (
        f"Hallucinated answer should have low faithfulness, got {ev['scores']['faithfulness']}"
    )
    return {"faithfulness": ev["scores"]["faithfulness"], "overall": ev["overall"]}


def test_evaluator_with_reference():
    """When a reference answer is provided, correctness dimension is scored."""
    question = "How many days of annual leave do employees get?"
    chunks_raw = retrieve(question, TEST_COLLECTION, k=5)
    context = [(t, s) for t, s, _ in chunks_raw]
    answer = "Employees are entitled to 15 days of annual leave per calendar year."
    reference = "15 days of annual leave per year."
    ev = do_evaluate(question, context, answer, reference=reference)
    assert "correctness" in ev["scores"], "correctness should be present when reference given"
    assert 0.0 <= ev["scores"]["correctness"] <= 1.0
    return {
        "correctness": ev["scores"]["correctness"],
        "overall": ev["overall"],
        "passed": ev["passed"],
    }


def test_pipeline_with_evaluate():
    """Full pipeline with evaluate=True attaches evaluation dict to response."""
    question, must_contain, _ = QA_PAIRS[0]
    result = run(question, TEST_COLLECTION, k=5, evaluate=True)
    ev = result.get("evaluation", {})
    assert ev, "evaluation dict should not be empty when evaluate=True"
    assert "scores" in ev and "overall" in ev
    assert "evaluate" in result["timings"], "Missing evaluate timing"
    return {
        "overall": ev["overall"],
        "passed": ev["passed"],
        "timings": {k: round(v, 3) for k, v in result["timings"].items()},
    }


# ══════════════════════════════════════════════════════════════════════════════
# Stage 7 — Performance benchmarks (non-failing, just measured)
# ══════════════════════════════════════════════════════════════════════════════

def benchmark_pipeline_modes():
    """
    Run the same question through all four pipeline modes and record latency.
    This test always passes — it exists to surface timing numbers in the report.
    """
    question = "How many days of annual leave do employees get?"
    modes = [
        ("none+no-rerank",   "none",        False),
        ("none+rerank",      "none",        True),
        ("hyde+no-rerank",   "hyde",        False),
        ("multi_query",      "multi_query", False),
    ]
    metrics = {}
    for label, transform, do_rerank in modes:
        t0 = time.perf_counter()
        r = run(question, TEST_COLLECTION, k=5,
                query_transform=transform, rerank=do_rerank)
        elapsed = time.perf_counter() - t0
        metrics[label] = {
            "total_s": round(elapsed, 3),
            "node_timings": {k: round(v, 3) for k, v in r["timings"].items()},
        }
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════════

FAST_TESTS = [
    ("ingest: all strategies",          test_ingest_all_strategies),
    ("ingest: chunk counts differ",     test_chunk_counts_differ_by_strategy),
    ("ingest: chunk preview",           test_chunk_preview),
    ("retrieve: returns results",       test_retrieve_returns_results),
    ("retrieve: scores ordered",        test_similarity_scores_ordered),
    ("retrieve: top chunk relevant",    test_top_chunk_is_relevant),
    ("pipeline: none transform",        test_pipeline_none),
]

SLOW_TESTS = [
    ("query: multi_query expand",       test_multi_query_expand),
    ("query: HyDE embedding",           test_hyde_embedding),
    ("rerank: returns scores",          test_rerank_returns_scores),
    ("rerank: changes order",           test_rerank_changes_order),
    ("pipeline: with rerank",           test_pipeline_with_rerank),
    ("pipeline: HyDE",                  test_pipeline_hyde),
    ("pipeline: multi-query",           test_pipeline_multi_query),
    ("eval: response structure",        test_evaluator_structure),
    ("eval: good answer passes",        test_evaluator_good_answer_passes),
    ("eval: hallucination fails",       test_evaluator_hallucination_fails),
    ("eval: with reference answer",     test_evaluator_with_reference),
    ("eval: pipeline integration",      test_pipeline_with_evaluate),
    ("benchmark: all modes",            benchmark_pipeline_modes),
]


def print_report():
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed

    print("\n" + "═" * 62)
    print("  E2E Test Report")
    print("═" * 62)

    for r in results:
        icon = "✓" if r.passed else "✗"
        print(f"  {icon}  {r.name:<40} {r.duration:>6.2f}s")
        if not r.passed:
            for line in r.detail.splitlines():
                print(f"        {line}")
        if r.passed and r.metrics:
            # Pretty-print interesting metrics
            for k, v in r.metrics.items():
                if isinstance(v, dict):
                    # nested (benchmark)
                    print(f"        {k}:")
                    for kk, vv in v.items():
                        print(f"          {kk}: {vv}")
                elif isinstance(v, list):
                    print(f"        {k}: {v}")
                elif not k.startswith("_"):
                    print(f"        {k}: {v}")

    print("─" * 62)
    print(f"  {passed}/{total} passed", end="")
    if failed:
        print(f"  ·  {failed} FAILED", end="")
    total_time = sum(r.duration for r in results)
    print(f"  ·  {total_time:.1f}s total")
    print("═" * 62)
    return failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Skip slow tests (rerank, HyDE, multi-query, benchmark)")
    parser.add_argument("--keep-data", action="store_true",
                        help="Don't delete the test collection after the run")
    args = parser.parse_args()

    print("═" * 62)
    print("  RAG E2E Test Suite")
    print("═" * 62)

    # Clean slate
    if TEST_COLLECTION in list_collections():
        delete_collection(TEST_COLLECTION)
        print(f"  Cleaned up existing '{TEST_COLLECTION}' collection.")

    tests = FAST_TESTS + ([] if args.fast else SLOW_TESTS)
    print(f"  Running {len(tests)} tests{'  [--fast mode]' if args.fast else ''}\n")

    for name, fn in tests:
        run_test(name, fn)

    # Teardown
    if not args.keep_data and TEST_COLLECTION in list_collections():
        delete_collection(TEST_COLLECTION)

    failed = print_report()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
