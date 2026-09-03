"""Assertions on Phase 11 / Lesson 01: the patterns, the scorer, and the tape.

The checks that need a recorded response skip cleanly until someone runs
`DEMO_MODE=live` once. Everything else -- the prompt builders, the request
adaptation, and the scorer's behaviour on real text -- runs offline.
"""

import json
import sys
from pathlib import Path

import pytest

DEMO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO))
sys.path.insert(0, str(DEMO.parents[3]))

from harness.cassette import Cassette  # noqa: E402
from harness.parity import load_reference  # noqa: E402
from run import CASES, CASSETTE, LESSON, MAX_TOKENS, MODEL, adapt_request  # noqa: E402

ref = load_reference(LESSON)

needs_tape = pytest.mark.skipif(
    not CASSETTE.exists(), reason="no cassette recorded yet (DEMO_MODE=live to cut one)"
)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["pattern"])
def test_every_case_renders_through_the_lessons_builder(case):
    """The claim: a pattern plus its variables produces a complete prompt."""
    prompt = ref.build_prompt(case["pattern"], case["variables"])
    assert prompt["pattern"] == case["pattern"]
    assert prompt["user"] and "{" not in prompt["user"], "a template slot went unfilled"
    assert prompt["system"]


def test_a_missing_variable_is_rejected_not_silently_rendered():
    """Half-rendered prompts are the classic failure this check exists to catch."""
    with pytest.raises(ValueError, match="Missing variables"):
        ref.build_prompt("persona", {"role": "an engineer"})


def test_adapt_request_drops_the_parameter_the_current_api_rejects():
    """`temperature` was removed on current models; sending it is a 400."""
    prompt = ref.build_prompt(CASES[0]["pattern"], CASES[0]["variables"])
    lesson_request = ref.format_anthropic_request(prompt)
    assert "temperature" in lesson_request, "the lesson stopped setting temperature"

    request, dropped = adapt_request(lesson_request)
    assert dropped == lesson_request["temperature"]
    assert "temperature" not in request
    assert request["model"] == MODEL
    assert request["max_tokens"] == MAX_TOKENS
    assert request["messages"] and request["system"]


def test_the_scorer_actually_discriminates():
    """A scorer that passes everything proves nothing about a prompt."""
    criteria = {"max_words": 5, "required_keywords": ["index"], "expected_format": "json"}
    good = ref.score_response('{"index": 1}', criteria)
    bad = ref.score_response("a much longer answer with no such term " * 3, criteria)
    assert good["composite_score"] > bad["composite_score"]
    assert good["format_valid"] and not bad["format_valid"]


def test_a_cassette_replays_the_same_bytes_twice(tmp_path):
    """Replay must be deterministic -- that is the whole point of committing it."""
    from harness.cassette import Recording
    from harness.tiers import LIVE

    tape = Cassette.load(tmp_path / "t.json")
    request = {"model": MODEL, "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}

    def fake(req, *, key):
        return Recording(key=key, request=req, text="recorded once", model=MODEL,
                         stop_reason="end_turn",
                         usage={"input_tokens": 10, "output_tokens": 3},
                         recorded_at="2026-09-02T00:00:00+00:00")

    first = tape.complete(request, run_mode=LIVE, recorder=fake)
    tape.save()

    reloaded = Cassette.load(tmp_path / "t.json")
    assert reloaded.complete(request).text == first.text == "recorded once"
    assert reloaded.complete(request).recorded_at == first.recorded_at


def test_a_changed_prompt_misses_the_tape_instead_of_replaying_a_stale_answer(tmp_path):
    from harness.cassette import CassetteError

    tape = Cassette.load(tmp_path / "t.json")
    with pytest.raises(CassetteError, match="no recording for this request"):
        tape.complete({"model": MODEL, "messages": [{"role": "user", "content": "new"}]})


@needs_tape
@pytest.mark.parametrize("case", CASES, ids=lambda c: c["pattern"])
def test_the_real_response_satisfies_the_criteria_the_prompt_asked_for(case):
    """What the demo is for: the pattern worked on a model that really answered."""
    tape = Cassette.load(CASSETTE)
    prompt = ref.build_prompt(case["pattern"], case["variables"])
    request, _ = adapt_request(ref.format_anthropic_request(prompt))

    scores = ref.score_response(tape.complete(request).text, case["criteria"])
    assert scores["composite_score"] >= 0.75, scores
    assert not scores.get("forbidden_violations")


@needs_tape
def test_the_tape_records_which_model_said_it_and_when():
    """A cassette without provenance cannot be judged stale."""
    tape = Cassette.load(CASSETTE)
    assert tape.models, "the tape records no model id"
    assert tape.recorded_dates, "the tape records no date"
    for recording in tape.interactions.values():
        assert recording.stop_reason != "refusal"
        assert recording.usage["output_tokens"] > 0
        # A committed tape must never carry a credential.
        assert "sk-ant" not in json.dumps(recording.request)
