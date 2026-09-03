"""Cassettes replace hand-written simulations; coverage replaces a hand-written table."""

import json
from pathlib import Path

import pytest

from harness import coverage
from harness.cassette import (Cassette, CassetteError, PRICE_PER_MTOK, Recording,
                              request_key)
from harness.tiers import LIVE, REPLAY

REQUEST = {"model": "claude-opus-5", "max_tokens": 16,
           "messages": [{"role": "user", "content": "hello"}]}


def recorder(text="recorded once", model="claude-opus-5"):
    def record(request, *, key):
        return Recording(key=key, request=request, text=text, model=model,
                         stop_reason="end_turn",
                         usage={"input_tokens": 1000, "output_tokens": 200},
                         recorded_at="2026-09-02T12:00:00+00:00")
    return record


# --------------------------------------------------------------------------
# cassettes
# --------------------------------------------------------------------------


def test_the_same_request_always_hashes_to_the_same_key():
    assert request_key(REQUEST) == request_key(dict(reversed(list(REQUEST.items()))))
    assert request_key(REQUEST) != request_key({**REQUEST, "max_tokens": 17})


def test_record_then_replay_returns_identical_text(tmp_path):
    tape = Cassette.load(tmp_path / "t.json")
    live = tape.complete(REQUEST, run_mode=LIVE, recorder=recorder())
    tape.save()

    replayed = Cassette.load(tmp_path / "t.json").complete(REQUEST, run_mode=REPLAY)
    assert replayed.text == live.text
    assert replayed.recorded_at == live.recorded_at
    assert replayed.model == live.model


def test_replaying_a_request_that_was_never_recorded_fails_loudly(tmp_path):
    """Silently falling through to a live call would bill CI on every push."""
    tape = Cassette.load(tmp_path / "t.json")
    with pytest.raises(CassetteError, match="no recording for this request"):
        tape.complete(REQUEST, run_mode=REPLAY)


def test_a_credential_never_reaches_a_committed_tape(tmp_path):
    tape = Cassette.load(tmp_path / "t.json")
    tape.complete({**REQUEST, "api_key": "sk-ant-secret"}, run_mode=LIVE,
                  recorder=recorder())
    tape.save()
    written = (tmp_path / "t.json").read_text()
    assert "sk-ant-secret" not in written
    assert "<redacted>" in written


def test_a_tape_records_which_model_said_it_and_when(tmp_path):
    """Provenance is what makes staleness visible instead of invisible."""
    tape = Cassette.load(tmp_path / "t.json")
    tape.complete(REQUEST, run_mode=LIVE, recorder=recorder(model="claude-sonnet-5"))
    assert tape.models == ["claude-sonnet-5"]
    assert tape.recorded_dates == ["2026-09-02"]


def test_cost_is_priced_from_the_model_that_answered(tmp_path):
    tape = Cassette.load(tmp_path / "t.json")
    recording = tape.complete(REQUEST, run_mode=LIVE, recorder=recorder())
    rate_in, rate_out = PRICE_PER_MTOK["claude-opus-5"]
    assert recording.cost_usd() == pytest.approx((1000 * rate_in + 200 * rate_out) / 1e6)


def test_an_unknown_model_reports_no_price_rather_than_a_wrong_one(tmp_path):
    tape = Cassette.load(tmp_path / "t.json")
    recording = tape.complete(REQUEST, run_mode=LIVE,
                              recorder=recorder(model="claude-from-the-future"))
    assert recording.cost_usd() is None
    assert tape.total_cost_usd() == 0.0


def test_saving_is_a_no_op_when_nothing_changed(tmp_path):
    path = tmp_path / "t.json"
    tape = Cassette.load(path)
    tape.complete(REQUEST, run_mode=LIVE, recorder=recorder())
    tape.save()
    before = path.stat().st_mtime_ns
    Cassette.load(path).save()
    assert path.stat().st_mtime_ns == before


def test_a_tape_from_a_future_format_is_refused(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"version": 99, "interactions": []}))
    with pytest.raises(CassetteError, match="cassette version"):
        Cassette.load(path)


# --------------------------------------------------------------------------
# coverage
# --------------------------------------------------------------------------


@pytest.fixture
def fake_repos(tmp_path):
    """A two-lesson reference tree and a demos tree, both on disk."""
    reference = tmp_path / "ref"
    for lesson in ("01-alpha/01-one", "01-alpha/02-two"):
        (reference / "phases" / lesson / "docs").mkdir(parents=True)
        (reference / "phases" / lesson / "docs" / "en.md").write_text(f"# {lesson}")
    (reference / "ROADMAP.md").write_text("roadmap")

    demos = tmp_path / "demos"
    built = demos / "phases/01-alpha/01-one"
    built.mkdir(parents=True)
    return reference, demos, built


def manifest_for(lesson, *, doc_sha=None):
    body = [f"lesson: phases/{lesson}", "title: t", "tier: T0",
            "runtime_seconds: 10", "deps_group: math", "proves: x"]
    if doc_sha is not None:
        body += [f"reference_doc: phases/{lesson}/docs/en.md",
                 f"reference_doc_sha256: {doc_sha}"]
    return "\n".join(body) + "\n"


def test_coverage_is_a_diff_of_two_trees(fake_repos):
    reference, demos, built = fake_repos
    (built / "demo.yaml").write_text(manifest_for("01-alpha/01-one"))

    statuses = coverage.survey(demos, reference)
    assert coverage.summary(statuses) == {"built": 1, "stale": 0, "missing": 1}
    assert "| **all** | **2** | **1** |" in coverage.phase_table(statuses)


def test_a_lesson_whose_doc_changed_is_flagged_stale(fake_repos):
    reference, demos, built = fake_repos
    (built / "demo.yaml").write_text(
        manifest_for("01-alpha/01-one", doc_sha="0000deadbeef0000")
    )
    statuses = coverage.survey(demos, reference)
    stale = [s for s in statuses if s.state == coverage.STALE]
    assert len(stale) == 1
    assert "doc changed" in stale[0].note


def test_a_matching_doc_hash_is_not_flagged(fake_repos):
    reference, demos, built = fake_repos
    lesson = "01-alpha/01-one"
    actual = coverage.doc_hash(reference / "phases" / lesson / "docs" / "en.md")
    (built / "demo.yaml").write_text(manifest_for(lesson, doc_sha=actual))
    assert coverage.summary(coverage.survey(demos, reference))["stale"] == 0


def test_a_demo_for_a_lesson_that_no_longer_exists_is_surfaced(fake_repos):
    reference, demos, _ = fake_repos
    orphan = demos / "phases/01-alpha/99-deleted"
    orphan.mkdir(parents=True)
    (orphan / "demo.yaml").write_text(manifest_for("01-alpha/99-deleted"))

    statuses = coverage.survey(demos, reference)
    orphans = [s for s in statuses if "no such lesson" in s.note]
    assert len(orphans) == 1
