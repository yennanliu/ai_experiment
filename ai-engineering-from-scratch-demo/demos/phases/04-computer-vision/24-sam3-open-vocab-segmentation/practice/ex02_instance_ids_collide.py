"""Exercise 2 — instance ids collide.

    **(Medium)** Build a "click-to-include / click-to-exclude" UI on top of SAM
    3: a text prompt returns candidate instances; user clicks keep which ones
    count as positive. Output the final concept set as JSON.

Reading of the exercise: a click-to-include UI is a selection keyed on something,
so the exercise is really asking what identifies an instance. The lesson's own
`run_multi_concept` answers badly: it calls `detect` once per concept and each
call numbers its instances from zero, so a three-concept utterance returns six
detections carrying only two distinct `instance_id` values. A UI keyed on that
id cannot address a click. There is no interactive session to ship, so the UI is
modelled as what it reduces to -- a set of kept keys applied to the candidate
list -- and the exercise's real content becomes which key works. Two further
faults surface on the way: `split_concepts` cuts inside ordinary noun phrases,
and the JSON round trip the exercise asks for does not preserve a detection.

Structure: `candidates` runs the lesson's own `run_multi_concept`; `composite`
is the (concept, instance_id) key; `select` applies a kept-key set and returns
the surviving detections; `emit` serialises them the way the exercise asks and
reads them back.
"""

from __future__ import annotations

import dataclasses
import json

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "24-sam3-open-vocab-segmentation"

HEIGHT, WIDTH = 64, 80
UTTERANCE = "cat and dog and bird"
PHRASES = ("salt and pepper shaker", "AT&T logo", "black-and-white cat", "gin and tonic; ice", "")

composite = lambda det: (det.concept, det.instance_id)                       # noqa: E731
ids = lambda dets: [d.instance_id for d in dets]                             # noqa: E731
keys = lambda dets: [composite(d) for d in dets]                             # noqa: E731
select = lambda dets, kept: [d for d in dets if composite(d) in kept]        # noqa: E731
splits = lambda ref: {p: ref.split_concepts(p) for p in PHRASES}             # noqa: E731


def candidates(np, ref, utterance) -> list:
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    return ref.run_multi_concept(ref.StubOpenVocabSeg(), frame, utterance)


def emit(ref, dets) -> tuple:
    """The exercise's JSON output, and what survives being read back into the dataclass."""
    payload = json.dumps([dataclasses.asdict(d) for d in dets])
    restored = [ref.ConceptDetection(**row) for row in json.loads(payload)]
    return payload, restored


def solve():
    try:
        import numpy as np
    except ImportError as exc:                      # pragma: no cover - T0 needs numpy
        raise practice.Skip(f"needs numpy: uv sync --extra math ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    found = candidates(np, ref, UTTERANCE)
    kept = {("cat", 0), ("bird", 1)}
    chosen = select(found, kept)
    payload, restored = emit(ref, chosen)
    empty = candidates(np, ref, "")
    return {"n": len(found), "ids": ids(found), "unique_ids": len(set(ids(found))),
            "unique_keys": len(set(keys(found))), "concepts": [d.concept for d in found],
            "kept": sorted(keys(chosen)), "chosen": len(chosen),
            "by_id": len([d for d in found if d.instance_id in {0}]),
            "splits": splits(ref), "payload_bytes": len(payload),
            "restored_equal": restored == chosen,
            "box_types": (type(chosen[0].box).__name__, type(restored[0].box).__name__),
            "empty": (len(empty), [d.concept for d in empty])}


def verify(result):
    splits_, kept = result["splits"], result["kept"]
    return [
        practice.Check(
            "ANSWER: the selection works, keyed on (concept, instance_id)",
            result["chosen"] == 2 and kept == [("bird", 1), ("cat", 0)]
            and result["unique_keys"] == result["n"],
            f"{UTTERANCE!r} returns {result['n']} candidates; clicking two of them keeps exactly "
            f"{result['chosen']}, {kept}. The composite key is unique across all "
            f"{result['unique_keys']} candidates, which is what makes the click addressable"),
        practice.Check(
            "FINDING: instance_id alone cannot address a click",
            result["unique_ids"] == 2 < result["n"] and result["by_id"] == 3,
            f"run_multi_concept calls detect once per concept and each call numbers from zero, so "
            f"{result['n']} detections carry ids {result['ids']} -- {result['unique_ids']} distinct "
            f"values for {result['n']} candidates. Clicking 'instance 0' selects "
            f"{result['by_id']} of them, one per concept: a UI keyed on the id the dataclass "
            "exposes would include three objects for every one the user pointed at"),
        practice.Check(
            "MECHANISM: split_concepts cuts inside ordinary noun phrases",
            splits_["salt and pepper shaker"] == ["salt", "pepper shaker"]
            and splits_["AT&T logo"] == ["AT", "T logo"],
            "the separators are replaced textually before any splitting, so "
            + "; ".join(f"{p!r} -> {splits_[p]}" for p in PHRASES[:3])
            + f"; and {PHRASES[3]!r} -> {splits_[PHRASES[3]]}. A hyphenated phrase survives because "
            "the rule matches ' and ' with spaces, so which prompts break is a matter of typography"),
        practice.Check(
            "CONTROL: an empty utterance is a concept",
            result["empty"] == (2, ["", ""]),
            f"split_concepts('') returns {splits_['']} rather than an empty list, so the pipeline "
            f"queries the model for the empty concept and gets {result['empty'][0]} detections back "
            f"with concept {result['empty'][1]}. A UI that trusts the candidate list would offer the "
            "user two anonymous boxes for a prompt they never typed"),
        practice.Check(
            "CONTROL: the JSON the exercise asks for does not round-trip",
            not result["restored_equal"] and result["box_types"] == ("tuple", "list"),
            f"serialising with dataclasses.asdict and reading back through ConceptDetection gives "
            f"{result['payload_bytes']:,} bytes whose detections compare unequal to the originals: "
            f"`box` leaves as a {result['box_types'][0]} and returns as a "
            f"{result['box_types'][1]}, and the dataclass's generated __eq__ compares by type. The "
            "output format is lossy against its own reader unless the box is coerced back"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
