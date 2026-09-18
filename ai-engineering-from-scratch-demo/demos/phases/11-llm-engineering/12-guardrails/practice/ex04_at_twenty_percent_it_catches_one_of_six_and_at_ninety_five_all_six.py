"""Exercise 4 — at 20% it catches one of six; at 95% it catches all six.

    **Build a hallucination detector for RAG.** Given a source document and a
    model response, check that every factual claim in the response can be
    traced to the source. Use sentence-level comparison: split both into
    sentences, compute word overlap between each response sentence and all
    source sentences, flag any response sentence with <20% overlap as
    potentially hallucinated. Test on 10 response/source pairs.

Reading of the exercise: built exactly as written -- split on sentence
punctuation, score each response sentence as the best word overlap against any
source sentence, flag below 0.20 -- over 10 pairs carrying 15 response
sentences, 9 supported and 6 hallucinated. The hallucinations are the realistic
kind: one digit or one word changed, not a change of subject.

**ANSWER: at the exercise's 20% threshold the detector flags 1 of the 6
hallucinated sentences and 0 of the 9 supported ones.** The one it catches is
the only one that changes the subject -- "Quantum entanglement violates
causality", against a source about shipping, scores 0.000.

**FINDING: the five it misses are the five that matter.** "2.4 million" for
"4.2 million" scores 0.857. "100 requests" for "1000 requests" scores 0.917.
"three days" for "three weeks" scores 0.909. A set of words has no place to put
"2.4 is not 4.2": the token is present, the fact is inverted.

**FINDING: dropping stop words changes nothing.** The 29 most common English
function words come out of both sides and the count stays 1 of 6, because the
overlap was never being carried by "the" and "was" -- every supported sentence
is copied verbatim and scores 1.000.

**CONTROL: 0.95 catches 6 of 6 with no false positives, and then flags every
faithful paraphrase.** Supported sentences score exactly 1.000 and the worst
hallucination scores 0.917, so any threshold in between separates them
perfectly on this set. Three faithful paraphrases of the same facts --
"Third-quarter revenue came to 4.2 million dollars" -- score 0.500, 0.364 and
0.429. The metric separates copied from not-copied, which is not what the
exercise says it measures.

**FINDING: the lesson already ships this function, and a digit test beats it.**
`check_relevance(input_text, output_text, threshold=0.15)` is the same word
overlap with a different denominator; pointed at source-versus-response it
fails 1 of the 10 pairs -- the same pair. Comparing the digit strings of the
two texts catches 2 of the 6 hallucinations exactly, and costs one regex.

Structure: `PAIRS` is the 10 source/response pairs with per-sentence labels,
`PARAPHRASE` the faithful-rewrite control, `overlap` is the exercise's metric,
`overlap` is the exercise's metric (best word overlap against any source
sentence), `score_all` applies it to every labelled sentence, `at` counts
catches and false positives at a threshold, and `shipped` runs the lesson's own
`check_relevance` over the same pairs beside a digit-set test.
"""

from __future__ import annotations

import re

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "12-guardrails"
THRESHOLD = 0.20
STOP = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "of", "in", "for", "on",
        "with", "at", "by", "from", "it", "this", "that", "to", "and", "or", "but", "as",
        "its", "their", "has", "have"}
REVENUE = "The Q3 revenue was 4.2 million dollars. The team grew to 48 engineers in Berlin."
LIMITS = "The API rate limit is 1000 requests per hour for paid plans."
TRAINED = "The model was trained on 12 billion tokens over three weeks."
SHIPPING = "Shipping is free above fifty euros. Returns are accepted within thirty days."
PAIRS = [
    (REVENUE, [("The Q3 revenue was 4.2 million dollars.", True),
               ("The team grew to 48 engineers in Berlin.", True)]),
    (REVENUE, [("The Q3 revenue was 2.4 million dollars.", False),
               ("The team grew to 48 engineers in Berlin.", True)]),
    ("Support is available on weekdays from nine to five. Refunds take five business days.",
     [("Support is available on weekdays from nine to five.", True),
      ("Support is also available on Sundays.", False)]),
    (LIMITS, [(LIMITS, True)]),
    (LIMITS, [("The API rate limit is 100 requests per hour for paid plans.", False)]),
    ("Our office is in Lisbon. The building has a roof terrace.",
     [("Our office is in Lisbon.", True),
      ("The building was designed by Napoleon.", False)]),
    (TRAINED, [(TRAINED, True)]),
    (TRAINED, [("The model was trained on 12 billion tokens over three days.", False)]),
    (SHIPPING, [("Shipping is free above fifty euros.", True),
                ("Returns are accepted within thirty days.", True)]),
    ("Shipping is free above fifty euros.",
     [("Quantum entanglement violates causality in the standard model.", False)]),
]
PARAPHRASE = [(REVENUE, "Third-quarter revenue came to 4.2 million dollars."),
              (LIMITS, "Paid plans may issue up to 1000 API requests each hour."),
              ("Our office is in Lisbon.", "The office is located in Lisbon, Portugal.")]


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def bag(sentence, drop_stop):
    found = set(re.findall(r"[a-z0-9.$]+", sentence.lower()))
    return found - STOP if drop_stop else found


def overlap(sentence, source, drop_stop=False):
    target = bag(sentence, drop_stop)
    shares = (len(target & bag(s, drop_stop)) / len(target) for s in sentences(source))
    return round(max(shares, default=0.0), 3) if target else 0.0


def score_all(drop_stop=False):
    return [(label, overlap(sentence, source, drop_stop))
            for source, claims in PAIRS for sentence, label in claims]


def at(scores, threshold):
    return (sum(1 for label, value in scores if not label and value < threshold),
            sum(1 for label, value in scores if label and value < threshold))


def digits(text):
    return set(re.findall(r"\d[\d.,]*", text))


def shipped(ref):
    relevance = [ref.check_relevance(source, " ".join(s for s, _ in claims))
                 for source, claims in PAIRS]
    return {"relevance_failed": sum(1 for r in relevance if not r.passed),
            "digit_catches": sum(1 for source, claims in PAIRS for sentence, label in claims
                                 if not label and digits(sentence) - digits(source))}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "guardrails")
    scores = score_all()
    supported = [v for label, v in scores if label]
    hallucinated = [v for label, v in scores if not label]
    caught, false_pos = at(scores, THRESHOLD)
    return {
        "pairs": len(PAIRS), "sentences": len(scores), "supported": len(supported),
        "hallucinated": len(hallucinated), "caught": caught, "false_pos": false_pos,
        "caught_nostop": at(score_all(True), THRESHOLD)[0],
        "worst_supported": min(supported), "best_hallucinated": max(hallucinated),
        "hallucinated_scores": sorted(hallucinated, reverse=True),
        "caught_95": at(scores, 0.95), "paraphrase": [overlap(p, s) for s, p in PARAPHRASE],
        **shipped(ref),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: at 20% it flags 1 of the 6 hallucinated sentences and 0 of 9 supported",
            all([result["caught"] == 1, result["false_pos"] == 0, result["pairs"] == 10]),
            f"over {result['pairs']} pairs carrying {result['sentences']} response sentences "
            f"({result['supported']} supported, {result['hallucinated']} hallucinated), the "
            f"exercise's <{THRESHOLD:.0%} rule catches {result['caught']} and "
            "wrongly flags 0. The one it catches is the one that changes the subject",
        ),
        practice.Check(
            "FINDING: the five it misses are the five that matter",
            all([max(result["hallucinated_scores"]) > 0.9,
                 sorted(result["hallucinated_scores"])[1] > THRESHOLD]),
            f"the hallucinated sentences score {result['hallucinated_scores']}: '2.4 million' "
            "for '4.2 million', '100 requests' for '1000', 'three days' for 'three weeks'. A "
            "set of words has no place to put '2.4 is not 4.2' -- the token is present and "
            "the fact is inverted",
        ),
        practice.Check(
            "FINDING: dropping stop words changes nothing",
            result["caught_nostop"] == result["caught"],
            f"taking the {len(STOP)} commonest function words out of both sides leaves the "
            f"count at {result['caught_nostop']} of {result['hallucinated']}. The overlap was "
            "never carried by 'the' and 'was': every supported sentence is copied verbatim "
            f"and scores {result['worst_supported']}",
        ),
        practice.Check(
            "CONTROL: 0.95 catches 6 of 6, and then flags every faithful paraphrase",
            all([result["caught_95"] == (result["hallucinated"], 0),
                 max(result["paraphrase"]) < 0.95,
                 min(result["paraphrase"]) > THRESHOLD]),
            f"supported sentences score {result['worst_supported']} and the worst "
            f"hallucination {result['best_hallucinated']}, so 0.95 separates them "
            f"{result['caught_95'][0]} of {result['hallucinated']} with "
            f"{result['caught_95'][1]} false positives -- but three faithful paraphrases "
            f"score {result['paraphrase']}. It separates copied from not-copied",
        ),
        practice.Check(
            "FINDING: the lesson already ships this function, and a digit test beats it",
            all([result["relevance_failed"] == result["caught"],
                 result["digit_catches"] == 2]),
            "`check_relevance(input_text, output_text, threshold=0.15)` is the same word "
            f"overlap with a different denominator; pointed at source-versus-response it "
            f"fails {result['relevance_failed']} of {result['pairs']} pairs -- the same pair. "
            f"Comparing digit strings catches {result['digit_catches']} of "
            f"{result['hallucinated']} exactly, and costs one regex",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
