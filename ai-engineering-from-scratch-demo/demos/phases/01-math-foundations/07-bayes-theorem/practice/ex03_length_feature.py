"""Exercise 3 — add a message-length feature alongside word counts.

    **Add features.** Extend the NaiveBayes class to also use message length
    (short/long) as a feature alongside word counts. Estimate P(short|spam) and
    P(short|ham) from the training data and fold it into the prediction score.

Reading of the exercise: "extend the class" means subclass it — this repo never
edits the reference (D5). The length feature is smoothed the same way words are,
because an all-short class would otherwise reintroduce exercise 2's log(0). The
check that matters is check 4: the feature has to *change* a decision on some
document, or it has been folded in without being used.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "07-bayes-theorem"
THRESHOLD = 4                             # words; <= is "short"

DOCS = ["win free money now", "free money click here", "claim your prize", "act now",
        "meeting at three tomorrow with the whole product team",
        "lunch tomorrow with the team if that suits everyone",
        "project deadline moved to friday afternoon please review", "sounds good"]
LABELS = ["spam", "spam", "spam", "spam", "ham", "ham", "ham", "ham"]


def make_model(ref, smoothing=1.0):
    class LengthAwareNaiveBayes(ref.NaiveBayes):
        """Adds a binary length feature; word handling is inherited untouched."""

        def train(self, documents, labels):
            super().train(documents, labels)
            self.length_counts = {}
            for document, label in zip(documents, labels):
                bucket = self._bucket(document)
                self.length_counts.setdefault(label, {"short": 0, "long": 0})
                self.length_counts[label][bucket] += 1

        @staticmethod
        def _bucket(document):
            return "short" if len(document.split()) <= THRESHOLD else "long"

        def p_length(self, bucket, cls):
            counts = self.length_counts[cls]
            total = counts["short"] + counts["long"]
            return (counts[bucket] + self.smoothing) / (total + 2 * self.smoothing)

        def predict_proba(self, document):
            scores = {}
            bucket = self._bucket(document)
            for cls in self.class_counts:
                score = self._log_prior(cls) + math.log(self.p_length(bucket, cls))
                for word in document.lower().split():
                    score += self._log_likelihood(word, cls)
                scores[cls] = score
            top = max(scores.values())
            exponentiated = {c: math.exp(s - top) for c, s in scores.items()}
            total = sum(exponentiated.values())
            return {c: v / total for c, v in exponentiated.items()}

    return LengthAwareNaiveBayes(smoothing=smoothing)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "bayes")
    extended = make_model(ref)
    extended.train(DOCS, LABELS)
    baseline = ref.NaiveBayes(smoothing=1.0)
    baseline.train(DOCS, LABELS)
    probes = ["free money", "tomorrow with the whole team please review", "click here"]
    return {
        "p_short_spam": extended.p_length("short", "spam"),
        "p_short_ham": extended.p_length("short", "ham"),
        "counts": extended.length_counts,
        "with": {p: extended.predict_proba(p)["spam"] for p in probes},
        "without": {p: baseline.predict_proba(p)["spam"] for p in probes},
        "subclass": issubclass(type(extended), ref.NaiveBayes),
    }


def verify(result):
    shifts = {p: result["with"][p] - result["without"][p] for p in result["with"]}
    biggest = max(shifts, key=lambda p: abs(shifts[p]))
    return [
        practice.Check("NaiveBayes is subclassed, not edited", result["subclass"],
                       "LengthAwareNaiveBayes inherits word handling unchanged"),
        practice.Check("P(short|spam) > P(short|ham), as the training data implies",
                       result["p_short_spam"] > result["p_short_ham"],
                       f"P(short|spam) = {result['p_short_spam']:.4f}, "
                       f"P(short|ham) = {result['p_short_ham']:.4f} from counts "
                       f"{result['counts']}"),
        practice.Check("the length feature is smoothed too — no class can hit log(0)",
                       0 < result["p_short_spam"] < 1 and result["counts"]["spam"]["long"] == 0,
                       f"spam is 100% short in this data ({result['counts']['spam']}), so an "
                       f"unsmoothed P(long|spam) would be 0 and any long document would score "
                       f"log(0) for spam — the same failure exercise 2 reproduces"),
        practice.Check("the feature actually changes a prediction",
                       abs(shifts[biggest]) > 0.01,
                       f"largest shift on {biggest!r}: P(spam) "
                       f"{result['without'][biggest]:.4f} → {result['with'][biggest]:.4f} "
                       f"({shifts[biggest]:+.4f})"),
        practice.Check("…in the direction the feature predicts, for every probe",
                       all((shifts[p] > 0) == (len(p.split()) <= THRESHOLD)
                           for p in shifts if abs(shifts[p]) > 1e-6),
                       "; ".join(f"{p!r} ({len(p.split())}w) {shifts[p]:+.4f}"
                                 for p in shifts)),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
