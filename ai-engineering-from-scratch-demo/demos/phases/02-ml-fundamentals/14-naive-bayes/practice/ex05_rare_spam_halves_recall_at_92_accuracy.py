"""Exercise 5 — trained on 0.9% spam, the filter keeps 92% accuracy and lets half the spam through.

    **Spam filter.** Build a complete spam classifier: tokenize raw email text,
    build vocabulary, create bag-of-words features, train MultinomialNB,
    evaluate with precision and recall (not just accuracy -- why?).

Reading of the exercise: there is no email corpus offline, so `make_emails`
writes labelled raw text deterministically: 6 to 29 words drawn from a spam
pool, a work pool and a shared pool of function words. Spam leans on the spam
pool at a varying rate (5 to 35% of words), and half of the spam is in capitals
with "!!!" appended. Ham uses spam words 3% of the time ("are you free"). The
pipeline is the one the exercise lists: a regex tokenizer (lower-cased), a
vocabulary from the 2000 training emails only, bag-of-words counts, and the
lesson's `MultinomialNB` (alpha=1). It is scored on 2000 fresh emails, 14.8%
of them spam.

**ANSWER: precision 0.924, recall 0.818, accuracy 0.963.** Accuracy is the
wrong headline because an always-ham filter already scores 0.852 with
recall 0.

**FINDING: accuracy hides a filter that misses half the spam.** Trained on
0.9% spam (all the ham kept, the spam subsampled), the filter scores 0.921
accuracy, still above the always-ham baseline. Its recall is 0.475, so half
the spam gets through, while precision rises to 0.993. Accuracy moved 4.2
points and recall 34.
The mechanism is the doc's "class imbalance" gotcha, measured: the log-prior
gap is 4.65 nats and the median spam email carries only 4.32 nats of word
evidence. Resetting the prior to 50/50 brings recall back to 0.879, at
precision 0.735.

**CONTROL: the doc's worked example reproduces from the lesson's code, and
mislabels its output.** Fit on counts free/money/meeting = 80/60/10 (spam) and
5/10/100 (ham) with priors 0.4/0.6, the lesson's `predict_log_proba([2, 1, 0])`
gives -3.108 and -8.841 (the doc: -3.109 and -8.838). The doc calls these
"log P(spam | email)", but they are joint log scores. The posterior is
P(spam | email) = 0.9968, not exp(-3.108) = 0.045.

Structure: `make_emails` writes text; `bag` featurizes; `evaluate` scores.
"""

from __future__ import annotations

import random
import re

import numpy as np

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "14-naive-bayes"
SPAM = ("free winner cash prize offer click claim urgent limited deal money credit bonus "
        "guaranteed act now discount").split()
WORK = ("meeting project schedule report review team lunch agenda deadline draft notes call "
        "update budget client slides").split()
COMMON = "the a you your to for and of in is this we please today it our on at with be".split()


def make_emails(n, seed, share=0.15):
    """(raw texts, labels): deterministic synthetic email."""
    rng, texts, labels = random.Random(seed), [], []
    for _ in range(n):
        spam = rng.random() < share
        hot, work = (rng.uniform(0.05, 0.35), 0.05) if spam else (0.03, 0.3)
        words = []
        for _ in range(rng.randint(6, 29)):
            r = rng.random()
            pool = SPAM if r < hot else WORK if r < hot + work else COMMON
            words.append(rng.choice(pool))
        text = " ".join(words)
        text = text.upper() + "!!!" if spam and rng.random() < 0.5 else text
        texts.append(text.capitalize() + ".")
        labels.append(int(spam))
    return texts, np.array(labels)


def tokenize(text):  # lower-cased words, digits and apostrophes
    return re.findall(r"[a-z0-9']+", text.lower())


def bag(texts, vocab):
    index, X = {w: j for j, w in enumerate(vocab)}, np.zeros((len(texts), len(vocab)))
    for i, text in enumerate(texts):
        for word in tokenize(text):
            if word in index:
                X[i, index[word]] += 1
    return X


def evaluate(model, X, y):
    pred = model.predict(X)
    tp, fp = np.sum((pred == 1) & (y == 1)), np.sum((pred == 1) & (y == 0))
    return {"precision": float(tp / (tp + fp)), "recall": float(tp / np.sum(y == 1)),
            "accuracy": float(np.mean(pred == y))}


def doc_example(ref):
    """The doc's free/money/meeting example, fit through the lesson's MultinomialNB."""
    X = np.array([[40, 30, 5], [40, 30, 5], [2, 4, 40], [2, 3, 30], [1, 3, 30]], float)
    model, email = ref.MultinomialNB(alpha=1.0).fit(X, np.array([1, 1, 0, 0, 0])), [[2.0, 1, 0]]
    return model.predict_log_proba(email)[0].tolist(), float(model.predict_proba(email)[0, 1])


def solve():
    ref = parity.load_reference(PHASE, LESSON, "naive_bayes")
    (train_text, y_tr), (test_text, y_te) = make_emails(2000, 1), make_emails(2000, 2)
    vocab = sorted({w for t in train_text for w in tokenize(t)})
    X_tr, X_te = bag(train_text, vocab), bag(test_text, vocab)
    full = ref.MultinomialNB(alpha=1.0).fit(X_tr, y_tr)
    ham, spam = np.flatnonzero(y_tr == 0), np.flatnonzero(y_tr == 1)
    rare = np.concatenate([ham, np.random.RandomState(0).choice(spam, len(ham) // 99, False)])
    skewed = ref.MultinomialNB(alpha=1.0).fit(X_tr[rare], y_tr[rare])
    scores = skewed.predict_log_proba(X_te)
    gap = float(skewed.class_log_prior_[0] - skewed.class_log_prior_[1])
    words = scores[:, 1] - scores[:, 0] + gap
    result = {"full": evaluate(full, X_te, y_te),
              "skewed": evaluate(skewed, X_te, y_te), "gap": gap,
              "evidence": float(np.median(words[y_te == 1])),
              "spam_share": float(y_te.mean()), "rare_share": float(y_tr[rare].mean())}
    skewed.class_log_prior_ = np.log([0.5, 0.5])
    result["reset"] = evaluate(skewed, X_te, y_te)
    result["doc"] = doc_example(ref)
    return result


def verify(result):
    full, skewed, reset = result["full"], result["skewed"], result["reset"]
    (ham_score, spam_score), posterior = result["doc"]
    return [
        practice.Check(
            "ANSWER: precision and recall, because always-ham already scores 0.86 accuracy",
            full["precision"] > 0.9 and full["recall"] > 0.8,
            f"precision {full['precision']:.3f}, recall {full['recall']:.3f}, accuracy "
            f"{full['accuracy']:.3f}; always-ham scores {1 - result['spam_share']:.3f} with "
            "recall 0",
        ),
        practice.Check(
            "FINDING: trained on 1% spam, accuracy stays high while half the spam gets through",
            skewed["accuracy"] > 1 - result["spam_share"] and skewed["recall"] < 0.6,
            f"{result['rare_share']:.1%} spam in training: accuracy {skewed['accuracy']:.3f}, "
            f"recall {skewed['recall']:.3f}, precision {skewed['precision']:.3f}; the prior gap "
            f"is {result['gap']:.2f} nats against a median spam evidence of "
            f"{result['evidence']:.2f}; a 50/50 prior gives recall {reset['recall']:.3f} at "
            f"precision {reset['precision']:.3f}",
        ),
        practice.Check(
            "CONTROL: the doc's spam example reproduces, but its scores are not posteriors",
            abs(spam_score + 3.109) < 0.01 and abs(ham_score + 8.838) < 0.01
            and posterior > 0.99,
            f"lesson's predict_log_proba gives {spam_score:.3f} (spam) and {ham_score:.3f} "
            f"(ham) against the doc's -3.109 and -8.838; P(spam | email) is {posterior:.4f}, "
            f"not exp({spam_score:.3f}) = {np.exp(spam_score):.3f}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
