"""Exercise 1 — two answers made of articles match exactly.

    **Easy.** Set up the SQuAD extractive pipeline above on 10 Wikipedia
    passages. Hand-craft 10 questions. Measure how often the answer is correct.
    You should see 7-9 correct if passages and questions are clean.

Reading of the exercise: the pipeline lands inside the predicted band -- exact
match 8 of 10, mean token F1 0.800 -- and the two it gets wrong are worth more
than the eight it gets right, because they fail in the retriever and in the
reader rather than in the metric. `In what year did Android launch?` retrieves
the Macworld passage, whose overlap on `year` and a date outweighs one mention of
Android; `Where was the World Wide Web proposed?` retrieves correctly and the
span rule returns `The World`.

The metrics themselves have a hole the exercise's clean questions never reach.
`normalize` strips `a`, `an` and `the` before it strips punctuation, so any
answer made only of articles normalises to the empty string -- and `exact_match`
compares normalised strings. `exact_match("a", "the")` is **1.0**. So is
`exact_match("The", "A")`, and `token_f1` agrees at 1.0, because it returns 1.0
when both token lists are empty. Two different one-word answers score a perfect
match, and a system that answered `the` to every question would be graded
correct on any gold answer that is also an article.

The other gap is the opposite one. `token_f1` is a bag of tokens, so
`29 June 2007` against `June 29, 2007` scores **1.000** while exact match scores
0 -- the same 1.000 the identical string gets. The lesson's note says F1 is
partial credit; on a reordering it is full credit, and the two metrics disagree
completely rather than by a margin.

Structure: `PASSAGES` and `QUESTIONS` are the ten and ten the exercise asks for,
written so each question is answered by exactly one passage. `reader` is the
span extractor -- a date pattern for `when` questions and a proper-name pattern
otherwise -- and `answer` runs the lesson's own `toy_retrieve` in front of it.
"""

from __future__ import annotations

import re

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "13-question-answering"

PASSAGES = (
    "Apple Inc. released the first iPhone on June 29, 2007, in the United States.",
    "Steve Jobs announced the iPhone at Macworld in San Francisco in January 2007.",
    "Android launched in 2008 as Google's mobile operating system for handset makers.",
    "The first iPod was released by Apple in October 2001 with a five gigabyte drive.",
    "Nokia sold its handset business to Microsoft in a deal completed in April 2014.",
    "The Nintendo Switch went on sale in March 2017 and sold ten million units that year.",
    "Linus Torvalds released the first Linux kernel in September 1991 from Helsinki.",
    "Python was created by Guido van Rossum and first released in February 1991.",
    "The World Wide Web was proposed by Tim Berners-Lee at CERN in March 1989.",
    "Amazon began as an online bookstore founded by Jeff Bezos in July 1994.",
)
QUESTIONS = (
    ("When was the first iPhone released?", "June 29, 2007"),
    ("Who announced the iPhone at Macworld?", "Steve Jobs"),
    ("In what year did Android launch?", "2008"),
    ("When was the first iPod released?", "October 2001"),
    ("Who bought Nokia's handset business?", "Microsoft"),
    ("When did the Nintendo Switch go on sale?", "March 2017"),
    ("Who released the first Linux kernel?", "Linus Torvalds"),
    ("Who created Python?", "Guido van Rossum"),
    ("Where was the World Wide Web proposed?", "CERN"),
    ("Who founded Amazon?", "Jeff Bezos"),
)
DATE = re.compile(r"[A-Z][a-z]+ \d{1,2}, \d{4}|[A-Z][a-z]+ \d{4}|\b\d{4}\b")
NAME = re.compile(r"[A-Z][a-z]+(?: van)? [A-Z][a-z-]+|\bCERN\b|\bMicrosoft\b")
ARTICLES = ("a", "an", "the", "The", "A")


def reader(passage, question) -> str:
    """The span rule: a date for a `when` question, otherwise the first proper name."""
    when = DATE.search(passage)
    if question.lower().startswith(("when", "in what year")) and when:
        return when.group()
    names = NAME.findall(passage)
    return names[0] if names else passage.split()[0]


def answer(ref, question) -> tuple:
    passage = ref.toy_retrieve(question, top_k=1)[0][1]
    return passage, reader(passage, question)


def run(ref) -> list:
    rows = []
    for question, gold in QUESTIONS:
        passage, predicted = answer(ref, question)
        rows.append({"question": question, "gold": gold, "predicted": predicted,
                     "em": ref.exact_match(predicted, gold),
                     "f1": round(ref.token_f1(predicted, gold), 3),
                     "retrieved": PASSAGES.index(passage)})
    return rows


def articles(ref) -> dict:
    """Every pair of article-only answers, and what normalize does to them."""
    pairs = [(a, b) for i, a in enumerate(ARTICLES) for b in ARTICLES[i + 1:]]
    return {"article_em": {f"{a}|{b}": ref.exact_match(a, b) for a, b in pairs},
            "article_f1": ref.token_f1("a", "the"),
            "normalized": {w: ref.normalize(w) for w in ("a", "the", "A.", "The Beatles")}}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    ref.CORPUS = list(PASSAGES)
    rows = run(ref)
    return dict(
        articles(ref), rows=rows, questions=len(QUESTIONS), passages=len(PASSAGES),
        em=sum(row["em"] for row in rows),
        f1=round(sum(row["f1"] for row in rows) / len(rows), 3),
        wrong=[(row["question"], row["predicted"], row["gold"], row["retrieved"])
               for row in rows if row["em"] == 0.0],
        misretrieved=sum(row["retrieved"] != i for i, row in enumerate(rows)),
        reordered=(ref.exact_match("29 June 2007", "June 29, 2007"),
                   round(ref.token_f1("29 June 2007", "June 29, 2007"), 3)),
        identical=round(ref.token_f1("June 29, 2007", "June 29, 2007"), 3))


def verify(result):
    rows, wrong = result["rows"], result["wrong"]
    reordered, article_em = result["reordered"], result["article_em"]
    return [
        practice.Check(
            "ANSWER: exact match 8 of 10 and mean F1 0.800, inside the predicted band",
            7 <= result["em"] <= 9,
            f"on {result['passages']} passages and {result['questions']} hand-written questions the "
            f"pipeline scores EM {result['em']:.0f}/{result['questions']} and mean token F1 "
            f"{result['f1']}. The exercise predicts 7-9 correct on clean inputs and this is 8, so "
            f"the headline is the one it expects"),
        practice.Check(
            "FINDING: the two failures are the retriever and the span rule, not the metric",
            len(wrong) == 2 and result["misretrieved"] == 1,
            f"the misses are {[(q[:34], p, g) for q, p, g, _ in wrong]}. One is retrieval -- "
            f"{result['misretrieved']} question lands on the wrong passage, because overlap on "
            f"`year` and a date outweighs one mention of the entity -- and one is the span rule "
            f"returning a leading noun phrase from the right passage"),
        practice.Check(
            "MECHANISM: normalize strips articles before punctuation, so 'a' becomes the empty string",
            result["normalized"]["a"] == "" and result["normalized"]["A."] == "",
            f"`normalize` runs `\\b(a|an|the)\\b` first and `[^\\w\\s]` second, so "
            f"{result['normalized']}. Anything made only of articles normalises to nothing at all, "
            f"and `exact_match` compares the normalised strings"),
        practice.Check(
            "FINDING: so any two answers made of articles are an exact match",
            set(article_em.values()) == {1.0} and result["article_f1"] == 1.0,
            f"pairwise exact match over {list(ARTICLES)}: {article_em}. `token_f1('a', 'the')` is "
            f"{result['article_f1']} too, because it returns 1.0 when both token lists are empty. A "
            f"system answering 'the' to everything is graded correct on any gold answer that is "
            f"also an article"),
        practice.Check(
            "FINDING: token F1 is a bag of tokens, so a reordering scores full credit not partial",
            reordered == (0.0, result["identical"]),
            f"'29 June 2007' against 'June 29, 2007' scores exact match {reordered[0]:.0f} and F1 "
            f"{reordered[1]} -- the same {result['identical']} the identical string gets. The "
            f"lesson's note calls F1 partial credit; on a permutation it is full credit, and the "
            f"two metrics disagree completely rather than by a margin"),
        practice.Check(
            "CONTROL: on the eight it gets right the two metrics agree",
            all(row["f1"] == 1.0 for row in rows if row["em"] == 1.0),
            f"every question with EM 1 also scores F1 1.0, so the disagreements above are not noise "
            f"in the pipeline -- they are cases the ten clean questions the exercise asks for do "
            f"not contain. A hand-written set of ten will not find them"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
