"""Exercise 1 — precision is no higher than recall.

    **Easy.** Run the pattern extractor in `code/main.py` on 5 news-article
    sentences. Hand-check precision.

Reading of the exercise: thirteen news-style sentences, hand-labelled with the
eleven triples a careful annotator would extract. The extractor predicts ten and
five are right: **precision 0.5000, recall 0.4545**. The lesson's closing note
says "rule-based RE = high precision, low recall". The second half holds; the
first does not, and the two numbers are within one triple of each other.

Four of the five false positives are clauses the sentence denies or hedges. "The
company denied that Carl Reed was born in Ohio" yields `(Carl Reed, place of
birth, Ohio)`; so do "It is unclear whether Elon Musk acquired Twitter", "Nobody
believes Bob Smith works at Initech" and "Reports suggest Jane Doe founded Acme".
The patterns match a clause and never look at what governs it.

The fifth false positive is also a false negative. "Steve Jobs founded Apple
Computer Inc" returns the object `Apple`, because the object group is
`([A-Z][A-Za-z]+(?: Inc)?)` -- which admits *Apple Inc* but not *Apple Computer
Inc*. One truncation is counted on both sides of the ledger.

The six misses are all syntax the pattern cannot express: a passive ("Twitter was
acquired by Elon Musk"), a relative clause ("Alice Chen, who works at Acme"), a
coordinated subject ("Sergey Brin, along with Larry Page, founded Google" -- two
triples, neither found), and a coordinated object ("CEO of Google and Alphabet").

Both directions are the same regex property. Every pattern requires subject, verb
and object adjacent, which excludes any paraphrase that puts something between
them, and admits any clause containing that adjacency however it is embedded.
Tightening one costs the other only because the same expression is doing both
jobs.

On `main()`'s own document the extractor is 8 for 8 with no error -- eight
sentences in the contiguous active form, each with a two-word name and a one-word
object. That is the evidence the note's claim rests on.

Structure: `NEWS` is one line per sentence with its gold triples; `predictions`
runs the lesson's extractor over the set; `pr` scores predicted against gold;
`hedged` marks the sentences whose relation is denied or hedged.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "26-relation-extraction-kg"

NEWS = """Tim Cook became CEO of Apple in 2011.|Tim Cook,P169,Apple
Satya Nadella was born in Hyderabad.|Satya Nadella,P19,Hyderabad
Marissa Mayer studied at Stanford University before joining Google.|Marissa Mayer,P69,Stanford University
Yann LeCun works at Meta.|Yann LeCun,P108,Meta
Steve Jobs founded Apple Computer Inc in a garage in 1976.|Steve Jobs,P112,Apple Computer Inc
Sergey Brin, along with Larry Page, founded Google in a garage.|Sergey Brin,P112,Google;Larry Page,P112,Google
Twitter was acquired by Elon Musk in a leveraged deal.|Elon Musk,P1830,Twitter
Alice Chen, who works at Acme, presented the roadmap.|Alice Chen,P108,Acme
Analysts said Sundar Pichai is CEO of Google and Alphabet.|Sundar Pichai,P169,Google;Sundar Pichai,P169,Alphabet
The company denied that Carl Reed was born in Ohio.|
It is unclear whether Elon Musk acquired Twitter last year.|
Nobody believes Bob Smith works at Initech any more.|
Reports suggest Jane Doe founded Acme before leaving the industry.|"""
ROWS = tuple((text, tuple(tuple(t.split(",")) for t in gold.split(";") if t))
             for text, _, gold in (line.partition("|") for line in NEWS.splitlines()))
HEDGED = tuple(i for i, (_, gold) in enumerate(ROWS) if not gold)
DOC = ("Tim Cook became CEO of Apple in 2011. Steve Jobs founded Apple in 1976. "
       "Larry Page founded Google with Sergey Brin. Sundar Pichai is CEO of Google. "
       "Satya Nadella was born in Hyderabad. Elon Musk acquired Twitter in 2022. "
       "Dario Amodei studied at Princeton University. Yann LeCun works at Meta.")


def gold_triples():
    """Every hand-labelled (sentence, subject, relation, object)."""
    return {(i, *triple) for i, (_, gold) in enumerate(ROWS) for triple in gold}


def predictions(ref):
    """Every (sentence, subject, relation, object) the lesson's extractor returns."""
    return {(i, t["subject"], t["relation"], t["object"])
            for i, (text, _) in enumerate(ROWS) for t in ref.extract(text)}


def pr(predicted, gold):
    """Precision, recall and the two error lists."""
    hits = predicted & gold
    return {"predicted": len(predicted), "gold": len(gold),
            "precision": round(len(hits) / len(predicted), 4),
            "recall": round(len(hits) / len(gold), 4),
            "false_positives": sorted(predicted - gold),
            "false_negatives": sorted(gold - predicted)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    gold, predicted = gold_triples(), predictions(ref)
    scored = pr(predicted, gold)
    own = ref.extract(DOC)
    return dict(
        scored,
        sentences=len(ROWS),
        hedged=len(HEDGED),
        from_hedged=sum(1 for row in scored["false_positives"] if row[0] in HEDGED),
        truncated=[row for row in scored["false_positives"] if row[0] not in HEDGED],
        own_doc=len(own),
        own_verified=len(ref.verify(own, DOC)),
        object_group="([A-Z][A-Za-z]+(?: Inc)?)",
    )


def verify(result):
    return [
        practice.Check(
            "ANSWER: precision 0.5000 and recall 0.4545, within one triple of each other",
            abs(result["precision"] - result["recall"]) < 1 / result["gold"],
            f"over {result['sentences']} news sentences carrying {result['gold']} hand-labelled "
            f"triples the extractor predicts {result['predicted']} and gets "
            f"{len(result['false_positives'])} wrong: precision {result['precision']}, recall "
            f"{result['recall']}. The lesson's note claims 'high precision, low recall'",
        ),
        practice.Check(
            "MECHANISM: four of the five false positives are denied or hedged clauses",
            result["from_hedged"] == len(HEDGED),
            f"{result['from_hedged']} of {len(result['false_positives'])} come from the "
            f"{result['hedged']} sentences whose relation is denied or hedged -- 'The company "
            "denied that Carl Reed was born in Ohio' yields (Carl Reed, place of birth, Ohio). "
            "The patterns match a clause and never look at what governs it",
        ),
        practice.Check(
            "FINDING: the fifth false positive is also a false negative",
            len(result["truncated"]) == 1,
            f"{result['truncated']} -- 'Steve Jobs founded Apple Computer Inc' returns the object "
            f"`Apple`, because the object group is `{result['object_group']}`, which admits "
            "*Apple Inc* and not *Apple Computer Inc*. One truncation counts on both sides",
        ),
        practice.Check(
            "FINDING: the misses are all syntax the pattern cannot express",
            len(result["false_negatives"]) == 6,
            f"{len(result['false_negatives'])} of {result['gold']} gold triples are missed: a "
            "passive, a relative clause, a coordinated subject that carries two triples, a "
            "coordinated object, and the truncation. Every one puts something between the "
            "subject, the verb and the object",
        ),
        practice.Check(
            "MECHANISM: both directions are the same regex property",
            result["from_hedged"] + len(result["truncated"]) == len(result["false_positives"]),
            "the patterns require subject, verb and object adjacent. That excludes any paraphrase "
            "with material between them and admits any clause containing the adjacency, however "
            "it is embedded. One expression is doing both jobs, so tightening either costs the "
            "other",
        ),
        practice.Check(
            "CONTROL: on the lesson's own document it is perfect",
            result["own_doc"] == result["own_verified"] == 8,
            f"`main()`'s document yields {result['own_doc']} triples with no error and "
            f"{result['own_verified']} surviving verification: eight sentences in the contiguous "
            "active form, each with a two-word name and a one-word object. That is the evidence "
            "the note's claim rests on",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
