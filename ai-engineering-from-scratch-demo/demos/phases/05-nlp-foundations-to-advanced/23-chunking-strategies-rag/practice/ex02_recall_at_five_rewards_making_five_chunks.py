"""Exercise 2 — recall@5 rewards making five chunks.

    **Medium.** Build a 30-query eval set over 5 documents. Measure recall@5 for
    recursive, semantic, and parent-document. Which wins? Does it match the blog
    posts?

Reading of the exercise: the eval set is 30 questions over 5 documents of 6
sections each, each question answered by a distinct section, scored with the
lesson's own hash embedder. Parent-document wins at **1.0000** -- which does
match the blog posts, and is not evidence for them.

Parent-document produces **5 chunks** for 5 documents, one per document. At k=5
the retriever returns the entire index, so recall@5 is 1.0000 by construction
and would be 1.0000 for any ranking function, including a constant one. The
metric is measuring how few chunks the strategy makes.

At k=1, where every arm returns one chunk, the strategies are indistinguishable:
0.5000 for fixed, 0.5333 for recursive and 0.5667 for semantic, sentence and
parent-document alike -- a spread of two questions in thirty, with
parent-document tied rather than best. The ordering the exercise asks for exists
only at the k where one arm has swallowed the index.

The same strategy is also both best and worst. `chunk_parent_child` returns one
row per *child*, so `main()`'s own index expression `[m["parent"] for m in pc]`
holds 15 entries and 5 distinct texts. Duplicates fill the top-5 slots, and that
index scores **0.7333** -- the lowest of any strategy -- while the deduplicated
one scores 1.0000. The difference between the winner and the loser is a `set()`.

The lesson's own demonstration cannot see any of this: on its 3 queries over one
contract, all five strategies score 3 of 3, including the duplicated
parent-document index with a single distinct document in it.

Structure: `CORPUS` is one line per section -- document, text, question, gold
span; `DOCUMENTS` reassembles the five documents; `index_for` builds each
strategy's index; `ranked` and `recall_at` score them; `coverage` is k over
index size.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "23-chunking-strategies-rag"

AT, CHUNK, PARENT, CHILD = 5, 300, 800, 200
CORPUS = """lease|Rent. The tenant pays two thousand dollars on the first of each month.|How much is the monthly rent?|two thousand dollars
lease|Deposit. A security deposit of four thousand dollars is held until the tenant vacates.|What is the security deposit?|four thousand dollars
lease|Pets. Cats are permitted with a one hundred dollar pet fee. Dogs are not permitted.|Are dogs allowed in the unit?|Dogs are not permitted
lease|Repairs. The landlord repairs structural faults. The tenant repairs damage they cause.|Who fixes structural faults?|landlord repairs structural
lease|Sublet. Subletting the unit requires written consent from the landlord.|Can the unit be sublet?|written consent from the landlord
lease|Utilities. The tenant pays for electricity and water; the landlord pays refuse.|Who pays for water?|tenant pays for electricity and water
handbook|Leave. Employees accrue eighteen days of paid leave each calendar year.|How many days of paid leave?|eighteen days
handbook|Remote. Staff may work remotely three days a week with manager approval.|How many remote days per week?|three days a week
handbook|Expenses. Receipts must be filed within thirty days of the expense being incurred.|What is the deadline for receipts?|within thirty days
handbook|Notice. Resignation requires four weeks written notice to the line manager.|How much notice to resign?|four weeks written notice
handbook|Training. Each employee receives a budget of twelve hundred dollars a year.|What is the training budget?|twelve hundred dollars
handbook|Review. Performance reviews take place twice a year in March and September.|When are performance reviews?|March and September
warranty|Coverage. The warranty covers manufacturing defects for twenty four months.|How long does the warranty last?|twenty four months
warranty|Exclusions. Water damage and accidental drops are excluded from coverage.|Is water damage covered?|Water damage and accidental
warranty|Claims. Claims require the original receipt and the serial number of the unit.|What is needed to file a claim?|original receipt and the serial
warranty|Repair. Approved repairs are completed within fifteen working days of receipt.|How long do repairs take?|fifteen working days
warranty|Transfer. The warranty transfers to a new owner once, with proof of sale.|Can the warranty be transferred?|transfers to a new owner once
warranty|Parts. Replacement parts carry a further six month warranty of their own.|How long are replacement parts covered?|further six month warranty
privacy|Retention. Account records are retained for seven years after closure.|How long are records kept?|seven years after closure
privacy|Sharing. Data is shared with processors under contract and never sold.|Is data ever sold?|processors under contract and never sold
privacy|Access. Subjects may request a copy of their data within one calendar month.|How fast is a data access request?|within one calendar month
privacy|Cookies. Analytics cookies expire after thirteen months unless refreshed.|When do analytics cookies expire?|thirteen months
privacy|Breach. Regulators are notified within seventy two hours of a confirmed breach.|How fast are breaches reported?|seventy two hours
privacy|Children. Accounts are not offered to anyone under the age of sixteen.|What is the minimum account age?|under the age of sixteen
loan|Interest. The fixed rate is six point five percent for the first three years.|What is the fixed interest rate?|six point five percent
loan|Fees. An arrangement fee of nine hundred dollars is added to the balance.|What is the arrangement fee?|nine hundred dollars
loan|Overpay. Borrowers may overpay ten percent of the balance each year penalty free.|How much can be overpaid?|ten percent of the balance
loan|Default. Two missed payments place the account in default and trigger recovery.|What causes default?|Two missed payments
loan|Term. The standard term is twenty five years and may be shortened on request.|What is the loan term?|twenty five years
loan|Insurance. Buildings insurance is mandatory for the life of the loan.|Is insurance required?|Buildings insurance is mandatory"""
ROWS = tuple(tuple(line.split("|")) for line in CORPUS.splitlines())
DOCUMENTS = {name: "\n\n".join(text for doc, text, _, _ in ROWS if doc == name)
             for name in dict.fromkeys(doc for doc, _, _, _ in ROWS)}
QUERIES = tuple((question, gold) for _, _, question, gold in ROWS)


def index_for(ref, strategy):
    """One flat index over all five documents, built by the named strategy."""
    out = []
    for text in DOCUMENTS.values():
        if strategy == "fixed":
            out += ref.chunk_fixed(text, CHUNK, 50)
        elif strategy == "recursive":
            out += ref.chunk_recursive(text, CHUNK)
        elif strategy == "semantic":
            out += ref.chunk_semantic(text)
        elif strategy == "sentence":
            out += ref.chunk_sentence(text, 3)
        else:
            rows = ref.chunk_parent_child(text, PARENT, CHILD)
            parents = [row["parent"] for row in rows]
            out += list(dict.fromkeys(parents)) if strategy == "parent" else parents
    return out


def ranked(ref, chunks, query):
    """(score, index) over the whole index, best first."""
    embedded = ref.hash_embed(query)
    return sorted(((ref.cosine(ref.hash_embed(c), embedded), i) for i, c in enumerate(chunks)),
                  reverse=True)


def recall_at(ref, chunks, k):
    """Fraction of questions whose gold span appears in a top-k chunk."""
    hits = sum(1 for query, gold in QUERIES
               if any(gold.lower() in chunks[i].lower() for _, i in ranked(ref, chunks, query)[:k]))
    return round(hits / len(QUERIES), 4)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    names = ("fixed", "recursive", "semantic", "sentence", "parent", "parent as main builds it")
    indexes = {name: index_for(ref, name) for name in names}
    return {
        "docs": len(DOCUMENTS), "queries": len(QUERIES),
        "size": {n: len(c) for n, c in indexes.items()},
        "distinct": {n: len(set(c)) for n, c in indexes.items()},
        "coverage": {n: round(min(1.0, AT / len(c)), 2) for n, c in indexes.items()},
        "at5": {n: recall_at(ref, c, AT) for n, c in indexes.items()},
        "at1": {n: recall_at(ref, c, 1) for n, c in indexes.items()},
    }


def verify(result):
    at5, at1, size = result["at5"], result["at1"], result["size"]
    dup = "parent as main builds it"
    return [
        practice.Check(
            "ANSWER: parent-document wins at 1.0000, which is the whole index",
            at5["parent"] == 1.0 and size["parent"] == AT,
            f"over {result['queries']} questions on {result['docs']} documents, recall@{AT} is "
            f"{at5}. Parent-document builds {size['parent']} chunks for {result['docs']} "
            f"documents, so at k={AT} the retriever returns the entire index: 1.0000 holds for "
            "any ranking function, including a constant one",
        ),
        practice.Check(
            "MECHANISM: the metric is reading index size",
            result["coverage"]["parent"] == 1.0 and result["coverage"]["semantic"] < 0.25,
            f"index sizes are {size} and the share of the index that k={AT} returns is "
            f"{result['coverage']}. Semantic sees 17% of its index and parent-document sees all "
            "of its own. Nothing about boundary quality is being compared",
        ),
        practice.Check(
            "FINDING: at equal coverage the strategies are indistinguishable",
            max(at1.values()) - min(at1.values()) <= 3 / result["queries"],
            f"at k=1 the same indexes score {at1} -- a spread of two questions in "
            f"{result['queries']}, with parent-document tied rather than best. The ordering the "
            f"exercise asks for exists only at the k where one arm has swallowed the index",
        ),
        practice.Check(
            "FINDING: the same strategy is both the best and the worst arm",
            at5[dup] == min(at5.values()) and at5["parent"] == max(at5.values()),
            f"`chunk_parent_child` returns one row per child, so `main()`'s own index expression "
            f"`[m['parent'] for m in pc]` holds {size[dup]} entries and "
            f"{result['distinct'][dup]} distinct texts. Duplicates fill the top-{AT} slots and it "
            f"scores {at5[dup]}, the lowest of any strategy, against {at5['parent']} "
            "deduplicated. The difference between winner and loser is a `set()`",
        ),
        practice.Check(
            "MECHANISM: and recursive, the arm with no degeneracy, wins on the honest comparison",
            at5["recursive"] > at5["semantic"] and at5["recursive"] > at5[dup],
            f"among the strategies whose index is larger than k, recursive scores "
            f"{at5['recursive']} against {at5['semantic']} for semantic and {at5['fixed']} for "
            f"fixed, at {size['recursive']}, {size['semantic']} and {size['fixed']} chunks. That "
            "is the comparison the exercise wanted, and parent-document is not in it",
        ),
        practice.Check(
            "CONTROL: the lesson's own demonstration cannot see any of this",
            result["distinct"][dup] < size[dup],
            "`main()` scores 3 queries over one contract and reports 3 of 3 for all five "
            f"strategies, including the duplicated parent index -- {result['distinct'][dup]} "
            f"distinct texts across {size[dup]} entries here, and exactly one distinct text "
            "there. A saturated eval ranks nothing",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
