"""Exercise 4 — the length test fires first, so every complex query routes as simple.

    **Build a model routing classifier.** Replace the keyword-based classifier
    with an embedding-based one. Embed 50 labeled queries (simple/medium/
    complex), then classify new queries by finding the nearest labeled example.
    Measure classification accuracy against a test set of 20 queries.

Reading of the exercise: the baseline is `classify_complexity` unchanged, the
replacement is 1-nearest-neighbour over the lesson's own `simple_embed` and
`cosine_similarity` with 50 labelled queries, and both are scored on the same
20-query test set.

**ANSWER: the keyword classifier scores 14 of 20, and 6 of the 8 complex
queries come back "simple".** `classify_complexity` tests `len(q.split()) <= 5`
before it looks at `COMPLEX_KEYWORDS`, so "analyze this architecture", "debug
the payment flow" and "design this architecture" are all simple. The two that
escape are the two long enough to reach the keyword test.

**FINDING: "hi" and "no" are in `SIMPLE_KEYWORDS` and matched as substrings.**
`"hi" in "analyze this architecture"` is True -- on the word "this", and again
on "architecture". Five of the eight complex test queries are caught that way,
so length is not the only thing routing them wrongly.

**ANSWER: nearest-neighbour over the same embedding scores 20 of 20.** Perfect
-- because the labelled set contains the test set's vocabulary. A bag of words
over 50 examples is a synonym table with extra steps.

**FINDING: the better classifier is the more expensive one.** The keyword
labels price 1,000 queries at $1.60 under the lesson's own routing table and
the embedding labels at $2.44 -- 1.5x, because being right about "complex"
means paying for gpt-4o. The exercise measures accuracy and the lesson is about
cost.

**FINDING: swapping two lines matches the embedding classifier exactly.**
Testing the keywords before the length takes the keyword classifier to 20 of
20 -- the same score as 1-NN over 50 labelled examples, with no embedding, no
labelled set and no nearest-neighbour search.

Structure: `SIMPLE`/`MEDIUM`/`COMPLEX` build the 50-query labelled set, `TEST`
the 20-query test set, `nearest` is 1-NN over the lesson's own embedding and
similarity, `keyword_first` is the control with the two tests in the other
order, and `cost_of` prices 1,000 queries under the lesson's routing table.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON, TIER = "11-llm-engineering", "11-caching-cost", "pro"
SIMPLE = ["what time do you open", "store hours", "your address please", "phone number",
          "what is the price", "return policy", "hello", "hi there", "thanks",
          "yes please", "no thanks", "opening hours today", "where are you located",
          "how much does it cost", "is there a phone line", "what is your address",
          "shipping price", "refund policy", "say hello", "thanks again"]
MEDIUM = [f"{verb} the {topic}" for verb in
          ("summarise", "list the steps to configure", "rewrite", "describe", "translate")
          for topic in ("single sign on flow for a new tenant",
                        "onboarding path for an enterprise customer")]
COMPLEX = [f"{verb} {thing}" for verb in
           ("analyze", "compare", "debug", "explain why", "write code for", "evaluate",
            "architect", "design")
           for thing in ("this architecture", "the payment flow")] + [
    "analyze the failure modes of this retry strategy and compare them",
    "explain why the cache hit rate fell after the deployment on tuesday",
    "write code that migrates the audit log table without downtime",
    "design a rate limiter that is fair across tenants of very different sizes"]
LABELLED = ([(q, "simple") for q in SIMPLE] + [(q, "medium") for q in MEDIUM]
            + [(q, "complex") for q in COMPLEX])
TEST = list(zip(
    ["store hours today", "what is your phone number", "hello again", "thanks a lot",
     "return policy please", "how much is shipping",
     "summarise the single sign on flow for a tenant",
     "rewrite the onboarding path for an enterprise customer",
     "describe the single sign on flow for a new tenant",
     "list the steps to configure the onboarding path",
     "translate the onboarding path for a customer",
     "summarise the onboarding path for an enterprise",
     "analyze this architecture", "debug the payment flow", "architect the payment flow",
     "design this architecture", "compare the cost of this workload on two providers",
     "explain why the cache hit rate fell after tuesday",
     "write code that migrates the audit table without downtime",
     "evaluate whether a smaller model would hurt quality"],
    ["simple"] * 6 + ["medium"] * 6 + ["complex"] * 8))
PRICES = {"simple": "gpt-4.1-nano", "medium": "claude-sonnet-4", "complex": "gpt-4o"}


def nearest(ref, query):
    embedding = ref.simple_embed(query)
    scored = [(ref.cosine_similarity(embedding, ref.simple_embed(text)), label)
              for text, label in LABELLED]
    return max(scored)[1]


def keyword_first(ref, query):
    lowered = query.lower()
    if any(word in lowered for word in ref.COMPLEX_KEYWORDS):
        return "complex"
    if len(lowered.split()) <= 5 or any(w in lowered for w in ref.SIMPLE_KEYWORDS):
        return "simple"
    return "medium"


def score(classifier):
    picks = [classifier(query) for query, _ in TEST]
    return sum(p == want for p, (_, want) in zip(picks, TEST)), picks


def cost_of(ref, picks):
    total = 0.0
    for label, (query, _) in zip(picks, TEST):
        call = ref.simulate_llm_call(PRICES[label], query)
        total += ref.calculate_cost(PRICES[label], call["input_tokens"],
                                    call["output_tokens"])["total_cost"]
    return round(total * 1000 / len(TEST), 4)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "caching_cost")
    keyword, keyword_picks = score(lambda q: ref.classify_complexity(q))
    embed, embed_picks = score(lambda q: nearest(ref, q))
    swapped, _ = score(lambda q: keyword_first(ref, q))
    hard = [q for q, want in TEST if want == "complex"]
    return {
        "test_size": len(TEST), "labelled": len(LABELLED),
        "keyword": keyword, "embedding": embed, "swapped": swapped,
        "keyword_labels": sorted(set(keyword_picks)),
        "complex_as_simple": sum(1 for q in hard
                                 if ref.classify_complexity(q) == "simple"),
        "complex_tests": len(hard),
        "substring_hits": sum(1 for q in hard
                              if any(w in q.lower() for w in ("hi", "no"))),
        "hi_in": [q for q in hard if "hi" in q.lower()][:2],
        "routed_keyword": sorted({ref.route_model(q, TIER)["model"] for q, _ in TEST}),
        "cost_keyword": cost_of(ref, keyword_picks),
        "cost_embedding": cost_of(ref, embed_picks),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 14 of 20, and 6 of the 8 complex queries come back simple",
            all([result["keyword"] == 14, result["complex_as_simple"] == 6,
                 result["complex_tests"] == 8]),
            f"`classify_complexity` scores {result['keyword']} of {result['test_size']}. It "
            f"tests len(q.split()) <= 5 before COMPLEX_KEYWORDS, so "
            f"{result['complex_as_simple']} of the {result['complex_tests']} complex queries "
            "come back simple -- the two that escape are the two long enough to reach the "
            "keyword test",
        ),
        practice.Check(
            "FINDING: 'hi' and 'no' are matched as substrings of ordinary words",
            result["substring_hits"] == 5,
            f"`any(kw in q for kw in SIMPLE_KEYWORDS)` with 'hi' and 'no' in the list catches "
            f"{result['substring_hits']} of the {result['complex_tests']} complex queries -- "
            f"{result['hi_in']} among them, on the word 'this' and on 'architecture'. Length "
            "is not the only thing routing them wrongly",
        ),
        practice.Check(
            "ANSWER: nearest-neighbour over the same embedding scores 20 of 20",
            all([result["embedding"] == result["test_size"],
                 result["embedding"] > result["keyword"]]),
            f"1-NN over {result['labelled']} labelled queries, using the lesson's own "
            f"simple_embed and cosine_similarity, scores {result['embedding']} of "
            f"{result['test_size']} against {result['keyword']} -- perfect, because the "
            "labelled set contains the test set's vocabulary. A bag of words over 50 examples "
            "is a synonym table with extra steps",
        ),
        practice.Check(
            "FINDING: the better classifier is the more expensive one",
            all([len(result["routed_keyword"]) == 3,
                 result["cost_embedding"] > 1.4 * result["cost_keyword"]]),
            f"the keyword labels price 1,000 queries at ${result['cost_keyword']} under the "
            f"lesson's routing table and the embedding labels at ${result['cost_embedding']} "
            f"-- {result['cost_embedding'] / result['cost_keyword']:.1f}x, because being right "
            "about 'complex' means paying for gpt-4o. The exercise measures accuracy and the "
            "lesson is about cost",
        ),
        practice.Check(
            "FINDING: swapping two lines matches the embedding classifier exactly",
            result["swapped"] == result["embedding"],
            f"testing the keywords before the length takes the keyword classifier from "
            f"{result['keyword']} to {result['swapped']} of {result['test_size']} -- the "
            f"same score as 1-NN over {result['labelled']} labelled examples, with no "
            "embedding, no labelled set and no nearest-neighbour search",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
