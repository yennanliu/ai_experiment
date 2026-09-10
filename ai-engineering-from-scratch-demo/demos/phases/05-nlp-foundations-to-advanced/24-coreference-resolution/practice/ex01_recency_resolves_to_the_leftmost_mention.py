"""Exercise 1 — recency resolves to the leftmost mention.

    **Easy.** Run the rule-based resolver in `code/main.py` on 5 hand-crafted
    paragraphs. Measure mention-link accuracy against ground truth.

Reading of the exercise: seven paragraphs, twelve pronouns, one gold antecedent
each. `resolve` gets **8 of 12** right, and the four failures are four distinct
mechanisms rather than four hard cases.

`recency_score` is not recency. Sentence distance costs 2.0 per sentence, and
the within-sentence term is `max(0, mention_token - candidate_token) * 0.01` --
clipped at zero. A sentence-initial pronoun has token index 0, so that term is 0
for **every** candidate in an earlier sentence, and all candidates in the nearest
sentence tie exactly. `max` returns the first of them, so the resolver picks the
*leftmost* mention of the nearest sentence: "Apple sued Google over the patent.
It appealed the ruling" links `It` to Apple.

Capitalisation manufactures the competition. `extract_mentions` treats any
capitalised alphabetic token as a named entity, and the first word of every
sentence is capitalised, so `Analysts` and `Engineers` enter the candidate list
as people or organisations. Both then win on recency and take a pronoun that
belonged to a real entity two sentences back.

Gender is a twenty-name lookup. `FEMALE_FIRST` and `MALE_FIRST` hold ten first
names each; every other name infers `u`, and `agreement_score` treats `u` as
compatible with everything. "Priya Sharma presented the chip. Yusuf Demir asked a
question. She answered him directly." sends `She` to Yusuf Demir. Substituting
Mary Chen and James Lin -- both on the lists -- resolves the identical sentence
correctly, so the feature works exactly on the twenty names it knows.

Underneath, agreement can rank but never veto. Its whole range is 3.0, from +2.0
for a compatible pair down to -1.0 for a gender clash, against -2.0 per sentence
of distance: a mismatched antecedent one sentence nearer still wins.

Structure: `PARAGRAPHS` is one line per paragraph -- text, then the gold
antecedent for each pronoun in order; `links_for` runs the lesson's pipeline;
`score` compares against gold and returns the misses; `spurious` counts the
sentence-initial named entities.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "24-coreference-resolution"

PARAGRAPHS = """John Smith opened the meeting. He thanked the team. The company had grown quickly. It hired forty people last year.|John Smith;The company
Sarah Patel joined the bank in March. She now leads the group. Her group ships the payments product.|Sarah Patel;Sarah Patel
Apple sued Google over the patent. It appealed the ruling within a month.|Google
Satya Nadella spoke at the conference. He praised the agency. Analysts liked his tone.|Satya Nadella;Satya Nadella
The firm bought a device from the supplier. It shipped the device in June. Engineers tested it that week.|The firm;a device
Priya Sharma presented the chip. Yusuf Demir asked a question. She answered him directly.|Priya Sharma;Yusuf Demir
Nokia announced the phone. Nokia also cut prices. It blamed the market.|Nokia"""
ROWS = tuple((text, tuple(gold.split(";")))
             for text, gold in (line.split("|") for line in PARAGRAPHS.splitlines()))
LISTED = "Mary Chen presented the chip. James Lin asked a question. She answered him directly."
UNLISTED = ROWS[5][0]


def links_for(ref, text):
    """(pronoun, antecedent text) for one paragraph, through the lesson's own pipeline."""
    mentions = ref.extract_mentions(text)
    return [(pronoun["text"], antecedent["text"] if antecedent else "<none>")
            for pronoun, antecedent in ref.resolve(mentions)]


def score(ref):
    """Correct links and the misses, as (pronoun, predicted, gold)."""
    hits, misses = 0, []
    for text, gold in ROWS:
        for (pronoun, got), want in zip(links_for(ref, text), gold):
            hits += got == want
            if got != want:
                misses.append((pronoun, got, want))
    return hits, misses


def spurious(ref):
    """Named entities produced only because a sentence starts with a capital letter."""
    named = {"John Smith", "Sarah Patel", "Apple", "Google", "Satya Nadella",
             "Priya Sharma", "Yusuf Demir", "Nokia", "March", "June"}
    return [m["text"] for text, _ in ROWS for m in ref.extract_mentions(text)
            if m["type"] == "ne" and m["span"][1] == 0 and m["text"] not in named]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    hits, misses = score(ref)
    feats = {g: {"gender": g, "number": "sg"} for g in "fmun"}
    return {
        "paragraphs": len(ROWS),
        "links": sum(len(gold) for _, gold in ROWS),
        "hits": hits,
        "misses": misses,
        "spurious": spurious(ref),
        "listed": links_for(ref, LISTED),
        "unlisted": links_for(ref, UNLISTED),
        "known_names": len(ref.FEMALE_FIRST) + len(ref.MALE_FIRST),
        "unknown": sorted({m["text"] for text, _ in ROWS for m in ref.extract_mentions(text)
                           if m["type"] == "ne" and m["features"]["gender"] == "u"}),
        "compatible": ref.agreement_score({"features": feats["f"]}, {"features": feats["u"]}),
        "clash": ref.agreement_score({"features": feats["f"]}, {"features": feats["m"]}),
        "sentence": ref.recency_score({"span": (1, 0)}, {"span": (0, 0)}),
        "tie": ref.recency_score({"span": (1, 0)}, {"span": (0, 4)}),
    }


def verify(result):
    misses = dict((pronoun, (got, want)) for pronoun, got, want in result["misses"])
    return [
        practice.Check(
            "ANSWER: 8 of 12 links, and the four failures are four different mechanisms",
            result["hits"] < result["links"],
            f"over {result['paragraphs']} paragraphs and {result['links']} pronouns the resolver "
            f"gets {result['hits']} right. The misses are {result['misses']} -- a tie-break, two "
            "capitalisation artefacts and a gender lookup, not four hard cases",
        ),
        practice.Check(
            "MECHANISM: `recency_score` returns the leftmost mention of the nearest sentence",
            result["tie"] == result["sentence"],
            f"the within-sentence term is `max(0, mention_token - candidate_token) * 0.01`, "
            f"clipped at zero, so a sentence-initial pronoun scores {result['tie']} against a "
            f"candidate at token 4 and {result['sentence']} against one at token 0 -- identical. "
            f"All candidates in the nearest sentence tie and `max` takes the first: "
            f"{misses['It'][0]} instead of {misses['It'][1]} in 'Apple sued Google'",
        ),
        practice.Check(
            "FINDING: capitalisation manufactures the competition",
            len(result["spurious"]) == 2,
            f"any capitalised alphabetic token becomes a named entity, and every sentence starts "
            f"with one: {result['spurious']} enter the candidate list as entities. Both then win "
            f"on recency -- `his` goes to {misses['his'][0]} rather than {misses['his'][1]}, `it` "
            f"to {misses['it'][0]} rather than {misses['it'][1]}",
        ),
        practice.Check(
            "FINDING: gender is a twenty-name lookup",
            result["listed"] != result["unlisted"],
            f"`FEMALE_FIRST` and `MALE_FIRST` hold {result['known_names']} first names between "
            f"them and every other name infers `u`, which `agreement_score` treats as compatible "
            f"with everything -- {result['unknown']} here. The same sentence resolves "
            f"{result['unlisted']} with unlisted names and {result['listed']} with listed ones",
        ),
        practice.Check(
            "MECHANISM: agreement can rank but never veto",
            abs(result["compatible"] - result["clash"]) < 2 * abs(result["sentence"]),
            f"the whole range of `agreement_score` is {result['compatible']} for a compatible "
            f"pair down to {result['clash']} for a gender clash, a span of "
            f"{result['compatible'] - result['clash']}, against {result['sentence']} per sentence "
            "of distance. A mismatched antecedent one sentence nearer still wins",
        ),
        practice.Check(
            "CONTROL: the easy links are all correct, so the resolver is not simply broken",
            result["hits"] >= 8,
            f"{result['hits']} of {result['links']} land, including every case where one "
            "compatible entity precedes the pronoun with nothing between them. The rules work; "
            "what fails is every situation where two candidates compete",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
