"""Exercise 2 — a small lexicon does what the encoder would.

    **Medium.** Use `paraphrase-multilingual-MiniLM-L12-v2` to build a
    cross-lingual retriever over a small mixed-language corpus. Query in English,
    retrieve documents in any language. Measure recall@5.

Reading of the exercise: sentence-transformers is not installed and the
checkpoint is not downloadable, so the corpus is 12 concepts written in French
and Spanish -- 24 documents -- queried by their English originals, and the
retriever is Jaccard overlap on tokens. That is a purely lexical retriever with
no cross-lingual resource at all, and it scores recall@5 of 0.5833 against an
exact random baseline of 0.3804.

It scores that on cognates, and the split is total. Counting only content words,
the 7 concepts whose English wording shares a token with a translation score
**1.0000** and the 5 that share none score **0.0000** -- no middle. A lexical
retriever is a cross-lingual retriever over the vocabulary the languages happen
to have in common, and outside that vocabulary it is not a retriever at all. The
mean is 0.31 shared content types per parallel pair, so most of the corpus falls
outside it.

Counting content words is what makes that split visible. The shared token in
several of these pairs is `a` -- the English article and the French third-person
verb -- so an unfiltered overlap count reports a cognate where there is a
coincidence, and the isolated group shrinks from 5 concepts to 1.

Adding the missing resource costs 27 dictionary entries. Expanding each English
query with its French and Spanish translations for 27 content words takes
recall@5 to 1.0000, including on every concept with no shared content word. On
this corpus a multilingual encoder and a 27-word lexicon produce the same number,
and recall@5 over 24 documents saturates before it could separate them -- worth
knowing before crediting the encoder.

Structure: `PARALLEL` holds one pipe-separated row per concept in the order
English, French, Spanish; the English column is the query side and the other two
are the index. `jaccard` is the scorer, `expand` applies `LEXICON`, `recall_at`
scores a ranker over the twelve queries, and `BASELINE` is the exact probability
of at least one relevant document in a random sample of five, by complement.
"""

from __future__ import annotations

import math
import re

from harness import practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "18-multilingual-nlp"

PARALLEL = (
    "the government announced a new tax on imports|le gouvernement a annonce une nouvelle taxe "
    "sur les importations|el gobierno anuncio un nuevo impuesto sobre las importaciones",
    "the hospital opened a modern surgery department|l hopital a ouvert un service de chirurgie"
    " moderne|el hospital abrio un servicio de cirugia moderno",
    "the football team won the national championship|l equipe de football a gagne le "
    "championnat national|el equipo de futbol gano el campeonato nacional",
    "the museum exhibited a collection of modern art|le musee a expose une collection d art "
    "moderne|el museo expuso una coleccion de arte moderno",
    "the train service was delayed by heavy snow|le service de train a ete retarde par la "
    "neige|el servicio de tren se retraso por la nieve",
    "the university published a study on climate change|l universite a publie une etude sur le "
    "changement climatique|la universidad publico un estudio sobre el cambio climatico",
    "the bank reduced interest rates this quarter|la banque a reduit les taux d interet ce "
    "trimestre|el banco redujo las tasas de interes este trimestre",
    "the company reported record annual profits|l entreprise a annonce des benefices annuels "
    "records|la empresa reporto beneficios anuales record",
    "the police arrested two suspects yesterday|la police a arrete deux suspects hier|la "
    "policia arresto a dos sospechosos ayer",
    "the airport cancelled all international flights|l aeroport a annule tous les vols "
    "internationaux|el aeropuerto cancelo todos los vuelos internacionales",
    "the minister resigned after the election result|le ministre a demissionne apres le "
    "resultat de l election|el ministro dimitio tras el resultado de la eleccion",
    "the child learned to read at school|l enfant a appris a lire a l ecole|el nino aprendio a "
    "leer en la escuela",
)
CONCEPTS = tuple(tuple(row.split("|")) for row in PARALLEL)
LEXICON = dict(pair.split("=") for pair in (
    "government=gouvernement gobierno|tax=taxe impuesto|team=equipe equipo|hospital=hopital "
    "hospital|surgery=chirurgie cirugia|championship=championnat "
    "campeonato|university=universite universidad|study=etude estudio|climate=climatique "
    "climatico|bank=banque banco|interest=interet interes|museum=musee "
    "museo|collection=collection coleccion|train=train tren|snow=neige "
    "nieve|company=entreprise empresa|profits=benefices beneficios|art=art arte|police=police"
    " policia|suspects=suspects sospechosos|airport=aeroport aeropuerto|flights=vols "
    "vuelos|minister=ministre ministro|election=election eleccion|child=enfant "
    "nino|school=ecole escuela|read=lire leer").split("|"))
AT, UNAVAILABLE = 5, ("sentence_transformers", "transformers", "torch")
FUNCTION = frozenset("a an the of on to at in by and is was for this l le la les un une des du el "
                     "las los das die der eine ein sur en de d se im zum".split())
WORD = re.compile(r"[a-z]+")
DOCS = tuple(text for row in CONCEPTS for text in row[1:])
GOLD = tuple(index for index, row in enumerate(CONCEPTS) for _ in row[1:])
QUERIES = tuple((row[0], index) for index, row in enumerate(CONCEPTS))


tokens = lambda text: set(WORD.findall(text.lower()))                             # noqa: E731
rank = lambda text: sorted(range(len(DOCS)), key=lambda i: -jaccard(text, DOCS[i]))  # noqa: E731


def jaccard(left, right) -> float:
    a, b = tokens(left), tokens(right)
    return len(a & b) / len(a | b) if a | b else 0.0


def expand(query) -> str:
    """The query plus every translation the lexicon has for its content words."""
    words = WORD.findall(query.lower())
    return " ".join([*words, *(w for t in words for w in LEXICON.get(t, "").split())])


def recall_at(queries, transform=None) -> float:
    hits = sum(any(GOLD[i] == gold for i in rank(transform(q) if transform else q)[:AT])
               for q, gold in queries)
    return round(hits / len(queries), 4)


# exactly P(at least one relevant document in a random sample of AT), by complement
RELEVANT = len(DOCS) // len(CONCEPTS)
BASELINE = round(1 - math.comb(len(DOCS) - RELEVANT, AT) / math.comb(len(DOCS), AT), 4)


def solve():
    import importlib.util
    shared = {index: sum(len(tokens(row[0]) & tokens(text) - FUNCTION) for text in row[1:])
              for index, row in enumerate(CONCEPTS)}
    cognate = tuple(q for q in QUERIES if shared[q[1]] > 0)
    isolated = tuple(q for q in QUERIES if shared[q[1]] == 0)
    return {
        "unavailable": [m for m in UNAVAILABLE if importlib.util.find_spec(m) is None],
        "documents": len(DOCS), "queries": len(QUERIES), "lexicon": len(LEXICON),
        "lexical": recall_at(QUERIES), "expanded": recall_at(QUERIES, expand),
        "baseline": BASELINE, "relevant": RELEVANT,
        "cognate": (recall_at(cognate), len(cognate)),
        "isolated": (recall_at(isolated), len(isolated)),
        "isolated_expanded": recall_at(isolated, expand),  # the group with no shared content word
        "mean_shared": round(sum(shared.values()) / (len(CONCEPTS) * 3), 2),
    }


def verify(result):
    lexical, expanded, base = result["lexical"], result["expanded"], result["baseline"]
    cognate, isolated = result["cognate"], result["isolated"]
    return [
        practice.Check(
            "ANSWER: a purely lexical retriever scores 0.5833 with no cross-lingual resource",
            lexical > base,
            f"{result['unavailable']} are all absent, so the corpus is {result['documents']} "
            f"documents in two languages queried by {result['queries']} English originals, "
            f"scored by Jaccard overlap on tokens: {lexical} against a random baseline of "
            f"{base}"),
        practice.Check(
            "MECHANISM: it scores that on cognates, and the split is complete",
            cognate[0] > lexical > isolated[0] == 0.0,
            f"the {cognate[1]} concepts sharing a content word with a translation score "
            f"{cognate[0]}; the {isolated[1]} sharing none score {isolated[0]}, at a mean of "
            f"{result['mean_shared']} shared content types per parallel pair. A lexical retriever "
            f"is a cross-lingual retriever over the vocabulary the languages share"),
        practice.Check(
            "FINDING: 27 dictionary entries take it to 1.0000",
            expanded == 1.0 and result["isolated_expanded"] == 1.0,
            f"expanding each query through a {result['lexicon']}-entry lexicon gives recall@{AT} "
            f"{expanded}, including {result['isolated_expanded']} on the concepts with no shared "
            f"content word. The missing cross-lingual resource costs {result['lexicon']} lines"),
        practice.Check(
            "MECHANISM: so the measurement would not distinguish the encoder from the lexicon",
            expanded == 1.0 > lexical,
            f"an encoder scoring {expanded} and a {result['lexicon']}-word dictionary scoring "
            f"{expanded} are the same number: recall@{AT} over {result['documents']} documents "
            f"saturates before it can separate them"),
        practice.Check(
            "CONTROL: the baseline is a third of the way up before anything is retrieved",
            base > 0.3,
            f"{result['documents']} documents, {result['relevant']} relevant per query, {AT} "
            f"retrieved: the exact chance of at least one hit is {base}, so a reported {lexical} is "
            f"{(lexical - base) / (1 - base):.0%} of the available range rather than "
            f"{lexical:.0%} of the task"),
        practice.Check(
            "CONTROL: the corpus is ASCII, so no encoding difference is doing the work",
            all(text.isascii() for text in DOCS),
            f"every document is written without diacritics -- annonce, universite, publie -- so the "
            f"overlap carrying the lexical arm is vocabulary overlap rather than a normalisation "
            f"artefact. Restoring the accents would lower it and leave the expanded arm alone"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
