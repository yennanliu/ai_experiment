"""Exercise 1 — recall-only ROUGE is maximised by the article.

    **Easy.** Run TextRank on 5 news articles. Compare the top-3 sentences to a
    reference summary. Measure ROUGE-L. You should see 30-45 ROUGE-L on
    CNN/DailyMail-style articles.

Reading of the exercise: ROUGE-L is not in the lesson. `code/main.py` ships
`rouge_n`, which is recall against the reference and nothing else -- no precision
term, no length penalty -- so it rises with every sentence added and is maximised
by returning the whole article. Across the five articles written below it goes
0.1322, 0.2453, 0.3348, 0.4513, 0.5272 at k = 1 to 5 and 0.6584 at k = 7, which
is the article itself. Ranking summarisers on it ranks the one that summarises
least.

ROUGE-L, implemented here as the longest-common-subsequence F-measure the
exercise names, does have a precision term, and it stops paying for length: the
same sweep gives 0.1674, 0.2410, 0.2941, 0.3418, 0.3577 and 0.3589. From k=5 to
the full article `rouge_n` gains 25% and ROUGE-L gains 0.3%. On two of the five
articles ROUGE-L peaks strictly before the end.

The exercise's own number is a near miss. The top-3 summary it specifies scores
ROUGE-L 0.2941 -- just under the 30 of its predicted 30-45 band -- and the band
opens at k=4, 0.3418. That is a small discrepancy and it is the whole
measurement: the number the exercise quotes belongs to a different k than the one
it tells you to take.

Structure: `ARTICLES` is five (article, reference) pairs written in the shape the
exercise describes -- seven sentences, a reference naming the facts a summary
should carry. `rouge_l` is the LCS F-measure with the standard beta of 1.2;
`sweep` runs the lesson's own `textrank` at each k and scores both metrics.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "12-text-summarization"

BETA, TOPS = 1.2, (1, 2, 3, 4, 5, 7)
BAND = (0.30, 0.45)
ARTICLES = (
    ("Researchers at a Canadian university published a paper on efficient transformers. The paper "
     "introduces a new attention variant that runs in linear time. The authors trained models up "
     "to 1 billion parameters on public data. Benchmarks show the new attention matches standard "
     "attention on most tasks. The authors released training code and weights on GitHub. Several "
     "research labs have already replicated the main results. The paper has been accepted at "
     "NeurIPS.",
     "Researchers introduced a linear-time attention variant, trained up to 1 billion parameters, "
     "matched standard attention on benchmarks, and released code and weights."),
    ("The city council approved a new transit plan on Tuesday evening. The plan adds three bus "
     "routes across the eastern districts. Construction on the first route begins in March. "
     "Officials estimate the work will cost 40 million dollars. Residents raised concerns about "
     "noise during the consultation. The council promised a review after the first year. "
     "Opposition members voted against the proposal.",
     "The council approved a transit plan adding three bus routes, costing 40 million dollars, "
     "with construction beginning in March and a review after one year."),
    ("A storm system moved across the northern coast overnight. Wind speeds reached 90 kilometres "
     "per hour in exposed areas. Ferries were cancelled and two bridges closed to high vehicles. "
     "Power was lost to roughly 12000 homes. Crews restored most connections by midday. "
     "Forecasters expect conditions to ease by Thursday. No injuries have been reported.",
     "A storm brought 90 kilometre winds to the northern coast, cancelling ferries and cutting "
     "power to 12000 homes, with most connections restored by midday and no injuries reported."),
    ("The company reported quarterly revenue of 2 billion dollars. Revenue grew 14 percent "
     "compared with the same quarter last year. Operating margin narrowed slightly on higher "
     "shipping costs. The chief executive said demand remained strong in Europe. The board "
     "approved a dividend of 30 cents per share. Shares rose 4 percent in after hours trading. "
     "Guidance for the next quarter was left unchanged.",
     "The company reported 2 billion dollars in quarterly revenue, up 14 percent, approved a 30 "
     "cent dividend, and left guidance unchanged as shares rose 4 percent."),
    ("A hospital trial tested a new treatment for chronic pain. The trial enrolled 300 patients "
     "across four sites. Half received the treatment and half received a placebo. Patients on the "
     "treatment reported lower pain scores after eight weeks. Side effects were mild and occurred "
     "in 6 percent of participants. The results were published in a peer reviewed journal. A "
     "larger trial is planned for next year.",
     "A trial of 300 patients across four sites found a new chronic pain treatment lowered pain "
     "scores after eight weeks with mild side effects in 6 percent, and a larger trial is planned."),
)


def lcs(left, right) -> int:
    table = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i, a in enumerate(left, 1):
        for j, b in enumerate(right, 1):
            table[i][j] = table[i - 1][j - 1] + 1 if a == b else max(table[i - 1][j], table[i][j - 1])
    return table[-1][-1]


def rouge_l(hypothesis, reference) -> float:
    """The LCS F-measure the exercise names, which the lesson does not ship."""
    hyp, ref = hypothesis.lower().split(), reference.lower().split()
    common = lcs(hyp, ref)
    if not common:
        return 0.0
    precision, recall = common / len(hyp), common / len(ref)
    return (1 + BETA ** 2) * precision * recall / (recall + BETA ** 2 * precision)


def summaries(ref, top_k) -> list:
    return [" ".join(ref.textrank(article, top_k=top_k)) for article, _ in ARTICLES]


def sweep(ref) -> dict:
    rows = {}
    for top_k in TOPS:
        made = summaries(ref, top_k)
        rows[top_k] = {
            "recall": round(sum(ref.rouge_n(s, r, 1) for s, (_, r) in zip(made, ARTICLES))
                            / len(ARTICLES), 4),
            "rouge_l": round(sum(rouge_l(s, r) for s, (_, r) in zip(made, ARTICLES))
                             / len(ARTICLES), 4),
            "per_article": [round(rouge_l(s, r), 3) for s, (_, r) in zip(made, ARTICLES)]}
    return rows


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = sweep(ref)
    whole = TOPS[-1]
    peaks = [max(TOPS, key=lambda k: rows[k]["per_article"][i]) for i in range(len(ARTICLES))]
    return {
        "rows": rows, "articles": len(ARTICLES), "has_rouge_l": hasattr(ref, "rouge_l"),
        "ships": sorted(n for n in dir(ref) if n.startswith("rouge")),
        "recall": [rows[k]["recall"] for k in TOPS], "curve": [rows[k]["rouge_l"] for k in TOPS],
        "peaks": peaks, "early": sum(p < whole for p in peaks),
        "gain": {"recall": round(rows[whole]["recall"] / rows[5]["recall"] - 1, 4),
                 "rouge_l": round(rows[whole]["rouge_l"] / rows[5]["rouge_l"] - 1, 4)},
        "sentences": len(ref.sentence_split(ARTICLES[0][0])),
    }


def verify(result):
    rows, recall, curve = result["rows"], result["recall"], result["curve"]
    whole, gain = TOPS[-1], result["gain"]
    return [
        practice.Check(
            "ANSWER: the metric the lesson ships rises with every sentence and peaks on the article",
            recall == sorted(recall) and rows[whole]["recall"] == max(recall),
            f"`rouge_n` is recall against the reference with no precision term, so over "
            f"{result['articles']} articles it reads {dict(zip(TOPS, recall))} at k = {list(TOPS)} "
            f"-- and k={whole} is the whole article, which is where it is highest. Ranking "
            f"summarisers on it ranks the one that summarises least"),
        practice.Check(
            "MECHANISM: ROUGE-L is not in the lesson at all",
            not result["has_rouge_l"] and result["ships"] == ["rouge_n"],
            f"`code/main.py` ships {result['ships']} and the exercise asks for ROUGE-L. The LCS "
            f"F-measure has to be written first, which is also the point: the precision term is the "
            f"difference between the two metrics and it is the term that is missing"),
        practice.Check(
            "FINDING: with a precision term the curve stops paying for length",
            gain["recall"] > 20 * gain["rouge_l"],
            f"from k=5 to the whole article, `rouge_n` gains {gain['recall']:+.1%} and ROUGE-L gains "
            f"{gain['rouge_l']:+.1%}. The ROUGE-L curve is {dict(zip(TOPS, curve))} -- flat across "
            f"its last two points where the recall-only curve climbs a quarter of its own value"),
        practice.Check(
            "FINDING: on two of the five articles ROUGE-L peaks strictly before the end",
            0 < result["early"] < result["articles"],
            f"per-article peaks fall at k = {result['peaks']}, so {result['early']} of "
            f"{result['articles']} articles score best at a summary shorter than the source. "
            f"`rouge_n` peaks at k={whole} on every one of them, because it cannot do otherwise"),
        practice.Check(
            "FINDING: the top-3 the exercise specifies scores just under the band it predicts",
            rows[3]["rouge_l"] < BAND[0] < rows[4]["rouge_l"] < BAND[1],
            f"the top-3 summary scores ROUGE-L {rows[3]['rouge_l']}, below the "
            f"{BAND[0] * 100:.0f}-{BAND[1] * 100:.0f} the exercise predicts; the band opens at k=4, "
            f"{rows[4]['rouge_l']}. The number quoted belongs to a different k than the one the "
            f"exercise tells you to take"),
        practice.Check(
            "CONTROL: the articles are the length TextRank needs to rank anything",
            result["sentences"] == whole,
            f"each article is {result['sentences']} sentences, and `textrank` returns the input "
            f"unranked when `n <= top_k`. At k={whole} the summary is the article by that guard "
            f"rather than by the scoring, which is why it is the right control: it is the output of "
            f"doing no work at all"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
