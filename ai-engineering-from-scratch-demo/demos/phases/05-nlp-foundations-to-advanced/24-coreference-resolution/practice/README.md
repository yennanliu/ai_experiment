<!-- generated:start -->
# 05-nlp-foundations-to-advanced / 24-coreference-resolution

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/24-coreference-resolution/) · upstream spec
`phases/05-nlp-foundations-to-advanced/24-coreference-resolution/docs/en.md`

```bash
uv run demo practice run 24-coreference-resolution --ex 1
uv run demo explain 24-coreference-resolution --ex 1
uv run pytest demos/phases/05-nlp-foundations-to-advanced/24-coreference-resolution
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run the rule-based resolver in `code/main.py` on 5 hand-crafted paragraphs. Measure men… | code | T0 | `ex01_recency_resolves_to_the_leftmost_mention.py` |
| 2 | Medium. Use a pretrained neural coref model on a news article. Compare clusters against your… | code | T0 | `ex02_the_cluster_metric_is_the_flattering_one.py` |
| 3 | Hard. Build a coref-enhanced NER pipeline: NER first, then merge via coref clusters. Measure… | code | T0 | `ex03_coverage_can_only_go_up_including_when_it_is_wrong.py` |
<!-- generated:end -->

## Answers

One rule-based resolver, three questions about it. The link accuracy in exercise
1 is the honest number; exercise 2's cluster metric flatters the same output, and
exercise 3's coverage metric cannot go down at all. Every failure traces to four
bad links, and each of the four has a different cause.

All three run at **T0** — `code/main.py` imports only `re`.

### 1 — Recency resolves to the leftmost mention

Seven paragraphs, twelve pronouns. **8 of 12.**

| pronoun | predicted | gold | cause |
|---|---|---|---|
| It | Apple | **Google** | tie broken by document order |
| his | Analysts | **Satya Nadella** | sentence-initial capital |
| it | Engineers | **a device** | sentence-initial capital |
| She | Yusuf Demir | **Priya Sharma** | name not on the gender list |

**MECHANISM: `recency_score` is not recency.** Sentence distance costs 2.0, and
the within-sentence term is `max(0, mention_token - candidate_token) * 0.01` —
clipped at zero. A sentence-initial pronoun has token index 0, so that term is 0
for *every* candidate in an earlier sentence: all of them tie, and `max` returns
the first. The resolver picks the **leftmost** mention of the nearest sentence.

**FINDING: capitalisation manufactures the competition.** Any capitalised
alphabetic token becomes a named entity, and every sentence begins with one.
`Analysts` and `Engineers` enter the candidate list as entities and win on
recency.

**FINDING: gender is a twenty-name lookup.** `FEMALE_FIRST` and `MALE_FIRST` hold
ten names each; every other name infers `u`, which `agreement_score` treats as
compatible with everything. The same sentence resolves `She → Yusuf Demir` with
unlisted names and `She → Mary Chen` with listed ones.

**MECHANISM: agreement can rank but never veto.** Its whole range is +2.0 to
−1.0, against −2.0 per sentence of distance — a mismatched antecedent one
sentence nearer still wins.

### 2 — The cluster metric is the flattering one

No neural coref is installed, so the system compared is the lesson's own
`clusters`; the annotation is exercise 1's gold links extended by string
identity.

| measure | value |
|---|---:|
| mention-link accuracy | **0.6667** |
| B³ precision / recall / F1, all mentions | 0.8725 / 0.8333 / **0.8525** |
| B³ over non-singleton gold mentions | 0.8768 / **0.7536** / 0.8106 |

**ANSWER: the cluster metric is more flattering than the link accuracy, on the
same output** — because 11 of the 21 gold clusters are singletons, and a
singleton is a cluster no system can get wrong.

**FINDING: where it fails is one merged cluster per bad link** — `Apple + It`,
`Analysts + his`, `Engineers + it`, `Yusuf Demir + She + him`. A link error is a
cluster error for every mention on both sides; it is never local.

**FINDING: and it fails the other way for a reason link accuracy cannot see.**
`resolve` iterates only pronouns and `clusters` unions only what it returned, so
no two non-pronoun mentions are ever joined — string identity included. Nokia
comes back `[Nokia] [the phone] [Nokia, It]` against gold `[Nokia, Nokia, It]
[the phone]`. **Every entity named twice is two clusters.**

**MECHANISM: the two failures partly cancel** — 22 predicted clusters against 21
gold, from a system that mis-links a third of its pronouns.

**CONTROL: `main()` prints only clusters above size one**, so the singletons
never reach the screen.

### 3 — Coverage can only go up, including when it is wrong

18 gold entity-sentence pairs:

| pipeline | pairs | precision | recall |
|---|---:|---:|---:|
| NER only | 9 | **1.0000** | 0.5000 |
| NER + coref merge | 16 | 0.9375 | **0.8333** |

**ANSWER: two thirds more coverage, at 0.0625 of precision.** 7 attributions are
added and 6 are right; the one that is not is `Apple` taking the sentence that
belongs to Google.

**MECHANISM: the added coverage inherits the pronoun link accuracy.** Every
addition comes from a link, and 0.6667 of the links are correct.

**FINDING: coverage is monotone, so the metric cannot see a wrong merge.**
Merging only adds sentences — 0 NER-only attributions are lost — so coverage
rises when the link is right and rises when it is wrong. Reported alone, as the
exercise asks, it scores a correct merge and a false one identically.

**FINDING: the three pairs still missing come from the same four links** —
`Google`, `Satya Nadella`, `Priya Sharma`. The false addition and the misses are
one failure counted twice.

**MECHANISM: the NER layer has no types at all.** 4 of its 13 entities are months
(`March`, `June`) or plural common nouns that began a sentence (`Analysts`,
`Engineers`). "NER, then merge via coref" is merging dates into entity clusters
before anything is measured.

**CONTROL: all of the precision loss belongs to the merge.** NER-only is 1.0000
by construction, and the number the exercise asks for is the one that hides that.
