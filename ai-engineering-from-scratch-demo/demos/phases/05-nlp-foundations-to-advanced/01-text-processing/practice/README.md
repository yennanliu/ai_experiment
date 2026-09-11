<!-- generated:start -->
# 05-nlp-foundations-to-advanced / 01-text-processing

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/01-text-processing/) · upstream spec
`phases/05-nlp-foundations-to-advanced/01-text-processing/docs/en.md`

```bash
uv run demo practice run 01-text-processing --ex 1
uv run demo explain 01-text-processing --ex 1
uv run pytest demos/phases/05-nlp-foundations-to-advanced/01-text-processing
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Extend `tokenize` to keep URLs as single tokens. Test: `tokenize("Visit https://example… | code | T0 | `ex01_alternation_order_beats_the_pattern.py` |
| 2 | Medium. Implement Porter step 1b. If a word contains a vowel and ends in `ed` or `ing`, remov… | code | T0 | `ex02_the_four_rules_the_wording_omits.py` |
| 3 | Hard. Build a lemmatizer that uses WordNet as a lookup table but falls back to your Porter st… | code | T0 | `ex03_the_fallback_is_the_small_half.py` |
<!-- generated:end -->

## Answers

Three exercises whose stated acceptance criteria all pass on the broken answer.
Exercise 1 names one test sentence, and it is one of the five (of twelve) that
the standard buggy URL pattern happens to get right. Exercise 2 names
`hopping -> hop`, and the reading of step 1b that gets `hopping -> hop` scores
**5 of the 16** examples Porter published. Exercise 3 asks for a fallback worth
**4 rows out of 48**, against **15** for the piece its own wording — "WordNet as
a lookup table" — quietly stands in for. In each case the number that settles
the exercise is not the number the exercise asks for.

All three run at **T0** on the standard library alone: the lesson's `code/main.py`
imports nothing but `re`.

### 1 — Alternation order beats the pattern

**ANSWER: four tokens, exactly one of them the URL.**

| tokenizer | `Visit https://example.com today.` |
|---|---|
| the lesson's `WORD_RE` | 10 tokens, **0** URLs — `['Visit', 'https', ':', '/', '/', 'example', '.', 'com', 'today', '.']` |
| with a URL rule **first** | 4 tokens, **1** URL — `['Visit', 'https://example.com', 'today', '.']` |

**MECHANISM: placement decides it, not the pattern.** Appending the same
`https?://\S+` *after* the word rule leaves the output **byte-identical to the
lesson's on all 12** labelled sentences — 0/12 URLs kept whole. `re` alternation
is ordered and first-match-wins at each position, so `[A-Za-z]+` matches `https`
at offset 6 before the URL branch is ever tried. The lesson says "add patterns
before the general ones" without saying why; this is why.

**FINDING: the exercise's own test passes on the standard broken pattern.**

| URL rule | whole URLs recovered | sentences missed |
|---|---:|---|
| none (the lesson) | 0 / 12 | all |
| `https?://\S+` | **5 / 12** | 2, 3, 4, 6, 7, 8, 12 |
| refuse a trailing `.,;:!?)]}"'` | 11 / 12 | 9 |
| + re-attach a balanced `)` | **12 / 12** | — |

`\S+` eats the sentence-final stop, the comma, the closing bracket and the
closing quote — it destroys **7 of 12** punctuation boundaries. Sentence 1 is
not among its misses, because its URL happens to be followed by a space. That
sentence is the one the exercise names as its test.

**FINDING: the last one needs a second pass, not a better character class.**
The trimmed pattern's single miss is `https://example.com/path_(v2` — a `(`
the path itself opened. No trailing-character rule can decide it: `)` closes
the URL in sentence 4 (`(see https://example.com)`) and closes nothing in
sentence 9. Re-attaching a `)` only when the URL carries an unmatched `(`
reaches 12/12 and leaves sentences 4 and 12 alone.

**CONTROL: character coverage ranks nothing.** All four arms satisfy
`"".join(tokens) == the input with whitespace removed`. The obvious round-trip
test calls the 0/12 arm and the 12/12 arm equally correct.

**FINDING: sharding a URL erases the scheme, twice.** As tokens, `http://` and
`https://` differ in one place. As stems and as lemmas they are identical:

| | `http://example.com` | `https://example.com` |
|---|---|---|
| tokens | `['http', ':', …]` | `['https', ':', …]` |
| stems | `['http', ':', '/', '/', 'example', '.', 'com']` | **same** |
| lemmas | `['http', ':', '/', '/', 'example', '.', 'com']` | **same** |

`stem_step_1a` strips the final `s` of the shard `https`, and
`lemmatize(_, "NOUN")` strips it again — so the lesson's own pipeline erases
the distinction on both paths.

### 2 — The four rules the wording omits

**ANSWER: the full step scores 16/16 on Porter's own examples; the exercise's
sentence, transcribed, scores 5/16.**

The grading set is the worked-example table Porter published with the rules —
the file the lesson's Further Reading links to — so each miss names an omitted
rule rather than a taste. The 11 misses fall into exactly four groups.

| omitted rule | words | literal reading gives | correct |
|---|---|---|---|
| vowel test is on the **stem** | `bled`, `sing` | `bl`, `s` | `bled`, `sing` |
| `eed` is its own measure-gated rule | `feed`, `agreed` | `fe`, `agre` | `feed`, `agree` |
| `*d` carries a **not (L, S, Z)** exception | `falling`, `hissing`, `fizzed` | `fal`, `his`, `fiz` | `fall`, `hiss`, `fizz` |
| `AT/BL/IZ` and `(m=1, *o)` restore an `e` | `conflated`, `troubled`, `sized`, `filing` | `conflat`, `troubl`, `siz`, `fil` | `conflate`, `trouble`, `size`, `file` |

Both cases the wording names pass either way: `hopping -> hop` in both arms.

**MECHANISM: "contains a vowel" is about the stem, and can empty a word.**
`bled` and `sing` carry their only vowel *inside* the suffix — which is why
Porter put them in the table. Read over the whole word, the rule strips it. On
the words `ing` and `ed` themselves it returns the **empty string**.

**FINDING: the lesson's own `stem_step_1a` guards `ss`, and a literal step 1b
un-guards it two functions later.** `stem_step_1a` returns `process`, `miss`
and `pass` untouched via an explicit `endswith("ss")` rule. Run three families
of four forms each through it and then through step 1b:

| step 1b | distinct stems for 3 families | stems |
|---|---:|---|
| full Porter | **3** | `miss`, `pass`, `process` |
| the literal reading | **6** | `mis`, `miss`, `pas`, `pass`, `proces`, `process` |

The literal rule splits every base form off its own inflections — the exact
opposite of what a stemmer is for, and it does so by undoing a guard the same
stemmer applies one step earlier.

**CONTROL: step 1b is a phase, not a stemmer.** The step-1b-correct output for
`treated` is `treate` and for `during` is `dure`; the restored `e` is step 5a's
to remove, and the exercise does not ask for step 5a. Grading step 1b against a
full stemmer's output would mark both wrong. On the lesson's own prose — 586
word types, 43 ending in `ed`/`ing` — the two readings disagree on **15**,
including `string -> str`, `thing -> th` and `speed -> spe`.

### 3 — The fallback is the small half

Neither `nltk` nor `spacy` is installed, and the lesson ships no tagged corpus,
so both are supplied and said so: the corpus is **48 hand-labelled
`(word, POS, lemma)` triples**, and WordNet is its own published `morphy(7WN)`
detachment rules run over a vocabulary. That vocabulary is the answer key's own
range — the most favourable dictionary morphy can be given here — so its score
is a **ceiling**, not an estimate of real WordNet.

**ANSWER: the hybrid wins, at 31/48.**

| arm | accuracy |
|---|---:|
| the lesson's `lemmatize` | 12 / 48 = 0.250 |
| Porter (`stem_step_1a` + step 1b) | 14 / 48 = 0.292 |
| morphy alone | 27 / 48 = 0.562 |
| **hybrid** (morphy, else Porter) | **31 / 48 = 0.646** |
| morphy + a 15-entry exception list | **42 / 48 = 0.875** |

**MECHANISM: the hybrid's score is a sum, not a synergy.** Morphy abstains on
**20** of 48 rows and the stemmer is consulted on exactly those, so the hybrid
is `27 + 4 = 31` by construction. It cannot change a row WordNet covers,
however wrong that row is.

**FINDING: the entire fallback gain is exercise 2's double-consonant rule.**
The four rows the stemmer rescues are `planned`, `running`, `sitting`,
`stopped` — every one a doubled final consonant, which morphy cannot reach
because `stopp` and `runn` are not words. Porter scores **14/48 overall but
4/20 on the abstentions**: it is weakest exactly where it is called.

**FINDING: the exercise asks for the small half of the repair.** A 15-entry
exception list of the kind real WordNet ships is worth **+15** rows; the Porter
fallback is worth **+4**. "WordNet as a lookup table" is a description of that
exception list, and the exercise names it and then asks for the other fix.

**FINDING: WordNet is rules *plus a membership test*, and the membership test
is the part the lesson's lemmatizer lacks.** On the six nouns that are already
lemmas:

| word | `lemmatize(w, "NOUN")` | morphy |
|---|---|---|
| `gas` | `ga` | `gas` |
| `class` | `clas` | `class` |
| `analysis` | `analysi` | `analysis` |
| `news` | `new` | `news` |
| `series` | `serie` | `series` |
| `lens` | `len` | `lens` |

Morphy scores 6/6 by checking the word itself against the dictionary first. The
lesson scores 0/6: its `NOUN` branch strips a final `s` unconditionally —
though `stem_step_1a`, two functions above it, guards `ss` explicitly.

**CONTROL: the lesson's lemmatizer scores below the lesson's own stemmer.** On
lemmatization — the task the lesson's rule of thumb says lemmatizers win —
`lemmatize` gets 12/48 and the stemmer gets 14/48. It also disagrees with
itself about case: `Cats -> cat`, `Dogs -> Dog`, `Walking -> Walk`,
`Walked -> walked`. The table path lowercases, the two suffix paths do not, and
the final fallback does.

### A note on file lengths

The three files run 132 / 148 / 142 lines of code, over D14's 120-line target
and under its 150-line ceiling. The overrun is fixture, not machinery: exercise
1 carries 12 sentences each labelled with the URL span a human would draw,
exercise 2 carries Porter's 16-example table plus his four conditions (`cons`,
`measure`, `*d`, `*o`) written out, and exercise 3 carries a 48-row tagged
corpus, morphy's published detachment rules and a 15-entry exception list —
all three of which the exercise assumes exist and none of which ship. Exercise 3
imports exercise 2's `porter` via `practice.load_module` rather than carrying a
second copy.
