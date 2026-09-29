<!-- generated:start -->
# 19-capstone-projects / 17-personal-ai-tutor

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/17-personal-ai-tutor/) · upstream spec
`phases/19-capstone-projects/17-personal-ai-tutor/docs/en.md`

```bash
uv run demo practice run 17-personal-ai-tutor --ex 1
uv run demo explain 17-personal-ai-tutor --ex 1
uv run pytest demos/phases/19-capstone-projects/17-personal-ai-tutor
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run the efficacy study with and without the adaptive learner model (random concept order). Re… | code | T0 | `ex01_adaptive_wins_by_1_69_mastery_points_and_a_learner_who_never_learns_scores_9_51.py` |
| 2 | Add a multimodal probe: the same concept question delivered as text, voice, and photo. Measur… | code | T0 | `ex02_the_preferred_modality_finishes_7_8_turns_sooner_and_a_10pct_channel_miss_rate_erases_it.py` |
| 3 | Build a parent dashboard: topics practiced, mastery trajectories, upcoming concepts, safety e… | code | T0 | `ex03_the_learner_record_has_no_timestamp_and_the_web_half_masters_a_lesson_on_one_right_answer.py` |
| 4 | Add a language-switch mode: the tutor accepts Spanish input and teaches in Spanish. Measure X… | code | T0 | `ex04_an_english_filter_catches_1_of_8_spanish_messages_and_the_lesson_ships_no_guard_to_measure.py` |
| 5 | Stress the memory privacy: verify that learner A cannot see learner B's data even through a v… | code | T0 | `ex05_a_hash_deduped_voice_memory_leaks_on_4_of_4_attacks_and_the_web_half_shares_one_store.py` |
<!-- generated:end -->

## Answers

Every exercise runs the lesson's `code/main.py`. It is a simulated study:
Bayesian knowledge tracing (BKT) with fixed parameters, an 11-concept
algebra DAG, a four-branch Socratic policy, and a learner whose chance of a
right answer is a sigmoid of ability plus 1.5 times the tutor's own mastery
estimate. There is no LLM, voice, photo, memory store or safety filter in
the code, so exercises 2-5 build those pieces around the reference loop
and measure what it does with them. Exercises 3 and 5 also read the
lesson's TypeScript web half (`code/ts/src/`). External facts were read on
2026-09-29 from the COPPA rule (16 CFR 312.2, 312.6 and 312.10, at
law.cornell.edu/cfr/text/16/312.*), the X-Guard paper (arXiv:2504.08848)
and the Llama Guard 4 model card (huggingface.co/meta-llama/Llama-Guard-4-12B).

### 1 — adaptive wins by 1.69 mastery points, and a learner who never learns scores 9.51

**Adaptive beats random concept order by +1.69 mastery points out of 11
(paired 95% CI +0.68 to +2.71).** The protocol is `main()`'s own: 10
learners, 60 turns, the same seeds. `main.py` ships a round-robin baseline,
not a random one, so the random arm was added with the reference functions.

| arm | mastery sum (of 11) | expected post-test |
|---|---:|---:|
| adaptive (`run_adaptive`) | 9.81 | 74.6% |
| round-robin (`run_baseline`) | 8.45 | |
| random order | 8.12 | 69.4% |

The pre-test is 52.3% in both arms, so the gain gap is 5.2 points out of
100. Across 200 fresh 10-learner cohorts the delta averages +1.29 (sd
0.39, range +0.09 to +2.21), and adaptive never loses a cohort. It does
lose single learners: 2 of 10 do better on round-robin. Adaptive stops as
soon as every concept reads 0.85, and 7 of 10 learners finish early (47.7
of 60 turns on average).

**The study measures the tutor's belief, and that belief cannot tell
learning from guessing.** The outcome is the BKT estimate, and the
simulator feeds that estimate back in as knowledge. A learner at mastery 0.2
already answers 52.5-57.4% right, against BKT's `p_guess` of 0.15. So two
right answers in a row read 0.927, which counts as mastered. A learner who
never learns and answers 55% right forever scores 9.51 under the adaptive
policy, more than real learners average on round-robin. The lesson's
worked example does not match the code either: "Use It" shows 0.62 -> 0.77
after a right answer, but `bkt_update` gives 0.918. Getting 0.77 needs a
prior of 0.320.

### 2 — the preferred modality finishes 7.8 turns sooner, and a 10% channel miss rate erases it

**Yes on clean channels, and the size of the effect is set by the channel,
not the preference.** Each learner does the curriculum through the
reference `run_adaptive` once per modality. Two numbers are assumed: a
preferred modality adds 0.3 logits (about +7 points of P(correct)), and
voice loses 10% and photo 5% of right answers to misreads.

| condition | preferred minus other modalities (turns to finish) |
|---|---:|
| clean channels | -7.8 (sd 2.4; faster in 100/100 cohorts) |
| misreads on: text-preferrers | -15.7 |
| misreads on: photo-preferrers | -9.2 |
| misreads on: voice-preferrers | -2.0 |

Over 300 voice-preferring learners, the preference is worth -7.9 turns on
a clean channel and is gone at a 10% miss rate. That is the point where
(1 - miss) x P(correct | preferred) falls back to the unpreferred P(correct).

**With no preference at all, channel error alone looks like one.** Set the
preference to 0 and keep the misreads: voice-preferrers take 7.1 more
turns in "their" modality and text-preferrers 7.2 fewer. The learner model
cannot tell the difference, because `bkt_update` takes a bare boolean. A
right answer lost to the channel at mastery 0.648 leaves 0.277 instead of
0.927. The 10-learner study also splits 4/3/3 across the three
preferences, so each per-modality estimate rests on 3-4 learners.

### 3 — the learner record has no timestamp to delete by, and the web half masters a lesson on one right answer

**The dashboard is `dashboard()` in the file.** Here is the week-1 report
for the study's first learner:

- **Topics:** 8 of 11 practised in 30 turns, with attempts and right answers.
- **Trajectories:** replayed from `history` through `bkt_update`; each ends at the stored mastery.
- **Upcoming:** `distributive_property`, the only concept whose prerequisites are met.
- **Safety events:** 6 guardrail hits on 3 session days, shown as (date, category) counts.

None of the 6 raw messages appears in the output, although two carry a
name, an address and a phone number. The COPPA parts are tested too:

- An unverified requester gets only "parent verification required" (312.6(a)(3)).
- The parent's delete leaves 0 items (312.6(a)(2)).
- With the lesson's 1-year timeframe written down, the purge keeps 30, 10 and 0 turns at days 365, 368 and 370 (312.10).

The rule sets no fixed period. It requires a written timeframe and forbids
keeping data indefinitely, so the lesson's "1 year" is a policy the
operator must justify, not the law.

**The lesson's learner record cannot support this.** `LearnerState` has 3
fields: `learner_id`, `mastery` and `history`. There is no timestamp, no
guardrail log and no link to a parent, so every date and event here had
to be added. Reading the record also changes it. `mastery` is a
`defaultdict`, so the reference's own `mastery_sum` adds the 3 concepts
the learner never practised (8 entries become 11). The two halves of the
lesson also draw different trajectories. The TypeScript store scores
`0.3 * score + 0.7 * observed`, so one right answer gives exactly 0.7,
which equals `MASTERY_THRESHOLD`. Python's BKT needs two right answers.

### 4 — an English filter catches 1 of 8 Spanish messages, and the lesson ships no guard to measure

**The Spanish mode works end to end.** The pieces:

- Every one of the 4 actions `socratic_policy` can return has Spanish wording, and none is flagged.
- The language detector labels 30 of 30 messages correctly.
- The Spanish signal reader parses 6 of 6 ("no entiendo por qué x = 2", "es seis", ...).

X-Guard translates with mBART-50 and then judges with a 3B model. Here a
word-level dictionary stands in for the translator and an English age
filter stands in for the judge.

| guard | English | Spanish | code-switched | false alarms (4 safe per language) |
|---|---:|---:|---:|---:|
| English age filter alone | 8/8 | 1/8 | 5/8 | 0 |
| translate, then filter (X-Guard shape) | 8/8 | 8/8 | 8/8 | 0 |

The 8/8 is partly by construction, because the dictionary knows these
words. On 4 held-out inflections ("me voy a matar", "quiero morirme", ...)
the translation step catches 0. A translate-first guard covers only what
its translator covers, so with the real system that test belongs on
mBART-50's output. The English filter's one Spanish hit is the loanword in
"cuéntame una historia sexy". An English-only signal reader gets 2 of 6.

**The lesson has nothing to measure yet.** Its `code/` (Python and
TypeScript) has 0 lines about a guard, safety, moderation or language. The
doc names X-Guard only once, in this exercise. The stack it describes is
Llama Guard 4 (5 mentions), and Llama Guard 4 already lists Spanish among
its 7 non-English languages. X-Guard's 132 languages matter for
low-resource languages and code-switching, not for Spanish.

### 5 — a hash-deduped voice memory leaks on 4 of 4 attacks, and the web half shares one store

**Verified for a store scoped to the session's authenticated learner.**
Learners A and B come from the reference `run_adaptive`. From A's session,
four re-ingest attacks target B: replay B's exact clip, a clip saying
"this is learner_b", a clip in B's voice, and a clip asking for
"learner_b's mistakes". Results:

- A sees none of B's data in any of the 4.
- Each attempt is logged with its target and raises an alert.
- Two benign re-ingests by A raise no alert, and A still reaches its own memory.
- Alerted clips are quarantined rather than stored, so 0 of B's clips end up in A's memory. This matters because a child's voice recording is personal information under 312.2.

**The obvious design leaks on every attack.** A shared store that dedupes
clips by hash and takes identity from the clip (a named learner, then the
speaker) hands A learner B's record on 4 of 4 attacks. What leaks is B's
semantic memory, the concepts B got wrong: `combining_like_terms`,
`equality`, `negative_numbers` and `number_line`.

**The lesson's web half has no learner scope at all.** It has two routes,
`/lesson/next` and `/lesson/:id/submit`. `server.ts` and `mastery.ts`
mention "learner" 0 times, and `index.ts` builds one `MasteryStore` for
the whole server. Every caller reads and writes the same mastery, so A
sees B's progress without any attack. The Python `LearnerState` is
isolated per object: changing A's mastery leaves B's unchanged.
