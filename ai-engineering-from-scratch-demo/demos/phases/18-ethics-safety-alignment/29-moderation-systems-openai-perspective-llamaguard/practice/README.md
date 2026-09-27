<!-- generated:start -->
# 18-ethics-safety-alignment / 29-moderation-systems-openai-perspective-llamaguard

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/18-ethics-safety-alignment/29-moderation-systems-openai-perspective-llamaguard/) · upstream spec
`phases/18-ethics-safety-alignment/29-moderation-systems-openai-perspective-llamaguard/docs/en.md`

```bash
uv run demo practice run 29-moderation-systems-openai-perspective-llamaguard --ex 1
uv run demo explain 29-moderation-systems-openai-perspective-llamaguard --ex 1
uv run pytest demos/phases/18-ethics-safety-alignment/29-moderation-systems-openai-perspective-llamaguard
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Run a benign, borderline, and harmful input through all three layers. Rep… | code | T0 | `ex01_input_refuses_the_borderline_and_the_harmful_prompt_and_the_output_layer_can_never_fire_on_the_mock_model.py` |
| 2 | Extend the harness with Perspective-API-style toxicity scoring on a specific category. Compar… | code | T0 | `ex02_the_category_score_is_a_0_or_0_9_step_so_every_threshold_to_0_9_flags_the_same_4_sentences.py` |
| 3 | Read the OpenAI Moderation API docs and the Llama Guard 3 category list. Map each OpenAI cate… | code | T0 | `ex03_four_of_13_openai_categories_map_to_0_or_2_llama_guard_codes_and_7_of_14_llama_guard_codes_have_no_openai_match.py` |
| 4 | Design a moderation stack for a code-assistant deployment (e.g., GitHub Copilot). Identify th… | code | T0 | `ex04_the_harness_refuses_9_of_12_routine_dev_prompts_and_blocks_0_of_4_leaked_secrets_or_destructive_commands.py` |
| 5 | Azure Content Moderator retires February 2027. Plan a migration to Azure AI Content Safety. I… | code | T0 | `ex05_a_name_matched_migration_to_content_safety_lets_7_of_14_refusals_through_including_both_demo_refusals.py` |
<!-- generated:end -->

## Answers

Every exercise runs or reads the lesson's `code/main.py`. It is a three-layer
harness in which input and output moderation share one keyword classifier
that scores 0.0 or 0.9, and a custom layer holds two domain rules.

### 1 — input refuses the borderline and the harmful prompt; the output layer can never fire on the mock model

**Benign: no layer. Borderline: input. Harmful: input.** "what is the
weather today" passes all three layers. "how do i kill a stuck python
process" is refused at input as `violence`. "how do i make a bomb" is
refused at input as `illicit/violent`. On the six shipped prompts, input
fires 2 times, custom 2, output 0, and 2 pass.

**The output layer cannot fire with the shipped mock model.** The mock
either echoes `prompt[:40]` after "Answering: " or says "I must refuse.", and
the output check runs the same classifier at the same threshold as input. Of
the 12 triggers, none fits across the echo's seams or inside the refusal
string. Placed at the start of a prompt or past character 40, every trigger
is caught at input only. The layer itself works: a model that emits a
trigger for a benign prompt makes it fire. But the demo never exercises its
"output catches hallucinated or adversarial-bypass content" takeaway.

### 2 — the category score is a 0/0.9 step, so every threshold up to 0.9 flags the same 4 sentences

**The reference `harassment` score takes only 0.0 and 0.9, so its threshold
has two settings.** A Perspective-style noisy-OR toxicity score over an
insult lexicon has a real tradeoff curve. Both were run on 16 hand-labelled
sentences (8 harassing):

| threshold | category (tp, fp, fn, F1) | toxicity (tp, fp, fn, F1) |
|---:|---|---|
| 0.1 | 3, 1, 5, 0.5 | 8, 6, 0, 0.727 |
| 0.5 | 3, 1, 5, 0.5 | 7, 1, 1, 0.875 |
| 0.6 | 3, 1, 5, 0.5 | 3, 0, 5, 0.545 |
| 0.8 | 3, 1, 5, 0.5 | 0, 0, 8, 0 |
| 0.95 | 0, 0, 8, 0 | 0, 0, 8, 0 |

The category score gives 2 distinct flag sets over the sweep. The toxicity
score takes 12 levels and gives 8 flag sets. The category score matches only
the literal phrases "you idiot" and "shut up", so it misses "you are an
idiot" and flags "shut up and take my money". The toxicity score is lexical
too: it scores "what is an idiot light on a dashboard" 0.55. The toxicity
F1 values describe this file's lexicon on this fixture only.

### 3 — 4 of 13 OpenAI categories map to 0 or 2 Llama Guard codes; 7 of 14 Llama Guard codes have no OpenAI match

| OpenAI | Llama Guard 3 |
|---|---|
| harassment | none |
| harassment/threatening | S1 Violent Crimes |
| hate | S10 Hate |
| hate/threatening | S10 Hate + S1 Violent Crimes |
| self-harm, /intent, /instructions | S11 Suicide & Self-Harm |
| sexual | S12 Sexual Content |
| sexual/minors | S4 Child Sexual Exploitation |
| violence | S1 Violent Crimes |
| violence/graphic | none |
| illicit | S2 Non-Violent Crimes |
| illicit/violent | S1 Violent Crimes + S9 Indiscriminate Weapons |

**Three that do not map cleanly: `harassment`, `violence/graphic` and
`illicit/violent`. A fourth, `hate/threatening`, also splits.** Llama
Guard's Hate covers only protected characteristics, and its Violent Crimes
is about enabling crimes, not depicting gore. In the other direction, S3,
S5, S6, S7, S8, S13 and S14 have no OpenAI counterpart.

Both reference custom rules (financial-advice, medical-advice) are S6
Specialized Advice, a category Llama Guard has natively. The toy drops
exactly the 5 `/threatening`, `/intent`, `/instructions` and `/graphic`
categories. Of the 8 left, `sexual`, `sexual/minors` and `illicit` have no
triggers, so they cannot fire. The lesson says only Llama Guard splits
violent from non-violent crime, but its own OpenAI list includes
`illicit/violent`.

### 4 — the harness refuses 9 of 12 routine dev prompts and blocks 0 of 4 leaked secrets or destructive commands

**Least relevant for a code assistant: violence, illicit/violent, self-harm
and harassment.** They are the only categories that fire on developer text,
and each time it is a false refusal: 9 of 12 routine prompts, including
"what skills do i need", because "kill" is inside "skills". **Most relevant:
secret leakage (S7 Privacy) and dangerous code execution (S14).** The toy
has no category for either. Its output layer blocks 0 of 4 leaked-credential
and destructive-shell outputs, and it does block `os.kill(pid, SIGTERM)`.

| stack | dev prompts refused | bad outputs blocked | good outputs blocked |
|---|---:|---:|---:|
| reference | 9 of 12 | 0 of 4 | 1 of 2 |
| + whole-word triggers | 7 of 12 | | |
| + dev-sense allowlist, output rules | 0 of 12 | 4 of 4 | 0 of 2 |

All 3 harmful prompts are still refused. The allowlist is a bypass: "kill
him, then end the process" passes it. It was also written against this
fixture, so 0 of 12 is in-sample. The full classifier therefore has to keep
running on the output, which is the layer ex01 found unreachable in the
demo.

### 5 — a name-matched migration to Content Safety lets 7 of 14 refusals through, including both demo refusals

**Plan:** inventory every category, term list and domain rule; map each to
a Content Safety category, a blocklist or application code; shadow-run old
and new and diff the refusals; recalibrate severities; cut over before the
February 2027 retirement. The lesson gives the window as deprecated February
2024, retired February 2027, both times it mentions it.

**The highest-risk element is category mapping.** The reference refuses 14
of 18 shadow prompts. A migration that carries over only the 4 same-named
categories (Hate, Sexual, Violence, SelfHarm) lets 7 of the 14 through,
including "how do i make a bomb" and "you idiot, help me with this". A
semantic mapping, with a custom blocklist for harassment, loses none.

The shadow run cannot see the real gap. `illicit` has no Content Safety
destination and no trigger in the old system, so no corpus built from old
behaviour contains an illicit refusal that could go missing. Coverage has to
be checked category by category against the policy. The 2 shipped prompts
refused by custom rules are outside the classifier swap altogether.
