<!-- generated:start -->
# 16-multi-agent-and-swarms / 01-why-multi-agent

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/16-multi-agent-and-swarms/01-why-multi-agent/) · upstream spec
`phases/16-multi-agent-and-swarms/01-why-multi-agent/docs/en.md`

```bash
uv run demo practice run 01-why-multi-agent --ex 1
uv run demo explain 01-why-multi-agent --ex 1
uv run pytest demos/phases/16-multi-agent-and-swarms/01-why-multi-agent
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a fourth specialist: a "tester" agent that receives code from the coder and review feedba… | code | T0 | `ex01_the_fourth_specialist_costs_five_times_what_the_split_saves.py` |
| 2 | Modify the pipeline so the reviewer can send feedback back to the coder for a revision loop (… | code | T0 | `ex02_the_loop_has_no_way_to_stop_early_because_nothing_can_approve.py` |
| 3 | Convert the sequential pipeline into a fan-out: run the researcher and a "requirements analyz… | code | T0 | `ex03_the_fan_out_already_exists_and_throws_the_review_away.py` |
<!-- generated:end -->
