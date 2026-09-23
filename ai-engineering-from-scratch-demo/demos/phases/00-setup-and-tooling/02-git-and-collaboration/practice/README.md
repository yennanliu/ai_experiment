<!-- generated:start -->
# 00-setup-and-tooling / 02-git-and-collaboration

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/00-setup-and-tooling/02-git-and-collaboration/) · upstream spec
`phases/00-setup-and-tooling/02-git-and-collaboration/docs/en.md`

```bash
uv run demo practice run 02-git-and-collaboration --ex 1
uv run demo explain 02-git-and-collaboration --ex 1
uv run pytest demos/phases/00-setup-and-tooling/02-git-and-collaboration
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Fork this repo, clone your fork, create a branch called `my-progress`, make a file, commit it… | code | T0 | `ex01_the_workflow_pushes_to_the_repo_it_later_says_you_cannot.py` |
| 2 | Create a `.gitignore` that excludes model checkpoint files (`.pt`, `.pth`, `.safetensors`) | code | T0 | `ex02_the_rule_that_hides_weights_also_hides_a_source_package.py` |
| 3 | Look at the commit history of this repo with `git log --oneline` and read how lessons were added | code | T0 | `ex03_the_checkout_that_grades_this_has_one_commit_in_it.py` |
<!-- generated:end -->
