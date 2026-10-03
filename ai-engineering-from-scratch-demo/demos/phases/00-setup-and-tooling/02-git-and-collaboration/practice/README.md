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
| 1 | Fork this repo, clone your fork, create a branch called `my-progress`, make a file, commit it… | code | T0 | `ex01_a_fork_never_receives_the_next_lesson.py` |
| 2 | Create a `.gitignore` that excludes model checkpoint files (`.pt`, `.pth`, `.safetensors`) | code | T0 | `ex02_the_curriculum_writes_checkpoints_the_three_rules_miss.py` |
| 3 | Look at the commit history of this repo with `git log --oneline` and read how lessons were added | code | T0 | `ex03_a_shallow_clone_says_every_lesson_arrived_at_once.py` |
<!-- generated:end -->

## Answers

The lesson has no `code/`, so every exercise runs real git in throwaway
repositories under a temp directory, with an isolated config (no global or
system file) and fixed identity and dates. Nothing reads this checkout's
history. All three are **T0**, and skip with a remedy if `git` is absent.

### 1 — a fork never receives the next lesson

The fork is `git clone --bare` of a course repo; Step 4 then runs verbatim.

**ANSWER: the push lands** -- the fork's `my-progress` equals the reader's
`HEAD`, and its tree holds the new file.

**FINDING: the course moves on, the reader does not.** After the course adds
lesson 02, `git fetch --all` in the clone does not bring it in: the only remote
is `origin`, the fork. The lesson says `upstream` **0** times, `remote add`
**0** and `sync` **0**.

**FINDING: no upstream.** `git push origin my-progress` without `-u` leaves
`my-progress@{upstream}` unresolved; the next bare `git push` exits **128**,
"has no upstream branch". `push -u` appears **0** times in the lesson.

**CONTROL:** `git remote add upstream` + `fetch` brings lesson 02 in; `push -u`
sets the upstream to `origin/my-progress`.

### 2 — the curriculum writes checkpoints the three rules miss

**ANSWER:** `*.pt`, `*.pth`, `*.safetensors`; `git check-ignore` claims the
three samples and leaves `train.py` tracked.

**FINDING: 3 of the 6 checkpoint names the curriculum writes get through.**
Each name is confirmed by its literal in that lesson's source:

| written by | name | ignored |
|---|---|:-:|
| 03/11 | `mnist_mlp.pt` | yes |
| 19/47 | `ckpt.pt` | yes |
| 19/47 | `ckpt.pt.<random>.tmp` (atomic save) | **no** |
| 19/37 | `gpt2-stub.safetensors` | yes |
| 19/80 | `rank0.bin` (torch.save) | **no** |
| 19/80 | `rank0.bin.tmp` | **no** |

**FINDING:** a `weights.pt` committed before the `.gitignore` stays tracked --
`git status` reports ` M weights.pt` after retraining.

**CONTROL:** after `git rm --cached` it is `D  weights.pt` and ignored.

### 3 — a shallow clone says every lesson arrived at once

The history is rebuilt from the **12** real phase-00 lessons, one commit each.

| clone | `log --oneline` lines | distinct adding commits |
|---|---:|---:|
| full | 12 | 12 |
| `--depth 1` via `file://` | **1** | **1** |
| `--depth 1` via plain path | 12 | 12 |

**ANSWER:** on a full history, one line and one adding commit per lesson.

**FINDING:** on a depth-1 clone -- which is what a CI checkout with
`fetch-depth: 1` is -- `--diff-filter=A` names one commit as the adder of all
12 lessons; the only hint is a `grafted` decoration.

**FINDING:** `--depth 1` on a plain local path is ignored with a warning; only a
`file://` URL makes it shallow.

**CONTROL:** the shallow tip still lists all **12** lesson directories.
