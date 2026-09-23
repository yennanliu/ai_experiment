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

## Answers

### 1 — the workflow pushes to the repo it later says you cannot

Five of the six steps run locally and need nothing but git. In a throwaway
repository: `git init -b main`, `git checkout -b my-progress`, write
`notes.md`, `git add`, `git commit` — the branch is `my-progress`, one commit
ahead of `main`, with `notes.md` in its tree. `git push origin my-progress`
then exits **128** with

```
fatal: 'origin' does not appear to be a git repository
```

which is the honest shape of the sixth step: it is the first one that needs a
second machine, an account on it, and write access to a repository there.

The lesson gets the order wrong. Line **62**, in "The daily workflow", is
`git push origin main`. Line **78**, sixteen lines later, is "You can't push to
the course repo itself — only maintainers have write access." A reader working
top to bottom runs the failing command before reaching the sentence that
explains why it failed.

It also never says how the push is authenticated. Step 4 clones over anonymous
HTTPS and pushes two lines later. The words token, ssh, credential,
authenticate, PAT and password appear **zero** times in the lesson. The one
step that fails for a reason a reader cannot guess is the one step with no
instructions.

And the Use It table, which promises "you need exactly these commands", lists
six. Build It demonstrates five it omits — `config`, `fetch`, `merge`, `pull`,
`status` — and lists `git log --oneline`, which no Build It block uses. The
list is neither a superset nor a subset of what the lesson teaches.

### 2 — the rule that hides weights also hides a source package

`*.pt`, `*.pth`, `*.safetensors` in a `.gitignore`, checked with
`git check-ignore` against a real index rather than read by eye: all three
sample weight files are claimed, `train.py` beside them is not.

Except the repository already ships it. Lines **49**, **50** and **52** of the
root `.gitignore` are exactly those three patterns, and lines 51, 53 and 54 add
`*.onnx`, `*.bin` and `*.h5`. The exercise asks the reader to write a file that
was already in the checkout the previous exercise told them to clone.

Read the rest of that file and there is a real problem in it. Line **44** is
`data/` and line **45** is `models/`. A bare `dir/` pattern matches at every
depth, so:

```
$ git check-ignore -v --no-index \
    phases/01-math-foundations/01-linear-algebra-intuition/code/models/encoder.py
.gitignore:45:models/   phases/.../code/models/encoder.py
```

In a repository teaching machine learning, `models/` is the most likely name
for a source package. Adding one would make it invisible to `git add`, with no
error — `git status` simply would not mention it.

The last thing worth knowing about `.gitignore` is what it cannot do. Commit
`model.pt`, then add `*.pt` to `.gitignore`, then change the file:

```
$ git status --porcelain model.pt
 M model.pt
$ git check-ignore model.pt        # claims nothing, exit 1
```

Ignoring is consulted for untracked paths only. A reader who does this exercise
after committing a checkpoint has changed nothing about the checkpoint.

### 3 — the checkout that grades this has one commit in it

`git log --oneline` reports 1112 commits on the machine writing this, and
**one** on the machine grading it. So the exercise is answered from the tree.

The order lessons were added is written into the directory names: **20** phase
directories and **523** lesson directories, **zero** of them without a
two-digit prefix. The sequence sorts out of the names with no commits present
at all, which is the only form of the answer that survives the checkout doing
the grading.

That a shallow clone is lossless about files and total about history is easy to
show. Three commits in a source repository, cloned with `--depth=1`:

| | source | clone |
|---|---:|---:|
| `git log --oneline` | 3 | **1** |
| `git ls-files` | identical | identical |
| `HEAD^{tree}` | identical | identical |

Nothing in the working copy says the history is missing. Only `git log` does,
and it does so by simply being short — which is the failure mode worth
remembering, because a script that reads `git log` does not error, it just
answers a smaller question.

That clone is this exercise's grader. The workflow runs **two**
`actions/checkout` steps: the reference checkout pins `fetch-depth: 1`
explicitly, and the repository checkout takes the action's default, which is
also 1.

One ambiguity is worth naming. "This repo" is two repositories here — the
lesson text lives in the reference checkout and the solution in the practice
repository, with different git directories and different HEADs. `git log
--oneline` answers a different question in each, and the exercise does not say
which.
