<!-- generated:start -->
# 00-setup-and-tooling / 10-terminal-and-shell

Solutions to all 4 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/00-setup-and-tooling/10-terminal-and-shell/) · upstream spec
`phases/00-setup-and-tooling/10-terminal-and-shell/docs/en.md`

```bash
uv run demo practice run 10-terminal-and-shell --ex 1
uv run demo explain 10-terminal-and-shell --ex 1
uv run pytest demos/phases/00-setup-and-tooling/10-terminal-and-shell
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Install tmux, create a session with three panes, and run `htop` in one, `watch -n1 date` in a… | code | T0 | `ex01_trainenv_fills_two_panes_and_a_rerun_makes_five.py` |
| 2 | Add the aliases from `code/shell_aliases.sh` to your shell config and reload with `source ~/.… | code | T0 | `ex02_memhogs_fallback_never_fires_and_killtraining_misses_torchrun.py` |
| 3 | Create a fake training log with `for i in $(seq 1 100); do echo "epoch $i loss: $(echo "scale… | code | T0 | `ex03_bc_drops_the_leading_zero_so_a_number_regex_finds_one_loss.py` |
| 4 | Set up an SSH config entry for a server you have access to (or use `localhost` to practice th… | code | T0 | `ex04_syncto_sends_your_laptop_home_path_and_nests_the_folder.py` |
<!-- generated:end -->
## Answers

The lesson's code is one bash file, `shell_aliases.sh`, read from the reference
`phases/` tree and sourced into `bash --noprofile --norc` with an empty HOME.
Commands it would hand to the machine (`tmux`, `ps`, `rsync`) are shadowed by
recording shell functions, so nothing touches the host; `ssh -G` resolves an
SSH config without connecting. All four are **T0**, stdlib only, and skip
cleanly where `bash` (or `ssh`) is absent.

### 1 — `trainenv` fills two of its three panes, and a rerun makes five

**ANSWER:** the lesson's `trainenv train` makes three panes —
`new-session`, `split-window -h`, `split-window -v` — with pane 1 running
`watch -n1 nvidia-smi`, pane 2 `htop`, pane 0 a bare shell. Send
`watch -n1 date` to pane 1 and `python script.py` to pane 0; detach with
`C-b d`, reattach with the lesson's `ta train`.

**FINDING:** one pane runs nothing and one needs an NVIDIA GPU.

**FINDING: running it twice leaves five panes** —
`[shell, nvidia-smi, htop, nvidia-smi, htop]`: `new-session` fails on the
existing name, but both splits and both `send-keys` still run.

**FINDING:** 3 calls hardcode window 0 (`train:0.N`), which does not exist
under the common `set -g base-index 1`.

### 2 — `memhogs`' macOS fallback never fires

**ANSWER:** sourcing defines **22 aliases and 10 functions**, and a second
`source` (the reload) changes neither count.

**FINDING: `memhogs` prints nothing with a BSD `ps`, exit 0.** In
`ps aux --sort=-%mem | head -11 || ps aux -m | head -11` the `||` tests
`head`, which succeeds, so the `-m` branch never runs.

**FINDING: `killtraining` (`pkill -f "python.*train"`) matches 3 of 3
bystanders** — a server with `--config configs/train.yaml`, `pytest
tests/test_trainer.py`, a kernel under `~/training-notes` — **and 0 of 3**
`torchrun` / `accelerate launch` / `deepspeed` jobs.

**FINDING:** in a non-interactive shell the alias `diskuse` exits **127**;
the function `lastexp` exits 0. Scripts get the functions, never the aliases.

**FINDING:** the doc's rc line `source phases/00-setup-and-tooling/...` is
relative: from HOME, where a new shell starts, it exits 1 and defines 0 aliases.

### 3 — `bc` drops the leading zero

**ANSWER:** `grep 'loss:' fake_train.log | awk '{print $4}'` gives all 100
values; `| tail -n 5` before the `awk` keeps `.0104 .0103 .0102 .0101 .0100`.

**FINDING: 99 of 100 values have no leading zero** (`.5000`), so the usual
`grep -oE 'loss: [0-9]+\.[0-9]+'` finds **1** line; `awk` and `float()` read
them all.

**FINDING: 38 of 100 are truncated, not rounded** — `1/6` is `.1666`.

**FINDING:** the lesson's `taillog` reads `logs/*.log`, so beside
`fake_train.log` it prints nothing.

**CONTROL:** where `bc` exists, the exercise's own loop (without `sleep`)
writes the modelled file byte for byte.

### 4 — `syncto` sends your laptop's home path and nests the folder

**ANSWER:** `Host gpu / HostName localhost / User learner / Port 2222 /
IdentityFile ~/.ssh/gpu_key` resolves (`ssh -G`) to localhost, learner, 2222.

**FINDING:** the usage example `syncto gpu ~/data ./data` expands `~`
locally: with HOME=/Users/learner the call is
`rsync -avz --progress ./data gpu:/Users/learner/data`. The doc's own
`rsync ... user@gpu-box-ip:~/data/` keeps it for the remote side.

**FINDING:** `syncto` drops the doc's trailing slash, so a real `rsync`
lands `data/data/a.txt`.

**FINDING:** appended below a `Host *` / `Port 22` block, the same entry
resolves to port **22** — ssh takes the first value it sees.
