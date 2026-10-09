<!-- generated:start -->
# 00-setup-and-tooling / 11-linux-for-ai

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/00-setup-and-tooling/11-linux-for-ai/) · upstream spec
`phases/00-setup-and-tooling/11-linux-for-ai/docs/en.md`

```bash
uv run demo practice run 11-linux-for-ai --ex 1
uv run demo explain 11-linux-for-ai --ex 1
uv run pytest demos/phases/00-setup-and-tooling/11-linux-for-ai
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | SSH into any Linux machine (or open WSL2) and navigate to your home directory. Create a proje… | code | T0 | `ex01_the_fifteen_commands_are_thirteen_and_touch_is_not_one.py` |
| 2 | Install `htop` with apt, run it, and identify which process is using the most memory. | code | T0 | `ex02_the_biggest_virt_is_the_smallest_process.py` |
| 3 | Start a tmux session, run `sleep 300` inside it, detach, list sessions, and reattach. | code | T0 | `ex03_ctrl_b_capital_d_does_not_detach.py` |
| 4 | Use `df -h` to check available disk space, then use `du -sh ~/.cache/*` to find what's taking… | code | T0 | `ex04_du_star_skips_the_hidden_half_of_the_cache.py` |
| 5 | Transfer a file from your local machine to a remote one using `scp`, then do the same transfe… | code | T0 | `ex05_the_lessons_rsync_does_not_resume.py` |
<!-- generated:end -->

## Answers

This lesson has no `code/`: it is a survival guide of shell commands, and every
exercise asks for something done on a remote Linux box — SSH, `apt`, tmux, `scp`.
None of that runs in CI, so each solution runs the same mechanism on a
deterministic scale-down, on Linux and macOS alike: real system calls, real
`ps`/`du`/`df`, a real pseudo-terminal, and rsync's delta algorithm written out.
The lesson's own commands and claims are read from `docs/en.md` and checked
against what those runs show.

### 1 — the "15 commands" are 13, and `touch` is not one of them

In a temp dir, with the umask pinned to Ubuntu's 022: `mkdir my-project`, three
`touch`es, `ls -la`.

**ANSWER: 5 rows for 3 files** — `.`, `..`, and three 0-byte `-rw-r--r--` files
(mode 0666 masked by the umask; under umask 077 the same `touch` gives
`-rw-------`).

**FINDING:** "These are the 15 commands that cover 95%" — the code blocks of
Essential Commands use **13**: pwd, ls, cd, mkdir, cp, mv, rm, cat, head, tail,
less, grep, find.

**FINDING:** `touch`, the command this exercise is built on, occurs **0** times in
the lesson's code blocks and inline code before the exercises — not in Essential
Commands, not in the Quick Reference Card.

**CONTROL:** the permission example is right: `-rwxr-xr--` is 754, and `chmod 755`
and `chmod 644` read back as the comments say.

### 2 — the process with the biggest VIRT uses the least memory

Two real processes, read through `ps -o vsz=,rss=` (htop's VIRT and RES):

| process | RSS (RES) | VSZ (VIRT) vs the worker |
|---|---:|---:|
| reserver: maps 1 GiB, never touches it | 13 MiB | **+896 MiB** |
| worker: fills 128 MiB | **143 MiB** | — |
| idle interpreter | 13 MiB | |

**ANSWER: the worker, by RES** (~130 MiB more resident).

**FINDING: by VIRT the answer is the reserver** — the process doing nothing.
CUDA and JAX reserve tens of GiB of address space at start-up, so on a GPU box
sorting by VIRT puts them first whatever they hold.

**FINDING: the lesson never says which column to read.** RES, VIRT, RSS, VSZ and
%MEM occur 0 times; htop is "Interactive process viewer (q to quit)". htop sorts
by CPU% until told otherwise (`M`, or F6).

### 3 — a hang-up kills `sleep 300` unless it left the session; Ctrl+B D does not detach

`sleep 300` on a real pty (`pty.fork`, as sshd does), then the pty is hung up by
closing its master, as a dropped SSH connection does:

| how `sleep 300` was started | after the hang-up |
|---|---|
| directly on the terminal | killed by **signal 1 (SIGHUP)** |
| as a job under a shell on the terminal | killed |
| with `setsid` (new session, as the tmux server is) | **alive** |

**ANSWER:** what tmux buys is the `setsid`: its server is in no terminal's
session, so the hang-up never reaches it.

**FINDING: the lesson's detach key is wrong.** "Ctrl+B, then D — Detach". tmux
binds `d` to `detach-client`; `D` is `choose-client -Z`, a client picker.

**FINDING: the split labels are swapped relative to tmux.** The lesson says `%`
splits "vertically" and `"` "horizontally"; tmux binds `%` to `split-window -h`,
"Split window horizontally", and `"` to `split-window`, "Split window
vertically". The lesson names the divider, tmux the layout.

### 4 — `du -sh ~/.cache/*` skips the hidden half of the cache

A labelled stand-in cache, measured by the real `du` through `sh`:

| entry | content | `du -sk` |
|---|---|---:|
| `pip/` | 500 files x 100 bytes = 50,000 bytes | 2,000 KiB |
| `huggingface/` | 4 MiB blob + 2 GiB sparse `.safetensors` | 4,096 KiB |
| `.hidden/` | 8 MiB | 8,192 KiB — **not listed by `*`** |
| `.cache` itself | | 14,288 KiB |

**ANSWER:** huggingface 4 MiB, pip ~2 MiB — 6,096 KiB, against 14,288 KiB for the
directory.

**FINDING: the shell's `*` skips dotfiles**, here more than half the cache.

**FINDING: `du` counts blocks.** 50,000 bytes of tiny files take 2,000 KiB (41x,
a 4 KiB block each); the 2 GiB sparse file takes 0. `ls -l` and `du` answer
different questions.

**FINDING:** the lesson's `du -h --max-depth=1 / 2>/dev/null | sort -hr` uses a
GNU-only flag (BSD `du` spells it `-d`), and `2>/dev/null` hides the error, so on
a Mac it prints nothing. `du` is not in the lesson's macOS-to-Linux table.

**CONTROL:** `df -Pk` agrees with `statvfs`; summing `st_blocks` reproduces `du`.

### 5 — rsync resends one block, and the lesson's command does not resume

rsync's delta algorithm (rolling weak checksum + MD5 per 1,024-byte block) on a
1 MiB float32 checkpoint:

| second transfer | scp | rsync on the wire | literal bytes |
|---|---:|---:|---:|
| 1 byte flipped mid-file | 1,048,576 | **25,596** (41x less) | 1,024 |
| 1 byte inserted at the front | 1,048,576 | 24,577 | **1** |

**ANSWER:** scp resends the file; rsync sends one block plus a 20 KB checksum
list it pays even when nothing changed. The first transfer costs both the whole
file.

**FINDING: an insertion is cheaper than an edit.** The rolling checksum finds all
1,024 shifted blocks; a fixed-offset compare would match none.

**FINDING: the lesson's rsync does not resume.** It says rsync "resumes on
failure" and "handles interrupted connections", but its only command is
`rsync -avz --progress` — no `--partial` (or `-P`), so an interrupted transfer
deletes the partial file and restarts from zero.

**FINDING:** `-z` on weights saves little: zlib gets the checkpoint to **92.8%**.

**CONTROL:** both reconstructions are byte-exact.
