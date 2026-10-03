<!-- generated:start -->
# 00-setup-and-tooling / 08-editor-setup

Solutions to all 4 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/00-setup-and-tooling/08-editor-setup/) · upstream spec
`phases/00-setup-and-tooling/08-editor-setup/docs/en.md`

```bash
uv run demo practice run 08-editor-setup --ex 1
uv run demo explain 08-editor-setup --ex 1
uv run pytest demos/phases/00-setup-and-tooling/08-editor-setup
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Install VS Code and all extensions listed in Step 2 | code | T0 | `ex01_the_recommendations_live_where_vs_code_never_looks.py` |
| 2 | Copy the `settings.json` from this lesson into your VS Code config | code | T0 | `ex02_format_on_save_is_python_only_in_the_file.py` |
| 3 | Open a Python file and verify that Pylance shows type hints and Black formats on save | code | T0 | `ex03_auto_save_skips_format_on_save.py` |
| 4 | If you have access to a remote machine, set up Remote SSH and open a folder on it | code | T0 | `ex04_the_example_host_is_unroutable_by_design.py` |
<!-- generated:end -->

## Answers

The lesson's code is two VS Code JSON files, and CI has no editor, so every
exercise reads `code/vscode/*.json` with `json.loads` and the doc's own snippets
and commands, and derives what the editor would do. All four are **T0**, stdlib only.

### 1 — Step 2 installs 8 of 13, and the 13 never prompt

**ANSWER:** 8 well-formed, duplicate-free `code --install-extension` lines, one
per row of Step 2's table.

**FINDING: Step 2 installs 8 of the 13 recommended extensions.** It never
installs Dev Containers, Jupyter cell tags, Jupyter slideshow, YAML or Even
Better TOML.

**FINDING: the recommendations sit where VS Code never looks.** The doc (en and
zh, twice each) cites `code/.vscode/extensions.json` and promises a prompt; the
lesson ships `code/vscode/` — no dot — and has **0** `.vscode` directories.
VS Code reads recommendations only from `.vscode/` at the opened folder.

**CONTROL:** `extensions.json` is strict JSON with one key.

### 2 — the file's formatOnSave is Python-only, the doc's is global

**ANSWER: it copies as-is** — strict JSON, 30 top-level keys, no comments.

**FINDING: copying the file and copying the doc's snippet give different
editors.** Step 3's "key settings" put `editor.formatOnSave: true` at the top
level (every language); the file sets it only inside `[python]`. 4 of the 5
snippet keys match the file.

**CONTROL:** every extension the settings configure (Pylance, Black, Ruff,
GitLens) is recommended; the Step 4 terminal snippet matches 4 of 4 keys.

### 3 — the 1-second auto-save skips format-on-save

| save | Black formats | organize imports |
|---|---|---|
| explicit (Cmd+S) | yes | yes |
| auto, after 1000 ms | **no** | **no** |

**ANSWER:** Pylance shows return-type hints only (`functionReturnTypes` on,
`variableTypes` off, `basic` checking); Black runs on an explicit save.

**FINDING: the auto-save the lesson sells disables the auto-format it sells.**
VS Code runs `formatOnSave` only if "the file must not be saved after delay",
and `codeActionsOnSave: "explicit"` only on explicit saves. With
`files.autoSave: afterDelay`, the edit you forget to save — the doc's reason for
auto-save — is saved unformatted. "Never think about formatting again" holds
only if you press Cmd+S.

**CONTROL:** `--line-length 88` equals the first ruler and Black's default.

### 4 — the SSH config is sound, but its host is unroutable by design

**ANSWER:** keep the `Host gpu-box` block (4 known keywords, `IdentityFile`
matching `ssh-keygen -t ed25519`) and replace `HostName`; Remote SSH is
recommended.

**FINDING: `203.0.113.50` is RFC 5737 TEST-NET-3** (`ipaddress`: not global).
The doc says `gpu-box` then "connects instantly" and never says to replace it.

**FINDING: `ForwardAgent yes` is aimed at shared boxes** — the doc names lab
servers, Vast.ai and cloud VMs, exactly where `ssh_config(5)` says agent
forwarding needs caution.

**CONTROL:** the `ssh-keygen -C` email is at `example.com`, also reserved.
