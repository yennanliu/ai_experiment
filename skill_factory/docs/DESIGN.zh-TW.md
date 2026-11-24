# Skill Factory — 在 `skill_factory/` 打造 Uber 風格的 Agent Skill 平台

> 來源：Uber Engineering，《How Uber Evals 1 Million Agent Skill Invocations a Month》（2026/11/23）
> — @UberEng 的 X 長文，完整文章：<https://www.uber.com/us/en/blog/optimizing-ubers-agent-skills/>。
> 姊妹篇：Uber Engineering，《Running a Software Factory Efficiently at Uber Scale》（2026/08/27）
> — <https://www.uber.com/us/en/blog/efficient-software-factory/>。
> Skill 格式規範：<https://agentskills.io>。
>
> English version: [DESIGN.en.md](./DESIGN.en.md)

---

## 0. 摘要（TL;DR）

Uber 把 agent skill 當成**軟體產品**來經營：放在 Git、有 owner 與分級、由 spec 產生、經過 lint、
做評測（有 skill 與沒 skill 都測）、量測每次呼叫的成本，並且由 **agent 自己回報**的問題持續改善。

本文件描述一個縮小版、可以在 `skill_factory/` 由一個人或小團隊、以 Claude Code（也可搭配
Codex / OpenCode / Gemini CLI）做出來的版本。它會先把本 repo 既有的東西納入管理：
`~/.claude/skills/lc-add`、`~/.claude/skills/lc-coach`、`claude_code/skills_poc/.claude/skills/*`，
以及 `../CLAUDE.md` 裡的 104 人力銀行自動投遞流程。

目標不是 5,000 個 skill，而是：**留下來的每個 skill 都證明有用、成本已知、而且會在我們發現問題之前
就自己變好。**

---

## 1. Uber 做了什麼（重點整理）

| 面向 | Uber 的做法 | 關鍵數字 / 細節 |
|---|---|---|
| 規模 | 跨 Claude Code、OpenCode、Codex 與內部 agent 的共用 skill 市集 | 1 個（2025/10）→ 5,000+（2026/10）；每月 >100 萬次呼叫 |
| 格式 | Agent Skills 規範：一個資料夾，含 `SKILL.md` + 選用的 scripts 與 references | 三階段載入：name+description（約 100 tokens）常駐；本文（<500 行）使用時才載入；scripts/references 需要時才讀 |
| 發佈 | 一個 Git repo **就是**市集 | 免費得到 ownership、review、歷史紀錄 |
| 分級 | Always-on（約 100 個）、Core（平台團隊 review）、Domain（各團隊自管，1 位同儕核准 + CI） | |
| 啟動器 | `aifx` CLI：啟動各種 agent、安裝 skill/plugin/MCP、六層設定合併 | 公司 → 團隊 → repo → 目錄 → 使用者 → 本機；後面的層只能「加」 |
| 撰寫 | **Skill Workshop**：先寫 *spec*（目標、輸出、必用/禁用工具、目標 agent/模型），再編譯出 `SKILL.md` 與測試 prompt | 模型或 agent 改版時重新編譯 |
| 探索 | `find-skills` skill；依評測與使用次數排序；品質分數 | |
| 遙測 | 從 agent hooks 送出 OpenTelemetry 事件；不含 prompt 文字與檔案內容 | 每個 skill 12 項指標（使用者、留存、觸及、共用、成本、context、延遲、錯誤、工具、版本、agent、安裝數） |
| 單次成本 | 呼叫後的 session 成本快照 − 呼叫前快照；每日總額 ÷ 呼叫次數 | |
| 評測 | 約 5% golden（固定輸入輸出 + 工具 mock）、約 80% 真實 prompt 以**軌跡（trajectory）**評分、約 15% 由作者評分 | 通過率 70–90% 的評測比 100% 的更有用 |
| 消融（Ablation） | 用真實 prompt 重跑：有 skill / 沒 skill / 拿掉一部分；盲評 judge | 最大效益來自讓 agent 取得它自己找不到的東西（內部文件、資料倉儲查詢） |
| 無塵室 | 在遠端沙箱（Minion、AES、devpods）跑評測 | 抓出藏在筆電上的依賴 |
| Linter | 從好的 skill 歸納出的規則 | 步驟編號、一步一個目標、每步有驗證、錯誤處理表 |
| 降本手段 | Code Mode（用 script 取代 MCP 工具定義傾倒）、便宜模型的 subagent、載入時先跑 script、把輪詢移出主 session | MCP 預設載入 5–7 萬 tokens；tracing skill −46%；search skill −32% |
| 回饋 | 常駐的 `log-papercut` skill — agent 自己提交結構化的「小痛點」報告 | 三個月數千筆、修了數百筆 |
| 人類訊號 | 被 review 留言、反覆修改或被退回的 PR 會回饋成評測案例 | |

### Uber 明確提出的心得
1. Skill 放在 Git。
2. 先寫 spec 再寫 skill，讓意圖撐過模型更替。
3. **有 skill 和沒 skill 都要測**；有些 skill 應該退役。
4. 量測每個 skill 的成本。
5. 無塵室評測能抓出筆電環境的假設。
6. 讓 agent 回報摩擦點，並導給 owner 處理。

### 1.1 更大的圖像：Uber 的「軟體工廠」（2026/08 文章）

Skill 那篇只是整個系統的一角。較早的這篇說明 skill 運作所在的**工廠**，以及 Uber 如何在用量成長時
維持低成本。

| 面向 | Uber 的做法 | 關鍵數字 / 細節 |
|---|---|---|
| 採用率 | 本機 + 雲端 agent 遍佈整個 SDLC | >70% 的 PR 歸屬於 agent；每週活躍使用者 ×7、請求數 ×9.4（2026/02→08），總支出自 4 月起大致持平 |
| 成本公式 | 支出 = 使用者 × 每人 session 數 × **每 session 輪數 × 每輪請求數 × 每請求 tokens** × 每 token 價格 | 優化中間三項；前兩項是「希望增加」的採用率 |
| 託管 agent（Managed agents） | 由系統而非人啟動的 agent：code review（uReview）、CI 自我修復、端到端 PR、警報分流、除錯、維護 | 每個 agent 先定成果指標 → 評測基準 → 選出達標中最便宜的模型（Pareto） |
| 模型選擇 | 以數千個真實 PR 建立的 Uber SWE Benchmark | 依基準為每個 agent 選模型，而不是憑習慣 |
| 預設使用 subagent | 主模型負責規劃與檢查；明確的工作交給便宜模型的 subagent | 被稱為「影響最大的槓桿」 |
| 工具 | 單一 MCP gateway（>1,000 個 server）；MCP 工具以 **CLI 指令**形式提供；tool search；code-mode plugin | 預載 >100 個工具 = 每輪 5–7 萬 tokens；code mode 在簡單查詢省 55–71%、批次工作省 >90% |
| 資訊落地（Grounding） | AI Context Graph（2,400 萬節點、8,000 萬邊、30+ 個系統） | 有落地：38 秒且正確；沒落地：20 分鐘、3 個錯誤、答錯。「沒有落地的 agent 不是便宜地失敗，而是緩慢地失敗。」 |
| Context 與快取 | 即使是 1M context 模型也在 40 萬 tokens 自動壓縮；reasoning effort 預設 Medium；互動 session 用 1 小時快取 TTL、subagent 用 5 分鐘 | 快取讀取只要輸入價的 0.1× |
| 可視性 | status line 即時顯示成本；session analyzer 偵測 **16 種反模式**；Slack 在預期支出 50/80/100% 時提醒 | 不設硬上限，改用可視化與提醒 |
| 量測 | 量測優化效果時固定模型；每個 agent 有成果指標 | 每個合併 PR / 每次 review / 每個警報的成本，搭配 revert 率、F1、MTTR 作為品質指標 |

Session analyzer 會找的反模式（文章中的例子）：Sonnet 就夠卻用了 Opus、大型 MCP 回傳（例如 40 KB）
一直留在 context、休息後快取過期、第一個 prompt 之前就預載約 10 萬 tokens 的指示與工具。

**這篇為我們的設計增加了什麼：** skill 只是其中一個槓桿。我們還需要 (a) 以 *session* 為單位的成本分析，
不只是以 skill 為單位；(b) 即時成本顯示；(c)「託管 agent」路徑——skill 依排程無頭（headless）執行，
並有自己的成果指標；(d) 一層 grounding/context，讓 skill 不用浪費輪數在搜尋上。

---

## 2. 對照：Uber → `skill_factory`

| Uber 元件 | 我們的版本 | 工作量 |
|---|---|---|
| Skill 市集 repo | 本 repo 的 `skill_factory/skills/`，以 Claude Code plugin marketplace 形式發佈 | S |
| 分級 | `spec.yaml` 的 `tier:` 欄位 — `always-on`、`core`、`domain`、`experimental` | S |
| `aifx` 啟動器 | `sf` — 小型 Python CLI（`sf install`、`sf new`、`sf lint`、`sf eval`、`sf report`） | M |
| 六層設定 | 三層：`skill_factory/profiles/default.yaml` → repo 的 `.skillfactory.yaml` → `~/.skillfactory.local.yaml` | S |
| Skill Workshop | `sf new` + `sf compile`：spec → `SKILL.md` + 評測 prompt，由 Claude 自己執行 | M |
| OTel 遙測 | Claude Code hooks → 本機 JSONL（階段 1），可選 OTel collector（階段 3） | S → M |
| 儀表板 | `sf report` → 靜態 HTML（或發佈成 artifact） | M |
| 評測框架 | `claude -p --output-format stream-json` 執行器 + 軌跡評分 + LLM judge | L |
| 無塵室 | Docker 容器或空 `$HOME` 的 git worktree | M |
| Linter | `sf lint` — 確定性規則 + 選用的 LLM 審查 | S |
| `log-papercut` | 常駐 skill，寫入 `papercuts/inbox.jsonl` | S |
| `find-skills` | 搜尋 `skills/*/spec.yaml` 與使用統計的 skill | S |
| Status line 成本計數 | Claude Code `statusLine` script 顯示 session 成本 | S |
| Session analyzer（16 種反模式） | `sf analyze` 掃描 transcript — 見 9.1 節 | M |
| 託管 agent | 透過 `claude -p` 在 cron / CI 上無頭執行 skill（例如每晚的 104 投遞、PR review），並有成果指標 | M |
| MCP gateway + CLI 化 | 把用到的 MCP 工具（Playwright 等）包成 `skills/*/scripts/` 下的 script | S–M |
| AI Context Graph | `context/` 資料夾放 repo 地圖 + `lookup` script，需要時才載入（不需要圖資料庫） | M |
| Uber SWE Benchmark | 我們的評測套件，跨模型重跑以選出最便宜且達標的模型 | （沿用 §7） |

---

## 3. Repo 結構

```
skill_factory/
├── README.md
├── docs/
│   ├── DESIGN.en.md
│   └── DESIGN.zh-TW.md         ← 本文件
├── .claude-plugin/
│   └── marketplace.json        ← 讓 repo 能被當作 Claude Code plugin marketplace 安裝
├── plugins/
│   ├── always-on/              ← plugin = 一組 skill，每個分級一個
│   │   ├── .claude-plugin/plugin.json
│   │   └── skills/ -> ../../skills（symlink 或產生）
│   ├── core/
│   └── domain-<name>/
├── skills/
│   ├── lc-add/
│   │   ├── spec.yaml           ← 唯一真實來源（人寫）
│   │   ├── SKILL.md            ← 由 spec 編譯（進版控、要 review）
│   │   ├── scripts/            ← 確定性程式碼（"Code Mode"）
│   │   ├── references/         ← 需要時才載入
│   │   └── evals/
│   │       ├── golden/*.yaml   ← 固定輸入 → 預期輸出，工具 mock
│   │       ├── real/*.yaml     ← 真實 prompt + 預期軌跡
│   │       └── judged/*.yaml   ← 作者評分集，用來校準 LLM judge
│   ├── lc-coach/
│   ├── log-papercut/
│   ├── find-skills/
│   └── job-apply-104/          ← 把 ../CLAUDE.md 的 104 流程做成 skill
├── hooks/
│   ├── telemetry.py            ← PostToolUse / SessionStart / Stop hook → events.jsonl
│   └── settings.snippet.json
├── sf/                         ← CLI（Python，用 uv 管理）
│   ├── cli.py
│   ├── compile.py
│   ├── lint.py
│   ├── eval/
│   │   ├── runner.py
│   │   ├── trajectory.py
│   │   ├── judge.py
│   │   └── ablation.py
│   └── report.py
├── profiles/
│   └── default.yaml
├── papercuts/
│   └── inbox.jsonl             ← 不進版控，或分流後再 commit
├── telemetry/                  ← 不進版控的原始事件
└── .github/workflows/ci.yml    ← 每個 PR 跑 lint + golden 評測
```

---

## 4. Skill spec（先寫這個）

Uber 心得 #2：真實來源是 spec，不是文字敘述。模型換了就重新編譯。

```yaml
# skills/lc-add/spec.yaml
name: lc-add
version: 1.3.0
tier: domain            # always-on | core | domain | experimental
owner: jim
domain: leetcode
goal: >
  依照既有慣例，把 LeetCode 解答歸檔進 repo。
triggers:               # 會變成 description（約 100 tokens，常駐載入）
  - "add LC <number> to <folder>"
  - "file this solution and update the README"
outputs:
  - 符合 house layout 的 Python 檔（docstring、含 IDEA 區塊的 V0、時間/空間複雜度）
  - 依排序插入 README 的一列
tools:
  required: [Read, Write, Edit, Bash]
  forbidden: [WebSearch]      # slug 查詢必須用本機 script
  scripts:
    - scripts/find_slug.py    # 確定性：取代一次模型輪次
targets:
  agents: [claude-code, codex]
  models: [claude-sonnet-5-5, claude-opus-5-5]
budget:
  max_body_lines: 500
  target_cost_usd: 0.15       # 每次呼叫，`sf report` 用來警示
evals:
  min_pass_rate: 0.7
  max_pass_rate: 0.95         # >95% 表示評測可能太簡單
```

`sf compile skills/lc-add` 會把 spec + linter 規則 + 2–3 個範例 skill 交給 Claude，產出：
- `SKILL.md`（frontmatter `name`/`description` + 編號步驟），
- `evals/real/*.yaml` 的 prompt 草稿，再由作者修改。

編譯後的 `SKILL.md` 進版控並像程式碼一樣被 review。重新編譯一定走 PR，絕不默默改變。

---

## 5. 發佈與安裝（`sf`，我們的 `aifx`）

### 5.1 Claude Code plugin marketplace
Claude Code 可以從 Git 上的 marketplace（`.claude-plugin/marketplace.json`）安裝 plugin，每個 plugin
可以帶 skill、hook 和指令。用這個機制，Claude Code 端**不需要自己寫同步程式**：

```bash
/plugin marketplace add <git-url-或路徑>/skill_factory
/plugin install always-on@skill-factory
/plugin install core@skill-factory
```

> 實作前請對照 Claude Code 最新文件確認 marketplace/plugin schema；各版本有變動過。

### 5.2 其他 agent 用 `sf install`
Codex / OpenCode / Gemini CLI 則用 `sf install --agent codex`，把 `skills/<name>/` 複製或 symlink 到該
agent 的 skill 目錄，並寫入該 agent 專屬的 frontmatter。這就是 `aifx`「處理各 agent 差異」的工作。

### 5.3 分層 profile
```yaml
# profiles/default.yaml          （第 1 層：「公司」）
plugins: [always-on]
# <repo>/.skillfactory.yaml       （第 2 層：repo）
plugins: [core, domain-leetcode]
# ~/.skillfactory.local.yaml      （第 3 層：使用者/本機）
plugins: [experimental]
```
照抄 Uber 的規則：**後面的層只能加、不能移除**。這樣 always-on 才真的永遠開著。

---

## 6. Linter（`sf lint`）

確定性檢查，快到可以放在 pre-commit hook。先從這些開始：

| 規則 | 原因 |
|---|---|
| Frontmatter 有 `name` 與 `description`；description ≤ 約 100 tokens 且說明*何時*使用 | 第一階段載入成本與觸發準確度 |
| 本文 ≤ 500 行 | 第二階段預算 |
| 步驟有編號；一步一個目標 | Uber linter 規則 |
| 每步結尾有驗證（「確認 X 存在」、「檢查 exit code」） | Uber linter 規則 |
| 指令具體（放在 code block），不是「跑一下測試」 | Uber linter 規則 |
| 有錯誤處理表（`症狀 → 動作`） | Uber linter 規則 |
| skill/plugin 內不得有 `.mcp.json` | Uber 擋掉了 — 5–7 萬 tokens 的工具定義 |
| 不得含 secret / 帳密（regex + `gitleaks`） | `../CLAUDE.md` 已經警告過 |
| 不得使用共用暫存檔路徑（`/tmp/commit_msg.txt`）— 用 `mktemp` | Uber 真實 papercut：並行 session 互相覆蓋 |
| `SKILL.md` 引用的 script 存在且可執行 | 斷鏈類 bug |
| `spec.yaml` 與 `SKILL.md` 同步（SKILL.md 註解裡存 spec 的 hash） | 偵測繞過 spec 的手動修改 |

選用的 `sf lint --llm`：讓 Claude 依同樣規則審查較模糊的項目。

---

## 7. 評測

### 7.1 每個 skill 三組套件（Uber 的 5 / 80 / 15 比例）

**Golden（約 5%）** — 固定輸入、精確預期輸出、工具 mock。便宜；每個 PR 都跑。
```yaml
# evals/golden/two-sum.yaml
prompt: "add LC 1 to Hash_table"
fixtures: { repo: fixtures/mini_repo }
mocks:
  scripts/find_slug.py: { stdout: "two-sum" }
expect:
  files_created: ["Hash_table/two-sum.py"]
  readme_contains: "| 1 | [Two Sum]"
```

**軌跡（約 80%）** — 真實 prompt（從自己的歷史紀錄挑），評的是*agent 做了什麼*：
```yaml
# evals/real/contest-q1.yaml
prompt: "file yesterday's contest Q1, it's in scratch.py"
expect_trajectory:
  must_call_skill: lc-add
  ordered:
    - { tool: Read, path_glob: "**/scratch.py" }
    - { tool: Bash, cmd_regex: "find_slug.py" }
    - { tool: Write }
    - { tool: Bash, cmd_regex: "python .*\\.py" }   # smoke test
    - { tool: Edit, path_glob: "README.md" }
  forbidden: [WebSearch]
  max_turns: 12
```

**作者評分（約 15%）** — 開放式輸出（例如 `lc-coach` 的判定）。作者手動評一組，用來校準 LLM judge，
judge 與作者的一致率本身也要追蹤。

### 7.2 執行器
Claude Code 的 headless 模式會串流每一次工具呼叫，所以不用改 agent 就能評軌跡：

```bash
claude -p "$PROMPT" \
  --output-format stream-json --verbose \
  --model claude-sonnet-5-5 \
  --plugin-dir ./plugins/core \
  --max-turns 20 \
  > runs/$SKILL/$CASE/$RUN.jsonl
```

`sf/eval/runner.py` 每個案例跑 **N 次（預設 5）**，因為 agent 不是確定性的；接著 `trajectory.py` 解析
`tool_use` 事件（包含 `Skill` 工具呼叫本身）並為每次執行評分。回報的是**通過率**，不是單純過/不過。
目標 70–90%：100% 代表評測很可能看不到真正的變化。

### 7.3 消融（這個 skill 值得它的 tokens 嗎？）
`sf eval --ablate lc-add` 用真實 prompt 重跑三種版本：
1. 有 skill，
2. 沒有 skill，
3. 一次拿掉一個段落（例如拿掉錯誤處理表）。

**盲評 judge**（A/B 順序打亂，模型不知道哪個是哪個）挑出較好的答案；同時記錄每組的成本、輪數、工具錯誤。
決策規則：

| 結果 | 動作 |
|---|---|
| 贏且成本 ≤ 基準 | 保留 |
| 贏但成本高很多 | 優化（第 9 節） |
| 平手 | 縮短或退役 — 模型變強了 |
| 輸 | 退役 |
| 拿掉某段落沒有變差 | 刪掉該段落 |

### 7.4 無塵室
在只有 fixture repo、空 `$HOME`、沒有 `~/.claude` 的容器裡跑評測：
```bash
docker run --rm -e ANTHROPIC_API_KEY \
  -v "$PWD/fixtures/mini_repo:/work" -v "$PWD/plugins:/plugins:ro" \
  skill-factory-eval:latest \
  claude -p "$PROMPT" --plugin-dir /plugins/core --output-format stream-json --verbose
```
這能抓出「在我電腦上可以跑」的 skill：寫死的 `/Users/jliu/...` 路徑、已登入的 `gh`、快取檔案、
只有 macOS 才有的 `sed -i ''`。

### 7.5 CI
每個動到 `skills/**` 的 PR 執行 `.github/workflows/ci.yml`：
1. `sf lint`（阻擋），
2. golden 評測（阻擋），
3. 對變動的 skill 跑軌跡評測，N=3（不阻擋，以 PR 留言貼出與 `main` 的通過率差異）。

---

## 8. 遙測與每個 skill 的成本

### 8.1 收集事件（階段 1：本機、不需基礎設施）
Claude Code hooks 會在 stdin 收到 JSON，事件包含 `SessionStart`、`PreToolUse`、`PostToolUse`、
`UserPromptSubmit`、`Stop`。呼叫 skill 會以 `Skill` 工具呼叫的形式出現。

```json
// hooks/settings.snippet.json → 合併進 ~/.claude/settings.json 或隨 plugin 發佈
{
  "hooks": {
    "PostToolUse": [{ "matcher": "Skill", "hooks": [{ "type": "command", "command": "python3 $SF_HOME/hooks/telemetry.py skill" }] }],
    "SessionStart": [{ "hooks": [{ "type": "command", "command": "python3 $SF_HOME/hooks/telemetry.py session_start" }] }],
    "Stop":         [{ "hooks": [{ "type": "command", "command": "python3 $SF_HOME/hooks/telemetry.py stop" }] }]
  }
}
```

`telemetry.py` 會在 `telemetry/events.jsonl` 追加一行：
```json
{"ts":"2026-11-24T10:02:11Z","event":"skill","session":"…","skill":"lc-add","version":"1.3.0",
 "trigger":"model|user","agent":"claude-code","model":"claude-sonnet-5-5","cwd_hash":"…"}
```
隱私規則（來自 Uber）：**不記錄 prompt 文字、不記錄檔案內容**。cwd 做 hash。

### 8.2 單次呼叫成本
Uber：成本 = 呼叫*之後*的 session 成本快照 − 呼叫*之前*的快照。取得 session 成本有兩種方式：
- **OTel**：設定 `CLAUDE_CODE_ENABLE_TELEMETRY=1` 與 OTel exporter，Claude Code 會為每個 API 請求送出
  成本/tokens 事件。跑一個本機 collector（或直接用 console/file exporter），以 session id + 時間做 join。
- **Transcript**：`Stop` hook 會拿到 `transcript_path`；把 skill 呼叫到下一次 skill 呼叫 / session 結束
  之間每則 assistant 訊息的 token 用量加總，乘上模型價格。

接著 `sf report` 以 skill × 日 計算：呼叫次數、不同 session 數、每次成本（平均與 p90）、每次輪數、
每次工具錯誤數、模型觸發 vs 使用者觸發比例、使用中的版本、共用（同 session 出現的其他 skill）。

### 8.3 12 項指標的縮小版
| Uber 指標 | 我們怎麼取得 |
|---|---|
| 使用者 / 觸及 / 安裝數 | 不同的 `cwd_hash` 或機器（單人：不同的 repo） |
| 留存 | 在 ≥2 個不同週使用過 |
| 共用 | 同一 session 中的 skill |
| 成本、context、延遲 | OTel 或 transcript |
| 錯誤、工具 | skill 時間窗內帶錯誤結果的 `PostToolUse` |
| 版本、agent | 事件欄位 |

---

## 9. 降本手段（先量測再套用，套用後再量一次）

1. **用程式碼取代 LLM（"Code Mode"）。** 任何確定性的步驟（slug 查詢、README 排序、104 流程中的
   表單選擇器）都變成 `scripts/` 裡的 script。模型只要呼叫一個指令，不用一步步推理。不要為了幾個工具就
   裝一個 MCP server — 把呼叫包進 script。
2. **便宜的 worker。** 在 skill 裡把大量/機械性工作派給便宜模型的 subagent（例如用 Haiku 5.5 處理
   「收集這一頁的職缺連結」）。Uber 有一個 skill −46%、另一個 0% — 所以每次改動都要量。
3. **載入時先跑 script。** 如果 skill 一開始一定會跑 `git status` 或列目錄，就預先算好注入，省掉一整輪
   模型（Uber：4 輪 → 3 輪，−32%）。
4. **把輪詢移出主 session。** 等 CI、等部署、等 104 的「應徵完成」頁面，應該在背景 agent 或 script
   迴圈裡做，而不是在昂貴的 session 裡一輪一輪重複。
5. **以評測基準選模型。** 每個 skill 的評測套件在 Haiku / Sonnet / Opus 上各跑一次，把能達到
   `min_pass_rate` 的最便宜模型設為預設（`targets.models[0]`）。新模型推出時重跑。量測其他優化時要固定模型。
6. **先落地再搜尋。** 如果評測顯示 agent 花很多輪在 `grep`/`find`/網路搜尋，就加一個 `references/`
   檔或 lookup script 直接給答案。Uber 有落地的版本 38 秒，沒落地的 20 分鐘。
7. **調整 context 與快取。** 讓常駐文字保持精簡（description、CLAUDE.md、常駐 MCP）。注意
   `../CLAUDE.md` 會把整份 104 指南載入 `ai_experiment/` 底下的*每一個* session — 把它移進
   `job-apply-104` skill 本身就是一次降本。

### 9.1 Session analyzer（`sf analyze`）
以 skill 為單位的數字會漏掉發生在 skill 之間的浪費。`sf analyze` 讀取 `~/.claude/projects/*/` 裡的
transcript（CLI 本來就會寫的 JSONL），從 Uber 提到的反模式開始偵測：

| 反模式 | 偵測方式 |
|---|---|
| 簡單工作用了昂貴模型 | Opus 輪次輸出很短、只有 Read/Grep 工具呼叫 |
| 大型工具結果留在 context | 任何 > 20 KB 的工具結果，之後很多輪都還在 context 裡 |
| 休息後快取過期 | 輪次間隔 > 快取 TTL，接著出現大量快取寫入 |
| 預載 context 過重 | 使用者第一個 prompt 之前，第一個請求的輸入 tokens > 3 萬 |
| 在主 session 輪詢 | 連續 ≥3 次幾乎相同的 Bash 呼叫（`gh pr checks`、`sleep`） |
| 搜尋打轉 | 第一次 Edit 前有 > 8 次 Grep/Glob/find |
| 重試迴圈 | 同一個失敗指令執行 ≥ 3 次 |

輸出：每週報告，依成本排序 session，列出找到的反模式與建議修正（常常是「把這個做成 skill」或
「把這個移到 script」）。

### 9.2 即時成本顯示
Claude Code 的 `statusLine` 指令會在 stdin 收到 session 資訊（含成本）的 JSON，並輸出一行文字。
顯示 `$成本 · 模型 · 目前的 skill`。Uber 的做法：不設硬上限，只讓成本可見並適時提醒（例如在自己每日
預算的 50/80/100% 時通知）。

### 9.3 從 skill 到託管 agent
當 skill 能穩定通過評測，就讓它**不需要人來啟動**：
```bash
# cron / GitHub Actions
claude -p "/job-apply-104 page=1 dry_run=false max=20" \
  --model claude-haiku-5-5 --output-format stream-json --verbose > runs/$(date +%F).jsonl
```
每個託管 agent 在 spec 裡有成果指標（`outcome: 每次成功應徵的成本`、
`quality: 之後撤回 / 被視為濫投的比例`），它的執行紀錄和其他 session 一樣進入遙測與 papercut。
任何會對外產生影響的動作都要保留人工檢查點。

---

## 10. Papercuts：讓 agent 自己回報 bug

一個**常駐**的 skill `log-papercut`：

```markdown
---
name: log-papercut
description: Use when a skill, script or tool behaved unexpectedly, was confusing, or needed a workaround. Files a short structured report. Use it even if you recovered.
---
1. 在 `$SF_HOME/papercuts/inbox.jsonl` 寫入一行 JSON，包含：
   skill, skill_version, tried, expected, happened, suggested_fix, severity (low|med|high)。
2. 不得包含 secret、prompt 文字，或超過 3 行錯誤摘錄的檔案內容。
3. 用 `tail -n1` 確認已寫入，然後繼續使用者的任務。
```

每週循環（`sf papercuts`）：依 skill 分組 → 分流 → 根因分析 → 修正 → **把案例加進該 skill 的評測** →
PR → 量測是否退步 → 結案。每個修好的 papercut 都變成一個回歸測試。

Uber 的例子很適合直接拿來當 lint 規則：共用暫存檔、秒與毫秒搞混、缺少 key 時默默回傳錯的值、
永遠在等一個不存在的 CI。

**人類訊號**：當你退回或大幅修改某個 skill 產出的結果時，那個 prompt 就變成一個評測案例
（`sf eval add --from-session <id>`）。

---

## 11. 探索：`find-skills`

一個小 skill：給它任務描述，它搜尋 `skills/*/spec.yaml`（goal、triggers、domain），並用 `sf report`
的資料排序：評測通過率 × 使用量 × 新近度，扣掉成本。在有人寫出重複 skill 之前先回答「是不是已經有了？」。
品質分數：

```
quality = 0.4·eval_pass_rate_norm + 0.3·ablation_win_rate + 0.2·retention + 0.1·(1 − cost_vs_target)
```

---

## 12. 路線圖

| 階段 | 交付內容 | 完成條件 |
|---|---|---|
| **0 — 起步**（半天） | Repo 結構、`marketplace.json`，搬入 `lc-add`、`lc-coach`、`skills_poc/*`；為每個寫 `spec.yaml` | 可以從本 repo `/plugin install` |
| **1 — Lint + 遙測**（1–2 天） | `sf lint`、pre-commit hook、`hooks/telemetry.py`、`log-papercut` | 每次 skill 呼叫都進入 `events.jsonl` |
| **2 — 評測**（3–5 天） | 執行器、軌跡評分、每個 skill 的 golden 套件、各 5–10 個真實 prompt | 每個 skill 都有通過率 |
| **3 — 成本 + 報表**（2–3 天） | 以 OTel 或 transcript 算單次成本、`sf report` HTML | 一頁看到每個 skill 的單次成本與通過率 |
| **4 — Spec 編譯器**（2–3 天） | `sf new`、`sf compile`、spec↔SKILL hash 檢查 | 新 skill 都從 spec 開始 |
| **5 — 消融 + 無塵室**（3–5 天） | Docker 評測映像、`sf eval --ablate`、盲評 judge | 每個 core skill 都有保留/縮短/退役的判定 |
| **6 — CI + 探索**（2 天） | GitHub Actions、`find-skills`、品質分數 | PR 顯示通過率差異 |
| **7 — 工廠效率**（2–4 天） | Status line 成本、`sf analyze` session analyzer、每個 skill 的模型基準 | 每週報告最高成本 session 與反模式 |
| **8 — 託管 agent**（2–3 天） | 第一個有成果指標、依排程無頭執行的 skill | 一個 skill 依排程執行並回報每個成果的成本 |
| **之後** | Skill 版本間的 A/B 測試、依模型編譯、任務路由 | — |

### 第一批納管的 skill
| Skill | 為什麼適合先做 |
|---|---|
| `lc-add` | 輸出明確可檢查 → golden 評測容易寫 |
| `lc-coach` | 開放式 → 用來練作者評分套件與 judge 校準 |
| `python-code-reviewer`、`fastapi-master`（來自 `claude_code/skills_poc`） | 很好的消融對象：現在還贏得過基礎模型嗎？ |
| `job-apply-104`（來自 `../CLAUDE.md`） | 瀏覽器自動化 → 測試 Code Mode（把選擇器移進 Playwright script）、把輪詢移出主 session、無塵室（需要登入；絕不能含帳密） |
| `log-papercut`、`find-skills` | 平台 skill，常駐 |

---

## 13. 風險與待解問題

| 風險 | 緩解方式 |
|---|---|
| 評測成本（N 次 × 案例 × 模型） | 每個 PR 只跑 golden；軌跡評測只跑變動的 skill；每晚用便宜模型跑完整套件 |
| 非確定性讓差異充滿雜訊 | 以 N≥5 回報通過率與信賴區間；小差異不阻擋 |
| LLM judge 偏誤 | A/B 盲評順序、用作者評分集校準、追蹤 judge 與作者的一致率 |
| 遙測洩漏隱私 | 不記錄 prompt/檔案內容；路徑做 hash；`telemetry/` 不進版控 |
| 有副作用的 skill（104 流程會送出真的應徵） | 評測使用 mock/fixture 或 dry-run 旗標；CI 絕不對正式網站執行 |
| Agent/格式漂移（Claude Code、Codex 改 skill schema） | Spec 是真實來源；`sf compile` 依 agent 編譯；CI 中固定版本 |
| 罕見、高風險流程很難評測（Uber 也還沒解決） | Skill 內設人工檢查點；每次執行都留紀錄供事後檢視 |

---

## 14. 名詞表

- **Skill** — 含 `SKILL.md`（+ scripts、references）的資料夾，agent 在相關時載入。
- **Plugin** — 一起安裝的一組 skill（與 hook/指令）；我們每個分級一個。
- **軌跡（Trajectory）** — agent 依序做出的工具呼叫清單；用來取代（或補充）只看最終答案的評分。
- **消融（Ablation）** — 拿掉 skill 或其中一部分後重跑，以量測它的貢獻。
- **Papercut** — agent 提交的小型、結構化摩擦報告。
- **無塵室（Clean room）** — 不含開發者本機狀態的評測環境。
- **託管 agent（Managed agent）** — 由排程或系統事件啟動、不需人類開場的 agent，帶有自己的成果指標。
- **落地（Grounding）** — 預先提供 agent 需要的事實與脈絡，讓它不必花輪數去搜尋。
