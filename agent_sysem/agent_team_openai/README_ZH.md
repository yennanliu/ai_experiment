# Agent Team（代理團隊）

專門化代理協調框架 - 多代理協調模式的最小化、優雅實現。

## 功能特性

- **專門化代理**：分析師、開發者、審查者、文檔編寫者，各具特定提示詞（~2500 tokens vs ~16000 用於單體）
- **智慧路由**：關鍵字匹配 + LLM 備選方案用於任務分類
- **協調模式**：
  - **管道模式**：順序執行，每個輸出輸入給下一個
  - **中樞輻射模式**：中央協調器協調所有代理
  - **並行合併**：並發執行結果合併
- **令牌預算管理**：追蹤和控制 API 成本
- **最小上下文傳遞**：僅代理間傳遞摘要

## 安裝

### 前置條件
- Python 3.11+
- OpenAI API 密鑰（在 `.env` 中設置 `OPENAI_API_KEY`）

### 設置步驟

```bash
# 初始設置 - 安裝所有依賴項並創建虛擬環境
uv sync

# 設置後，使用以下命令運行：
uv run agent-team "您的任務"
uv run python main.py
```

### 何時使用哪個命令

| 命令 | 使用時機 | 目的 |
|------|---------|------|
| `uv sync` | 首次設置、更新 `pyproject.toml` 後、或 `.venv` 損壞 | 安裝/更新所有依賴項並同步鎖定文件 |
| `uv run <cmd>` | 設置後運行腳本/CLI | 在虛擬環境中運行命令，無需激活 |
| `source .venv/bin/activate` | （可選）想以交互方式使用 Python | 手動激活虛擬環境 |
| `uv pip install <pkg>` | 需要快速添加依賴項 | 直接 pip 安裝（更新鎖定文件） |

### 配置

在項目根目錄創建 `.env` 文件：

```bash
OPENAI_API_KEY=sk-xxxxxxxxxxxx
```

CLI 會自動加載此文件，或手動設置：

```bash
export OPENAI_API_KEY=sk-xxxxxxxxxxxx
uv run agent-team "您的任務"
```

## 使用方式

### 命令行界面 (CLI)

```bash
# 簡單任務，自動路由
uv run agent-team "寫一個 Python 函數來驗證電子郵件"

# 指定協調模式
uv run agent-team --pattern pipeline "分析需求並實現功能"

# 指定代理
uv run agent-team --agents analyst,developer,reviewer "構建 REST API"

# 完整開發工作流
uv run agent-team --workflow dev "創建用戶身份驗證系統"

# 並行代碼審查
uv run agent-team --workflow review "def foo(): return eval(input())"

# 顯示詳細輸出
uv run agent-team --verbose "您的任務"
```

### Python API

```python
from agent_team import Orchestrator, AgentRole, OrchestrationPattern

# 初始化
orchestrator = Orchestrator()

# 簡單單代理任務
response = orchestrator.simple(
    "寫一個函數來驗證電子郵件",
    role=AgentRole.DEVELOPER,
)

# 完整開發工作流（分析師 -> 開發者 -> 審查者）
state = orchestrator.analyze_and_implement(
    "使用令牌桶算法創建速率限制器"
)

# 自定義管道
state = orchestrator.run(
    "為此 API 編寫文檔",
    pattern=OrchestrationPattern.PIPELINE,
    agents=[AgentRole.ANALYST, AgentRole.DOC_WRITER],
)

# 從多個角度進行並行審查
state = orchestrator.review_from_all_angles(code)
```

## 架構

```
┌─────────────────────────────────────────────────────────┐
│                     協調器                              │
│  ┌─────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  路由器  │  │ 令牌預算     │  │  模式選擇器      │   │
│  └─────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ 管道模式 │    │中樞模式  │    │ 並行模式 │
    └──────────┘    └──────────┘    └──────────┘
          │               │               │
          ▼               ▼               ▼
    ┌─────────────────────────────────────────┐
    │           專門化代理                    │
    │ 分析師│開發者│審查者│文檔編寫者         │
    └─────────────────────────────────────────┘
```

## 配置選項

```python
from agent_team.orchestrator import Orchestrator, OrchestratorConfig

config = OrchestratorConfig(
    model="gpt-4o",                    # 要使用的模型
    token_budget=100_000,              # 令牌預算上限
    max_retries=2,                     # 失敗重試次數
    auto_select_pattern=True,          # 自動選擇協調模式
)

orchestrator = Orchestrator(config)
```

### 可用模型

- `gpt-4o`（推薦）- 最強大，最佳質量
- `gpt-4-turbo` - 平衡速度和成本
- `gpt-3.5-turbo` - 最快和最便宜

## 輸出和日誌

### 目錄結構

每個任務會創建組織化的輸出和日誌：

```
output/
  └── YYYYMMDD_HHMMSS/           # 時間戳目錄
      └── {job_id}/              # 唯一的任務ID（8個字符）
          ├── metadata.json       # 執行元數據和統計信息
          └── output.txt          # 完整的執行輸出

log/
  └── YYYYMMDD_HHMMSS/
      └── {job_id}/
          └── output.log          # 時間戳日誌文件（無顏色）
```

### 輸出文件

**metadata.json** - 包含：
- 任務ID和時間戳
- 任務、模式、使用的代理
- 每個代理的令牌使用情況
- 成功狀態和錯誤

**output.txt** - 包含：
- 執行摘要
- 每個代理的完整輸出
- 令牌使用分解

**output.log** - 包含：
- 時間戳日誌條目
- 所有 INFO、WARNING、ERROR 日誌
- 清晰的格式（無 ANSI 代碼）

### 彩色日誌

日誌在終端中以顏色顯示，易於閱讀：
- 🟢 **INFO** - 綠色
- 🟡 **WARNING** - 黃色
- 🔴 **ERROR** - 紅色
- 🔵 **DEBUG** - 青色

顏色代碼從日誌文件中去除以保證清晰的輸出。

### 自定義輸出/日誌目錄

```bash
# 保存到自定義位置
uv run agent-team --output-dir ./results --log-dir ./logs "您的任務"

# 或使用默認設置（output/ 和 log/）
uv run agent-team "您的任務"
```

## 令牌節省

通過使用具有特定提示詞的專門化代理而不是單一的單體代理：

| 方式 | 每個代理的令牌 | 開銷 |
|------|--------------|------|
| 單體 | ~16,000 | 高 |
| 專門化 | ~2,500 | 低 |
| **節省** | **~80%** | |

## 工作流程

### 開發工作流 (`--workflow dev`)
1. **分析師**：分解需求、識別約束
2. **開發者**：編寫代碼實現
3. **審查者**：檢查質量、安全性、最佳實踐

### 審查工作流 (`--workflow review`)
1. 所有代理並行執行代碼審查
2. 整合來自多個視角的反饋

### 文檔工作流 (`--workflow docs`)
1. **分析師**：提取關鍵概念
2. **文檔編寫者**：創建清晰的文檔

## 示例

### 簡單函數開發

```bash
uv run agent-team --workflow dev "創建一個 JSON 驗證函數"
```

輸出：
- 代碼實現
- 代碼審查反饋
- 改進建議

### 代碼審查

```bash
uv run agent-team --workflow review "
def process_user(data):
    query = f'SELECT * FROM users WHERE id = {data}'
    return db.execute(query)
"
```

### 並行執行

```python
from agent_team import Orchestrator, OrchestrationPattern, AgentRole

orchestrator = Orchestrator()

state = orchestrator.run(
    "分析系統架構並提供改進建議",
    pattern=OrchestrationPattern.PARALLEL_MERGE,
    agents=[AgentRole.ANALYST, AgentRole.DEVELOPER, AgentRole.REVIEWER]
)

print(f"令牌使用：{state.total_tokens}")
print(f"最終輸出：{state.responses[-1].content}")
```

## 常見問題

**Q: 為什麼我的任務被路由到特定代理？**
A: 系統使用關鍵字匹配進行快速路由。對於複雜任務，會調用 LLM 進行智慧分類。

**Q: 我可以使用其他 LLM 模型嗎？**
A: 目前系統使用 OpenAI 模型。計劃支持其他提供商。

**Q: 如何追蹤令牌使用量？**
A: 每個執行状態都包含 `total_tokens` 和 `tokens_used` 信息。

**Q: 什麼是令牌預算？**
A: 它是總令牌使用量的限制。達到後，進一步的請求將被拒絕。

## 許可證

MIT

## 貢獻

歡迎提交 Issues 和 Pull Requests！
