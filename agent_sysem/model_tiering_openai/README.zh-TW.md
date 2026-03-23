# 模型分層成本優化

OpenAI API 的智能模型路由，以優化成本並保持品質。

## 概述

本模組實現多種路由策略，根據任務複雜度自動選擇合適的 OpenAI 模型（GPT-3.5-turbo、GPT-4 或 GPT-4-turbo），相比使用 GPT-4 完成所有請求，最高可節省 **71% 的成本**。

## 安裝

```bash
cd model_tiering_openai
uv sync
```

## 配置

### 設定 OpenAI API 金鑰

1. 建立 `.env` 檔案：

```bash
cp .env.example .env
```

2. 編輯 `.env` 並新增您的 OpenAI API 金鑰：

```
OPENAI_API_KEY=sk-your-api-key-here
```

3. 執行指令碼時會自動載入 `.env` 檔案。

**安全提示：** 千萬不要將 `.env` 檔案提交到版本控制系統。`.gitignore` 已設定排除此檔案。

## 快速開始

### 執行演示

```bash
# 快速演示（包含路由範例和成本估計）
uv run demo

# 完整功能演示套件
uv run example
```

### 程式碼中使用

```python
from main import HybridRouter, get_recommended_model

# 快速推薦（無需 API 呼叫）
model = get_recommended_model("將 hello 翻譯成西班牙語")
print(model)  # ModelTier.GPT_3_5

# 完整路由與執行（需要設定 OPENAI_API_KEY）
router = HybridRouter()
result = router.execute("總結這篇文章")
print(f"使用模型: {result.model_used.name}, 成本: ${result.cost:.6f}")
```

## 路由策略

### 1. 規則型路由

零成本的路由，使用關鍵詞配對。適合定義完善的任務模式。

```python
from main import RuleBasedRouter

router = RuleBasedRouter()
decision = router.route("將這段文字翻譯成日文")
# -> ModelTier.GPT_3_5（符合翻譯規則）
```

### 2. LLM 輔助分類

使用 GPT-3.5-turbo（~$0.0005/請求）來分類任務複雜度和類別。

```python
from main import LLMClassifierRouter

router = LLMClassifierRouter()
decision = router.route("設計分散式系統")
# -> ModelTier.GPT_4_TURBO（分類為專家級別）
```

### 3. 動態品質提升

以 GPT-3.5-turbo 開始，若品質不足則自動升級。

```python
from main import DynamicEscalationRouter

router = DynamicEscalationRouter(quality_threshold=70.0)
result = router.execute_with_escalation("複雜推理任務")
# 若 GPT-3.5-turbo 的回應品質 < 70 會自動升級
```

### 4. 混合型路由（推薦）

生產級路由器，結合所有策略。

```python
from main import HybridRouter

router = HybridRouter(
    use_llm_classification=True,
    enable_escalation=True,
    quality_threshold=70.0,
)
result = router.execute("您的任務")
print(router.get_stats())  # 查看路由統計
```

## 自訂規則

新增特定領域的路由規則：

```python
from main import RuleBasedRouter, RoutingRule, ModelTier

router = RuleBasedRouter()

# 醫療任務需要最高準確度
router.add_rule(RoutingRule(
    name="medical",
    condition=lambda t: "診斷" in t.lower(),
    model=ModelTier.GPT_4_TURBO,
    priority=95,
))
```

## 成本估計

```python
from main import CostEstimator

estimates = CostEstimator.estimate_monthly_cost(
    requests_per_day=1000,
    avg_input_tokens=500,
    avg_output_tokens=1000,
)
print(f"全使用 GPT-4-Turbo: ${estimates['all_gpt4_turbo']:.2f}/月")
print(f"智能分層: ${estimates['smart_tiering']:.2f}/月")
print(f"節省: {estimates['savings_vs_gpt4_turbo']}%")
```

## 模型選擇準則

| 複雜度 | 模型 | 使用案例 |
|--------|------|---------|
| 簡單 | GPT-3.5-turbo | 簡單問答、翻譯、數據提取 |
| 簡單 | GPT-3.5-turbo | 格式化、摘要、基本任務 |
| 中等 | GPT-4 | 程式碼生成、分析、多步驟 |
| 複雜 | GPT-4 | 除錯、審核、詳細分析 |
| 專家級 | GPT-4-turbo | 安全性、證明、架構設計 |

## 典型分佈

根據分析，約 70% 的任務不需要最強大的模型：

- **GPT-3.5-turbo**: 70%（簡單 + 基本任務）
- **GPT-4**: 25%（中等 + 複雜任務）
- **GPT-4-turbo**: 5%（專家級任務）

## 成本效益

對於每天 100 個請求，平均 500 個輸入 token 和 1000 個輸出 token：

- 全使用 GPT-4-Turbo: **$105.00/月**
- 全使用 GPT-4: **$225.00/月**
- 全使用 GPT-3.5-turbo: **$5.25/月**
- 智能分層: **$65.17/月** ← 推薦
- 相比 GPT-4-Turbo 節省: **37.9%**
- 相比 GPT-4 節省: **71.0%**

## 主要特性

✨ **四種路由策略**
- 規則型路由（零成本）
- LLM 輔助分類
- 動態品質提升
- 混合智能路由

📊 **成本監控**
- 準確的成本追蹤
- 月度成本估計
- 節省百分比分析

🎯 **品質保證**
- 自動品質評估
- 動態模型升級
- 統計追蹤

🔧 **易於集成**
- 簡單的 Python API
- 支援自訂規則
- 完整的文件和範例

## 常見問題

### 執行示例需要 API 金鑰嗎？

不一定。規則型路由完全不需要 API 呼叫。只有在使用 LLM 輔助分類或動態提升功能時才需要。

### 如何確認 API 金鑰正確？

```bash
# 檢查 .env 檔案
cat .env | grep OPENAI_API_KEY

# 在 Python 中測試
from main import HybridRouter
router = HybridRouter()
result = router.execute("2+2=?", max_tokens=10)
print(f"成本: ${result.cost:.6f}")
```

### 如何追蹤成本？

```python
router = HybridRouter()
# ... 執行多個任務 ...
stats = router.get_stats()
print(f"總成本: ${stats['total_cost']:.2f}")
print(f"模型分佈: {stats['model_distribution']}")
```

## 參考資源

- [OpenAI 文件](https://platform.openai.com/docs)
- [定價資訊](https://platform.openai.com/pricing)
- [API 速率限制](https://platform.openai.com/account/rate-limits)
- [原始指南（中文）](https://yennj12.js.org/yennj12_blog_V4/posts/model-tiering-cost-optimization-guide-zh/)

## 命令快速參考

```bash
# 安裝依賴
uv sync

# 執行快速演示
uv run demo

# 執行完整演示
uv run example

# 直接使用 Python
python main.py
python example.py
```

## 安全提示

- 🔒 永遠不要在程式碼中硬編碼 API 金鑰
- 🔐 使用環境變數存儲所有憑證
- 📁 不要提交 `.env` 檔案到版本控制
- 🔄 定期輪換 API 金鑰
- 📊 監控 OpenAI 控制台以防止意外費用

## 許可

此專案基於模型分層成本優化概念。詳見參考資源。
