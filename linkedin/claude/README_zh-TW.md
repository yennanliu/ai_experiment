# LinkedIn 自動投履歷工具

使用 Chrome DevTools Protocol (CDP) 自動化 LinkedIn 求職申請。
連接現有瀏覽器 session，無需重新登入。

## 快速開始

### 1. 安裝 chrome-cdp-skill

```bash
cd ~/.claude/skills
git clone https://github.com/pasky/chrome-cdp-skill.git chrome-cdp
```

### 2. 安裝 Node.js 22+

```bash
nvm install 22
nvm use 22
```

### 3. 設定 Chrome

```bash
# 開啟 Chrome 並啟用遠端除錯
# 前往：chrome://inspect/#remote-debugging
# 開啟開關
```

### 4. 登入 LinkedIn

在 Chrome 開啟 LinkedIn 並確保已登入。

### 5. 執行自動化

```bash
# 設定環境
source ~/.nvm/nvm.sh && nvm use 22
CDP="~/.claude/skills/chrome-cdp/skills/chrome-cdp/scripts/cdp.mjs"

# 列出 Chrome 分頁以取得目標 ID
$CDP list

# 導航至職缺搜尋頁面（將 TARGET 替換為你的分頁 ID 前綴）
$CDP nav TARGET "https://www.linkedin.com/jobs/search/?keywords=Software%20Engineer&location=China&f_AL=true&f_TPR=r604800&f_WT=2,3"

# 使用包裝腳本執行簡易指令
./cdp.sh jobs TARGET      # 列出職缺
./cdp.sh apply TARGET     # 點擊 Easy Apply
./cdp.sh next TARGET      # 點擊下一步
./cdp.sh review TARGET    # 點擊檢視
./cdp.sh submit TARGET    # 送出申請
./cdp.sh close TARGET     # 關閉對話框
```

## 包裝腳本 (cdp.sh)

```bash
# 設定目標一次
export LINKEDIN_TARGET=4BF87A

# 基本指令
./cdp.sh list              # 列出 Chrome 分頁
./cdp.sh nav               # 導航至職缺搜尋
./cdp.sh shot              # 截圖
./cdp.sh jobs              # 列出可投職缺

# 申請流程
./cdp.sh nextjob           # 點擊下一個未投職缺
./cdp.sh apply             # 點擊 Easy Apply 按鈕
./cdp.sh next              # 點擊下一步
./cdp.sh review            # 點擊檢視
./cdp.sh submit            # 送出申請
./cdp.sh close             # 關閉成功對話框

# 處理問題
./cdp.sh yes               # 回答「是」（工作授權問題）

# 完整自動化
./cdp.sh auto              # 自動完成一個申請
```

## 申請流程

| 步驟 | 進度 | 動作 |
|------|------|------|
| 聯絡資訊 | 0% | 已預填 → 下一步 |
| 履歷 | 25% | 選擇履歷 → 下一步 |
| 工作經歷 | 50% | 已預填 → 下一步 |
| 學歷 | 75% | 已預填 → 檢視 |
| 附加資訊 | (選填) | 訊息 → 檢視 |
| 檢視 | 100% | 確認 → 送出 |
| 成功 | 完成 | 關閉對話框 |

## 重要選擇器

| 元素 | 選擇器 |
|------|--------|
| Easy Apply 按鈕 | `button.jobs-apply-button` |
| 申請對話框 | `.jobs-easy-apply-modal` |
| 下一步按鈕 | `button[aria-label='Continue to next step']` |
| 檢視按鈕 | `button[aria-label='Review your application']` |
| 送出按鈕 | `button[aria-label='Submit application']` |
| 關閉按鈕 | `button[aria-label='Dismiss']` |
| 職缺卡片 | `[data-job-id]` |

## URL 參數

| 參數 | 說明 | 值 |
|------|------|-----|
| `keywords` | 搜尋關鍵字 | `Software%20Engineer` |
| `location` | 地點 | `China`、`Taiwan` |
| `f_AL` | 僅限 Easy Apply | `true` |
| `f_TPR` | 發布日期 | `r86400` (24小時)、`r604800` (一週) |
| `f_WT` | 遠端工作 | `1` (現場)、`2` (遠端)、`3` (混合) |

## 檔案說明

| 檔案 | 說明 |
|------|------|
| `cdp.sh` | CDP 指令包裝腳本 |
| `linkedin_auto_apply.js` | 瀏覽器端輔助函式 |
| `linkedin_automation_controller.js` | 主控制器與設定 |
| `CLAUDE.md` | 詳細自動化指南 |
| `README.md` | 英文快速入門指南 |
| `README_zh-TW.md` | 繁體中文指南（本檔案） |

## 系統需求

- Chrome 瀏覽器（已啟用遠端除錯）
- Node.js 22+（內建 WebSocket 支援）
- LinkedIn 帳號（已登入）
- 已安裝 chrome-cdp-skill

## 疑難排解

| 問題 | 解決方案 |
|------|----------|
| WebSocket is not defined | 使用 Node.js 22+：`nvm use 22` |
| 訊息彈窗擋住畫面 | 導航時加上 `&refresh=true` |
| 對話框未出現 | 點擊後等待 2-3 秒 |
| 有額外問題需回答 | 使用 `./cdp.sh yes` 回答 |

## 相關資源

- 104.com.tw 自動化：參見專案 CLAUDE.md
- chrome-cdp-skill：https://github.com/pasky/chrome-cdp-skill

---

## 使用範例

### 投遞 10 個中國軟體工程師職缺

```bash
# 1. 設定環境
source ~/.nvm/nvm.sh && nvm use 22
export LINKEDIN_TARGET=4BF87A  # 替換為你的分頁 ID

# 2. 導航至職缺搜尋
./cdp.sh nav "$LINKEDIN_TARGET" "https://www.linkedin.com/jobs/search/?keywords=Software%20Engineer&location=China&f_AL=true&f_TPR=r604800&f_WT=2,3"

# 3. 投遞 10 個職缺
for i in {1..10}; do
  echo "=== 投遞第 $i 個職缺 ==="
  ./cdp.sh auto "$LINKEDIN_TARGET"
  sleep 3
done
```

### 手動逐步投遞

```bash
# 1. 查看職缺清單
./cdp.sh jobs

# 2. 點擊下一個未投職缺
./cdp.sh nextjob

# 3. 點擊 Easy Apply
./cdp.sh apply

# 4. 完成申請步驟
./cdp.sh next    # 重複數次
./cdp.sh review
./cdp.sh submit

# 5. 關閉成功對話框
./cdp.sh close
```
