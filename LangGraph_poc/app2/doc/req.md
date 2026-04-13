## AI Email Reply Assistant — Simple Proposal (xxx AI)
Date: 2026-04-09  
Client: my_awesome company
Channels: Outlook email + Notion knowledge base

### 1) Executive summary
The client wants an AI-assisted workflow to **pre-generate reply email drafts** that follow a **fixed format, wording, and consistent tone**, using **historical email threads saved in Notion** (and/or other databases) as reference.  

We propose a **human-in-the-loop “Draft Generator”** that:
- Ingests a new inbound email (initially via copy/paste into Notion)
- Retrieves similar historical cases from Notion (RAG)
- Generates a reply draft that matches the company style and required template
- Lets the user review/edit before sending via Outlook

This approach minimizes process change while delivering immediate time savings and tone consistency, with a clear upgrade path to deeper Outlook/Notion automation later.

### 2) Current situation (as understood)
- **Current flow**: Outlook inbound email → sales reads → drafts in Notion (fills some structured info) → internal confirmation → replies in Outlook.
- **Email volume**: ~20–40 emails/day.
- **Main categories** (example): New PO / Order changes / Wrong order / Issue handling (roughly ~50% order, ~50% wrong order).
- **Historical record**: saved in **Notion** (client confirmed).
- **Pain points**:
  - Reply speed depends on individual experience
  - Tone/wording inconsistency (esp. for new staff)
  - Hard to reuse “good” historical replies

### 3) Target outcome
- **Faster replies** with consistent formatting/tone
- **Template-driven** replies (not generic chatbot output)
- **Reusable knowledge** from historic email responses
- **Human control**: user must review/edit before sending

### 4) Proposed workflow (simple + elegant)
We recommend starting with the lightest workflow change (matches the client’s existing behavior).

#### Phase 1 (MVP): Notion-first, paste-to-generate
1. **New email received** in Outlook.
2. User creates/opens a Notion record (existing practice) and **pastes email content**.
3. User clicks **“Generate reply draft”** (Notion button / integration trigger).
4. AI:
   - Classifies email type (e.g., New PO / Wrong Order / Change Order)
   - Retrieves similar historical cases from Notion (same customer + similar issue)
   - Generates a reply in the **required company template** and tone
   - Writes back to Notion as **Draft Reply**
5. User reviews/edits in Notion and copies final content to Outlook to send.

This phase delivers value without requiring Outlook automation permissions immediately, and fits the client’s stated preference: **AI drafts automatically, but humans still paste/review/send**.

#### Optional Phase 2 (Streamline): Outlook draft creation
If Microsoft Graph access is allowed, we can reduce manual copy/paste:
- Automatically create an **Outlook Draft Reply** with the generated content
- Include links back to the Notion record + retrieved reference cases
- User reviews the draft in Outlook and sends

#### Optional Phase 3 (Automation): Inbox monitoring + notifications
If desired and permitted:
- Monitor inbox (webhook/polling) and auto-create a Notion page per email
- Notify user via Teams/Slack or Notion mentions
- Still requires **human approval** before sending

### 5) Solution design (high-level)
#### 5.1 Components
- **Notion Knowledge Base**
  - Stores historical emails + final replies
  - Stores templates and “gold standard” examples
- **AI Orchestration Service (Backend)**
  - Receives incoming email text from Notion trigger
  - Performs retrieval + drafting
  - Returns structured outputs back to Notion
- **Retrieval (RAG) Layer**
  - Indexes historical cases (email + reply + metadata)
  - Finds most similar cases based on topic/customer/type
- **LLM Draft Generator**
  - Generates draft reply strictly following the provided template and style constraints

#### 5.2 Data model (recommended minimal structure in Notion)
Create/standardize a Notion database (if not already structured) with fields:
- **Date** (YYYYMMDD) / **Thread ID** (optional)
- **Customer name**
- **Email type** (New PO / Change / Wrong Order / Other)
- **Inbound email** (raw text)
- **Draft reply (AI)**
- **Final reply (sent)**
- **Status** (New / Drafted / Reviewed / Sent)
- **Sensitive flags** (e.g., contains bank account / contract / NDA content)

This structure makes retrieval and continuous improvement much more reliable.

### 6) Ensuring “same format, same style” (key requirement)
We will use a **template-first generation** approach:
- **Reply Template Library** per email type (e.g., Wrong Order, New PO, Change Order)
- **Few-shot examples**: 5–10 “gold standard” replies per type to anchor tone and phrasing
- **Output constraints**:
  - Fixed sections and order (greeting, summary, required confirmations, next steps, closing)
  - Controlled length and wording patterns
  - Language rules (Chinese/English) as needed

We will also generate a **“Checklist” block** for the user (internal-only) that highlights:
- Missing info needed to finalize reply (e.g., PO number, pickup date, quantity)
- Risk flags (mentions bank account, NDA, pricing tables)

### 7) Security & privacy (baseline)
Because emails may include sensitive information (e.g., bank account, NDA):
- **Human approval required** before sending (no auto-send in initial scope)
- **Data minimization**: only send necessary content to the model
- **Redaction options** (configurable):
  - Detect and mask bank accounts / IDs / sensitive patterns before LLM call
- **Audit logging**:
  - Who generated, edited, and approved each reply
  - Which references were retrieved

Deployment options depend on client constraints:
- Public cloud LLM (fastest) vs. private/VPC vs. on-prem (if required)

### 8) Implementation options (to match IT constraints)
We propose three tiers to keep decisions simple:

- **Tier A — Notion-triggered MVP (recommended start)**
  - Manual paste email into Notion
  - One-click generate draft
  - No Outlook API required

- **Tier B — Add Outlook draft creation**
  - Requires Microsoft Graph OAuth permissions
  - AI writes the draft back to Outlook

- **Tier C — End-to-end workflow automation**
  - Inbox monitoring + auto Notion record + notifications
  - Still human approval to send

### 9) Pilot / PoC proposal (fast path)
#### Input needed from client
- 20–50 historical examples: **(inbound email → final reply)** pairs (anonymized is fine)
- Notion sample database structure (or export)
- 3–5 templates / “gold standard” replies (or we help extract them from history)

#### PoC deliverables
- Notion database + “Generate Draft” workflow
- Retrieval over historical cases (RAG)
- Draft replies in consistent format for top 2–3 email types
- Basic metrics dashboard (time saved, edit distance / acceptance rate)

#### Success criteria (suggested)
- **30–50% reduction** in time-to-first-draft
- **Tone consistency**: fewer manual rewrites for style
- **Adoption**: user accepts AI draft with minor edits in >50–70% of cases (target depends on email complexity)

### 10) Suggested timeline (indicative)
- **Week 1**: Data intake + Notion structure alignment + template extraction
- **Week 2**: RAG indexing + draft generation MVP + review loop
- **Week 3 (optional)**: Outlook draft integration (if Graph access approved)

### 11) Open questions / assumptions
To finalize scope and integration:
- **Outlook environment**: Microsoft 365? Web/Desktop? Graph API allowed?
- **Notion**: Workspace-level API access available? Database vs free text pages?
- **Language**: Chinese only vs bilingual?
- **Sensitive data policy**: any content not allowed to be processed by external LLM?
- **Approval flow**: who must approve before sending in special cases (VIP/complaints/legal)?

---
### Appendix A — Minimal user instructions (MVP)
1. Paste inbound email into Notion record.
2. Click “Generate Draft”.
3. Review/edit Draft Reply.
4. Copy to Outlook and send.

