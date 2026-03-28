# POC / side projects aligned with Waymo — Software Engineer, ML Tooling

**Role reference:** 

- W Software Engineer, Machine Learning Tooling (Taipei / Hsinchu)



Use this list to build **credible, demo-able** work that matches what the posting emphasizes: **ML performance tooling**, **metrics + storage**, **debugging / bisect workflows**, and **good UX for internal users** (not just training models).

---

## What the JD rewards (map your README / resume bullets here)

| Theme | What to show in a POC |
|--------|------------------------|
| Monitor & diagnose | Regressions, compatibility checks, failure modes, traces tied to a model + run |
| Empower rapid iteration | CLI + small UI, diff two runs, “why slower?” breakdowns |
| Metrics DB + query UX | Schema design, pipelines into Postgres/SQLite + optional object/parquet layer, simple dashboards |

**Stated bar:** strong software + DS&A, plus **data visualization** experience. **Nice-to-have:** relational + document/NoSQL, **data quality / governance**, **accessibility**, infra for **model diagnosis** and **tooling**.

---

## Tier A — Weekend to ~1 week (portfolio-friendly)

1. **Model run registry + “perf diff” CLI**  
   Log each training or export run (Git SHA, deps hash, latency p50/p95, memory). CLI: `compare-run <id_a> <id_b>` prints table + optional ASCII chart.  
   *Shows:* versioning discipline, perf awareness, developer-facing tooling.

2. **Minimal “model compatibility matrix”**  
   Small service or script: given `model.onnx` / `torch.jit` + runtime version matrix, run a smoke inference and record pass/fail + errors. Store results; simple HTML table or Streamlit view with red/green cells.  
   *Shows:* compatibility diagnosis, structured logging, visualization.

3. **Latency breakdown dashboard**  
   Instrument a dummy inference pipeline (preprocess → model → postprocess). Push timings to Prometheus + Grafana *or* a static dashboard from Parquet/CSV.  
   *Shows:* observability thinking, visualization, where tooling helps ML engineers.

4. **Bisect helper for ML configs**  
   Given a folder of YAML configs and a scoring script, automate “which config change caused metric X to drop?” (git bisect style or grid over checkpoints).  
   *Shows:* bisect/debug narrative from the JD, automation, reproducibility.

---

## Tier B — 1–3 weeks (strong signal)

5. **Metrics warehouse for experiments**  
   Postgres (or SQLite for demo) with normalized tables: `experiment`, `run`, `metric`, `artifact`. Optional **dbt** or SQL views for “latest run per experiment”. Add **Great Expectations** or simple checks for null rates / schema drift.  
   *Shows:* relational modeling, **data quality**, governance-light patterns.

6. **Failure corpus + triage UI**  
   Collect bad inputs (images, tensors, or tabular rows) that caused errors; store metadata, stack traces, model version. Simple React or Streamlit UI: filter, tag root cause, link to run.  
   *Shows:* diagnosis tooling, internal-customer UX, search/filter.

7. **ONNX / TorchScript runner as a tiny “lab” API**  
   FastAPI service: upload model + sample batch → returns latency, outputs hash, maybe numerical diff against a golden output. Persist jobs in Redis/Postgres.  
   *Shows:* infra adjacent to models, API design, async jobs.

8. **Release notes generator from the metrics DB**  
   Query last N runs; auto-summarize “accuracy +2.1%, latency +8% (p95)” and emit Markdown for PRs.  
   *Shows:* glue between data and engineering workflow, “track releases” from the JD.

---

## Tier C — deeper (if you want one flagship project)

9. **End-to-end “ML perf observatory”**  
   Combine registry + dashboards + alerting (e.g. “p95 latency regression > 10% vs main”). Add role-based views and **basic a11y** (keyboard nav, labels, contrast) on the UI.  
   *Shows:* product-shaped internal tooling, accessibility mention from “We prefer”.

10. **Edge / embedded-style perf sandbox (scaled down)**  
    Profile the same small model on CPU vs GPU vs quantized; store power/latency curves; visualize trade-offs.  
    *Shows:* performance culture relevant to automotive compute (without needing hardware access).

---

## Stretch details that echo “We prefer”

- **Data governance:** document retention, PII policy for logs, explicit “bronze/silver” layers if you use a lake metaphor.  
- **Accessibility:** one screen reader–friendly dashboard or documented WCAG checklist for your UI.  
- **Non-relational touch:** metrics blobs in object storage or a document collection for nested traces, *with* a clear sync path into SQL for reporting.

---

## How to present each POC on GitHub

- **Problem → user** (ML engineer trying to ship faster / debug a regression).  
- **One architecture diagram** (data flow: runs → storage → UI/CLI).  
- **Repro:** `docker compose up` or single `make demo` with synthetic data.  
- **Limitations** honestly stated (scope, no production SLA).

This mirrors the posting’s focus on **tooling, metrics, diagnosis, and iteration** for ML software—not on publishing novel models.
