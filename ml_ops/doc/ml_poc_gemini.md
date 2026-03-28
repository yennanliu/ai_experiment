This document outlines a Proof of Concept (PoC) and side project roadmap specifically tailored to the **Software Engineer, Machine Learning Tooling** role at Waymo (Taipei/Hsinchu).

The role focuses on the **Compute Team**, emphasizing performance optimization, hardware-software co-design (custom silicon/accelerators), and building infrastructure to monitor/diagnose model compatibility.

---

# Doc: MLOps / ML Tooling Project List (Waymo Target)

**Location:** `@ml_ops/waymo-poc-projects.md`  
**Focus Areas:** ML Performance, Accelerator Tooling, Model Compatibility, Data Governance.

## 1. High-Impact PoCs (Direct JD Alignment)

### A. The "Performance Profiler" Dashboard
* **Goal:** Build a tool to visualize and diagnose model latency and throughput across different simulated hardware targets.
* **Core Tasks:**
    * Create a Python-based CLI that traces model execution (using `torch.profiler` or `NSight`).
    * Store performance metrics (FLOPs, memory bandwidth, kernel execution time) in a relational database (PostgreSQL).
    * Build a **Streamlit** or **Dash** frontend to "bisect" performance regressions between model versions.
* **Waymo Tie-in:** Directly maps to "Build infrastructure and tooling to monitor and diagnose ML model performance issues."

### B. Hardware-Aware Model Compatibility Checker
* **Goal:** A CI/CD gate that checks if a model (ONNX/TensorRT) is compatible with a specific "custom silicon" constraint (e.g., limited SRAM, specific op support).
* **Core Tasks:**
    * Define a JSON schema representing hardware constraints (supported dtypes, max tensor size).
    * Write a linter that parses an ONNX graph and flags unsupported layers or memory-heavy operations.
    * Automate this via GitHub Actions to simulate a "model release" workflow.
* **Waymo Tie-in:** Maps to "Monitor and diagnose ML model compatibility" and "Track releases."

### C. ML Metrics & Versioning Database
* **Goal:** Implement a centralized system for data governance and performance tracking.
* **Core Tasks:**
    * Design a normalized SQL schema to track `Model_ID`, `Hardware_Target`, `Latency_P99`, and `Compiler_Version`.
    * Implement an API (FastAPI) to query this data for "bisection" (finding which change caused a performance dip).
* **Waymo Tie-in:** Maps to "Build and sustain a database of ML model performance and other metrics."

---

## 2. Advanced Side Projects (The "Extra Mile")

### D. Kernel-Level Optimization Tooling
* **Goal:** Build a script to automate the "tuning" of ML kernels for performance.
* **Tech:** Use **Triton** or **TVM** to write a simple fused activation kernel and compare its performance against a baseline using your own profiling tool.
* **Waymo Tie-in:** Shows interest in "system-level compute architectures" and "optimization of hardware and software."

### E. Synthetic Performance Benchmarking Suite
* **Goal:** Create a "Golden Suite" of models (ResNet, Transformer, YOLO) and a harness that runs them against various "virtual" accelerators to generate a leaderboard.
* **Tech:** Docker, Prometheus, Grafana.
* **Waymo Tie-in:** Demonstrates the ability to "Partner and empower ML engineers to optimize the performance of their model."

---

## 3. Skills to Highlight in Repo
If you build these, ensure your `README.md` mentions:
* **Infrastructure as Code (IaC):** Use Terraform or Ansible to set up your DB/Dashboard.
* **Data Governance:** Show how you handle data quality for your performance metrics.
* **C++/Python Interop:** If possible, write the profiling logic in C++ and the dashboard in Python.

---

## 4. Suggested Tech Stack for these PoCs
* **Languages:** Python (Primary), C++ (for Profiling), TypeScript (for UI).
* **ML:** PyTorch, ONNX, TensorRT.
* **Data:** PostgreSQL (Relational), Redis (Caching).
* **Tools:** Docker, GitHub Actions, Streamlit, Weights & Biases (for inspiration).