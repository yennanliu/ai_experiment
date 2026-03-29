"""FastAPI dashboard for ML Profiler with Chart.js visualizations."""

import json
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from .db import ProfileDB

app = FastAPI(title="ML Profiler Dashboard")

CHART_JS_CDN = "https://cdn.jsdelivr.net/npm/chart.js"


def get_html_template(content: str, title: str = "ML Profiler", scripts: str = "") -> str:
    """HTML template with Chart.js."""
    return f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="{CHART_JS_CDN}"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 20px; background: #f0f2f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: #1a1a2e; margin-bottom: 10px; }}
        h2 {{ color: #16213e; margin-top: 30px; }}
        .nav {{ margin-bottom: 20px; padding: 15px; background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .nav a {{ margin-right: 20px; color: #4a90d9; text-decoration: none; font-weight: 500; }}
        .nav a:hover {{ text-decoration: underline; }}
        .metrics {{ display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 20px; }}
        .metric {{ background: white; padding: 20px 30px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); min-width: 150px; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #4a90d9; }}
        .metric-label {{ color: #666; font-size: 0.9em; }}
        .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; margin-bottom: 20px; }}
        .chart-container {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        table {{ border-collapse: collapse; width: 100%; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden; }}
        th, td {{ border-bottom: 1px solid #eee; padding: 12px 15px; text-align: left; }}
        th {{ background: #4a90d9; color: white; font-weight: 500; }}
        tr:hover {{ background: #f8f9fa; }}
        .regression {{ color: #dc3545; font-weight: bold; }}
        .improvement {{ color: #28a745; font-weight: bold; }}
        .stable {{ color: #6c757d; }}
        .badge {{ padding: 3px 8px; border-radius: 4px; font-size: 0.8em; }}
        .badge-danger {{ background: #f8d7da; color: #721c24; }}
        .badge-success {{ background: #d4edda; color: #155724; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="nav">
            <a href="/">📊 Dashboard</a>
            <a href="/bisect">🔍 Bisection</a>
            <a href="/api/results">API: Results</a>
            <a href="/api/models">API: Models</a>
        </div>
        {content}
    </div>
    {scripts}
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Main dashboard with charts."""
    try:
        db = ProfileDB()
        results = db.get_all(limit=200)
        model_names = db.get_models()
    except Exception as e:
        return get_html_template(f"<h1>Database Error</h1><p>{e}</p>")

    if not results:
        return get_html_template("""
            <h1>⚡ ML Profiler Dashboard</h1>
            <p>No data yet. Run profiling first:</p>
            <pre style="background:#fff;padding:15px;border-radius:8px;">ml-profiler profile --model demo_model --demo</pre>
        """)

    # Metrics
    total = len(results)
    unique_models = len(model_names)
    avg_time = sum(r.total_time_ms for r in results) / total

    metrics_html = f"""
    <div class="metrics">
        <div class="metric">
            <div class="metric-value">{total}</div>
            <div class="metric-label">Total Profiles</div>
        </div>
        <div class="metric">
            <div class="metric-value">{unique_models}</div>
            <div class="metric-label">Unique Models</div>
        </div>
        <div class="metric">
            <div class="metric-value">{avg_time:.2f}ms</div>
            <div class="metric-label">Avg Latency</div>
        </div>
    </div>
    """

    # Prepare chart data - Performance over time
    time_data = {}
    for r in sorted(results, key=lambda x: x.created_at):
        date_str = r.created_at.strftime("%m/%d %H:%M")
        if r.model_name not in time_data:
            time_data[r.model_name] = {"labels": [], "data": []}
        time_data[r.model_name]["labels"].append(date_str)
        time_data[r.model_name]["data"].append(round(r.total_time_ms, 2))

    # Prepare chart data - Model comparison (avg time)
    model_stats = {}
    for r in results:
        if r.model_name not in model_stats:
            model_stats[r.model_name] = []
        model_stats[r.model_name].append(r.total_time_ms)

    model_labels = list(model_stats.keys())
    model_avgs = [round(sum(v) / len(v), 2) for v in model_stats.values()]

    # Colors for charts
    colors = ['#4a90d9', '#50c878', '#ff6b6b', '#ffd93d', '#6c5ce7', '#a29bfe']

    charts_html = """
    <div class="charts">
        <div class="chart-container">
            <h3>Performance Over Time</h3>
            <canvas id="timeChart"></canvas>
        </div>
        <div class="chart-container">
            <h3>Model Comparison (Avg Latency)</h3>
            <canvas id="modelChart"></canvas>
        </div>
    </div>
    """

    # Table
    rows = ""
    for r in results[:30]:
        rows += f"""<tr>
            <td>{r.id}</td>
            <td><strong>{r.model_name}</strong></td>
            <td>{r.model_version}</td>
            <td>{r.hardware_target}</td>
            <td>{r.total_time_ms:.2f}</td>
            <td>{r.total_params:,}</td>
            <td>{r.created_at.strftime('%Y-%m-%d %H:%M')}</td>
        </tr>"""

    table_html = f"""
    <h2>Recent Profiles</h2>
    <table>
        <tr>
            <th>ID</th>
            <th>Model</th>
            <th>Version</th>
            <th>Hardware</th>
            <th>Time (ms)</th>
            <th>Params</th>
            <th>Date</th>
        </tr>
        {rows}
    </table>
    """

    # Chart.js scripts
    time_datasets = []
    for i, (model, data) in enumerate(time_data.items()):
        color = colors[i % len(colors)]
        time_datasets.append({
            "label": model,
            "data": data["data"],
            "borderColor": color,
            "backgroundColor": color + "40",
            "tension": 0.3,
            "fill": False,
        })

    # Use first model's labels (they should be similar across models)
    time_labels = list(time_data.values())[0]["labels"] if time_data else []

    scripts = f"""
    <script>
        // Time series chart
        new Chart(document.getElementById('timeChart'), {{
            type: 'line',
            data: {{
                labels: {json.dumps(time_labels)},
                datasets: {json.dumps(time_datasets)}
            }},
            options: {{
                responsive: true,
                plugins: {{ legend: {{ position: 'bottom' }} }},
                scales: {{
                    y: {{ title: {{ display: true, text: 'Time (ms)' }} }}
                }}
            }}
        }});

        // Model comparison chart
        new Chart(document.getElementById('modelChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(model_labels)},
                datasets: [{{
                    label: 'Avg Time (ms)',
                    data: {json.dumps(model_avgs)},
                    backgroundColor: {json.dumps(colors[:len(model_labels)])},
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    y: {{ title: {{ display: true, text: 'Time (ms)' }}, beginAtZero: true }}
                }}
            }}
        }});
    </script>
    """

    return get_html_template(
        f"<h1>⚡ ML Profiler Dashboard</h1>{metrics_html}{charts_html}{table_html}",
        scripts=scripts
    )


@app.get("/bisect", response_class=HTMLResponse)
async def bisect_page():
    """Version bisection page."""
    try:
        db = ProfileDB()
        model_names = db.get_models()
    except Exception as e:
        return get_html_template(f"<h1>Database Error</h1><p>{e}</p>")

    if not model_names:
        return get_html_template("""
            <h1>🔍 Version Bisection</h1>
            <p>No models found. Profile some models first.</p>
        """)

    # Build comparison for each model
    content = "<h1>🔍 Version Bisection</h1>"
    content += "<p>Compare performance across versions to detect regressions.</p>"

    for model_name in model_names:
        results = db.compare_versions(model_name)
        if not results:
            continue

        # Group by version
        versions = {}
        for r in results:
            if r.model_version not in versions:
                versions[r.model_version] = []
            versions[r.model_version].append(r)

        if len(versions) < 2:
            continue

        sorted_versions = sorted(versions.keys())
        baseline_ver = sorted_versions[0]
        baseline_times = [r.total_time_ms for r in versions[baseline_ver]]
        baseline_avg = sum(baseline_times) / len(baseline_times)

        content += f"<h2>{model_name}</h2>"
        content += "<table><tr><th>Version</th><th>Avg Time (ms)</th><th>Runs</th><th>vs Baseline</th><th>Status</th></tr>"

        for ver in sorted_versions:
            times = [r.total_time_ms for r in versions[ver]]
            avg = sum(times) / len(times)
            diff_pct = ((avg - baseline_avg) / baseline_avg) * 100

            if ver == baseline_ver:
                status = '<span class="stable">baseline</span>'
                diff_str = "-"
            elif diff_pct > 10:
                status = '<span class="badge badge-danger">⚠️ REGRESSION</span>'
                diff_str = f'<span class="regression">+{diff_pct:.1f}%</span>'
            elif diff_pct < -10:
                status = '<span class="badge badge-success">✅ FASTER</span>'
                diff_str = f'<span class="improvement">{diff_pct:.1f}%</span>'
            else:
                status = '<span class="stable">stable</span>'
                diff_str = f'<span class="stable">{diff_pct:+.1f}%</span>'

            content += f"<tr><td>{ver}</td><td>{avg:.2f}</td><td>{len(times)}</td><td>{diff_str}</td><td>{status}</td></tr>"

        content += "</table>"

    return get_html_template(content)


@app.get("/api/results")
async def api_results(limit: int = 100):
    """Get all results as JSON."""
    db = ProfileDB()
    results = db.get_all(limit=limit)
    return [{
        "id": r.id,
        "model": r.model_name,
        "version": r.model_version,
        "hardware": r.hardware_target,
        "time_ms": r.total_time_ms,
        "params": r.total_params,
        "created_at": r.created_at.isoformat(),
        "kernels": json.loads(r.top_kernels_json) if r.top_kernels_json else [],
    } for r in results]


@app.get("/api/models")
async def api_models():
    """Get list of models."""
    db = ProfileDB()
    return db.get_models()


@app.get("/api/compare/{model_name}")
async def api_compare(model_name: str):
    """Compare versions of a model."""
    db = ProfileDB()
    results = db.compare_versions(model_name)

    versions = {}
    for r in results:
        if r.model_version not in versions:
            versions[r.model_version] = {"times": [], "params": r.total_params}
        versions[r.model_version]["times"].append(r.total_time_ms)

    comparison = {}
    sorted_versions = sorted(versions.keys())

    if sorted_versions:
        baseline_avg = sum(versions[sorted_versions[0]]["times"]) / len(versions[sorted_versions[0]]["times"])

        for v, d in versions.items():
            avg = sum(d["times"]) / len(d["times"])
            diff_pct = ((avg - baseline_avg) / baseline_avg) * 100
            comparison[v] = {
                "avg_time_ms": round(avg, 2),
                "runs": len(d["times"]),
                "params": d["params"],
                "vs_baseline_pct": round(diff_pct, 1),
                "status": "regression" if diff_pct > 10 else ("improvement" if diff_pct < -10 else "stable"),
            }

    return comparison
