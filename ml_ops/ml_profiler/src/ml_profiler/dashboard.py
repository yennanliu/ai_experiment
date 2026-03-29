"""FastAPI dashboard for ML Profiler."""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from .db import ProfileDB

app = FastAPI(title="ML Profiler Dashboard")


def get_html_template(content: str, title: str = "ML Profiler") -> str:
    """Simple HTML template."""
    return f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; margin: 40px; background: #f5f5f5; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background: #4a90d9; color: white; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        tr:hover {{ background: #f0f7ff; }}
        .metric {{ display: inline-block; background: white; padding: 20px; margin: 10px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #4a90d9; }}
        .metric-label {{ color: #666; }}
        .nav {{ margin-bottom: 20px; }}
        .nav a {{ margin-right: 20px; color: #4a90d9; text-decoration: none; }}
        .nav a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="nav">
        <a href="/">Dashboard</a>
        <a href="/api/results">API: All Results</a>
        <a href="/api/models">API: Models</a>
    </div>
    {content}
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Main dashboard page."""
    try:
        db = ProfileDB()
        results = db.get_all(limit=100)
        model_names = db.get_models()
    except Exception as e:
        return get_html_template(f"<h1>Database Error</h1><p>{e}</p>")

    if not results:
        return get_html_template("""
            <h1>ML Profiler Dashboard</h1>
            <p>No data yet. Run profiling first:</p>
            <pre>ml-profiler profile --model demo_model --demo</pre>
        """)

    # Metrics
    total = len(results)
    unique_models = len(model_names)
    avg_time = sum(r.total_time_ms for r in results) / total

    metrics_html = f"""
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
        <div class="metric-label">Avg Time</div>
    </div>
    """

    # Table
    rows = ""
    for r in results[:50]:
        rows += f"""<tr>
            <td>{r.id}</td>
            <td>{r.model_name}</td>
            <td>{r.model_version}</td>
            <td>{r.hardware_target}</td>
            <td>{r.total_time_ms:.2f}</td>
            <td>{r.total_params:,}</td>
            <td>{r.created_at.strftime('%Y-%m-%d %H:%M')}</td>
        </tr>"""

    table_html = f"""
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

    return get_html_template(f"""
        <h1>ML Profiler Dashboard</h1>
        {metrics_html}
        <h2>Recent Profiles</h2>
        {table_html}
    """)


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

    return {
        v: {
            "avg_time_ms": sum(d["times"]) / len(d["times"]),
            "runs": len(d["times"]),
            "params": d["params"],
        }
        for v, d in versions.items()
    }
