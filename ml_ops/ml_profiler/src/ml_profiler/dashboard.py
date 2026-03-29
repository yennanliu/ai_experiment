"""Streamlit dashboard for ML Profiler."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .db import ProfileDB


def load_data():
    """Load data from database."""
    try:
        db = ProfileDB()
        results = db.get_all(limit=500)
        if not results:
            return pd.DataFrame()

        data = []
        for r in results:
            data.append({
                "id": r.id,
                "model": r.model_name,
                "version": r.model_version,
                "hardware": r.hardware_target,
                "time_ms": r.total_time_ms,
                "cpu_time_ms": r.cpu_time_ms,
                "cuda_time_ms": r.cuda_time_ms,
                "memory_mb": r.memory_allocated_mb,
                "params": r.total_params,
                "flops": r.total_flops,
                "input_shape": r.input_shape,
                "created_at": r.created_at,
            })
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return pd.DataFrame()


def main():
    st.set_page_config(
        page_title="ML Profiler Dashboard",
        page_icon="⚡",
        layout="wide",
    )

    st.title("⚡ ML Profiler Dashboard")
    st.markdown("Monitor and analyze ML model performance metrics")

    # Load data
    df = load_data()

    if df.empty:
        st.warning("No profiling data found. Run `ml-profiler profile` to generate data.")
        st.code("ml-profiler profile --model resnet18 --version 1.0.0", language="bash")
        return

    # Sidebar filters
    st.sidebar.header("Filters")
    models = ["All"] + sorted(df["model"].unique().tolist())
    selected_model = st.sidebar.selectbox("Model", models)

    if selected_model != "All":
        df = df[df["model"] == selected_model]

    # Metrics overview
    st.header("📊 Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Profiles", len(df))
    with col2:
        st.metric("Unique Models", df["model"].nunique())
    with col3:
        st.metric("Avg Time (ms)", f"{df['time_ms'].mean():.2f}")
    with col4:
        st.metric("Avg Memory (MB)", f"{df['memory_mb'].mean():.2f}")

    # Performance over time
    st.header("📈 Performance Over Time")

    if len(df) > 1:
        fig = px.line(
            df.sort_values("created_at"),
            x="created_at",
            y="time_ms",
            color="model",
            markers=True,
            title="Inference Time Over Time",
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Time (ms)",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Model comparison
    st.header("🔄 Model Comparison")
    col1, col2 = st.columns(2)

    with col1:
        model_stats = df.groupby("model").agg({
            "time_ms": "mean",
            "memory_mb": "mean",
            "params": "first",
        }).reset_index()

        fig = px.bar(
            model_stats,
            x="model",
            y="time_ms",
            title="Average Inference Time by Model",
            color="model",
        )
        fig.update_layout(yaxis_title="Time (ms)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            model_stats,
            x="model",
            y="memory_mb",
            title="Average Memory Usage by Model",
            color="model",
        )
        fig.update_layout(yaxis_title="Memory (MB)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Version comparison (bisection helper)
    st.header("🔍 Version Bisection")
    st.markdown("Compare performance across versions to identify regressions")

    if selected_model != "All":
        version_df = df.groupby("version").agg({
            "time_ms": ["mean", "std", "count"],
            "memory_mb": "mean",
        }).reset_index()
        version_df.columns = ["version", "time_mean", "time_std", "runs", "memory_mean"]

        if len(version_df) > 1:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=version_df["version"],
                y=version_df["time_mean"],
                error_y=dict(type="data", array=version_df["time_std"]),
                name="Inference Time",
            ))
            fig.update_layout(
                title=f"Version Comparison: {selected_model}",
                xaxis_title="Version",
                yaxis_title="Time (ms)",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Regression detection
            if len(version_df) >= 2:
                latest = version_df.iloc[-1]
                previous = version_df.iloc[-2]
                diff_pct = ((latest["time_mean"] - previous["time_mean"]) / previous["time_mean"]) * 100

                if diff_pct > 10:
                    st.error(f"⚠️ Performance regression detected: {diff_pct:.1f}% slower than previous version")
                elif diff_pct < -10:
                    st.success(f"✅ Performance improvement: {abs(diff_pct):.1f}% faster than previous version")
                else:
                    st.info(f"ℹ️ Performance stable: {diff_pct:+.1f}% change from previous version")
        else:
            st.info("Add more versions to enable comparison")
    else:
        st.info("Select a specific model to compare versions")

    # Raw data table
    st.header("📋 Raw Data")
    st.dataframe(
        df[["model", "version", "hardware", "time_ms", "memory_mb", "params", "created_at"]]
        .sort_values("created_at", ascending=False),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
