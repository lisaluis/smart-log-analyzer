import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json, os

# ── Page Config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Log Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    .main-header h1 { font-size: 2.4rem; font-weight: 800; margin: 0; }
    .main-header p  { font-size: 1rem; opacity: 0.8; margin-top: 0.5rem; }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #0f3460;
        margin-bottom: 1rem;
    }
    .failure-card { border-left: 4px solid #e74c3c !important; }
    .normal-card  { border-left: 4px solid #27ae60 !important; }
    .result-failure {
        background: linear-gradient(135deg, #c0392b, #e74c3c);
        color: white; padding: 1.5rem; border-radius: 12px;
        text-align: center; font-size: 1.6rem; font-weight: bold;
    }
    .result-normal {
        background: linear-gradient(135deg, #1e8449, #27ae60);
        color: white; padding: 1.5rem; border-radius: 12px;
        text-align: center; font-size: 1.6rem; font-weight: bold;
    }
    .stButton > button {
        background: linear-gradient(135deg, #0f3460, #533483);
        color: white; border: none; border-radius: 8px;
        font-weight: 600; font-size: 1.1rem;
        padding: 0.7rem 2rem; width: 100%;
    }
    .stButton > button:hover { opacity: 0.9; }
    div[data-testid="stSidebar"] { background: #1a1a2e; }
    div[data-testid="stSidebar"] * { color: white !important; }
</style>
""", unsafe_allow_html=True)

# ── Config ─────────────────────────────────────────────────────────────
API_URL = os.environ.get("API_URL", "https://smart-log-analyzer-n3v3.onrender.com")

# ── State ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Header ────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔍 Smart Log Analyzer</h1>
    <p>System Failure Prediction using Machine Learning & MLOps</p>
    <p style="font-size:0.85rem; opacity:0.6;">Lisa Luis & Shannon Coelho | MSC AI, Goa University</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    api_url = st.text_input("API Endpoint", value=API_URL)

    st.markdown("---")
    st.markdown("## 📊 Quick Stats")

    if st.session_state.history:
        total     = len(st.session_state.history)
        failures  = sum(1 for r in st.session_state.history if r["prediction"] == 1)
        normals   = total - failures
        fail_rate = (failures / total * 100) if total else 0

        st.metric("Total Predictions", total)
        st.metric("Failures Detected", failures, delta=f"{fail_rate:.1f}% rate")
        st.metric("Normal Readings", normals)
    else:
        st.info("Run a prediction to see stats")

    st.markdown("---")
    st.markdown("## 🔗 Links")
    st.markdown("- [API Health Check](" + api_url + "/health)")
    st.markdown("- [API Metrics](" + api_url + "/metrics)")
    st.markdown("- [Model Info](" + api_url + "/info)")

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

# ── Tabs ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Predict", "📈 History", "📊 Analytics", "ℹ️ Model Info"])

# ═════════════════════ TAB 1 — PREDICT ═══════════════════════════════
with tab1:
    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("### 📥 Sensor Input")
        st.caption("Enter machine sensor readings to predict failure probability")

        # Feature ranges (from AI4I dataset statistics)
        air_temp  = st.slider("🌡️ Air Temperature (K)",
                              min_value=295.0, max_value=305.0, value=298.1,
                              step=0.1, help="Ambient temperature in Kelvin")

        proc_temp = st.slider("🔥 Process Temperature (K)",
                              min_value=305.0, max_value=315.0, value=308.6,
                              step=0.1, help="Internal machine temperature in Kelvin")

        rot_speed = st.slider("⚙️ Rotational Speed (RPM)",
                              min_value=1168, max_value=2886, value=1551,
                              step=1, help="Rotations per minute")

        torque    = st.slider("🔩 Torque (Nm)",
                              min_value=3.8, max_value=76.6, value=42.8,
                              step=0.1, help="Force applied to machine components")

        tool_wear = st.slider("🔧 Tool Wear (min)",
                              min_value=0, max_value=253, value=0,
                              step=1, help="Time in minutes since last tool change")

        st.markdown("")
        predict_btn = st.button("🚀 Predict Failure", use_container_width=True)

    with col_result:
        st.markdown("### 🎯 Prediction Result")

        if predict_btn:
            payload = {
                "air_temp":        air_temp,
                "process_temp":    proc_temp,
                "rotational_speed":rot_speed,
                "torque":          torque,
                "tool_wear":       tool_wear
            }

            with st.spinner("Analyzing sensor data..."):
                try:
                    response = requests.post(
                        f"{api_url}/predict",
                        json=payload,
                        timeout=60
                    )
                    result = response.json()

                    if "error" in result:
                        st.error(f"API Error: {result['error']}")
                    else:
                        pred = result["prediction"]
                        prob = result["failure_probability"]
                        status = result["status"]

                        if pred == 1:
                            st.markdown(f"""
                            <div class="result-failure">
                                ⚠️ FAILURE PREDICTED<br>
                                <span style="font-size:1rem; opacity:0.9">Failure Probability: {prob:.1%}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="result-normal">
                                ✅ NORMAL OPERATION<br>
                                <span style="font-size:1rem; opacity:0.9">Failure Probability: {prob:.1%}</span>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown("")

                        # Gauge chart
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=round(prob * 100, 1),
                            title={"text": "Failure Risk (%)"},
                            gauge={
                                "axis": {"range": [0, 100]},
                                "bar":  {"color": "#e74c3c" if pred == 1 else "#27ae60"},
                                "steps": [
                                    {"range": [0,  40],  "color": "#d5f5e3"},
                                    {"range": [40, 70],  "color": "#fef9e7"},
                                    {"range": [70, 100], "color": "#fadbd8"},
                                ],
                                "threshold": {
                                    "line":  {"color": "red", "width": 3},
                                    "thickness": 0.75,
                                    "value": 70
                                }
                            }
                        ))
                        fig.update_layout(height=280, margin=dict(t=40, b=10))
                        st.plotly_chart(fig, use_container_width=True)

                        # Latency
                        st.caption(f"⚡ Response time: {result.get('latency_seconds', 0)*1000:.1f} ms")

                        # Save to history
                        st.session_state.history.append({
                            "timestamp":    datetime.now().strftime("%H:%M:%S"),
                            "air_temp":     air_temp,
                            "process_temp": proc_temp,
                            "rot_speed":    rot_speed,
                            "torque":       torque,
                            "tool_wear":    tool_wear,
                            "prediction":   pred,
                            "probability":  prob,
                            "status":       status
                        })

                except requests.exceptions.Timeout:
                    st.warning("⏳ API is waking up (free tier takes ~30s). Please wait and try again.")
                except Exception as e:
                    st.error(f"Connection error: {e}")
        else:
            st.info("👈 Adjust the sliders and click **Predict Failure** to analyze sensor readings")

            # Feature importance (approximate, for display)
            features = ["Air Temp", "Process Temp", "Rot. Speed", "Torque", "Tool Wear"]
            importance = [0.12, 0.10, 0.22, 0.31, 0.25]

            fig = px.bar(
                x=importance, y=features, orientation="h",
                title="📊 Feature Importance (Random Forest)",
                color=importance, color_continuous_scale="Blues",
                labels={"x": "Importance", "y": "Feature"}
            )
            fig.update_layout(height=300, showlegend=False,
                              coloraxis_showscale=False, margin=dict(t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

# ═════════════════════ TAB 2 — HISTORY ═══════════════════════════════
with tab2:
    st.markdown("### 📋 Prediction History")

    if not st.session_state.history:
        st.info("No predictions yet. Go to the **Predict** tab to get started!")
    else:
        df_hist = pd.DataFrame(st.session_state.history)

        # Color-coded table
        def color_status(val):
            color = "#fadbd8" if val == "FAILURE" else "#d5f5e3"
            return f"background-color: {color}"

        styled = df_hist[["timestamp", "air_temp", "process_temp", "rot_speed",
                           "torque", "tool_wear", "probability", "status"]].style.applymap(
            color_status, subset=["status"]
        ).format({"probability": "{:.1%}"})

        st.dataframe(styled, use_container_width=True)

        # Download
        csv = df_hist.to_csv(index=False)
        st.download_button(
            "⬇️ Download History as CSV",
            data=csv,
            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# ═════════════════════ TAB 3 — ANALYTICS ═════════════════════════════
with tab3:
    st.markdown("### 📊 Prediction Analytics")

    if len(st.session_state.history) < 2:
        st.info("Make at least 2 predictions to see analytics.")
    else:
        df_hist = pd.DataFrame(st.session_state.history)
        col1, col2 = st.columns(2)

        with col1:
            # Pie chart
            counts = df_hist["status"].value_counts()
            fig = px.pie(values=counts.values, names=counts.index,
                         title="Normal vs Failure Distribution",
                         color=counts.index,
                         color_discrete_map={"NORMAL": "#27ae60", "FAILURE": "#e74c3c"})
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Probability over time
            fig2 = px.line(df_hist, x="timestamp", y="probability",
                           title="Failure Probability Over Time",
                           markers=True,
                           color_discrete_sequence=["#0f3460"])
            fig2.add_hline(y=0.5, line_dash="dash", line_color="red",
                           annotation_text="Risk Threshold")
            fig2.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig2, use_container_width=True)

        # Scatter: Torque vs Tool Wear
        fig3 = px.scatter(df_hist, x="torque", y="tool_wear",
                          color="status", size="probability",
                          title="Torque vs Tool Wear (colored by status)",
                          color_discrete_map={"NORMAL": "#27ae60", "FAILURE": "#e74c3c"})
        st.plotly_chart(fig3, use_container_width=True)

# ═════════════════════ TAB 4 — MODEL INFO ════════════════════════════
with tab4:
    st.markdown("### ℹ️ Model & Project Information")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🤖 ML Model")
        st.markdown("""
| Parameter | Value |
|-----------|-------|
| Algorithm | Random Forest |
| Dataset | AI4I Predictive Maintenance |
| Training samples | ~8,000 |
| Test samples | ~2,000 |
| Features | 5 sensor readings |
| Target | Machine failure (binary) |
""")
        st.markdown("#### 🏗️ MLOps Stack")
        st.markdown("""
| Tool | Purpose |
|------|---------|
| **MLflow** | Experiment tracking |
| **Docker** | Containerization |
| **GitHub Actions** | CI/CD pipeline |
| **Render** | Cloud deployment |
| **Prometheus** | Metrics collection |
| **Grafana** | Monitoring dashboard |
""")

    with col2:
        # Live model info from API
        try:
            info_resp = requests.get(f"{api_url}/info", timeout=10)
            info = info_resp.json()
            st.markdown("#### 📈 Live Model Metrics")
            metrics = info.get("metrics", {})
            if metrics:
                m1, m2 = st.columns(2)
                m1.metric("Accuracy",  f"{metrics.get('accuracy', 0):.2%}")
                m2.metric("F1 Score",  f"{metrics.get('f1_score', 0):.2%}")
                m3, m4 = st.columns(2)
                m3.metric("Precision", f"{metrics.get('precision', 0):.2%}")
                m4.metric("Recall",    f"{metrics.get('recall', 0):.2%}")
                if "roc_auc" in metrics:
                    st.metric("ROC AUC", f"{metrics.get('roc_auc', 0):.4f}")
            else:
                st.info("Metrics not available — run evaluate.py first")
        except:
            st.warning("⏳ API is starting up, model metrics unavailable right now")

        st.markdown("#### 📡 API Status")
        try:
            health_resp = requests.get(f"{api_url}/health", timeout=10)
            health = health_resp.json()
            if health.get("status") == "healthy":
                st.success("✅ API is healthy and responding")
            else:
                st.warning("⚠️ API status unknown")
        except:
            st.error("❌ API is not reachable (may be starting up)")
