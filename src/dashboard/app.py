"""Dark Pool Detection — Streamlit Dashboard."""

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from src.alerts.engine import AlertEngine
from src.data.simulator import OrderBookSimulator
from src.data.yfinance_feed import YFinanceFeed
from src.detection.dark_volume import DarkVolumeReconstructor
from src.pipeline import DarkPoolPipeline

st.set_page_config(page_title="Dark Pool Detection", page_icon="🌑", layout="wide")

st.markdown("""
<style>
.metric-card { border: 1px solid #0f3460; border-radius: 12px; padding: 1rem; }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🌑 Dark Pool Detection")
mode = st.sidebar.radio("Mode", ["Live Simulation", "Yahoo Finance Data"])

n_tickers = st.sidebar.slider("Tickers", 1, 10, 5)
n_ticks = st.sidebar.slider("Ticks", 100, 50000, 5000, step=100)
seed = st.sidebar.number_input("Seed", 0, 999, 42)
vpin_threshold = st.sidebar.slider("VPIN Toxicity", 0.5, 1.0, 0.8, 0.05)
iceberg_confidence = st.sidebar.slider("Iceberg Min Confidence", 0.3, 1.0, 0.6, 0.05)
dark_share_threshold = st.sidebar.slider("Dark Share Alert %", 20.0, 80.0, 40.0, 5.0)

run_button = st.sidebar.button("🔄 Run Detection", type="primary", use_container_width=True)

if "results" not in st.session_state:
    st.session_state.results = None

st.title("🌑 Dark Pool Detection Dashboard")
st.caption("Real-time dark pool activity monitoring & order flow analysis")

if run_button or st.session_state.results is None:
    with st.spinner("Running dark pool detection pipeline..."):
        pipeline = DarkPoolPipeline()
        results = pipeline.run_simulation(n_ticks=n_ticks, n_tickers=n_tickers, seed=seed)
        alert_engine = AlertEngine(thresholds={
            "vpin": {"toxic": vpin_threshold, "elevated": vpin_threshold - 0.2},
            "iceberg": {"min_confidence": iceberg_confidence, "min_hidden_volume": 5000},
            "dark_volume": {"anomaly_zscore": 3.0, "dark_share_pct": dark_share_threshold},
            "ml": {"dark_trade_prob": 0.7, "iceberg_fill_prob": 0.6},
        })
        alert_list = alert_engine.run_checks(results)
        st.session_state.results = results
        st.session_state.alerts = alert_list

results = st.session_state.results

if results is None:
    st.info("Click 'Run Detection' to start analyzing dark pool activity.")
    st.stop()

r = results
sim = r["simulation"]
vp = r["vpin"]
ice = r["iceberg"]
dark = r["dark_analysis"]
score = r["detection_score"]

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Detection Score", f"{score['overall']:.1f}/100")
col2.metric("Dark Pool Activity", f"{sim.get('dark_pct', 0):.1f}%")
col3.metric("VPIN", f"{vp.get('current_vpin', 0):.3f}")
col4.metric("Active Icebergs", ice.get("n_active", 0))
col5.metric("Alerts", len(st.session_state.get("alerts", [])))

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "⚡ VPIN", "🧊 Iceberg", "🌑 Dark Volume"])

with tab1:
    st.subheader("Simulation Summary")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write(f"• **Total Trades:** {sim['total_trades']}")
        st.write(f"• **Dark Pool:** {sim['dark_trades']} ({sim['dark_pct']:.1f}%)")
        st.write(f"• **Iceberg:** {sim['iceberg_trades']} ({sim['iceberg_pct']:.1f}%)")
        st.write(f"• **Informed:** {sim['informed_trades']} ({sim['informed_pct']:.1f}%)")
    with col_b:
        if sim.get("trader_types"):
            types_df = pd.DataFrame(list(sim["trader_types"].items()), columns=["Type", "Count"])
            st.bar_chart(types_df.set_index("Type"))

with tab2:
    st.subheader("VPIN — Order Flow Toxicity")
    vpin_data = r.get("vpin_history", [])
    if vpin_data:
        vpin_df = pd.DataFrame(vpin_data)
        if "vpin" in vpin_df.columns:
            st.line_chart(vpin_df.set_index("time")["vpin"])
    st.metric("Current VPIN", f"{vp.get('current_vpin', 0):.4f}")
    st.metric("Mean", f"{vp.get('mean', 0):.4f}")
    st.metric("Max", f"{vp.get('max', 0):.4f}")
    st.metric("Toxic %", f"{vp.get('toxic_pct', 0):.1f}%")

with tab3:
    st.subheader("Iceberg Detection")
    st.metric("Active Icebergs", ice.get("n_active", 0))
    st.metric("Native", ice.get("n_native", 0))
    st.metric("Synthetic", ice.get("n_synthetic", 0))
    st.metric("Est. Hidden Vol", f"{ice.get('total_estimated_hidden', 0):,}")
    st.metric("Avg Confidence", f"{ice.get('avg_confidence', 0):.2%}")

with tab4:
    st.subheader("Dark Volume Analysis")
    st.metric("Dark Trades", dark.get("n_dark", 0))
    st.metric("Dark Volume %", f"{dark.get('dark_volume_pct', 0):.1f}%")
    st.metric("Avg Dark Size", f"{dark.get('avg_dark_trade_size', 0):.0f}")

st.markdown("---")
st.caption("Dark Pool Detection System v0.1 | Research use only")
