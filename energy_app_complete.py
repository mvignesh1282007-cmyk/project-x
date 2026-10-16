"""
====================================================================================================
PROJECT TITLE: EnerSim AI — Factory Energy Spike, Tariff Penalty & Carbon Cost Forecaster
ARCHITECTURE: Machine Learning (Random Forest & Isolation Forest) + Synthetic Industrial Load Streamer
AUTHOR: Industrial AI Systems Engineer
ENVIRONMENT: 100% Offline, Python 3.9+, Streamlit, Plotly, Pandas, Scikit-Learn
HOW TO RUN:
    1. pip install streamlit plotly pandas scikit-learn
    2. python -m streamlit run energy_app_complete.py
====================================================================================================
"""

import time
import random
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, IsolationForest

# ==================================================================================================
# MODULE 1: INDUSTRIAL HEAVY EQUIPMENT & ENERGY PHYSICS SIMULATOR
# ==================================================================================================

EQUIPMENT_PROFILES = {
    "Electric Arc Furnace / Smelter": {"base_kw": 280.0, "power_factor": 0.82, "var_kw": 45.0},
    "Hydraulic Stamping Press (500 Ton)": {"base_kw": 120.0, "power_factor": 0.85, "var_kw": 30.0},
    "CNC 5-Axis Milling Centers (x4)": {"base_kw": 65.0, "power_factor": 0.91, "var_kw": 12.0},
    "Industrial Air Compressors & Chillers": {"base_kw": 95.0, "power_factor": 0.88, "var_kw": 15.0},
    "Automated Conveyor & Paint Line": {"base_kw": 45.0, "power_factor": 0.93, "var_kw": 8.0}
}

class FactoryEnergySimulator:
    """Simulates real-time electrical telemetry for heavy manufacturing plants."""
    def __init__(self):
        self.history = []
        self.max_history = 45 # rolling buffer of time ticks
        
        # Pre-seed history with 25 steady ticks
        curr_time = time.time() - 50.0
        for _ in range(25):
            t_str = time.strftime("%H:%M:%S", time.localtime(curr_time))
            curr_time += 2.0
            self.history.append({
                "time": t_str,
                "power_kw": round(random.uniform(320.0, 390.0), 1),
                "power_factor": round(random.uniform(0.86, 0.92), 2),
                "co2_rate_kg": round(random.uniform(140.0, 175.0), 1),
                "peak_breach": False,
                "anomaly": False
            })

    def record_tick(self, power_kw: float, power_factor: float, peak_limit: float, is_anomaly: bool):
        t_str = time.strftime("%H:%M:%S")
        co2_rate = round(power_kw * 0.475, 1) # ~0.475 kg CO2 per kWh grid average
        is_breach = power_kw > peak_limit

        self.history.append({
            "time": t_str,
            "power_kw": round(power_kw, 1),
            "power_factor": round(power_factor, 2),
            "co2_rate_kg": co2_rate,
            "peak_breach": is_breach,
            "anomaly": is_anomaly
        })

        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_history_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

# ==================================================================================================
# MODULE 2: OFFLINE AI LOAD FORECASTER & ANOMALY DETECTOR
# ==================================================================================================

@st.cache_resource
def train_offline_energy_ai():
    """
    Trains offline models on realistic industrial energy load distributions.
    Model 1: Random Forest Regressor for 24-Hour Horizon Load Forecasting.
    Model 2: Isolation Forest for detecting phantom idle machinery waste & reactive power drift.
    """
    np.random.seed(42)
    n_samples = 1200
    
    # Synthetic operational feature matrix
    hour_of_day = np.random.uniform(0, 24, n_samples)
    shift_index = np.where((hour_of_day >= 8) & (hour_of_day <= 18), 1, 0) # 1 = peak shift
    furnace_status = np.random.choice([0, 1], n_samples, p=[0.25, 0.75])
    press_status = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    pf_reading = np.random.uniform(0.78, 0.96, n_samples)
    
    # Physics relationship: Power (kW) = baseline + operational load + shift surge
    base = 110.0 + (shift_index * 80.0) + (furnace_status * 280.0) + (press_status * 120.0)
    noise = np.random.normal(0, 15.0, n_samples)
    total_kw = np.clip(base + noise, 90.0, 750.0)
    
    X = np.column_stack([hour_of_day, shift_index, furnace_status, press_status, pf_reading])
    y = total_kw
    
    forecaster = RandomForestRegressor(n_estimators=45, max_depth=6, random_state=42)
    forecaster.fit(X, y)
    
    anomaly_detector = IsolationForest(n_estimators=50, contamination=0.05, random_state=42)
    anomaly_detector.fit(np.column_stack([total_kw, pf_reading]))
    
    return forecaster, anomaly_detector

# ==================================================================================================
# MODULE 3: STREAMLIT INDUSTRIAL WEB DASHBOARD
# ==================================================================================================

st.set_page_config(
    page_title="EnerSim AI | Factory Energy Spike & Carbon Forecaster",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background-color: #1a1f2c;
        border: 1px solid #2e384d;
        border-radius: 8px;
        padding: 14px;
        margin-top: 5px;
    }
    .status-normal {
        background-color: rgba(46, 204, 113, 0.15);
        color: #2ecc71;
        border: 1px solid #2ecc71;
        padding: 6px 14px;
        border-radius: 16px;
        font-weight: bold;
    }
    .status-danger {
        background-color: rgba(231, 76, 60, 0.15);
        color: #e74c3c;
        border: 1px solid #e74c3c;
        padding: 6px 14px;
        border-radius: 16px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

if "energy_sim" not in st.session_state:
    st.session_state.energy_sim = FactoryEnergySimulator()

sim = st.session_state.energy_sim
forecaster, anomaly_detector = train_offline_energy_ai()

# --- SIDEBAR: PLANT MACHINERY DISPATCH & TARIFF CONTROLS ---
st.sidebar.image("https://img.icons8.com/fluency/96/electricity.png", width=65)
st.sidebar.title("⚡ Plant Energy Dispatch")
st.sidebar.caption("Real-Time Industrial Energy Management System")

st.sidebar.subheader("1. Active Manufacturing Equipment")
st.sidebar.caption("Toggle heavy industrial loads On/Off:")

equipment_states = {}
total_active_load = 50.0 # lighting & basic baseload
weighted_pf = 0.95

for eq_name, specs in EQUIPMENT_PROFILES.items():
    active = st.sidebar.toggle(eq_name, value=True if "Arc Furnace" in eq_name or "Compressors" in eq_name or "Milling" in eq_name else False)
    equipment_states[eq_name] = active
    if active:
        load = specs["base_kw"] + random.uniform(-specs["var_kw"]*0.4, specs["var_kw"]*0.4)
        total_active_load += load
        weighted_pf = min(weighted_pf, specs["power_factor"] + random.uniform(-0.02, 0.02))

st.sidebar.divider()
st.sidebar.subheader("2. Grid Tariff & Peak Demand Limits")
peak_demand_threshold = st.sidebar.number_input("Peak Demand Penalty Limit (kW)", min_value=200.0, max_value=800.0, value=480.0, step=20.0)
base_tariff_kwh = st.sidebar.number_input("Standard Electricity Tariff ($/kWh)", min_value=0.05, max_value=0.50, value=0.12, step=0.01)
penalty_multiplier = st.sidebar.slider("Peak-Hour Demand Surge Penalty (x)", 1.5, 4.0, 2.5, step=0.5)

# --- RUN AI MODELS ---
is_breach = total_active_load > peak_demand_threshold
anomaly_score = anomaly_detector.predict([[total_active_load, weighted_pf]])[0]
is_anomaly = (anomaly_score == -1)

# Record current telemetry tick
sim.record_tick(total_active_load, weighted_pf, peak_demand_threshold, is_anomaly)
df_history = sim.get_history_df()

# Financial calculations
current_hourly_cost = total_active_load * base_tariff_kwh
penalty_hourly_cost = (current_hourly_cost * penalty_multiplier) if is_breach else current_hourly_cost
monthly_projected_cost = penalty_hourly_cost * 24 * 30
potential_savings = ((current_hourly_cost * penalty_multiplier) - current_hourly_cost) * 24 * 30 if is_breach else 0.0

# --- HEADER SECTION ---
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("⚡ EnerSim AI: Factory Energy Spike & Carbon Forecaster")
    st.markdown("**Facility:** Heavy Manufacturing Plant Cell #1 • **Target:** Peak-Tariff Avoidance & Carbon Abatement • **Execution:** 100% Offline")

with col_h2:
    st.write("")
    if is_breach:
        st.markdown('<div class="status-danger">🔴 PEAK TARIFF SURGE DETECTED</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-normal">🟢 LOAD OPTIMAL & COMPLIANT</div>', unsafe_allow_html=True)

st.divider()

# --- TOP EXECUTIVE KPI ROW ---
col_k1, col_k2, col_k3, col_k4 = st.columns(4)

with col_k1:
    st.metric(
        "Current Power Demand",
        f"{total_active_load:.1f} kW",
        delta=f"{'+' if is_breach else ''}{total_active_load - peak_demand_threshold:.1f} kW vs Limit",
        delta_color="inverse" if is_breach else "normal"
    )
    st.caption(f"Peak Ceiling: {peak_demand_threshold:.0f} kW")

with col_k1 if False else col_k2:
    st.metric(
        "Power Factor (PF)",
        f"{weighted_pf:.2f}",
        delta="Inductive Load" if weighted_pf < 0.90 else "Healthy Reactive",
        delta_color="normal" if weighted_pf >= 0.90 else "inverse"
    )
    st.caption("Target > 0.90 to avoid utility surcharge")

with col_k3:
    st.metric(
        "Est. Hourly Electricity Cost",
        f"${penalty_hourly_cost:.2f} / hr",
        delta=f"+{(penalty_multiplier-1)*100:.0f}% Peak Penalty" if is_breach else "Standard Base Tariff",
        delta_color="inverse" if is_breach else "normal"
    )
    st.caption("Real-Time Utility Draw")

with col_k4:
    st.metric(
        "Carbon Intensity Rate",
        f"{total_active_load * 0.475:.1f} kg CO2/hr",
        delta=f"{total_active_load * 0.475 * 24 * 30 / 1000.0:.1f} Tons / month",
        delta_color="inverse"
    )
    st.caption("Grid Emission Factor: 0.475 kg/kWh")

st.divider()

# --- REAL-TIME TELEMETRY STRIP CHART ---
st.subheader("📈 Live Factory Power Demand vs. Utility Penalty Threshold")

fig_power = go.Figure()

# Peak Limit Red Line
fig_power.add_hline(
    y=peak_demand_threshold,
    line_dash="dash",
    line_color="#e74c3c",
    line_width=2.5,
    annotation_text=f"Peak Tariff Penalty Threshold ({peak_demand_threshold} kW)",
    annotation_font_color="#e74c3c"
)

# Power History Line
fig_power.add_trace(go.Scatter(
    x=df_history["time"],
    y=df_history["power_kw"],
    mode="lines+markers",
    name="Plant Load (kW)",
    line=dict(color="#f1c40f", width=3),
    fill="tozeroy",
    fillcolor="rgba(241, 196, 15, 0.08)"
))

fig_power.update_layout(
    height=340,
    margin=dict(l=20, r=20, t=30, b=20),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15, 23, 42, 0.8)",
    xaxis=dict(title="Timestamp (HH:MM:SS)", gridcolor="#22304a"),
    yaxis=dict(title="Power Draw (kW)", range=[0, max(peak_demand_threshold * 1.35, 600)], gridcolor="#22304a"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_power, use_container_width=True)

# --- LOWER SECTION: 24-HOUR FORECAST CONE & ENERGY SAVINGS ACTION ---
col_bot1, col_bot2 = st.columns([1.3, 1])

with col_bot1:
    st.subheader("🔮 24-Hour AI Predictive Load Forecast")
    
    # Generate 24-hour ahead horizon using trained Random Forest
    hours_ahead = np.arange(0, 24)
    shifts = np.where((hours_ahead >= 8) & (hours_ahead <= 18), 1, 0)
    furnace_flag = 1 if equipment_states.get("Electric Arc Furnace / Smelter", False) else 0
    press_flag = 1 if equipment_states.get("Hydraulic Stamping Press (500 Ton)", False) else 0
    pf_val = weighted_pf
    
    X_forecast = np.column_stack([hours_ahead, shifts, np.full(24, furnace_flag), np.full(24, press_flag), np.full(24, pf_val)])
    pred_kw = forecaster.predict(X_forecast)
    
    fig_fc = go.Figure()
    fig_fc.add_hline(y=peak_demand_threshold, line_dash="dot", line_color="#e74c3c", annotation_text="Peak Limit")
    fig_fc.add_trace(go.Scatter(
        x=[f"{h:02d}:00" for h in hours_ahead],
        y=pred_kw,
        mode="lines+markers",
        name="Predicted Load (kW)",
        line=dict(color="#3498db", width=2.5)
    ))
    fig_fc.update_layout(
        height=240,
        margin=dict(l=20, r=20, t=25, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.8)",
        xaxis=dict(title="24-Hour Horizon", gridcolor="#22304a"),
        yaxis=dict(title="Forecasted kW", gridcolor="#22304a"),
        showlegend=False
    )
    st.plotly_chart(fig_fc, use_container_width=True)

with col_bot2:
    st.subheader("💡 Automated Load Shedding & Curtailment Guide")
    
    if is_breach:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid #e74c3c;">
            <b>⚠️ ACTIVE CURTAILMENT PROTOCOL TRIGGERED:</b><br/>
            Current power draw exceeds threshold by <b>{total_active_load - peak_demand_threshold:.1f} kW</b>.<br/><br/>
            <b>Recommended AI Action:</b><br/>
            👉 <b>Stagger Smelter / Arc Furnace Cycle:</b> Delay heat cycle by 25 minutes until off-peak window (after 18:00).<br/>
            👉 <b>Cost Avoidance:</b> Saves <b>${potential_savings:,.0f} / month</b> in utility capacity surge fees.<br/>
            👉 <b>Power Factor:</b> Switch capacitor bank #2 ON to raise PF from {weighted_pf:.2f} to >0.92.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid #2ecc71;">
            <b>✅ NORMAL OPERATING DISPATCH:</b><br/>
            All machinery running comfortably within contracted power envelope.<br/><br/>
            • <b>Headroom Remaining:</b> {peak_demand_threshold - total_active_load:.1f} kW available.<br/>
            • <b>Projected Monthly Bill:</b> ${monthly_projected_cost:,.0f} (Base Rate).<br/>
            • <b>Carbon Abatement:</b> Facility operating within Tier-1 ESG compliance.
        </div>
        """, unsafe_allow_html=True)

# Footer
st.caption("EnerSim AI • Offline Factory Energy Spike & Carbon Forecaster • Powered by Python, Random Forest & Isolation Forest")
