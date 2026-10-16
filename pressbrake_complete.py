"""
PressBrake Springback AI: Sheet Metal V-Die Bending & Elastic Recovery Compensator
================================================================================
A Complete Physics-Based Sheet Metal Forming Simulation Engine
Built for Mechanical Engineering Final Year Project / Technical Defense.

Core Engineering Formulations:
1. Ludwik-Hollomon Hardening Law: sigma = K * epsilon^n
2. Moment of Resistance & Elastic Recovery (Springback Factor Ks):
   Ks = (theta_f / theta_loaded) = 4*(R_i / t * sigma_y / E)^3 - 3*(R_i / t * sigma_y / E) + 1
3. Neutral Axis Shift (k-factor): r_neutral = R_i + k * t
4. CNC Press Brake Punch Tonnage Sizing: F_bend = (C * L * t^2 * sigma_uts) / V_opening
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="PressBrake Springback AI - Sheet Metal Simulator",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 26px; font-weight: 700; color: #1E293B; border-bottom: 2px solid #059669; padding-bottom: 8px; margin-bottom: 15px; }
    .metric-card { background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px; }
</style>
""", unsafe_allow_html=True)

# Sheet Metal Alloy Plasticity Database (Ludwik-Hollomon K & n)
SHEET_MATERIALS = {
    "Deep Drawing Steel (DC04 / AISI 1008)": {
        "E": 210000.0, "sigma_y": 180.0, "sigma_uts": 310.0, "K": 520.0, "n": 0.22, "density": 7.85
    },
    "High-Strength Automotive Steel (DP600)": {
        "E": 210000.0, "sigma_y": 380.0, "sigma_uts": 620.0, "K": 980.0, "n": 0.16, "density": 7.85
    },
    "Aerospace Aluminum (Al 2024-T3)": {
        "E": 73000.0, "sigma_y": 290.0, "sigma_uts": 440.0, "K": 690.0, "n": 0.17, "density": 2.78
    },
    "Austenitic Stainless Steel (SS 304)": {
        "E": 195000.0, "sigma_y": 240.0, "sigma_uts": 580.0, "K": 1250.0, "n": 0.42, "density": 8.00
    }
}

# -------------------------------------------------------------
# PLASTIC FORMING & SPRINGBACK COMPUTATION
# -------------------------------------------------------------
def compute_springback_physics(mat_name, sheet_thick_mm, bend_len_mm, target_angle_deg, punch_radius_mm, die_v_mm):
    mat = SHEET_MATERIALS[mat_name]
    E = mat["E"]
    sy = mat["sigma_y"]
    s_uts = mat["sigma_uts"]
    
    # Radius-to-thickness ratio
    r_over_t = punch_radius_mm / sheet_thick_mm
    
    # Springback factor Ks via Gardner / Sachs analytical formulation
    # Ks = 4*(r/t * sy/E)^3 - 3*(r/t * sy/E) + 1
    term = (r_over_t * sy) / E
    Ks = 4.0 * (term**3) - 3.0 * term + 1.0
    Ks = np.clip(Ks, 0.70, 0.995)
    
    # Angle under load vs final angle
    # theta_final = Ks * theta_loaded
    # Target angle is the final desired bend (e.g. 90 deg)
    loaded_angle_deg = target_angle_deg / Ks
    springback_delta_deg = loaded_angle_deg - target_angle_deg
    
    # Recommended CNC overbend target
    cnc_punch_angle = target_angle_deg - springback_delta_deg
    
    # Neutral axis k-factor
    k_factor = 0.33 if r_over_t < 2.0 else 0.45
    bend_allowance = (np.pi / 180.0) * target_angle_deg * (punch_radius_mm + k_factor * sheet_thick_mm)
    
    # Press brake bending force tonnage: F = (1.33 * L * t^2 * UTS) / V [in Newtons]
    force_n = (1.33 * bend_len_mm * (sheet_thick_mm**2) * s_uts) / max(die_v_mm, 5.0)
    force_tonnes = force_n / 9806.65 # metric tonnes
    
    return Ks, loaded_angle_deg, springback_delta_deg, cnc_punch_angle, bend_allowance, force_tonnes

# -------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📐 **Sheet Metal Stock**")
    mat_choice = st.selectbox("Sheet Metal Material:", list(SHEET_MATERIALS.keys()))
    thickness = st.slider("Sheet Thickness $t$ (mm):", 0.8, 6.0, 2.0, 0.2)
    flange_len = st.slider("Bending Length $L$ (mm):", 100.0, 1000.0, 400.0, 50.0)
    
    st.markdown("---")
    st.markdown("### 🔨 **V-Die & Punch Tooling**")
    target_deg = st.slider("Target Bend Angle ($^\\circ$):", 45.0, 135.0, 90.0, 1.0)
    r_punch = st.slider("Punch Tip Radius $R_p$ (mm):", 0.5, 8.0, 2.0, 0.5)
    v_width = st.slider("V-Die Opening $V$ (mm):", 8.0, 50.0, 16.0, 2.0)

# Run calculation
Ks, loaded_deg, delta_deg, cnc_angle, ba_mm, tonnage = compute_springback_physics(
    mat_choice, thickness, flange_len, target_deg, r_punch, v_width
)

# -------------------------------------------------------------
# MAIN LAYOUT
# -------------------------------------------------------------
st.markdown('<div class="main-header">📐 PressBrake Springback AI: Sheet Metal V-Die Bending & Elastic Recovery Compensator</div>', unsafe_allow_html=True)

tab_vis, tab_stress, tab_def = st.tabs([
    "🕹️ 2D Tooling Simulation & Springback Animation",
    "📈 Plasticity Stress Curves & Tonnage",
    "🎓 Mechanical Defense & Examiner Q&A"
])

# -------------------------------------------------------------
# TAB 1: 2D VISUALIZATION
# -------------------------------------------------------------
with tab_vis:
    st.markdown("#### **Interactive Punch Penetration & Elastic Springback Opening**")
    
    c_p1, c_p2, c_p3, c_p4 = st.columns(4)
    c_p1.metric("Springback Angle (Δθ)", f"+{delta_deg:.2f}°", "Elastic Recovery", delta_color="inverse")
    c_p2.metric("CNC Compensated Overbend", f"{cnc_angle:.2f}°", "Punch Target Angle")
    c_p3.metric("Required Press Tonnage", f"{tonnage:.2f} Tonnes", f"Force: {tonnage*9.81:.1f} kN")
    c_p4.metric("Bend Allowance (BA)", f"{ba_mm:.2f} mm", "Unfolded Flat Pattern")
    
    # 2D Bending Geometry Plot
    # Under load vs Released
    arm_len = 60.0
    
    # Loaded coordinates (sharp angle = loaded_deg)
    half_load = np.radians(loaded_deg / 2.0)
    x_l_left = -arm_len * np.sin(half_load)
    y_l_left = arm_len * np.cos(half_load)
    x_l_right = arm_len * np.sin(half_load)
    y_l_right = arm_len * np.cos(half_load)
    
    # Released coordinates (springback angle = target_deg)
    half_rel = np.radians(target_deg / 2.0)
    x_r_left = -arm_len * np.sin(half_rel)
    y_r_left = arm_len * np.cos(half_rel)
    x_r_right = arm_len * np.sin(half_rel)
    y_r_right = arm_len * np.cos(half_rel)
    
    fig_bend = go.Figure()
    
    # V-Die Profile (Fixed Bottom Steel)
    v_half = v_width / 2.0
    v_depth = v_half / np.tan(np.radians(45.0))
    fig_bend.add_trace(go.Scatter(
        x=[-v_half - 15, -v_half, 0, v_half, v_half + 15],
        y=[0, 0, -v_depth, 0, 0],
        mode='lines',
        fill='tozeroy',
        fillcolor='#CBD5E1',
        line=dict(color='#475569', width=3),
        name='Lower V-Die Tooling'
    ))
    
    # Loaded Sheet (Green Dashed)
    fig_bend.add_trace(go.Scatter(
        x=[x_l_left, 0, x_l_right],
        y=[y_l_left - v_depth*0.7, -v_depth*0.7, y_l_right - v_depth*0.7],
        mode='lines+markers',
        line=dict(color='#059669', width=4, dash='dash'),
        name=f'Sheet Under Punch Load ({loaded_deg:.1f}°)'
    ))
    
    # Released / Springback Sheet (Red Solid)
    fig_bend.add_trace(go.Scatter(
        x=[x_r_left, 0, x_r_right],
        y=[y_r_left - v_depth*0.5, -v_depth*0.5, y_r_right - v_depth*0.5],
        mode='lines+markers',
        line=dict(color='#DC2626', width=5),
        name=f'Final Released Part after Springback ({target_deg:.1f}°)'
    ))
    
    fig_bend.update_layout(
        title="<b>V-Die Bending Profile: Loaded Plastic State vs. Elastic Springback Recovery</b>",
        xaxis=dict(title="Width (mm)", range=[-80, 80]),
        yaxis=dict(title="Height (mm)", range=[-30, 70]),
        height=480,
        template="plotly_white"
    )
    st.plotly_chart(fig_bend, use_container_width=True)

# -------------------------------------------------------------
# TAB 2: STRESS & HARDENING
# -------------------------------------------------------------
with tab_stress:
    st.markdown("#### **Ludwik-Hollomon Flow Stress & Through-Thickness Residual Stress**")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        # Flow curve
        eps = np.linspace(0.002, 0.40, 100)
        flow_stress = SHEET_MATERIALS[mat_choice]["K"] * (eps ** SHEET_MATERIALS[mat_choice]["n"])
        fig_flow = go.Figure()
        fig_flow.add_trace(go.Scatter(x=eps, y=flow_stress, mode='lines', line=dict(color='#2563EB', width=3), name='True Stress'))
        fig_flow.update_layout(title="<b>Ludwik-Hollomon Hardening Curve: sigma = K*eps^n</b>", xaxis_title="True Plastic Strain", yaxis_title="Flow Stress (MPa)", height=320, template="plotly_white")
        st.plotly_chart(fig_flow, use_container_width=True)
        
    with col_s2:
        st.info(f"""
        **Bending Force & Machine Tonnage Equation:**
        $$F_{{bend}} = \\frac{{1.33 \\cdot L \\cdot t^2 \\cdot \\sigma_{{UTS}}}}{{V}}$$
        - Material: **{mat_choice}**
        - Ultimate Tensile Strength: **{SHEET_MATERIALS[mat_choice]['sigma_uts']} MPa**
        - Sheet Thickness: **{thickness} mm** | Bending Length: **{flange_len} mm**
        - Selected V-Die Width: **{v_width} mm**
        - **Calculated Required Tonnage:** **{tonnage:.2f} Metric Tonnes**
        """)

# -------------------------------------------------------------
# TAB 3: DEFENSE
# -------------------------------------------------------------
with tab_def:
    st.markdown("### 🎓 **Final Year Project Defense Handbook**")
    with st.expander("⏱️ **2-Minute Presentation Pitch**", expanded=True):
        st.markdown("""
        > *"Respected Examiners, in sheet metal manufacturing, press brake forming is plagued by **elastic springback**—the tendency of metal to partially unbend when the punch is released.*
        >
        > *If a factory aims for a precise $90.0^\\circ$ flange, elastic unloading will cause the part to spring back to $92^\\circ$ or $94^\\circ$, failing assembly tolerances. Traditionally, operators rely on costly trial-and-error scrap.*
        >
        > *Our project, **PressBrake Springback AI**, applies **non-linear plasticity mechanics and Gardner-Sachs elastic recovery theory**:*
        > 1. *It models strain hardening using the **Ludwik-Hollomon law** ($\\sigma = K \\epsilon^n$).*
        > 2. *It computes the **springback factor ($K_s$)** and exact angular recovery ($\\Delta \\theta$).*
        > 3. *It calculates the precise **compensated overbending angle** for the CNC controller and sizes the required hydraulic tonnage.*
        >
        > *This delivers 'first-part-right' manufacturing, eliminating scrap in automotive and enclosure fabrication."*
        """)
        
    with st.expander("❓ **Anticipated Viva Q&A**", expanded=True):
        st.markdown("""
        **Q1: Why does higher-strength steel exhibit greater springback?**
        - *Answer:* "Springback is directly proportional to the ratio of yield strength to Young's modulus ($\\sigma_y / E$). High-strength steels like DP600 have a very high yield strength ($380\\text{ MPa}$ vs $180\\text{ MPa}$ for mild steel) while maintaining the same modulus of elasticity ($E = 210\\text{ GPa}$). Consequently, a much larger proportion of the total deformation energy is stored elastically rather than plastically."

        **Q2: What is the $k$-factor in bend allowance?**
        - *Answer:* "The $k$-factor describes the shift of the neutral axis toward the inner compression radius during bending. For sharp bends ($R/t < 2$), $k \\approx 0.33$; for larger radii, $k \\to 0.50$. It is essential for generating accurate unfolded flat sheet metal blanks."
        """)

st.caption("Developed for Mechanical Engineering Final Year Project Defense | Metal Forming & Plasticity")
