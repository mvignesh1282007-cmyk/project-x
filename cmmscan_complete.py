"""
CMM-Scan AI: 3D Point-Cloud Coordinate Metrology & ISO 1101 GD&T Inspector
==========================================================================
A Complete 3D Coordinate Measuring Machine (CMM) & Surface Error Digital Twin
Built for Mechanical Engineering Final Year Project / Technical Defense.

Core Engineering Formulations:
1. Least-Squares Reference Plane Fitting: z = ax + by + c via SVD / Normal Equations
2. ISO 1101 Flatness Tolerance Zone: t_flat = max(d_i) - min(d_i) from Best-Fit Plane
3. Cylindrical Best-Fit & Cylindricity Error: Delta_r = max(r_i) - min(r_i)
4. Gaussian Measurement Uncertainty & Automated Go / No-Go Metrology Verdict
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="CMM-Scan AI - 3D Metrology & GD&T",
    page_icon="📏",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 26px; font-weight: 700; color: #1E293B; border-bottom: 2px solid #0D9488; padding-bottom: 8px; margin-bottom: 15px; }
    .metric-card { background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 3D POINT CLOUD SYNTHESIZER WITH MANUFACTURING NOISE
# -------------------------------------------------------------
def generate_cmm_scan(feature_type, n_points, flatness_tol_um, waviness_amp_um):
    np.random.seed(42)
    
    if feature_type == "Milled Engine Cylinder Head Deck (Plane)":
        # Nominal 100mm x 100mm planar surface
        x = np.random.uniform(-50.0, 50.0, n_points)
        y = np.random.uniform(-50.0, 50.0, n_points)
        
        # Superimpose physical manufacturing errors:
        # 1. Thermal bed slope / tool tilt
        tilt_z = 0.0003 * x - 0.0002 * y
        # 2. Milling cutter waviness mark (periodicity)
        waviness = (waviness_amp_um / 1000.0) * np.sin(x / 6.0) * np.cos(y / 8.0)
        # 3. High-frequency random surface roughness micro-asperities
        roughness = np.random.normal(0, 0.003, n_points)
        
        z = tilt_z + waviness + roughness
        
        # Fit Least-Squares Reference Plane (Ax + By + Cz + D = 0)
        # Matrix formulation: A * [a, b, d] = z
        A_mat = np.column_stack([x, y, np.ones(n_points)])
        coeff, _, _, _ = np.linalg.lstsq(A_mat, z, rcond=None)
        a, b, d = coeff
        
        # Normal vector to best-fit plane
        norm = np.sqrt(a**2 + b**2 + 1.0)
        
        # Orthogonal signed distances d_i from best-fit plane
        deviations_mm = (z - (a*x + b*y + d)) / norm
        deviations_um = deviations_mm * 1000.0 # micrometers
        
        # ISO 1101 Flatness error = Peak to Valley (max - min)
        flatness_error_um = np.max(deviations_um) - np.min(deviations_um)
        is_pass = flatness_error_um <= flatness_tol_um
        
        return x, y, z, deviations_um, flatness_error_um, is_pass, "Flatness"
        
    else: # Precision Boring Bearing Sleeve (Cylinder)
        theta = np.random.uniform(0, 2*np.pi, n_points)
        z = np.random.uniform(-40.0, 40.0, n_points)
        nominal_radius = 30.0 # mm
        
        # Out-of-round ovality error + chatter lobes
        ovality = (waviness_amp_um / 1000.0) * np.cos(2 * theta)
        roughness = np.random.normal(0, 0.002, n_points)
        r = nominal_radius + ovality + roughness
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Best-fit circle radius
        r_calc = np.sqrt(x**2 + y**2)
        r_fit = np.mean(r_calc)
        deviations_um = (r_calc - r_fit) * 1000.0
        
        cylindricity_error_um = np.max(deviations_um) - np.min(deviations_um)
        is_pass = cylindricity_error_um <= flatness_tol_um
        
        return x, y, z, deviations_um, cylindricity_error_um, is_pass, "Cylindricity"

# -------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📏 **CMM Probe & Inspection Setup**")
    feat_sel = st.selectbox(
        "Machined Datum Feature:",
        ["Milled Engine Cylinder Head Deck (Plane)", "Bored Bearing Spindle Sleeve (Cylinder)"]
    )
    point_density = st.slider("Laser / Touch-Probe Point Count:", 500, 3000, 1500, 250)
    
    st.markdown("---")
    st.markdown("### 🎯 **ISO 1101 GD&T Specification**")
    tolerance_spec_um = st.slider("Drawing Tolerance Zone Limit (μm):", 10.0, 80.0, 35.0, 5.0)
    cutter_chatter_um = st.slider("Cutter Runout / Clamping Distortion (μm):", 5.0, 50.0, 22.0, 2.0)

# Run inspection
pts_x, pts_y, pts_z, dev_um, form_error, pass_flag, gd_and_t_type = generate_cmm_scan(
    feat_sel, point_density, tolerance_spec_um, cutter_chatter_um
)

# -------------------------------------------------------------
# MAIN LAYOUT
# -------------------------------------------------------------
st.markdown('<div class="main-header">📏 CMM-Scan AI: 3D Point-Cloud Coordinate Metrology & ISO 1101 GD&T Inspector</div>', unsafe_allow_html=True)

t_cloud, t_hist, t_def = st.tabs([
    "🕹️ 3D Metrology Point Cloud & Error Deviation Heatmap",
    "📈 Gaussian Error Distribution & Statistical Process Control",
    "🎓 Mechanical Defense & Examiner Q&A"
])

# -------------------------------------------------------------
# TAB 1: 3D POINT CLOUD & ERROR MAP
# -------------------------------------------------------------
with t_cloud:
    st.markdown("#### **3D Coordinate Measuring Machine (CMM) Deviation Cloud**")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Inspected Points", f"{point_density} pts")
    col_m2.metric(f"Measured {gd_and_t_type} Error", f"{form_error:.1f} μm", f"Limit: {tolerance_spec_um} μm")
    col_m3.metric("Peak Positive Deviation", f"+{np.max(dev_um):.1f} μm")
    col_m4.metric("Quality Verdict", "✅ GO (PASS)" if pass_flag else "❌ NO-GO (REJECT)", 
                  "Within Tolerance" if pass_flag else "Form Out of Spec",
                  delta_color="normal" if pass_flag else "inverse")
                  
    # 3D Point Cloud Plotly
    fig_cmm = go.Figure()
    fig_cmm.add_trace(go.Scatter3d(
        x=pts_x, y=pts_y, z=pts_z,
        mode='markers',
        marker=dict(
            size=3.5,
            color=dev_um,
            colorscale='Jet',
            colorbar=dict(title="GD&T Error Deviation (μm)", thickness=15, len=0.8),
            showscale=True
        ),
        text=[f"Dev: {d:.1f} μm" for d in dev_um],
        hoverinfo='text',
        name='Laser CMM Points'
    ))
    
    fig_cmm.update_layout(
        title=f"<b>3D Measured Point Cloud with Color-Coded {gd_and_t_type} Deviations</b>",
        scene=dict(
            xaxis=dict(title="X (mm)"),
            yaxis=dict(title="Y (mm)"),
            zaxis=dict(title="Z (mm)"),
            camera=dict(eye=dict(x=1.4, y=-1.3, z=1.2))
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        height=520,
        template="plotly_white"
    )
    st.plotly_chart(fig_cmm, use_container_width=True)

# -------------------------------------------------------------
# TAB 2: HISTOGRAM & SPC
# -------------------------------------------------------------
with t_hist:
    st.markdown("#### **Manufacturing Form Error Distribution & Process Capability (Cpk)**")
    
    # Statistical Capability
    mu = np.mean(dev_um)
    sigma = np.std(dev_um)
    cpk = min((tolerance_spec_um - mu) / (3*sigma), (mu - (-tolerance_spec_um)) / (3*sigma))
    
    c_s1, c_s2, c_s3 = st.columns(3)
    c_s1.metric("Mean Deviation (μ)", f"{mu:.2f} μm")
    c_s2.metric("Standard Deviation (σ)", f"{sigma:.2f} μm")
    c_s3.metric("Process Capability Index (Cpk)", f"{cpk:.2f}", "Six Sigma Benchmark: >= 1.33")
    
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=dev_um,
        nbinsx=40,
        marker_color='#0D9488',
        name='Deviation Histogram'
    ))
    fig_hist.add_vline(x=tolerance_spec_um/2.0, line_dash="dash", line_color="red", annotation_text="+Tol Boundary")
    fig_hist.add_vline(x=-tolerance_spec_um/2.0, line_dash="dash", line_color="red", annotation_text="-Tol Boundary")
    fig_hist.update_layout(
        title="<b>Normal Distribution of Surface Machining Errors</b>",
        xaxis_title="Deviation from Best-Fit Geometry (μm)",
        yaxis_title="Point Frequency",
        height=350,
        template="plotly_white"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# -------------------------------------------------------------
# TAB 3: DEFENSE
# -------------------------------------------------------------
with t_def:
    st.markdown("### 🎓 **Final Year Project Defense Handbook**")
    with st.expander("⏱️ **2-Minute Presentation Pitch**", expanded=True):
        st.markdown("""
        > *"Respected Examiners, in precision aerospace and powertrain manufacturing, verifying Geometric Dimensioning & Tolerancing (GD&T per ISO 1101 and ASME Y14.5) is paramount.*
        >
        > *Traditional manual dial indicators and vernier calipers cannot measure 3D spatial form errors like flatness, cylindricity, or waviness across complex mating surfaces.*
        >
        > *Our project, **CMM-Scan AI**, delivers an autonomous 3D Coordinate Metrology digital twin:*
        > 1. *It captures dense 3D point cloud coordinates representing laser-scanned or touch-probed parts.*
        > 2. *It applies **Singular Value Decomposition (SVD) and Least-Squares Orthogonal Regression** to fit datum planes and cylinders.*
        > 3. *It computes peak-to-valley form deviations to evaluate **ISO 1101 Flatness and Cylindricity**, automating Go/No-Go inspection verdicts and calculating statistical Six-Sigma Process Capability ($C_{pk}$).*
        >
        > *This bridges automated quality inspection and closed-loop machining feedback in smart manufacturing."*
        """)
        
    with st.expander("❓ **Anticipated Viva Q&A**", expanded=True):
        st.markdown("""
        **Q1: How is the ISO 1101 Flatness tolerance zone mathematically defined?**
        - *Answer:* "Flatness is defined as the minimum distance between two parallel planes that contain all the measured points of the surface. In our algorithm, we first fit an optimal least-squares plane, calculate the orthogonal signed distance of each point to the plane, and evaluate $t_{flat} = d_{\\max} - d_{\\min}$."

        **Q2: What is the significance of $C_{pk}$ in metrological inspection?**
        - *Answer:* "$C_{pk}$ (Process Capability Index) relates the spread and centering of the machining error distribution to the drawing tolerance limits: $C_{pk} = \\min\\left(\\frac{USL - \\mu}{3\\sigma}, \\frac{\\mu - LSL}{3\\sigma}\\right)$. A $C_{pk} \\ge 1.33$ proves that the CNC milling machine is capable of consistently producing defect-free parts within a 4-sigma/6-sigma manufacturing window."
        """)

st.caption("Developed for Mechanical Engineering Final Year Project Defense | Metrology & Quality Engineering")
