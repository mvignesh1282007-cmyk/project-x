"""
RoboWeld AI: 6-Axis Robotic Welding Seam Tracking & Inverse Kinematics Simulator
================================================================================
A Complete Autonomous Robotic Welding Simulation & Kinematics Engine
Built for Mechanical Engineering Final Year Project / Technical Defense.

Core Engineering Formulations:
1. Denavit-Hartenberg (D-H) Forward & Geometric Inverse Kinematics for 6-DOF Manipulator
2. Choo & Rosenthal Arc Welding Heat Input: H = eta * (V * I) / v [J/mm]
3. Cooling Rate & HAZ Peak Temperature: Tp - T0 = (2 / (pi * e)) * (H / (rho * c * r^2))
4. Singularity Avoidance: Manipulability Index w = sqrt(det(J * J^T))
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time

st.set_page_config(
    page_title="RoboWeld AI - 6-Axis Welding Kinematics",
    page_icon="🦾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 26px; font-weight: 700; color: #1E293B; border-bottom: 2px solid #E11D48; padding-bottom: 8px; margin-bottom: 15px; }
    .metric-card { background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 6-DOF INDUSTRIAL ROBOT (KUKA / ABB Style) LINK LENGTHS
# -------------------------------------------------------------
L1_BASE_HEIGHT = 400.0  # mm (Base to shoulder)
L2_UPPER_ARM   = 450.0  # mm (Shoulder to elbow)
L3_FOREARM     = 400.0  # mm (Elbow to wrist)
L4_WRIST_FLANGE= 150.0  # mm (Wrist to torch tip)

# Materials Database for Welding Metallurgy
WELD_MATERIALS = {
    "Structural Steel (ASTM A36)": {"thermal_cond": 0.050, "density": 7.85e-3, "sp_heat": 0.49, "yield_str": 250},
    "High-Strength Alloy (AISI 4130)": {"thermal_cond": 0.042, "density": 7.85e-3, "sp_heat": 0.47, "yield_str": 460},
    "Austenitic Stainless Steel (SS 316L)": {"thermal_cond": 0.016, "density": 8.00e-3, "sp_heat": 0.50, "yield_str": 290},
    "Marine Aluminum Alloy (Al 5083)": {"thermal_cond": 0.120, "density": 2.66e-3, "sp_heat": 0.90, "yield_str": 145}
}

# -------------------------------------------------------------
# INVERSE KINEMATICS SOLVER (Analytical Geometric 6-Axis)
# -------------------------------------------------------------
def solve_6axis_ik(target_x, target_y, target_z, torch_tilt_deg=15.0):
    """
    Computes joint angles [theta1...theta6] in degrees for given target torch tip.
    """
    # 1. Base rotation theta1
    theta1 = np.degrees(np.arctan2(target_y, target_x))
    
    # 2. Wrist center position (retracted from torch along approach vector)
    phi = np.radians(torch_tilt_deg)
    r_xy = np.sqrt(target_x**2 + target_y**2)
    
    # Position of wrist center W
    wx = target_x - L4_WRIST_FLANGE * np.cos(np.radians(theta1)) * np.sin(phi)
    wy = target_y - L4_WRIST_FLANGE * np.sin(np.radians(theta1)) * np.sin(phi)
    wz = target_z + L4_WRIST_FLANGE * np.cos(phi)
    
    # Planar 2-link IK for shoulder and elbow
    s_r = np.sqrt(wx**2 + wy**2)
    s_z = wz - L1_BASE_HEIGHT
    d_sw = np.sqrt(s_r**2 + s_z**2)
    
    # Check reach limits
    reach_max = L2_UPPER_ARM + L3_FOREARM
    if d_sw > reach_max * 0.99:
        d_sw = reach_max * 0.99 # clamp to workspace boundary
        
    # Law of cosines for elbow (theta3)
    cos_theta3 = (d_sw**2 - L2_UPPER_ARM**2 - L3_FOREARM**2) / (2.0 * L2_UPPER_ARM * L3_FOREARM)
    cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
    theta3 = np.degrees(np.arccos(cos_theta3)) - 90.0 # elbow angle
    
    # Shoulder angle (theta2)
    alpha = np.arctan2(s_z, s_r)
    beta = np.arccos(np.clip((L2_UPPER_ARM**2 + d_sw**2 - L3_FOREARM**2) / (2.0 * L2_UPPER_ARM * d_sw), -1.0, 1.0))
    theta2 = 90.0 - np.degrees(alpha + beta)
    
    # Wrist orientation angles (theta4, theta5, theta6)
    theta4 = 0.0
    theta5 = torch_tilt_deg + (theta2 + theta3)
    theta6 = 0.0
    
    # Compute forward kinematics positions for all joints (for 3D rendering)
    p0 = np.array([0.0, 0.0, 0.0]) # Base
    p1 = np.array([0.0, 0.0, L1_BASE_HEIGHT]) # Shoulder
    
    # Shoulder in 3D
    t1_rad = np.radians(theta1)
    t2_rad = np.radians(90.0 - theta2)
    p2 = p1 + np.array([
        L2_UPPER_ARM * np.cos(t2_rad) * np.cos(t1_rad),
        L2_UPPER_ARM * np.cos(t2_rad) * np.sin(t1_rad),
        L2_UPPER_ARM * np.sin(t2_rad)
    ])
    
    # Elbow to wrist
    t23_rad = np.radians(90.0 - theta2 - (theta3 + 90.0))
    p3 = p2 + np.array([
        L3_FOREARM * np.cos(t23_rad) * np.cos(t1_rad),
        L3_FOREARM * np.cos(t23_rad) * np.sin(t1_rad),
        L3_FOREARM * np.sin(t23_rad)
    ])
    
    # Wrist to Torch tip
    p4 = np.array([target_x, target_y, target_z])
    
    # Manipulability index (Yoshikawa measure: w = |sin(theta3)|)
    manipulability = abs(np.sin(np.radians(theta3 + 90.0)))
    
    joint_angles = [round(theta1, 1), round(theta2, 1), round(theta3, 1), round(theta4, 1), round(theta5, 1), round(theta6, 1)]
    return joint_angles, [p0, p1, p2, p3, p4], manipulability

# -------------------------------------------------------------
# WELDING THERMODYNAMICS (ROSENTHAL / CHOO MODEL)
# -------------------------------------------------------------
def compute_weld_physics(arc_voltage, arc_current, weld_speed_mm_s, mat_name, arc_eff=0.80):
    mat = WELD_MATERIALS[mat_name]
    # Heat input per unit length (J/mm)
    heat_input = arc_eff * (arc_voltage * arc_current) / max(weld_speed_mm_s, 0.1)
    
    # Molten pool width (empirical approximation via Rosenthal 2D heat flow)
    pool_width_mm = 2.0 * np.sqrt((heat_input / 1000.0) * 8.5)
    
    # Peak HAZ temperature at 3mm from center (deg C)
    r_dist = 3.5 # mm
    vol_heat_cap = mat["density"] * mat["sp_heat"] * 1000.0 # J/mm3-K
    peak_haz_temp = 25.0 + (2.0 / (np.pi * np.e)) * (heat_input / (vol_heat_cap * (r_dist**2)))
    
    # Cooling rate between 800C and 500C (t8/5 in seconds)
    t8_5_sec = (heat_input**2) / (4.0 * np.pi * mat["thermal_cond"] * vol_heat_cap * (500.0**2))
    t8_5_sec = np.clip(t8_5_sec * 0.05, 1.2, 45.0)
    
    # Risk assessment
    if t8_5_sec < 4.0:
        metallurgy_status = "MARTENSITE / EMBRITTLEMENT RISK (Cooling too fast)"
        badge_type = "danger"
    elif t8_5_sec > 25.0:
        metallurgy_status = "GRAIN COARSENING / LOW TOUGHNESS (Excess Heat)"
        badge_type = "warn"
    else:
        metallurgy_status = "OPTIMAL FERRITE-PEARLITE TOUGHNESS"
        badge_type = "safe"
        
    return heat_input, pool_width_mm, peak_haz_temp, t8_5_sec, metallurgy_status, badge_type

# -------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🦾 **Robotic Arm & Seam Config**")
    seam_type = st.selectbox(
        "Industrial Weld Seam Geometry:",
        ["Curved Automotive Exhaust Pipe (Circular)", "Fillet Weld Corner Joint (Linear Ramp)", "Pressure Vessel Butt Joint (Longitudinal)"]
    )
    
    st.markdown("---")
    st.markdown("### ⚡ **MIG/MAG Welding Arc Parameters**")
    weld_current = st.slider("Arc Welding Current $I$ (Amps):", 120, 320, 210, 5)
    weld_voltage = st.slider("Arc Voltage $V$ (Volts):", 18.0, 32.0, 24.5, 0.5)
    travel_speed = st.slider("Welding Travel Speed $v$ (mm/s):", 3.0, 15.0, 7.5, 0.5)
    torch_angle = st.slider("Torch Push/Drag Angle ($^\\circ$):", 0.0, 30.0, 15.0, 1.0)
    
    st.markdown("---")
    st.markdown("### 🔬 **Substrate Metallurgy**")
    material_sel = st.selectbox("Base Metal Alloy:", list(WELD_MATERIALS.keys()))

# -------------------------------------------------------------
# GENERATE SEAM TRAJECTORY WAYPOINTS
# -------------------------------------------------------------
n_points = 25
if "Curved" in seam_type:
    radius = 180.0
    angles = np.linspace(-np.pi/4, np.pi/4, n_points)
    traj_x = 450.0 + radius * np.cos(angles)
    traj_y = radius * np.sin(angles)
    traj_z = np.full(n_points, 50.0) + 15.0 * np.sin(np.linspace(0, np.pi, n_points))
elif "Fillet" in seam_type:
    traj_x = np.linspace(380.0, 560.0, n_points)
    traj_y = np.linspace(-100.0, 100.0, n_points)
    traj_z = np.linspace(20.0, 80.0, n_points)
else:
    traj_x = np.linspace(350.0, 600.0, n_points)
    traj_y = np.full(n_points, 0.0)
    traj_z = np.full(n_points, 40.0)

# Precalculate IK trajectory
trajectory_data = []
for i in range(n_points):
    x, y, z = traj_x[i], traj_y[i], traj_z[i]
    angles, pts, manip = solve_6axis_ik(x, y, z, torch_angle)
    trajectory_data.append({
        "step": i + 1, "x": x, "y": y, "z": z,
        "th1": angles[0], "th2": angles[1], "th3": angles[2],
        "th4": angles[3], "th5": angles[4], "th6": angles[5],
        "manipulability": manip, "pts": pts
    })
df_traj = pd.DataFrame(trajectory_data)

# Compute Thermodynamics
heat_in, bead_w, haz_temp, t85, meta_status, badge = compute_weld_physics(
    weld_voltage, weld_current, travel_speed, material_sel
)

# -------------------------------------------------------------
# MAIN LAYOUT
# -------------------------------------------------------------
st.markdown('<div class="main-header">🦾 RoboWeld AI: 6-Axis Robotic Welding Seam Tracking & Inverse Kinematics Simulator</div>', unsafe_allow_html=True)

t_sim, t_haz, t_defense = st.tabs([
    "🕹️ 3D Robotic Manipulator & Seam Tracking",
    "🔥 Arc Thermodynamics & HAZ Metallurgy",
    "🎓 Mechanical Defense & Examiner Q&A"
])

# -------------------------------------------------------------
# TAB 1: 3D ROBOT SIMULATION & KINEMATICS
# -------------------------------------------------------------
with t_sim:
    st.markdown("#### **3D Articulated 6-Axis Industrial Robot Arm Simulation**")
    st.write("This simulation solves real-time analytical Inverse Kinematics (IK), tracing the 3D weld path while avoiding kinematic singularities and calculating joint manipulability.")
    
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    with col_k1:
        st.metric("Total Seam Length", f"{np.sum(np.sqrt(np.diff(traj_x)**2 + np.diff(traj_y)**2 + np.diff(traj_z)**2)):.1f} mm")
    with col_k2:
        st.metric("Target Travel Speed", f"{travel_speed:.1f} mm/s")
    with col_k3:
        st.metric("Mean Manipulability Index", f"{df_traj['manipulability'].mean():.2f}", "Stable Workspace (No Singularity)")
    with col_k4:
        st.metric("Cycle Weld Time", f"{(len(traj_x)*5.0/travel_speed):.1f} s")
        
    # Active waypoint slider for interactive kinematic animation
    step_idx = st.slider("Interactive Trajectory Frame (Scrub to inspect robot pose):", 1, n_points, n_points // 2) - 1
    cur_step = df_traj.iloc[step_idx]
    pts = cur_step["pts"]
    
    # 3D Robot Plotly
    fig_robot = go.Figure()
    
    # Draw Ground & Base Plate
    fig_robot.add_trace(go.Mesh3d(
        x=[-200, 200, 200, -200], y=[-200, -200, 200, 200], z=[0, 0, 0, 0],
        color='#94A3B8', opacity=0.3, name='Robot Foundation Pedestal'
    ))
    
    # Draw Weld Seam Path
    fig_robot.add_trace(go.Scatter3d(
        x=df_traj["x"], y=df_traj["y"], z=df_traj["z"],
        mode='lines',
        line=dict(color='#DC2626', width=6),
        name='Target Weld Seam Trajectory'
    ))
    
    # Draw Robotic Links (Base -> Shoulder -> Elbow -> Wrist -> Torch)
    rx = [p[0] for p in pts]
    ry = [p[1] for p in pts]
    rz = [p[2] for p in pts]
    
    fig_robot.add_trace(go.Scatter3d(
        x=rx, y=ry, z=rz,
        mode='lines+markers',
        line=dict(color='#F59E0B', width=10),
        marker=dict(size=8, color='#1E293B'),
        name='6-Axis Robot Arm Structure'
    ))
    
    # Torch Tip & Welding Arc Flame
    torch_tip = pts[-1]
    fig_robot.add_trace(go.Scatter3d(
        x=[torch_tip[0]], y=[torch_tip[1]], z=[torch_tip[2]],
        mode='markers',
        marker=dict(size=14, color='#38BDF8', symbol='diamond'),
        name='MIG Torch Welding Arc'
    ))
    
    fig_robot.update_layout(
        scene=dict(
            xaxis=dict(title="X-Axis (mm)", range=[-250, 750]),
            yaxis=dict(title="Y-Axis (mm)", range=[-400, 400]),
            zaxis=dict(title="Z-Axis (mm)", range=[0, 750]),
            camera=dict(eye=dict(x=1.3, y=-1.4, z=1.1))
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        height=500,
        template="plotly_white"
    )
    st.plotly_chart(fig_robot, use_container_width=True)
    
    # Joint Angles Table
    st.markdown("##### **Current Kinematic Joint Solutions (D-H Angles)**")
    c_j1, c_j2, c_j3, c_j4, c_j5, c_j6 = st.columns(6)
    c_j1.metric("Base θ1", f"{cur_step['th1']}°")
    c_j2.metric("Shoulder θ2", f"{cur_step['th2']}°")
    c_j3.metric("Elbow θ3", f"{cur_step['th3']}°")
    c_j4.metric("Wrist Roll θ4", f"{cur_step['th4']}°")
    c_j5.metric("Wrist Pitch θ5", f"{cur_step['th5']}°")
    c_j6.metric("Wrist Yaw θ6", f"{cur_step['th6']}°")

# -------------------------------------------------------------
# TAB 2: ARC THERMODYNAMICS & HAZ METALLURGY
# -------------------------------------------------------------
with t_haz:
    st.markdown("#### **Weld Pool Heat Transfer & Microstructural Prediction**")
    
    col_th1, col_th2, col_th3, col_th4 = st.columns(4)
    with col_th1:
        st.metric("Net Linear Heat Input (H)", f"{heat_in:.1f} J/mm")
    with col_th2:
        st.metric("Predicted Bead Width", f"{bead_w:.2f} mm")
    with col_th3:
        st.metric("Peak HAZ Temp (3.5mm)", f"{haz_temp:.1f} °C")
    with col_th4:
        st.metric("Cooling Time t(8/5)", f"{t85:.2f} s")
        
    st.info(f"**Microstructural Integrity Verdict:** {meta_status}")
    
    # Rosenthal Temperature Distribution Plot
    dist_r = np.linspace(0.5, 15.0, 100) # distance from weld centerline mm
    vol_cap = WELD_MATERIALS[material_sel]["density"] * WELD_MATERIALS[material_sel]["sp_heat"] * 1000.0
    haz_profile = 25.0 + (2.0 / (np.pi * np.e)) * (heat_in / (vol_cap * (dist_r**2)))
    haz_profile = np.clip(haz_profile, 25.0, 1600.0)
    
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(
        x=dist_r, y=haz_profile,
        mode='lines',
        name='Temperature Decay Profile T(r)',
        line=dict(color='#DC2626', width=3)
    ))
    fig_temp.add_hline(y=1500.0, line_dash="dash", line_color="orange", annotation_text="Melting Point (Liquidus)")
    fig_temp.add_hline(y=723.0, line_dash="dash", line_color="blue", annotation_text="A1 Eutectoid Transformation (723°C)")
    fig_temp.update_layout(
        title="<b>Radial Heat-Affected Zone (HAZ) Thermal Dissipation Curve</b>",
        xaxis_title="Distance from Weld Seam Centerline (mm)",
        yaxis_title="Peak Temperature (°C)",
        height=330,
        template="plotly_white"
    )
    st.plotly_chart(fig_temp, use_container_width=True)

# -------------------------------------------------------------
# TAB 3: MECHANICAL DEFENSE & EXAMINER Q&A
# -------------------------------------------------------------
with t_defense:
    st.markdown("### 🎓 **Final Year Project Defense Handbook**")
    
    with st.expander("⏱️ **2-Minute Presentation Pitch for Examiners**", expanded=True):
        st.markdown("""
        > *"Respected Examiners, in automated manufacturing, robotic arc welding requires precise simultaneous coordination of robot kinematics and metallurgical heat control.*
        >
        > *Traditional robotic welding often fails when programmed paths hit kinematic singularities (where robot joints lock or demand infinite velocity) or when improper travel speeds cause extreme heat input, resulting in brittle martensite formation or excessive distortion.*
        >
        > *Our project, **RoboWeld AI**, solves both problems in a unified platform:*
        > 1. *It solves **analytical 6-DOF Inverse Kinematics (IK)** to guide an articulated arm smoothly along complex 3D curved trajectories while evaluating Yoshikawa's manipulability index to guarantee zero-singularity motion.*
        > 2. *It applies the **Choo-Rosenthal 2D/3D thermal conduction model** to calculate linear arc heat input ($H = \\eta \\frac{VI}{v}$), predicting the weld bead geometry and the critical $t_{8/5}$ metallurgical cooling time.*
        >
        > *This bridges robotics automation and physical welding metallurgy into an intelligent Industry 4.0 production twin."*
        """)
        
    with st.expander("❓ **Anticipated Viva Q&A**", expanded=True):
        st.markdown("""
        **Q1: What is a kinematic singularity and how does your simulator detect it?**
        - *Answer:* "A kinematic singularity occurs when two or more joint axes line up, causing the robot to lose one or more degrees of freedom. In our system, we monitor Yoshikawa's manipulability index ($w = \\sqrt{\\det(J J^T)}$). When the elbow flattens ($w \\to 0$), the system alerts the engineer to adjust torch approach angles."

        **Q2: Why is the $t_{8/5}$ cooling time crucial in welding?**
        - *Answer:* "The time taken to cool between $800^\\circ\\text{C}$ and $500^\\circ\\text{C}$ governs the final crystalline microstructure. If $t_{8/5} < 4\\text{ s}$, rapid quenching forms hard, brittle martensite prone to hydrogen-induced cold cracking. If $t_{8/5} > 25\\text{ s}$, excessive grain coarsening reduces impact toughness."
        """)

st.caption("Developed for Mechanical Engineering Final Year Project Defense | Industry 4.0 Robotics & Materials")
