# Project 2: RoboWeld AI (6-Axis Robotic Welding Seam Tracking & Kinematics)
### Comprehensive Technical Defense, Mechanical Formulas & Viva Documentation

---

## 1. Executive Summary & Problem Definition
In high-volume manufacturing (automotive exhaust, shipbuilding, pressure vessels), robotic MIG/MAG welding must maintain strict torch orientation and continuous travel speed along complex 3D seams. Two major engineering failures occur:
1. **Kinematic Singularities:** When robot joints line up, the inverse Jacobian becomes singular ($\det(J) \to 0$), demanding infinite joint velocity and causing mechanical jerks that ruin the weld seam.
2. **Thermal & Microstructural Degradation:** Unbalanced heat inputs lead to either brittle martensitic phase changes (cold cracking) or excessive heat-affected zone (HAZ) grain coarsening.

**RoboWeld AI** simulates 6-axis articulated manipulator kinematics while predicting arc thermodynamics and weld pool metallurgy in real time.

---

## 2. Core Mechanical & Physical Formulations

### A. 6-DOF Inverse Kinematics & Yoshikawa Manipulability
For a 6-axis manipulator, the relationship between end-effector Cartesian velocity $\dot{x}$ and joint velocities $\dot{\theta}$ is:

$$\dot{x} = J(\theta) \cdot \dot{\theta}$$

To assess posture distance from mechanical singularities, the **Yoshikawa Manipulability Measure** ($w$) is computed:

$$w = \sqrt{\det(J(\theta) \cdot J^T(\theta))}$$

When $w \to 0$, the robot is at a singular boundary.

### B. Choo & Rosenthal Arc Heat Input Equation
Linear heat input per unit length of weld ($H$ in $\text{J/mm}$):

$$H = \eta \cdot \frac{V \cdot I}{v}$$

Where:
- $\eta$: Arc thermal efficiency factor ($0.80$ for GMAW / MIG).
- $V$: Arc voltage (Volts).
- $I$: Welding current (Amps).
- $v$: Torch travel velocity ($\text{mm/s}$).

### C. Peak Temperature and $t_{8/5}$ Cooling Time
The peak temperature $T_p$ at distance $r$ from the fusion line:

$$T_p - T_0 = \frac{2}{\pi \cdot e} \cdot \frac{H}{\rho \cdot c_p \cdot r^2}$$

The critical metallurgical cooling time between $800^\circ\text{C}$ and $500^\circ\text{C}$ ($t_{8/5}$):

$$t_{8/5} = \frac{H^2}{4 \pi k \rho c_p (500^2)}$$

---

## 3. Anticipated Examiner Q&A

**Q: How does the system handle different plate alloys?**
- *Answer:* "The thermal engine loads material-specific thermal conductivity ($k$), volumetric heat capacity ($\rho c_p$), and yield strengths (e.g., ASTM A36 vs SS 316L). This ensures cooling rate predictions accurately reflect microstructural phase shifts for each alloy."
