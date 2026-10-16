# Project 8: HydroForming AI (Metal Extrusion Plastic Flow & Press Tonnage Simulator)
### Comprehensive Technical Defense, Mechanical Formulas & Viva Documentation

---

## 1. Executive Summary & Problem Definition
Direct metal extrusion is a high-tonnage bulk deformation process where metal billets are forced through shaped dies to produce constant cross-section profiles (aerospace channels, heat sinks, seamless tubes). Inaccurate extrusion pressure calculations lead to press stalling or internal **chevron center-burst cracking**.

**HydroForming AI** implements upper-bound plasticity and Johnson's semi-empirical formulation to model metal flow velocities, container friction, and hydraulic tonnage.

---

## 2. Core Mechanical & Physical Formulations

### A. Extrusion Ratio and True Strain
$$R = \frac{A_0}{A_f} = \left(\frac{D_0}{D_f}\right)^2$$
$$\epsilon = \ln(R)$$

### B. Total Extrusion Pressure (Johnson's Model)
The total ram pressure $p_{total}$ is the sum of deformation, redundant shear, and container wall friction:

$$p_{total} = \bar{\sigma} (a + b \ln R) + 2 m \frac{\bar{\sigma}}{\sqrt{3}} \frac{L(t)}{D_0}$$

Where:
- $\bar{\sigma}$: Flow stress of the alloy at processing temperature ($\text{MPa}$).
- $a, b$: Johnson constants characterizing die geometry and redundant shear work.
- $m$: Tresca friction factor ($0 \le m \le 1.0$).
- $L(t)$: Instantaneous unextruded billet length remaining in the container.

### C. Press Ram Force Tonnage
$$F_{ram} = \frac{p_{total} \cdot A_0}{9806.65}\quad [\text{Metric Tonnes}]$$

---

## 3. Anticipated Examiner Q&A

**Q: What is the difference between direct (forward) and indirect (backward) extrusion?**
- *Answer:* "In direct extrusion, the billet moves relative to the container wall, generating significant sliding friction that demands higher initial ram pressure. In indirect extrusion, the die moves into a stationary billet, eliminating container wall friction and reducing the required peak tonnage by 25–40%."
