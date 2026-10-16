# Project 4: PressBrake Springback AI (Sheet Metal V-Die Bending Compensator)
### Comprehensive Technical Defense, Mechanical Formulas & Viva Documentation

---

## 1. Executive Summary & Problem Definition
In precision sheet metal fabrication (HVAC enclosures, aircraft skins, automotive chassis), press brake air bending involves both plastic and elastic strain across the sheet thickness. Upon punch withdrawal, the elastic portion unloads, creating **angular springback ($\Delta \theta$)** that violates ISO 2768 tolerances.

**PressBrake Springback AI** is an analytical plasticity digital twin that calculates exact springback angles, punch penetration depths, and required machine tonnage.

---

## 2. Core Mechanical & Physical Formulations

### A. Ludwik-Hollomon Hardening Law
During plastic bending, true stress $\sigma$ scales with plastic strain $\epsilon$:

$$\sigma = K \cdot \epsilon^n$$

Where $K$ is the strength coefficient ($\text{MPa}$) and $n$ is the strain-hardening exponent.

### B. Elastic Recovery Springback Factor ($K_s$)
Using Gardner and Sachs' analytical moment-curvature formulation:

$$K_s = \frac{\theta_f}{\theta_{loaded}} = 4 \left( \frac{R_i}{t} \frac{\sigma_y}{E} \right)^3 - 3 \left( \frac{R_i}{t} \frac{\sigma_y}{E} \right) + 1$$

Where:
- $R_i$: Inside bend punch radius.
- $t$: Sheet metal thickness.
- $\sigma_y$: Material yield strength.
- $E$: Modulus of elasticity.

### C. Press Brake Machine Tonnage Equation
$$F_{bend} = \frac{C \cdot L \cdot t^2 \cdot \sigma_{UTS}}{V}$$

Where $C \approx 1.33$ for V-die air bending, $L$ is bend length, and $V$ is the die shoulder opening width.

---

## 3. Anticipated Examiner Q&A

**Q: How does the CNC machine use this springback prediction?**
- *Answer:* "The CNC press brake controller is programmed with the overbent angle $\theta_{loaded} = \theta_{target} / K_s$. The punch descends deeper into the V-die cavity to reach this overbend, so that upon elastic relaxation, the part settles precisely at the target $\theta_{target}$."
