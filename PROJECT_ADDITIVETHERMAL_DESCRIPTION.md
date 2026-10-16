# Project 3: AdditiveThermal AI (3D Printing Thermal Stress & Warpage Simulator)
### Comprehensive Technical Defense, Mechanical Formulas & Viva Documentation

---

## 1. Executive Summary & Problem Definition
In polymer and metal Additive Manufacturing (FDM, SLS, DED), local thermal gradients create non-uniform volumetric shrinkage. When a newly deposited layer cools from the liquidus/extrusion state down to the ambient bed temperature, its contraction is constrained by previous solidified layers, producing severe residual tensile stresses that cause **corner curling, dimensional warpage, and bed detachment**.

**AdditiveThermal AI** provides an analytical thermo-elastic digital twin that simulates the temperature field, through-thickness residual stresses, and 3D geometric curling deflection.

---

## 2. Core Mechanical & Physical Formulations

### A. 1D Transient Thermal Conduction
Through the build height $z$:

$$\rho C_p \frac{\partial T}{\partial t} = k \frac{\partial^2 T}{\partial z^2} - h_{conv} (T - T_\infty)$$

### B. Thermo-Elastic Residual Stress Model
Thermal contraction strain $\epsilon_{th}$ accumulating below the glass transition temperature $T_g$:

$$\epsilon_{th} = \alpha \cdot \Delta T = \alpha \cdot \max(T_g - T(z), 0)$$

The resulting residual tensile stress $\sigma_{res}$:

$$\sigma_{res}(z) = E(T) \cdot \epsilon_{th}(z)$$

Where:
- $\alpha$: Coefficient of Thermal Expansion (CTE, $\text{K}^{-1}$).
- $E$: Young's Modulus ($\text{MPa}$).
- $T_g$: Glass transition temperature ($^\circ\text{C}$).

### C. Cantilever Corner Curling Deflection
Applying classical bimetallic beam bending theory to the cooling part:

$$\delta_{warp} = \frac{\alpha \cdot \Delta T \cdot L^2}{2 \cdot H}$$

Where $L$ is the footprint length and $H$ is the total part build height.

---

## 3. Anticipated Examiner Q&A

**Q: What is the physical mechanism of bed detachment?**
- *Answer:* "The accumulated tensile stress across the upper layers generates a bending moment that applies high peeling shear stress at the interface between the bottom layer and the build plate. When this shear stress exceeds the adhesive shear strength of the PEI/glass coating ($\sim 1.5 - 2.0\text{ MPa}$), corner detachment occurs."
