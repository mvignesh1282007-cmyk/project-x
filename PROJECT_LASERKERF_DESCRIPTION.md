# Project 10: LaserKerf AI (Thermal Beam Kerf Width & Heat-Affected Zone Simulator)
### Comprehensive Technical Defense, Mechanical Formulas & Viva Documentation

---

## 1. Executive Summary & Problem Definition
In high-precision industrial sheet metal fabrication (aerospace titanium brackets, electronic enclosures, automotive chassis), CNC Fiber and $\text{CO}_2$ lasers provide narrow-tolerance non-contact cutting. However, excess linear energy density ($E = P / (v \cdot t)$) creates broad **Heat-Affected Zones (HAZ)** that alter substrate grain structures, reduce fatigue strength, and leave molten slag (dross) adherent to the bottom cut edge.

**LaserKerf AI** is an analytical thermodynamic and hydrodynamic simulation engine based on Rosenthal's classical moving point source equation, predicting kerf geometries and HAZ boundaries.

---

## 2. Core Mechanical & Physical Formulations

### A. Rosenthal 2D Moving Heat Source Formulation
For a continuous-wave laser of power $P$, optical absorptivity $\eta$, travel speed $v$, and sheet thickness $t$:

$$T(x, y) - T_0 = \frac{\eta P}{2 \pi k t} \exp\left( -\frac{v x}{2 \alpha} \right) K_0\left( \frac{v \sqrt{x^2 + y^2}}{2 \alpha} \right)$$

Where:
- $k$: Thermal conductivity ($\text{W/mm}\cdot\text{K}$).
- $\alpha = \frac{k}{\rho C_p}$: Thermal diffusivity ($\text{mm}^2/\text{s}$).
- $K_0$: Zero-order modified Bessel function of the second kind.
- $(x, y)$: Eulerian coordinates relative to moving beam center.

### B. Energy Balance Kerf Width Model
The minimum kerf width $w_{kerf}$ is obtained by balancing absorbed laser energy against the volumetric enthalpy of melting:

$$w_{kerf} = \frac{\eta P}{v \cdot t \cdot \left[ \rho C_p (T_{melt} - T_0) + \rho \Delta H_f \right]}$$

Where $\Delta H_f$ is the latent heat of fusion.

### C. Assist-Gas Hydrodynamic Shear Drag (Dross Ejection)
The molten film is stripped downwards by the coaxial gas jet:

$$\tau_{gas} = \frac{1}{2} C_f \rho_{gas} v_{gas}^2$$

If assist gas pressure is insufficient, surface tension retains the molten slag at the lower exit edge, forming dross burrs.

---

## 3. Anticipated Examiner Q&A

**Q: What is thermal diffusivity ($\alpha$) and why is it so critical in laser cutting?**
- *Answer:* "Thermal diffusivity $\alpha = k / (\rho C_p)$ measures the rate at which heat propagates through a solid relative to its heat storage capacity. Materials with low diffusivity, like titanium ($Ti-6Al-4V$, $\alpha \approx 2.8\text{ mm}^2/\text{s}$), trap heat locally, producing narrow, clean kerfs with minimal power. Conversely, high-diffusivity aluminum dissipates heat rapidly away into the sheet, requiring high laser powers to initiate melting."
