# Project 5: CastDefect AI (Foundry Solidification & Shrinkage Porosity Simulator)
### Comprehensive Technical Defense, Mechanical Formulas & Viva Documentation

---

## 1. Executive Summary & Problem Definition
In metal casting foundries, molten metal liquid-to-solid phase transition causes significant volumetric contraction ($\sim 3-7\%$). Without proper directional solidification towards a feeder reservoir (riser), the hot thermal center of the casting freezes last without liquid replenishment, creating **internal shrinkage cavities and micro-porosity** that cause catastrophic hydraulic pressure leakage or structural fracture.

**CastDefect AI** is an analytical casting solidification simulator using Chvorinov's modulus rules and the Niyama criterion to guarantee sound castings.

---

## 2. Core Mechanical & Physical Formulations

### A. Chvorinov's Solidification Rule
The total solidification time $t_s$ of a cast section:

$$t_s = B \cdot \left( \frac{V}{A} \right)^n = B \cdot M^n$$

Where:
- $V$: Volume of the metal section ($\text{cm}^3$).
- $A$: Cooling surface area in contact with mold sand ($\text{cm}^2$).
- $M = V / A$: Section modulus ($\text{cm}$).
- $B$: Mold-metal thermal constant ($\text{min/cm}^2$).
- $n$: Solidification exponent ($\approx 2.0$).

### B. Directional Solidification Feeder Rule
To ensure the riser feeds molten metal until the casting is fully frozen:

$$M_{riser} \ge 1.20 \cdot M_{casting}$$

### C. The Niyama Criterion for Micro-Porosity
The Niyama parameter $Ny$ combines local thermal gradient $G$ and cooling rate $\dot{T}$:

$$Ny = \frac{G}{\sqrt{\dot{T}}} = \frac{\|\nabla T\|}{\sqrt{\frac{\partial T}{\partial t}}}$$

A value of $Ny < 1.0\ ^\circ\text{C}^{0.5}\text{ s}^{0.5}/\text{mm}$ indicates severe feeding resistance through the dendritic mushy zone, resulting in micro-porosity.

---

## 3. Anticipated Examiner Q&A

**Q: How does mold material influence the constant $B$?**
- *Answer:* "The mold constant $B$ is inversely proportional to the thermal effusivity of the mold sand: $b_m = \sqrt{k_m \rho_m c_m}$. Metal chill molds have very high effusivity, leading to low $B$ and rapid chilling, whereas resin-bonded sand molds insulate the metal, increasing $B$."
