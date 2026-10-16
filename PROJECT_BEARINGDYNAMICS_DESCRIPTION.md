# Project 7: BearingDynamics AI (Spindle Bearing Fault & RUL Analyzer)
### Comprehensive Technical Defense, Mechanical Formulas & Viva Documentation

---

## 1. Executive Summary & Problem Definition
In high-precision machine tools (CNC milling, turning, grinders), spindle ball and roller bearings operate under extreme radial and axial loads. Sub-surface shear stresses induce micro-pitting, which progresses into severe spalling. 

**BearingDynamics AI** implements high-frequency resonance technique (HFRT) and Hilbert envelope analysis—the exact methodology benchmarked on the **Case Western Reserve University (CWRU) Bearing Dataset**—to identify defective inner/outer races and forecast Remaining Useful Life (RUL).

---

## 2. Core Mechanical & Physical Formulations

### A. Kinematic Bearing Defect Harmonics
For a bearing with pitch diameter $D$, ball diameter $d$, number of rolling elements $N$, contact angle $\phi$, and shaft rotational frequency $f_r = \frac{RPM}{60}$:

$$\gamma = \frac{d}{D} \cos \phi$$

1. **Ball Pass Frequency Outer Race (BPFO):**
   $$BPFO = \frac{N}{2} f_r (1 - \gamma)$$
2. **Ball Pass Frequency Inner Race (BPFI):**
   $$BPFI = \frac{N}{2} f_r (1 + \gamma)$$
3. **Ball Spin Frequency (BSF):**
   $$BSF = \frac{D}{2 d} f_r (1 - \gamma^2)$$
4. **Fundamental Train Frequency (Cage FTF):**
   $$FTF = \frac{f_r}{2} (1 - \gamma)$$

### B. Hilbert Transform Envelope Demodulation
Let $x(t)$ be the bandpass-filtered vibration signal around the bearing housing resonance. The analytic signal $z(t)$ is formed using the Hilbert transform $\mathcal{H}[x(t)]$:

$$z(t) = x(t) + j \mathcal{H}[x(t)]$$
$$Env(t) = |z(t)| = \sqrt{x(t)^2 + (\mathcal{H}[x(t)])^2}$$

Taking the FFT of $Env(t)$ reveals sharp peaks at $BPFO$ or $BPFI$.

### C. Statistical Health Indicators & Weibull RUL
$$\text{Kurtosis} = \frac{\frac{1}{N} \sum_{i=1}^N (x_i - \bar{x})^4}{\text{RMS}^4}$$

Weibull hazard reliability survival function:

$$R(t) = \exp \left( - \left(\frac{t}{\eta}\right)^\beta \right)$$

---

## 3. Anticipated Examiner Q&A

**Q: Why does the inner race defect (BPFI) produce sidebands, while the outer race (BPFO) does not?**
- *Answer:* "The outer race is stationary in the housing, so the defect stays in the load zone with constant amplitude. The inner race rotates with the shaft; when the defect passes through the maximum load zone, the pulse amplitude peaks, and when it is in the unloaded zone, it drops. This amplitude modulation by the shaft rotation frequency ($f_r$) generates characteristic sidebands at $BPFI \pm f_r$."
