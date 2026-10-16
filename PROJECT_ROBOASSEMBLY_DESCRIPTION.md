# Project 6: RoboAssembly AI (Assembly Precedence Graph & Collision-Free Planner)
### Comprehensive Technical Defense, Mechanical Formulas & Viva Documentation

---

## 1. Executive Summary & Problem Definition
In high-volume mechanical assembly (pumps, gearboxes, automotive engines), components must be mated sequentially without physical collisions. An unoptimized assembly sequence requires excessive robotic manipulator reorientations and fixture tilting, resulting in **prolonged cycle times and positioning backlash errors**.

**RoboAssembly AI** formulates assembly planning as a Directed Acyclic Graph (DAG) topological optimization problem, minimizing gripper tool reorientations and station cycle times.

---

## 2. Core Mechanical & Physical Formulations

### A. Geometric Contact Precedence DAG
A mechanical assembly of $N$ parts is modeled as a graph $G = (V, E)$, where $V$ represents parts $\{P_1, P_2, \dots, P_N\}$ and directed edges $E = \{(P_i, P_j)\}$ represent physical obstruction constraints: $P_i$ must be installed before $P_j$ can physically clear its insertion path.

### B. Multi-Objective Cost Function (Reorientation Penalty)
Let $\mathbf{S} = [s_1, s_2, \dots, s_N]$ be a valid topological permutation. The total cycle time $T_{cycle}$ is:

$$T_{cycle} = \sum_{i=1}^N t_{insert}(s_i) + \sum_{i=2}^N \lambda \cdot \Delta(\mathbf{a}_{s_i}, \mathbf{a}_{s_{i-1}})$$

Where:
- $t_{insert}$: Nominal pick-and-place insertion duration.
- $\mathbf{a}_k \in \{+X, -X, +Y, -Y, +Z, -Z\}$: Insertion approach vector.
- $\lambda$: Delay penalty for gripper/fixture reorientation ($\approx 10-15\text{ s}$).
- $\Delta = 0$ if $\mathbf{a}_{s_i} = \mathbf{a}_{s_{i-1}}$, and $1$ otherwise.

---

## 3. Anticipated Examiner Q&A

**Q: How do you handle circular dependencies in assembly?**
- *Answer:* "If a cycle exists in the precedence graph, the assembly is physically locked (e.g., a puzzle ring). In manufacturing design for assembly (DFA), cycles are broken by sub-assembling modular sub-units prior to final chassis placement."
