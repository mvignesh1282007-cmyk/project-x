# 🏭 Real AI in Mechanical Engineering & Manufacturing — 10 Project Suite

A comprehensive, production-grade suite of **10 Real AI Projects for Mechanical & Manufacturing Engineering**, built strictly on deep, learned neural architectures (PINNs, Deep Reinforcement Learning, Graph Neural Networks, Foundation Transformers, Denoising Diffusion, and Multi-Agent RL) benchmarked against classical engineering baselines.

Each project is completely self-contained in a single executable Python file (`.py`), accompanied by an interactive cyber-physical Streamlit dashboard, a 1-click batch launcher (`.bat`), and an exhaustive technical defense specification (`PROJECT_..._DESCRIPTION.md`).

---

## 🚀 Master Project Catalog

| # | Project Name | Real AI Core | Physical Digital Twin | Non-AI Classical Baseline | App File | Technical Guide | Port |
|---|---|---|---|---|---|---|---|
| **1** | **Inverse Heat-Source Identification** | **PINN** (Higher-Order Autograd PDE) + **Neural Operator** | 2D Goldak Double-Ellipsoid Conduction | Cubic Spline / Bilinear Interpolation | `pinn_heat_source.py` | `PROJECT_PINN_HEAT_SOURCE.md` | `:8522` |
| **2** | **Deep RL Adaptive Process Control** | **PPO Continuous Actor-Critic** | Thermal-Kinetic Composite Curing | Industrial PID with Anti-Windup | `deep_rl_process_control.py` | `PROJECT_DEEP_RL_PROCESS_CONTROL.md` | `:8518` |
| **3** | **GNN Variation Propagation & Localization** | **Message-Passing GNN** | Multi-Stage Tolerance Stack Matrix | Univariate SPC & Pearson Correlation | `gnn_variation_propagation.py` | `PROJECT_GNN_VARIATION_PROPAGATION.md` | `:8519` |
| **4** | **Industrial Time-Series Foundation Model** | **Pretrained Transformer Encoder** (PatchTST/MOMENT) | CNC High-Frequency Spindle Vibration | XGBoost / Random Forest from Scratch | `timeseries_foundation_model.py` | `PROJECT_TIMESERIES_FOUNDATION_MODEL.md` | `:8520` |
| **5** | **Diffusion Generative Defect Synthesis** | **Latent DDPM U-Net + DRAEM** | Additive LPBF Sub-Surface Cracks | Frozen Pretrained PatchCore | `diffusion_defect_synthesis.py` | `PROJECT_DIFFUSION_DEFECT_SYNTHESIS.md` | `:8521` |
| **6** | **CAD Feature Recognition $\to$ CAPP** | **B-Rep Graph Neural Network** | STEP B-Rep Topology & CAM Toolpaths | Rule-Based Face Heuristics | `brep_gnn_capp.py` | `PROJECT_BREP_GNN_CAPP.md` | `:8523` |
| **7** | **Manufacturing Copilot: VLM Drawing + Agent** | **Vision-Language Model + RAG** | GD&T Blueprints, CMM Inspection | Classical OCR / Regex Search | `manufacturing_copilot.py` | `PROJECT_MANUFACTURING_COPILOT.md` | `:8524` |
| **8** | **Uncertainty-Aware RUL Prognostics** | **Monte Carlo Dropout & Conformal** | Aero Turbofan Paris-Erdogan Fatigue | Fixed Time-Based Maintenance (TBM) | `uncertainty_rul_prognostics.py` | `PROJECT_UNCERTAINTY_RUL.md` | `:8525` |
| **9** | **Deep RL Production Scheduling Twin** | **PPO Actor-Critic Dispatcher** | Discrete-Event JSSP with Spindle Failure | Static FIFO / SPT & Static CP-SAT | `rl_production_scheduling.py` | `PROJECT_RL_PRODUCTION_SCHEDULING.md` | `:8526` |
| **10** | **Multi-Agent RL for AGV Fleet Twin** | **MAPPO / QMIX (CTDE)** | 10×10 Shop Floor Grid with Dynamic Obstacles | Multi-Agent A* with Reservation Tables | `marl_agv_fleet.py` | `PROJECT_MARL_AGV_FLEET.md` | `:8527` |

---

## ⚡ Quick Start: 1-Click Master Menu

To launch any project, run the interactive master menu:
```cmd
run_all_projects_menu.bat
```
Or launch any individual project directly:
```cmd
run_pinn_heat_source.bat            # Project 1 (Port 8522)
run_deep_rl_process_control.bat     # Project 2 (Port 8518)
run_gnn_variation_propagation.bat   # Project 3 (Port 8519)
run_timeseries_foundation_model.bat # Project 4 (Port 8520)
run_diffusion_defect_synthesis.bat  # Project 5 (Port 8521)
run_brep_gnn_capp.bat               # Project 6 (Port 8523)
run_manufacturing_copilot.bat       # Project 7 (Port 8524)
run_uncertainty_rul.bat             # Project 8 (Port 8525)
run_rl_production_scheduling.bat    # Project 9 (Port 8526)
run_marl_agv.bat                    # Project 10 (Port 8527)
```

---

## 🛠️ Tech Stack & Requirements

- **Python 3.10+ / 3.12+ / 3.13**
- **PyTorch** (`torch`, `torchvision`)
- **Streamlit** (Cyber-Physical Interactive HUDs)
- **Plotly** (3D Isothermal & 2D Engineering Visualizations)
- **NumPy, SciPy, Pandas, NetworkX, Scikit-Learn**

Install dependencies with:
```bash
pip install torch torchvision streamlit plotly numpy scipy pandas networkx scikit-learn
```

---

## 🎓 Academic Defense & Viva Voce Handbook

Every project folder includes a dedicated markdown defense guide containing:
1. **Mathematical & Thermodynamic Formulations** (PDEs, Loss functions, Bellman equations).
2. **2-Minute Pitch Script** crafted specifically for external university examiners.
3. **Anticipated Viva Voce Q&A** addressing non-AI baselines, convergence guarantees, and industrial shop-floor integration.
