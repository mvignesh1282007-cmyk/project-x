@echo off
title Real AI in Manufacturing - Master Project Suite Launcher
color 0a

:MENU
cls
echo ===============================================================================
echo     REAL AI IN MECHANICAL ENGINEERING & MANUFACTURING - 10 PROJECT SUITE
echo ===============================================================================
echo.
echo   [1]  Inverse Heat-Source Identification with PINN                 (Port 8522)
echo   [2]  Deep RL Adaptive Process Control (PPO vs PID)                (Port 8518)
echo   [3]  GNN Variation Propagation & Defect Localization              (Port 8519)
echo   [4]  Industrial Time-Series Foundation Model (Machine Health)      (Port 8520)
echo   [5]  Diffusion-Based Generative Defect Synthesis (Inspection)     (Port 8521)
echo   [6]  CAD Feature Recognition (B-Rep GNN -> CAPP Toolpath)         (Port 8523)
echo   [7]  Manufacturing Copilot (VLM Drawing Interpretation + RAG)     (Port 8524)
echo   [8]  Uncertainty-Aware RUL Prognostics & Risk-Based Maintenance   (Port 8525)
echo   [9]  Deep RL Production Scheduling (Shop-Floor Digital Twin)      (Port 8526)
echo   [10] Multi-Agent RL for AGV Fleet in Factory Digital Twin         (Port 8527)
echo.
echo   [0]  Exit
echo ===============================================================================
set /p choice="Enter your choice (1-10 or 0): "

if "%choice%"=="1" start run_pinn_heat_source.bat & goto MENU
if "%choice%"=="2" start run_deep_rl_process_control.bat & goto MENU
if "%choice%"=="3" start run_gnn_variation_propagation.bat & goto MENU
if "%choice%"=="4" start run_timeseries_foundation_model.bat & goto MENU
if "%choice%"=="5" start run_diffusion_defect_synthesis.bat & goto MENU
if "%choice%"=="6" start run_brep_gnn_capp.bat & goto MENU
if "%choice%"=="7" start run_manufacturing_copilot.bat & goto MENU
if "%choice%"=="8" start run_uncertainty_rul.bat & goto MENU
if "%choice%"=="9" start run_rl_production_scheduling.bat & goto MENU
if "%choice%"=="10" start run_marl_agv.bat & goto MENU
if "%choice%"=="0" exit

echo Invalid choice, try again.
pause
goto MENU
