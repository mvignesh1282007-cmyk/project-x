@echo off
title GNN Variation Propagation - Multi-Stage Manufacturing Digital Twin
color 0a
echo =====================================================================
echo    GNN Variation Propagation & Defect Source Localization
echo    Mechanical Engineering Final Year Project Defense
echo =====================================================================
echo Starting application on local server...
echo URL: http://localhost:8519
echo.
python -m streamlit run gnn_variation_propagation.py --server.headless true --server.port 8519
pause
