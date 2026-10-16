@echo off
title TopoGen AI - Generative Structural Topology Optimizer
echo ====================================================================
echo Starting TopoGen AI (100%% Offline FEA Topology Optimization Server)...
echo ====================================================================
echo.
python -m streamlit run topogen_complete.py --server.port 8505
pause
