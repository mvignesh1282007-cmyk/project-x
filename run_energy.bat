@echo off
title EnerSim AI - Factory Energy Spike & Carbon Forecaster
echo ====================================================================
echo Starting EnerSim AI (100% Offline Factory Energy Optimization Server)...
echo ====================================================================
echo.
python -m streamlit run energy_app_complete.py --server.port 8503
pause
