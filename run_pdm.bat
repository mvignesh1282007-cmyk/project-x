@echo off
title NeuroMaint AI - Industrial Predictive Maintenance Platform
echo ====================================================================
echo Starting NeuroMaint AI (100%% Offline Machine Health Server)...
echo ====================================================================
echo.
python -m streamlit run pdm_complete.py --server.port 8504
pause
