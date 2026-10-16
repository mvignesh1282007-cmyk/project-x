@echo off
title Uncertainty-Aware RUL Prognostics - Risk-Based Maintenance
color 0e
echo =====================================================================
echo    Uncertainty-Aware RUL Prognostics & Risk-Based Maintenance
echo    Mechanical Engineering Final Year Project Defense
echo =====================================================================
echo Starting application on local server...
echo URL: http://localhost:8525
echo.
python -m streamlit run uncertainty_rul_prognostics.py --server.headless true --server.port 8525
pause
