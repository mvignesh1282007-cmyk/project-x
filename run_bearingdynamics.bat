@echo off
title BearingDynamics AI - Bearing Fault & RUL Analyzer
color 0b
echo =====================================================================
echo    BearingDynamics AI: Spindle Bearing Diagnostics & RUL Analyzer
echo    Mechanical Engineering Final Year Project Defense
echo =====================================================================
echo Starting application on local server...
echo URL: http://localhost:8513
echo.
python -m streamlit run bearingdynamics_complete.py --server.headless true --server.port 8513
pause
