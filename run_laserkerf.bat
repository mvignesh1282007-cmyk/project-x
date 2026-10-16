@echo off
title LaserKerf AI - Laser Cutting & HAZ Simulator
color 06
echo =====================================================================
echo    LaserKerf AI: Thermal Beam Kerf Width & HAZ Simulator
echo    Mechanical Engineering Final Year Project Defense
echo =====================================================================
echo Starting application on local server...
echo URL: http://localhost:8516
echo.
python -m streamlit run laserkerf_complete.py --server.headless true --server.port 8516
pause
