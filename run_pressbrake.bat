@echo off
title PressBrake Springback AI - Sheet Metal Simulator
color 0a
echo =====================================================================
echo    PressBrake Springback AI: Sheet Metal V-Die Bending Simulator
echo    Mechanical Engineering Final Year Project Defense
echo =====================================================================
echo Starting application on local server...
echo URL: http://localhost:8510
echo.
python -m streamlit run pressbrake_complete.py --server.headless true --server.port 8510
pause
