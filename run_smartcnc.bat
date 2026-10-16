@echo off
title SmartCNC AI - Physics-Informed G-Code & Chatter Simulator
color 0b
echo =====================================================================
echo    SmartCNC AI: Physics-Informed G-Code Feed-Rate & Chatter Simulator
echo    Mechanical Engineering Final Year Project Defense
echo =====================================================================
echo Starting application on local server...
echo URL: http://localhost:8507
echo.
python -m streamlit run smartcnc_complete.py --server.headless true --server.port 8507
pause
