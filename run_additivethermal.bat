@echo off
title AdditiveThermal AI - 3D Printing Warpage & Thermal Stress
color 0d
echo =====================================================================
echo    AdditiveThermal AI: 3D Printing Warpage Simulator
echo    Mechanical Engineering Final Year Project Defense
echo =====================================================================
echo Starting application on local server...
echo URL: http://localhost:8509
echo.
python -m streamlit run additivethermal_complete.py --server.headless true --server.port 8509
pause
