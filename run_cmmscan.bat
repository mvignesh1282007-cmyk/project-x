@echo off
title CMM-Scan AI - 3D Metrology & GD&T
color 03
echo =====================================================================
echo    CMM-Scan AI: 3D Point-Cloud Coordinate Metrology & GD&T
echo    Mechanical Engineering Final Year Project Defense
echo =====================================================================
echo Starting application on local server...
echo URL: http://localhost:8515
echo.
python -m streamlit run cmmscan_complete.py --server.headless true --server.port 8515
pause
