@echo off
title RoboWeld AI - 6-Axis Robotic Welding Seam Tracking
color 0c
echo =====================================================================
echo    RoboWeld AI: 6-Axis Robotic Welding Seam Tracking & Kinematics
echo    Mechanical Engineering Final Year Project Defense
echo =====================================================================
echo Starting application on local server...
echo URL: http://localhost:8508
echo.
python -m streamlit run roboweld_complete.py --server.headless true --server.port 8508
pause
