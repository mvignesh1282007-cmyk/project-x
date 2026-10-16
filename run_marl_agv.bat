@echo off
title Multi-Agent RL AGV Fleet - Factory Digital Twin
color 0b
echo =====================================================================
echo    Multi-Agent RL AGV Fleet Coordination in Factory Digital Twin
echo    Mechanical Engineering Final Year Project Defense
echo =====================================================================
echo Starting application on local server...
echo URL: http://localhost:8527
echo.
python -m streamlit run marl_agv_fleet.py --server.headless true --server.port 8527
pause
