@echo off
title Deep RL Production Scheduling - Shop-Floor Digital Twin
color 0b
echo =====================================================================
echo    Deep RL Production Scheduling on a Shop-Floor Digital Twin
echo    Mechanical Engineering Final Year Project Defense
echo =====================================================================
echo Starting application on local server...
echo URL: http://localhost:8526
echo.
python -m streamlit run rl_production_scheduling.py --server.headless true --server.port 8526
pause
