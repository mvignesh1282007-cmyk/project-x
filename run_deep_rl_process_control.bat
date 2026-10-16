@echo off
title Deep RL Adaptive Process Control - Industrial Digital Twin
color 0b
echo =====================================================================
echo    Deep RL Adaptive Process Control (PPO vs. Industrial PID)
echo    Mechanical Engineering Final Year Project Defense
echo =====================================================================
echo Starting application on local server...
echo URL: http://localhost:8518
echo.
python -m streamlit run deep_rl_process_control.py --server.headless true --server.port 8518
pause
