@echo off
title RoboAssembly AI - Assembly Sequence Planner
color 09
echo =====================================================================
echo    RoboAssembly AI: Multi-Part Precedence Graph Assembly Planner
echo    Mechanical Engineering Final Year Project Defense
echo =====================================================================
echo Starting application on local server...
echo URL: http://localhost:8512
echo.
python -m streamlit run roboassembly_complete.py --server.headless true --server.port 8512
pause
