@echo off
title Manufacturing Copilot - VLM & Shop-Floor Agent
color 09
echo =====================================================================
echo    Manufacturing Copilot: VLM Drawing Interpretation & Shop-Floor Agent
echo    Mechanical Engineering Final Year Project Defense
echo =====================================================================
echo Starting application on local server...
echo URL: http://localhost:8524
echo.
python -m streamlit run manufacturing_copilot.py --server.headless true --server.port 8524
pause
