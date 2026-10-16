@echo off
title CastDefect AI - Solidification & Porosity Simulator
color 0e
echo =====================================================================
echo    CastDefect AI: Metal Foundry Solidification & Porosity Simulator
echo    Mechanical Engineering Final Year Project Defense
echo =====================================================================
echo Starting application on local server...
echo URL: http://localhost:8511
echo.
python -m streamlit run castdefect_complete.py --server.headless true --server.port 8511
pause
