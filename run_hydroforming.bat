@echo off
title HydroForming AI - Metal Extrusion Simulator
color 05
echo =====================================================================
echo    HydroForming AI: Metal Extrusion Plastic Flow & Press Tonnage
echo    Mechanical Engineering Final Year Project Defense
echo =====================================================================
echo Starting application on local server...
echo URL: http://localhost:8514
echo.
python -m streamlit run hydroforming_complete.py --server.headless true --server.port 8514
pause
