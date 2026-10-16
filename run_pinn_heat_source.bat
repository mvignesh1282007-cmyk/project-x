@echo off
title PINN Inverse Heat-Source Identification - Welding & AM
color 0c
echo =====================================================================
echo    PINN Inverse Heat-Source Identification in Welding & AM
echo    Mechanical Engineering Final Year Project Defense
echo =====================================================================
echo Starting application on local server...
echo URL: http://localhost:8522
echo.
python -m streamlit run pinn_heat_source.py --server.headless true --server.port 8522
pause
