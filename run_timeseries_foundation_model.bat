@echo off
title Industrial Foundation Model - Machine Health & Cross-Machine Transfer
color 0d
echo =====================================================================
echo    Industrial Time-Series Foundation Model for Machine Health
echo    Mechanical Engineering Final Year Project Defense
echo =====================================================================
echo Starting application on local server...
echo URL: http://localhost:8520
echo.
python -m streamlit run timeseries_foundation_model.py --server.headless true --server.port 8520
pause
