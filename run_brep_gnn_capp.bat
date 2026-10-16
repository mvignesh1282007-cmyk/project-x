@echo off
title CAD B-Rep GNN Feature Recognition & CAPP
color 0b
echo =====================================================================
echo    CAD B-Rep GNN Feature Recognition & Automatic Process Planning
echo    Mechanical Engineering Final Year Project Defense
echo =====================================================================
echo Starting application on local server...
echo URL: http://localhost:8523
echo.
python -m streamlit run brep_gnn_capp.py --server.headless true --server.port 8523
pause
