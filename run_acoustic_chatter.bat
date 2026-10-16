@echo off
title AcoustiCut AI - Spindle Chatter Stethoscope
color 0b
echo =====================================================================
echo    AcoustiCut AI: Acoustic Spindle Chatter Stethoscope
echo    Mechanical Engineering Final Year Project Defense
echo =====================================================================
echo Starting application on local server...
echo URL: http://localhost:8517
echo.
python -m streamlit run acoustic_chatter_complete.py --server.headless true --server.port 8517
pause
