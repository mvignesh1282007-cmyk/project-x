@echo off
title MetrologyAI - Optical Inspection & GD&T Comparator Gauge
echo ====================================================================
echo Starting MetrologyAI (100%% Offline Optical Metrology Gauge Server)...
echo ====================================================================
echo.
python -m streamlit run metrology_app_complete.py --server.port 8506
pause
