@echo off
title Diffusion Defect Synthesis - Few-Shot Quality Inspection
color 0c
echo =====================================================================
echo    Diffusion-Based Generative Defect Synthesis for Few-Shot Inspection
echo    Mechanical Engineering Final Year Project Defense
echo =====================================================================
echo Starting application on local server...
echo URL: http://localhost:8521
echo.
python -m streamlit run diffusion_defect_synthesis.py --server.headless true --server.port 8521
pause
