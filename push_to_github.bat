@echo off
title Push Real AI Projects Suite to GitHub
color 0a
echo ===============================================================================
echo     PUSH 10 REAL AI MANUFACTURING PROJECTS TO GITHUB
echo ===============================================================================
echo.
echo Target Repository: https://github.com/mvignesh1282007-cmyk/project-x.git
echo.

git remote remove origin 2>nul
git remote add origin https://github.com/mvignesh1282007-cmyk/project-x.git
git branch -M main

echo Uploading 10 projects and documentation to GitHub...
echo (A browser window will open to authorize your GitHub account if needed)
echo.
git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ===============================================================================
    echo [SUCCESS] All 10 projects and documentation are now live on GitHub!
    echo URL: https://github.com/mvignesh1282007-cmyk/project-x
    echo ===============================================================================
) else (
    echo.
    echo [NOTICE] If the push was canceled or failed:
    echo 1. Ensure you authorized the prompt in your browser.
    echo 2. Or generate a Personal Access Token at https://github.com/settings/tokens
)
echo.
pause
