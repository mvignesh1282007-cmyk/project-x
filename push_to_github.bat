@echo off
title Push Real AI Projects Suite to GitHub
color 0b
echo ===============================================================================
echo     PUSH 10 REAL AI MANUFACTURING PROJECTS TO GITHUB
echo ===============================================================================
echo.
echo Please enter your GitHub Repository URL (e.g. https://github.com/username/repo-name.git)
set /p REPO_URL="Repo URL: "

if "%REPO_URL%"=="" (
    echo [ERROR] No URL provided. Exiting.
    pause
    exit /b
)

echo.
echo Configuring remote origin...
git remote remove origin 2>nul
git remote add origin %REPO_URL%
git branch -M main

echo.
echo Pushing committed files to GitHub...
echo (If prompted, log in via your browser or paste your GitHub Personal Access Token 'ghp_...')
echo.
git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ===============================================================================
    echo [SUCCESS] All 10 projects and documentation successfully uploaded to GitHub!
    echo ===============================================================================
) else (
    echo.
    echo [ERROR] Push failed. Please verify that:
    echo 1. The repository exists on your GitHub account.
    echo 2. You used a Personal Access Token (ghp_...) instead of your account password.
)
echo.
pause
