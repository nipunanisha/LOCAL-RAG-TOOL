@echo off
title RAG Tool - Setup
mode con cols=80 lines=50
cls

echo.
echo  ================================================
echo    RAG Tool - Setup and Installation
echo  ================================================
echo.

set "SCRIPT_DIR=%~dp0"

echo  Starting installer...
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%setup.ps1"

if %ERRORLEVEL% neq 0 (
    echo.
    echo  [ERROR] Installation encountered errors. See messages above.
    echo.
)

pause
