@echo off
setlocal
title NOVA AI Data Analyst - Priyotosh Das (23053065)
cd /d "%~dp0"

echo ========================================================
echo   NOVA AI Data Analyst - Priyotosh Das (Roll No: 23053065)
echo   Starting Streamlit Application...
echo ========================================================
echo.

REM Use the Python installation available on this computer instead of
REM a hard-coded path from another Windows account.
where python >nul 2>nul
if not errorlevel 1 (
    python -m streamlit run app.py
) else (
    py -3 -m streamlit run app.py
)

if errorlevel 1 (
    echo.
    echo Streamlit could not start. Install the project packages with:
    echo   python -m pip install -r requirements.txt
)

pause
endlocal
