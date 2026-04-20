@echo off
REM ============================================================
REM  Churn Prediction - Scheduled Run
REM ============================================================
REM
REM  To schedule this with Windows Task Scheduler:
REM    1. Open Task Scheduler (Win+R -> taskschd.msc)
REM    2. Click "Create Basic Task"
REM    3. Set a trigger (e.g. Daily, Weekly)
REM    4. Action: "Start a Program"
REM    5. Program/script: full path to this .bat file
REM    6. Start in: full path to the project folder
REM
REM ============================================================

cd /d "%~dp0"

echo [%date% %time%] Starting churn prediction... >> run_log.txt

python ChurnPrediction.py

if %ERRORLEVEL% == 0 (
    echo [%date% %time%] Prediction completed successfully. >> run_log.txt
) else (
    echo [%date% %time%] ERROR: Prediction failed with code %ERRORLEVEL%. >> run_log.txt
)
