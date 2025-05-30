@echo off
title Launch Nova + Telegram Sync
setlocal enabledelayedexpansion

echo --------------------------------------------
echo 🚀 Launching ngrok tunnel on port 5678...
echo --------------------------------------------

start "ngrok" cmd /k "title ngrok tunnel & ngrok http 5678"
echo Waiting for ngrok to initialize...
timeout /t 5 >nul

echo.
echo --------------------------------------------
echo 🌐 Fetching public ngrok HTTPS URL...
echo --------------------------------------------

REM Use PowerShell to parse the HTTPS tunnel safely
for /f "tokens=* usebackq" %%i in (`powershell -Command ^
  "(Invoke-RestMethod -Uri http://127.0.0.1:4040/api/tunnels).tunnels | Where-Object { $_.proto -eq 'https' } | Select-Object -ExpandProperty public_url"`) do (
    set NGROK_URL=%%i
)

if not defined NGROK_URL (
    echo ❌ Could not retrieve HTTPS ngrok URL. Make sure ngrok is running and port 4040 is accessible.
    pause
    exit /b 1
)

echo ✅ Public tunnel: %NGROK_URL%

echo.
echo --------------------------------------------
echo 📝 Updating docker-compose.yml
echo --------------------------------------------

REM Backup existing docker-compose.yml
copy docker-compose.yml docker-compose.backup.yml >nul

(
    for /f "usebackq delims=" %%a in ("docker-compose.yml") do (
        echo %%a | findstr /C:"WEBHOOK_TUNNEL_URL=" >nul
        if errorlevel 1 (
            echo %%a
        ) else (
            echo       - WEBHOOK_TUNNEL_URL=%NGROK_URL%
        )
    )
) > docker-compose.tmp.yml

move /y docker-compose.tmp.yml docker-compose.yml >nul

REM Also patch N8N_EDITOR_BASE_URL
(
    for /f "usebackq delims=" %%a in ("docker-compose.yml") do (
        echo %%a | findstr /C:"N8N_EDITOR_BASE_URL=" >nul
        if errorlevel 1 (
            echo %%a
        ) else (
            echo       - N8N_EDITOR_BASE_URL=%NGROK_URL%
        )
    )
) > docker-compose.final.yml

move /y docker-compose.final.yml docker-compose.yml >nul

echo 🔄 Restarting Docker containers...
docker compose down
docker compose up -d --force-recreate

echo.
echo --------------------------------------------
echo 📬 Registering Telegram webhook
echo --------------------------------------------

set BOT_TOKEN=8172438472:AAG4MTzB_d5GP9vEWrhzLL3P2k0GUqkNVnY
set WEBHOOK_ID=c0eee8d7-a6cd-42b5-8ea9-47c43e79c2ec
set FULL_WEBHOOK=%NGROK_URL%/webhook/%WEBHOOK_ID%/webhook

curl -X POST "https://api.telegram.org/bot%BOT_TOKEN%/setWebhook" -H "Content-Type: application/x-www-form-urlencoded" -d "url=%FULL_WEBHOOK%"

echo --------------------------------------------
echo ✅ Telegram webhook set to:
echo %FULL_WEBHOOK%
echo --------------------------------------------

echo.
echo 🔚 All done. Press any key to close this window...
pause >nul
exit /b 0
