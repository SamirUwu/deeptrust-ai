@echo off
title Run API + Frontend

echo ================================
echo   Iniciando backend y frontend
echo ================================
echo.

echo [1/2] Iniciando API...
start /B python api.py

echo Esperando a que la API cargue...
timeout /t 10 /nobreak > nul

echo [2/2] Iniciando frontend...
start /B npm run dev

echo.
echo Servicios iniciados en segundo plano.
pause