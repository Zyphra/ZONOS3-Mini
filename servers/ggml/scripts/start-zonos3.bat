@echo off
rem start-zonos3 — double-click to launch with local model files and fetch ffmpeg if needed the server.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-zonos3.ps1" %*
if errorlevel 1 pause
