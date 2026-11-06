@echo off
setlocal
node "%~dp0desktop\launch.cjs" %*
set "SPYDER_LAUNCH_EXIT=%ERRORLEVEL%"
if not "%SPYDER_LAUNCH_EXIT%"=="0" pause
exit /b %SPYDER_LAUNCH_EXIT%
