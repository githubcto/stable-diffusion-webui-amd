@echo off

cd /d "%~dp0"

set PYTHON=py.exe -3.10
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=

@REM set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --models-dir D:/AI/models

call webui.bat
