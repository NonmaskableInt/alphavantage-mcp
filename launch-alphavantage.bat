@echo off
REM Launch script for AlphaVantage MCP Server (Windows)

REM Get the directory where this script is located
cd /d "%~dp0"

REM Check if uv is available
where uv >nul 2>&1
if errorlevel 1 (
    echo Error: 'uv' is not installed or not in PATH
    echo Install it from: https://docs.astral.sh/uv/getting-started/installation/
    exit /b 1
)

uv run alphavantage-mcp-server
