# Installs the PC side of the trading bot (the hourly local brain + CLM second opinion).
# - makes a private copy of the repo in %LOCALAPPDATA%\TradingBotRunner with its own Python venv
# - adds a hidden auto-start entry to your Startup folder (runs at every Windows login)
# - starts it right away
# Safe to re-run. To remove: run local\uninstall.ps1
param([string]$Repo = "https://github.com/EricsonArt/ai-trading-bot.git")
$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $PSScriptRoot
$Runner = Join-Path $env:LOCALAPPDATA "TradingBotRunner"

if (-not (Test-Path (Join-Path $Runner ".git"))) {
    git clone -q $Repo $Runner
}
Push-Location $Runner
git config credential.https://github.com.helper ""
git config --add credential.https://github.com.helper "!gh auth git-credential"
git config user.name "trading-bot-pc"
git config user.email "trading-bot-pc@users.noreply.github.com"
git pull -q
Pop-Location

$Py = Join-Path $Runner ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    python -m venv (Join-Path $Runner ".venv")
}
& $Py -m pip install -q --disable-pip-version-check -r (Join-Path $Runner "requirements.txt")

# Reuse already-downloaded price history so the first run is fast.
$Candles = Join-Path $Project ".cache\candles"
if (Test-Path $Candles) {
    New-Item -ItemType Directory -Force (Join-Path $Runner ".cache\candles") | Out-Null
    Copy-Item "$Candles\*" (Join-Path $Runner ".cache\candles") -Force
}

$Startup = [Environment]::GetFolderPath("Startup")
$Link = Join-Path $Startup "AI Trading Bot (brain).lnk"
$Shell = New-Object -ComObject WScript.Shell
$Sc = $Shell.CreateShortcut($Link)
$Sc.TargetPath = Join-Path $Runner ".venv\Scripts\pythonw.exe"
$Sc.Arguments = "-m bot local"
$Sc.WorkingDirectory = $Runner
$Sc.Description = "AI trading bot: local brain (Ollama) + CLM, syncs with GitHub"
$Sc.Save()

Start-Process -FilePath $Sc.TargetPath -ArgumentList $Sc.Arguments -WorkingDirectory $Runner -WindowStyle Hidden
Write-Host "Installed. Runner: $Runner  |  Auto-start: $Link  |  Log: $Runner\.cache\local.log"
