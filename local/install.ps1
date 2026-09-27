# Installs the PC side of the trading bot (the hourly local brain + CLM second opinion).
# - makes a private copy of the repo in "<project>\.runner" with its own Python venv
# - registers a hidden Task Scheduler task that starts it at every Windows login
# - starts it right away
# Safe to re-run. To remove: run local\uninstall.ps1
param([string]$Repo = "https://github.com/EricsonArt/ai-trading-bot.git")
$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $PSScriptRoot
$Runner = Join-Path $Project ".runner"
$TaskName = "AI Trading Bot (brain)"

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

$Pyw = Join-Path $Runner ".venv\Scripts\pythonw.exe"
$Action = New-ScheduledTaskAction -Execute $Pyw -Argument "-m bot local" -WorkingDirectory $Runner
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
$Trigger.Delay = "PT1M"  # give Ollama and the network a minute after login
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -Hidden
$Principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings `
    -Principal $Principal -Description "AI trading bot: local brain (Ollama) + CLM, syncs with GitHub" -Force | Out-Null

Start-ScheduledTask -TaskName $TaskName
Write-Host "Installed. Runner: $Runner  |  Auto-start task: $TaskName  |  Log: $Runner\.cache\local.log"
