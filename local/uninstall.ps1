# Stops the local brain and removes its auto-start entry. The cloud bot keeps running.
$Link = Join-Path ([Environment]::GetFolderPath("Startup")) "AI Trading Bot (brain).lnk"
Remove-Item $Link -ErrorAction SilentlyContinue
Get-CimInstance Win32_Process -Filter "Name = 'pythonw.exe'" |
    Where-Object { $_.CommandLine -like "*-m bot local*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Write-Host "Local brain stopped and removed from startup. (Folder %LOCALAPPDATA%\TradingBotRunner left in place.)"
