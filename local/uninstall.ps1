# Stops the local brain and removes its auto-start task. The cloud bot keeps running.
Unregister-ScheduledTask -TaskName "AI Trading Bot (brain)" -Confirm:$false -ErrorAction SilentlyContinue
Get-CimInstance Win32_Process -Filter "Name = 'pythonw.exe'" |
    Where-Object { $_.CommandLine -like "*-m bot local*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Write-Host "Local brain stopped and removed from auto-start. (The .runner folder is left in place.)"
