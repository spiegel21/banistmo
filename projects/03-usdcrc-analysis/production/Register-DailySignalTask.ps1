<#
  Register (or update) the Windows Scheduled Task that runs the USD/CRC daily
  signal notifier every weekday after the Costa Rica market close.

  Fires on the WORK COMPUTER'S local clock. Default is 13:50 local. Override with
  -Time "HH:mm" for a different local time. (For reference, Costa Rica is UTC-6
  year-round with no daylight saving, and MONEX trades ~09:00-16:00 CST.)

  Usage (run in PowerShell, no admin needed for a per-user task):
      cd projects\03-usdcrc-analysis\production
      powershell -ExecutionPolicy Bypass -File .\Register-DailySignalTask.ps1
      powershell -ExecutionPolicy Bypass -File .\Register-DailySignalTask.ps1 -Time 13:50 -Channel popup

  Remove it later with:
      Unregister-ScheduledTask -TaskName "USDCRC Daily Signal" -Confirm:$false
#>
param(
    [string]$Time = "13:50",                       # local clock time to fire
    [string]$TaskName = "USDCRC Daily Signal",
    [ValidateSet("auto", "outlook", "popup", "both", "none")]
    [string]$Channel = "",                          # blank = use config.py default
    [switch]$NoRefresh
)

$here = Split-Path -Parent $MyInvocation.MyCommand.Definition
$bat  = Join-Path $here "run_daily_signal.bat"
if (-not (Test-Path $bat)) { throw "Launcher not found: $bat" }

# Build the argument string passed through to run_notifier.py.
$argList = @()
if ($Channel)  { $argList += "--channel $Channel" }
if ($NoRefresh){ $argList += "--no-refresh" }
$argString = ($argList -join " ")

$action  = New-ScheduledTaskAction -Execute $bat -Argument $argString -WorkingDirectory $here
# Weekdays only — MONEX does not trade on weekends.
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $Time
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `                          # catch up if the laptop was asleep at fire time
    -DontStopOnIdleEnd `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Principal $principal -Force | Out-Null

Write-Host "Registered task '$TaskName' -> weekdays at $Time local." -ForegroundColor Green
Write-Host "Runs: $bat $argString"
Write-Host "Test it now with:  Start-ScheduledTask -TaskName '$TaskName'"
