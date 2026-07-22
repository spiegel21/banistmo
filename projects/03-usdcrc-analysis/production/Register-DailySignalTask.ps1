<#
  Register (or update) the Windows Scheduled Task that runs the USD/CRC daily
  signal notifier every weekday after the Costa Rica market close.

  Costa Rica is UTC-6 year-round (no daylight saving). MONEX trades ~09:00-16:00
  CST and BCCR posts the day's session shortly after; the default 16:45 fires
  after that, so the notification is for the NEXT session with today's data in
  hand. IMPORTANT: Task Scheduler uses the WORK COMPUTER'S local time. If that
  machine is not on CR time, pass -Time as the CR-close equivalent in local time
  (e.g. a machine in Colombia/UTC-5 would use "17:45" for 16:45 CR).

  Usage (run in PowerShell, no admin needed for a per-user task):
      cd projects\03-usdcrc-analysis\production
      powershell -ExecutionPolicy Bypass -File .\Register-DailySignalTask.ps1
      powershell -ExecutionPolicy Bypass -File .\Register-DailySignalTask.ps1 -Time 17:45 -Channel popup

  Remove it later with:
      Unregister-ScheduledTask -TaskName "USDCRC Daily Signal" -Confirm:$false
#>
param(
    [string]$Time = "16:45",                       # local clock time to fire (see note above)
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
