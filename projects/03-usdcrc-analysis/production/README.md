# Productionalizing the USD/CRC daily signal

A tiny, low-footprint daily job for a Windows work computer: once a day, just
after the Costa Rica market closes, it computes the recommended calendar
(quincena) position for the **next** session and tells you — by an **Outlook
e-mail**, a **desktop popup**, or both. Set it and forget it via Task Scheduler.

This is deliberately *not* the research pipeline. The backtests in `../src`
pull scipy / statsmodels / scikit-learn / weasyprint; none of that runs here.

## Why this can be so light

The recommended rule has two parts, and only one of them needs market data:

| Part | Where it comes from | Needs a data feed? |
|------|--------------------|--------------------|
| **Direction** (LONG vs SHORT USD) | pure calendar math — business days to the IVA/quincena deadline (`payment_calendar.td_to_iva_for`) | **No** |
| **Size** (1.0× vs 0.5×) + "as of" context | trailing ~20 sessions of MONEX volume | Yes, but stale-tolerant |

So the notification is *always correct on direction* even fully offline. The
BCCR refresh only sharpens the size multiplier and the context line. That is why
the whole thing runs on **pandas + numpy** (~2 seconds, ~150 MB peak, once a day).

## The pieces

```
production/
  config.py                 # all knobs; every one overridable by a USDCRC_* env var
  refresh_monex.py          # best-effort daily pull from BCCR CodCuadro=770 -> upsert monex_clean.csv
  notify.py                 # Outlook (win32com) + native popup (ctypes MessageBox); both optional
  run_notifier.py           # the entry point Task Scheduler runs: refresh -> signal -> deliver -> log
  run_daily_signal.bat      # Windows launcher (finds ..\.venv or PATH python)
  Register-DailySignalTask.ps1  # one command to create the weekday scheduled task
  requirements-prod.txt     # pandas, numpy (+ optional pywin32 / matplotlib)
  logs/                     # notifier.log (rotating) — gitignored
```

The notifier reuses `daily_signal.signal_for` / `render_text` **verbatim**, so
the message can never drift from the backtested rule.

## Design choices (the brainstorm)

- **Email vs popup?** Both are implemented; you choose per run or in config.
  Default is `auto`: send the Outlook mail, and only if Outlook isn't available
  (closed, not installed) fall back to a popup — so you always get *something*.
  The Outlook path uses `win32com` and sends from/to your own logged-in account,
  so there's **no SMTP server, password, or app-password** to manage. The popup
  path is pure standard library (`ctypes` → `MessageBoxW`) — zero dependencies.
- **Data refresh is best-effort, never fatal.** If BCCR is unreachable (VPN,
  outage, weekend) the run logs a warning and uses the last local CSV; the
  "As of last data" line makes any staleness obvious. Rows are *upserted* into
  `monex_clean.csv`, so the deep history the backtests need is preserved.
- **Scheduling.** A per-user Task Scheduler job, weekdays only (MONEX is closed
  weekends), `-StartWhenAvailable` so a sleeping laptop catches up.
- **Timing.** Costa Rica is **UTC-6 all year** (no DST). MONEX trades
  ~09:00–16:00 CST; BCCR posts the session shortly after. Default fire time is
  **16:45**, which gives you the next session's marching orders with today's
  data already in. ⚠️ Task Scheduler uses the *work computer's* local clock — if
  that machine isn't on CR time, set `-Time` to the CR-close equivalent in local
  time (see the PowerShell script's header).

## Setup (on the Windows work computer)

```powershell
cd projects\03-usdcrc-analysis
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r production\requirements-prod.txt
# for the Outlook channel also:  pip install pywin32
# for the PNG card also:         pip install matplotlib

# try it once (writes out\daily_signal.txt and pops/sends per config):
python production\run_notifier.py --channel popup

# schedule it (weekdays 16:45 local):
cd production
powershell -ExecutionPolicy Bypass -File .\Register-DailySignalTask.ps1
# e.g. Outlook e-mail, machine on Colombia time (17:45 = 16:45 CR):
#   .\Register-DailySignalTask.ps1 -Time 17:45 -Channel outlook
```

Run the task on demand to verify: `Start-ScheduledTask -TaskName "USDCRC Daily Signal"`.
Remove it with `Unregister-ScheduledTask -TaskName "USDCRC Daily Signal" -Confirm:$false`.

## Configuration

Edit `config.py`, or set an environment variable (same name, `USDCRC_` prefix) —
handy for giving the scheduled task its own settings without touching code:

| Setting | Default | Meaning |
|---------|---------|---------|
| `NOTIFY_CHANNEL` | `auto` | `auto` / `outlook` / `popup` / `both` / `none` |
| `EMAIL_TO` | `""` (self) | recipient; blank = your own Outlook account |
| `EMAIL_SUBJECT_PREFIX` | `USD/CRC signal` | subject line prefix |
| `ATTACH_PNG` | `true` | render + attach the PNG card (needs matplotlib) |
| `REFRESH_BEFORE_RUN` | `true` | pull fresh MONEX data before computing |
| `REFRESH_LOOKBACK_DAYS` | `45` | how many days back to re-fetch and upsert |
| `HTTP_TIMEOUT_SECONDS` | `60` | BCCR download timeout |
| `MONEX_EXPORT_URL` | BCCR CodCuadro=770 | the export template (`{start}`/`{end}`) |

Example command-line overrides:

```bat
python production\run_notifier.py --no-refresh          REM skip the BCCR fetch
python production\run_notifier.py --channel both        REM email AND popup
python production\run_notifier.py 2026-06-10            REM signal for a specific date
```

## What one run looks like

```
============================================================
  USD/CRC DAILY SIGNAL  -  real calendar quincena (recommended)
============================================================
  Session date     : 2026-07-22  (Wednesday)
  Days to deadline : -5 bd  ->  5 business day(s) past the deadline (supply cleared)

  POSITION         : LONG USD
  Size             : 0.5x  (half - volume regime disagrees)
  Notional         : USD 500,000  (of USD 1,000,000 full)

  Slow 20d volume  : 1.63x normal  (above normal (supply present))
  As of last data  : 2026-07-21  (close 452.60, VWAP 452.62)
============================================================
```

Every run appends one line to `logs/notifier.log`; failures are logged with a
traceback there, never swallowed.
