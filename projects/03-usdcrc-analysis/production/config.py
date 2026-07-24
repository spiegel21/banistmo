"""Configuration for the USD/CRC daily-signal production notifier.

Every knob here can be overridden by an environment variable (same name,
prefixed ``USDCRC_``) so you can tune a scheduled task without editing code —
e.g. ``USDCRC_NOTIFY_CHANNEL=popup``. Booleans accept 1/0/true/false/yes/no.

Only the two obvious ones usually need touching: ``NOTIFY_CHANNEL`` and, if you
use e-mail, ``EMAIL_TO``.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- paths ------------------------------------------------------------------ #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "out"
LOG_DIR = PROJECT_ROOT / "production" / "logs"


def _env(name: str, default: str) -> str:
    return os.environ.get(f"USDCRC_{name}", default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(f"USDCRC_{name}")
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


# --- notification ----------------------------------------------------------- #
# How to deliver the signal:
#   "auto"    -> try Outlook, fall back to a popup if Outlook is unavailable (recommended)
#   "outlook" -> Outlook e-mail only (win32com); errors if it fails
#   "popup"   -> native Windows message box only (zero extra dependencies)
#   "both"    -> Outlook e-mail AND a popup
#   "none"    -> write artifacts/log only, no notification (useful for a dry run)
NOTIFY_CHANNEL = _env("NOTIFY_CHANNEL", "auto")

# E-mail settings (only used when the channel involves Outlook).
# Leave EMAIL_TO blank to send the mail to yourself (the logged-in Outlook account).
EMAIL_TO = _env("EMAIL_TO", "")
EMAIL_SUBJECT_PREFIX = _env("EMAIL_SUBJECT_PREFIX", "USD/CRC signal")

# Render the one-page PNG and attach it to the e-mail / save it to out/.
# Requires matplotlib; if it is not installed we silently skip the PNG.
ATTACH_PNG = _env_bool("ATTACH_PNG", True)

# --- data refresh ----------------------------------------------------------- #
# Pull the latest MONEX session from BCCR before computing the signal.
# If the fetch fails (offline, VPN, BCCR down, weekend) we fall back to the last
# local data/monex_clean.csv and clearly label the "as of" date — the calendar
# direction is still correct; only the size multiplier may be one day stale.
REFRESH_BEFORE_RUN = _env_bool("REFRESH_BEFORE_RUN", True)

# BCCR "Resumen de negociación en MONEX" export (CodCuadro=770): the exact
# HTML-as-XLS matrix parse_monex.py reads. {start}/{end} are YYYY/MM/DD.
MONEX_EXPORT_URL = _env(
    "MONEX_EXPORT_URL",
    "https://gee.bccr.fi.cr/indicadoreseconomicos/Cuadros/frmVerCatCuadro.aspx"
    "?CodCuadro=770&Idioma=1&FecInicial={start}&FecFinal={end}&Filtro=0&Exportar=True",
)
# How many calendar days back to re-fetch and upsert on each run. A short window
# keeps the download tiny while still refreshing any late-revised recent rows;
# the deep history in monex_clean.csv is preserved (rows are merged, not replaced).
REFRESH_LOOKBACK_DAYS = int(_env("REFRESH_LOOKBACK_DAYS", "45"))
HTTP_TIMEOUT_SECONDS = int(_env("HTTP_TIMEOUT_SECONDS", "60"))
