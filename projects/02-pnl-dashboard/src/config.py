"""
Central configuration: filesystem paths and logging setup.

All modules import paths from here rather than hard-coding them, so the data
location can be overridden in one place (e.g. for tests via PNL_DATA_DIR).
"""
import logging
import os
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Allow overriding the data directory (used by tests and alternate deployments).
DATA_DIR = Path(os.environ.get("PNL_DATA_DIR", PROJECT_ROOT / "data"))
TEMPLATES_DIR = PROJECT_ROOT / "templates"

TRADES_PATH = DATA_DIR / "trades.csv"
INITIAL_POSITIONS_PATH = DATA_DIR / "initial_positions.csv"
BONDS_STATIC_PATH = DATA_DIR / "bonds_static.csv"
PORTFOLIO_PATH = DATA_DIR / "portfolio.csv"
PRICE_HISTORY_PATH = DATA_DIR / "price_history.csv"
PNL_HISTORY_PATH = DATA_DIR / "pnl_history.csv"
PRICES_DIR = DATA_DIR / "prices"
MANUAL_PRICES_PATH = PRICES_DIR / "manual_prices.csv"
MANUAL_HISTORY_PATH = PRICES_DIR / "manual_price_history.csv"
BLOOMBERG_TEMPLATE_PATH = TEMPLATES_DIR / "bloomberg_prices.xlsx"

# Timestamped copies of any file the UI overwrites are kept here.
BACKUPS_DIR = DATA_DIR / "backups"

# Sentinel portfolio name used when no portfolio filter is applied.
ALL_PORTFOLIOS = "all"
DEFAULT_PORTFOLIO = "default"


# ── logging ─────────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Return a module logger with a sane default handler/format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(os.environ.get("PNL_LOG_LEVEL", "INFO"))
        logger.propagate = False
    return logger
