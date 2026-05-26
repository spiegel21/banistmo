# banistmo

Personal finance & trading projects. Each project is self-contained under `projects/`.

## Projects

| # | Project | Description | Stack |
|---|---------|-------------|-------|
| 01 | [forex-strategy](projects/01-forex-strategy/) | EUR/USD moving-average crossover strategy with sentiment overlay | yfinance, pandas, matplotlib, VADER |
| 02 | [pnl-dashboard](projects/02-pnl-dashboard/) | Fixed-income P&L dashboard: positions, accruals, MTM, Bloomberg pricing | pandas, streamlit, xlwings |

## Structure

```
banistmo/
├── projects/
│   ├── 01-forex-strategy/   each project is fully self-contained
│   └── 02-pnl-dashboard/
├── README.md
├── CLAUDE.md
└── .gitignore
```

## Setup

Each project has its own `requirements.txt`. Create a virtual environment per project:

```bash
cd projects/<project-name>
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
