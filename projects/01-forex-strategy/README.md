# 01 — EUR/USD Forex Strategy

Moving-average crossover strategy on EUR/USD with sentiment analysis overlay. 20 years of historical data (2005–2025).

## Pipeline

```
plot.py          download EURUSD data + compute MA/RSI/MACD indicators
auto_trade.py    MA crossover signals + scipy parameter optimisation
email_alert.py   Gmail SMTP alerts on BUY/SELL signals
sentiment.py     FXStreet headline scraping + VADER/TextBlob sentiment
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/plot.py          # writes data/eurusd_data_with_indicators.csv
python src/auto_trade.py    # writes out/strategy_results.csv
python src/sentiment.py     # writes out/forex_sentiment_results.csv
```

## Results

Optimised parameters: MA41 / MA257 crossover.

| Metric | Value |
|--------|-------|
| Sharpe ratio | see out/strategy_results.csv |
| Max drawdown | see out/strategy_results.csv |

## Data

- `data/eurusd_data_with_indicators.csv` — 5,190 daily bars with indicators
- `out/strategy_results.csv` — full backtest with signals and returns
- `out/forex_sentiment_results.csv` — 18 FXStreet headlines with sentiment scores
- `out/trading_implications.txt` — strategy recommendation text
