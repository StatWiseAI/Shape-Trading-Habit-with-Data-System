# Trading Analytics Platform

A professional, broker-agnostic trade analytics pipeline built for serious
quantitative traders.  Upload any CSV or XLSX export from your broker and
get a structured statistical summary — instantly, with no configuration.

---

## Features

- **Auto-detects 10+ brokers** — NinjaTrader / Topstep, Interactive Brokers,
  Tradovate, TradeStation, Tastytrade, MetaTrader 4/5, Rithmic / Apex,
  Binance, Kraken, Webull
- **Generic enrichment** — no hardcoded instruments, no timezone assumptions
- **Descriptive statistics** — expectancy, profit factor, Sharpe, max drawdown,
  hold-time asymmetry, fee drag, streak analysis
- **JSON summary payload** — paste into Claude for deep statistical dialogue
- **Streamlit-ready** — built to be deployed as a web app (Step 3)
- **Fully tested** — pytest suite with 40+ assertions

---

## Repository structure

```
trading-platform/
│
├── pipeline.py               CLI entry point
├── requirements.txt
├── README.md
│
├── core/
│   ├── __init__.py
│   ├── ingest.py             File loading + broker detection
│   ├── etl.py                Feature engineering
│   ├── descriptive.py        Step 2a — descriptive statistics
│   └── adapters/
│       ├── __init__.py
│       └── brokers.py        Broker fingerprints + column remapping
│
├── tests/
│   └── test_pipeline.py      Full pytest suite
│
└── data/                     Put your export files here (git-ignored)
    └── .gitkeep
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/trading-platform.git
cd trading-platform
pip install -r requirements.txt
```

### 2. Run the pipeline

```bash
python pipeline.py --input data/trades_export.csv
```

Outputs written next to the input file:
- `clean_trades.csv`  — enriched trade-level DataFrame (28+ columns)
- `summary.json`      — analytical payload

Optional flags:

```bash
# Write outputs to a specific directory
python pipeline.py --input data/trades_export.csv --out-dir results/

# Print the full JSON summary to stdout
python pipeline.py --input data/trades_export.csv --print-summary
```

### 3. Run the test suite

```bash
pytest tests/ -v
```

With coverage:

```bash
pytest tests/ -v --cov=core --cov-report=term-missing
```

---

## Canonical schema

All broker exports are remapped to these internal column names before any
analysis runs.  If your broker is not yet supported, rename your columns to
match this schema and the pipeline will work without modification.

| Column          | Type    | Description                                  |
|-----------------|---------|----------------------------------------------|
| `instrument`    | str     | Ticker / contract name / trading pair        |
| `entered_at`    | str     | Entry timestamp (any format pandas can parse)|
| `exited_at`     | str     | Exit timestamp (optional)                    |
| `entry_price`   | float   | Entry fill price                             |
| `exit_price`    | float   | Exit fill price (optional)                   |
| `pnl_gross`     | float   | Gross PnL before costs                       |
| `fees`          | float   | Exchange / routing fees                      |
| `commissions`   | float   | Broker commissions                           |
| `size`          | float   | Contracts / shares / units                   |
| `direction`     | str     | `"Long"` or `"Short"`                        |
| `trade_id`      | any     | Unique trade identifier (optional)           |
| `duration_str`  | str     | Raw duration string (optional)               |

---

## Derived columns added by the pipeline

| Column             | Description                                           |
|--------------------|-------------------------------------------------------|
| `entered_at_dt`    | Parsed entry datetime                                 |
| `exited_at_dt`     | Parsed exit datetime                                  |
| `trade_date`       | Entry date                                            |
| `entry_hour`       | Integer hour of entry (0–23)                          |
| `hour_bin`         | 2-hour bucket label, e.g. `"08-10"`, `"14-16"`       |
| `day_of_week`      | 0 = Monday … 4 = Friday                              |
| `day_name`         | `"Monday"` … `"Friday"`                              |
| `duration_seconds` | Hold time in seconds                                  |
| `duration_minutes` | Hold time in minutes                                  |
| `net_pnl`          | `pnl_gross − fees − commissions`                     |
| `outcome`          | `"win"` / `"loss"` / `"breakeven"`                   |
| `trade_index`      | Chronological rank (1-based)                         |
| `pnl_per_minute`   | `net_pnl / duration_minutes` — efficiency proxy      |

---

## Adding a new broker

Edit `core/adapters/brokers.py` and add an entry to `BROKER_PROFILES`:

```python
{
    "broker": "MyBroker",
    "fingerprint": {
        # A set of column names that uniquely identifies this export.
        # Include at least 6-8 columns.  Case-insensitive matching is used.
        "Symbol", "EntryTime", "ExitTime", "Side",
        "FilledQty", "AvgPrice", "RealizedPnL", "Fee",
    },
    "column_map": {
        # source column  →  canonical column
        "Symbol":       "instrument",
        "EntryTime":    "entered_at",
        "ExitTime":     "exited_at",
        "AvgPrice":     "entry_price",
        "FilledQty":    "size",
        "Side":         "direction",
        "Fee":          "fees",
        "RealizedPnL":  "pnl_gross",
    },
    "direction_map": {"BUY": "Long", "SELL": "Short"},
    "date_format": None,    # None = let pandas infer
    "notes": "MyBroker trade history export.",
},
```

Then add a test case in `tests/test_pipeline.py` following the existing pattern.

---

## Roadmap

| Step | Status      | Description                                        |
|------|-------------|----------------------------------------------------|
| 1    | ✅ Complete | ETL — ingest, broker detection, enrichment         |
| 2a   | ✅ Complete | Descriptive statistics + JSON payload              |
| 2b   | Planned     | Inferential layer — bootstrap CI, KS test, Ljung-Box, CUSUM |
| 2c   | Planned     | Regression — GLM, quantile regression, survival analysis |
| 3    | Planned     | Streamlit app — file upload UI, charts, summary    |
| 4    | Planned     | Claude integration — paste JSON payload            |
| 5    | Planned     | PDF / HTML report export                           |

---

## Deployment (Streamlit Community Cloud)

1. Push the repo to GitHub (ensure `data/` is in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo, set main file to `app.py`
4. Deploy — Streamlit reads `requirements.txt` automatically

---

## License

MIT
