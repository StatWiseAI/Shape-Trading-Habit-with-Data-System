"""
core/adapters/brokers.py
========================
Broker auto-detection and column remapping.

Design principles
-----------------
- Zero hardcoded instrument knowledge. This layer only cares about column names.
- NinjaTrader is the canonical reference schema; all other brokers remap to it.
- Matching is case-insensitive and whitespace-tolerant.
- Unmapped columns are preserved with a  raw__  prefix so no data is lost.
- If no broker profile reaches the confidence threshold, a manual-mapping
  fallback is returned so the caller can prompt the user.

Canonical schema (internal names used throughout the pipeline)
--------------------------------------------------------------
  trade_id        str / int   unique identifier (optional)
  instrument      str         ticker, contract, pair, symbol
  entered_at      str         raw entry timestamp (parsed in etl.py)
  exited_at       str         raw exit  timestamp (parsed in etl.py, optional)
  entry_price     float
  exit_price      float       (optional — not all brokers export this)
  fees            float       exchange / routing fees
  commissions     float       broker commissions
  pnl_gross       float       gross PnL before costs
  size            float       contracts / shares / units / lots
  direction       str         "Long" | "Short"
  duration_str    str         raw duration string (optional)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# BROKER PROFILES
# ─────────────────────────────────────────────────────────────────────────────

BROKER_PROFILES: list[dict] = [

    # ── NinjaTrader 8 (Topstep / AMP / others) ───────────────────────────────
    {
        "broker": "NinjaTrader",
        "fingerprint": {
            "Id", "ContractName", "EnteredAt", "ExitedAt",
            "EntryPrice", "ExitPrice", "Fees", "PnL",
            "Size", "Type", "TradeDuration", "Commissions",
        },
        "column_map": {
            "Id":            "trade_id",
            "ContractName":  "instrument",
            "EnteredAt":     "entered_at",
            "ExitedAt":      "exited_at",
            "EntryPrice":    "entry_price",
            "ExitPrice":     "exit_price",
            "Fees":          "fees",
            "PnL":           "pnl_gross",
            "Size":          "size",
            "Type":          "direction",
            "TradeDuration": "duration_str",
            "Commissions":   "commissions",
        },
        "direction_map": {"Long": "Long", "Short": "Short"},
        "date_format": None,
        "notes": "NinjaTrader 8 Account Performance → Trades export. "
                 "Used by Topstep, Apex, AMP, and most futures prop firms.",
    },

    # ── Interactive Brokers — Flex Query (Trades section) ────────────────────
    {
        "broker": "InteractiveBrokers",
        "fingerprint": {
            "TradeID", "Symbol", "DateTime", "OpenDateTime",
            "Buy/Sell", "Quantity", "TradePrice",
            "ClosePrice", "FifoPnlRealized", "CommissionUSD",
        },
        "column_map": {
            "TradeID":         "trade_id",
            "Symbol":          "instrument",
            "OpenDateTime":    "entered_at",
            "DateTime":        "exited_at",
            "TradePrice":      "entry_price",
            "ClosePrice":      "exit_price",
            "CommissionUSD":   "commissions",
            "FifoPnlRealized": "pnl_gross",
            "Quantity":        "size",
            "Buy/Sell":        "direction",
        },
        "direction_map": {"BUY": "Long", "SELL": "Short"},
        "date_format": None,
        "notes": "Interactive Brokers Flex Query — Trades section.",
    },

    # ── Tradovate ─────────────────────────────────────────────────────────────
    {
        "broker": "Tradovate",
        "fingerprint": {
            "id", "name", "timestamp", "action",
            "qty", "price", "profit", "fees",
        },
        "column_map": {
            "id":        "trade_id",
            "name":      "instrument",
            "timestamp": "entered_at",
            "price":     "entry_price",
            "profit":    "pnl_gross",
            "qty":       "size",
            "action":    "direction",
            "fees":      "fees",
        },
        "direction_map": {"Buy": "Long", "Sell": "Short"},
        "date_format": None,
        "notes": "Tradovate Orders report CSV.",
    },

    # ── TradeStation ─────────────────────────────────────────────────────────
    {
        "broker": "TradeStation",
        "fingerprint": {
            "TradeNumber", "Symbol", "Account",
            "EntryDate", "ExitDate", "EntryPrice", "ExitPrice",
            "TradeType", "Contracts/Shares", "ProfitLoss",
        },
        "column_map": {
            "TradeNumber":      "trade_id",
            "Symbol":           "instrument",
            "EntryDate":        "entered_at",
            "ExitDate":         "exited_at",
            "EntryPrice":       "entry_price",
            "ExitPrice":        "exit_price",
            "ProfitLoss":       "pnl_gross",
            "Contracts/Shares": "size",
            "TradeType":        "direction",
            "Commission":       "commissions",
        },
        "direction_map": {"Long": "Long", "Short": "Short"},
        "date_format": "%m/%d/%Y %H:%M:%S",
        "notes": "TradeStation Strategy Performance Report — trade list.",
    },

    # ── Tastytrade ────────────────────────────────────────────────────────────
    {
        "broker": "Tastytrade",
        "fingerprint": {
            "Date", "Type", "Action", "Symbol",
            "Instrument Type", "Quantity", "Average Price",
            "Fees", "Value", "Net Liquidating Value Effect",
        },
        "column_map": {
            "Date":                         "entered_at",
            "Symbol":                       "instrument",
            "Average Price":                "entry_price",
            "Quantity":                     "size",
            "Action":                       "direction",
            "Fees":                         "fees",
            "Net Liquidating Value Effect": "pnl_gross",
        },
        "direction_map": {
            "BUY_TO_OPEN":  "Long",  "SELL_TO_OPEN":  "Short",
            "BUY_TO_CLOSE": "Long",  "SELL_TO_CLOSE": "Short",
        },
        "date_format": None,
        "notes": "Tastytrade transaction history export.",
    },

    # ── MetaTrader 4 / 5 ─────────────────────────────────────────────────────
    {
        "broker": "MetaTrader",
        "fingerprint": {
            "Ticket", "Open Time", "Close Time", "Type",
            "Size", "Item", "Open Price", "Close Price",
            "Commission", "Swap", "Profit",
        },
        "column_map": {
            "Ticket":      "trade_id",
            "Item":        "instrument",
            "Open Time":   "entered_at",
            "Close Time":  "exited_at",
            "Open Price":  "entry_price",
            "Close Price": "exit_price",
            "Size":        "size",
            "Type":        "direction",
            "Commission":  "commissions",
            "Swap":        "fees",
            "Profit":      "pnl_gross",
        },
        "direction_map": {
            "buy": "Long", "sell": "Short",
            "0":   "Long", "1":    "Short",
        },
        "date_format": "%Y.%m.%d %H:%M:%S",
        "notes": "MetaTrader 4/5 account statement (HTML→CSV or direct CSV).",
    },

    # ── Rithmic / Apex Trader Funding ────────────────────────────────────────
    {
        "broker": "Rithmic",
        "fingerprint": {
            "Account", "Buy/Sell", "Qty", "Symbol",
            "Entry Price", "Exit Price", "Entry Time",
            "Exit Time", "P&L", "Commission",
        },
        "column_map": {
            "Symbol":       "instrument",
            "Entry Time":   "entered_at",
            "Exit Time":    "exited_at",
            "Entry Price":  "entry_price",
            "Exit Price":   "exit_price",
            "Qty":          "size",
            "Buy/Sell":     "direction",
            "Commission":   "commissions",
            "P&L":          "pnl_gross",
        },
        "direction_map": {"Buy": "Long", "Sell": "Short"},
        "date_format": None,
        "notes": "Rithmic R|Trader Pro / Apex Trader Funding export.",
    },

    # ── Binance (USDⓈ-M futures trade history) ───────────────────────────────
    {
        "broker": "Binance",
        "fingerprint": {
            "Order ID", "Trade ID", "Symbol", "Side",
            "Price", "Executed Qty", "Commission",
            "Commission Asset", "Time",
        },
        "column_map": {
            "Order ID":     "trade_id",
            "Symbol":       "instrument",
            "Time":         "entered_at",
            "Price":        "entry_price",
            "Executed Qty": "size",
            "Side":         "direction",
            "Commission":   "commissions",
            "Realized PnL": "pnl_gross",
        },
        "direction_map": {"BUY": "Long", "SELL": "Short"},
        "date_format": None,
        "notes": "Binance USDⓈ-M futures trade history CSV.",
    },

    # ── Kraken ────────────────────────────────────────────────────────────────
    {
        "broker": "Kraken",
        "fingerprint": {
            "txid", "ordertxid", "pair", "time",
            "type", "ordertype", "price", "cost", "fee", "vol",
        },
        "column_map": {
            "txid":  "trade_id",
            "pair":  "instrument",
            "time":  "entered_at",
            "price": "entry_price",
            "vol":   "size",
            "type":  "direction",
            "fee":   "fees",
        },
        "direction_map": {"buy": "Long", "sell": "Short"},
        "date_format": None,
        "notes": "Kraken trades/ledger export.",
    },

    # ── Webull ────────────────────────────────────────────────────────────────
    {
        "broker": "Webull",
        "fingerprint": {
            "Order ID", "Symbol", "Side",
            "Filled Time", "Avg Price", "Filled Qty", "Commission",
        },
        "column_map": {
            "Order ID":    "trade_id",
            "Symbol":      "instrument",
            "Filled Time": "entered_at",
            "Avg Price":   "entry_price",
            "Filled Qty":  "size",
            "Side":        "direction",
            "Commission":  "commissions",
        },
        "direction_map": {"BUY": "Long", "SELL": "Short"},
        "date_format": None,
        "notes": "Webull order history export.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# DETECTION
# ─────────────────────────────────────────────────────────────────────────────

MIN_MATCH_THRESHOLD = 0.60   # fraction of fingerprint cols that must be present


def detect_broker(columns: list[str]) -> tuple[dict | None, float]:
    """
    Score every profile against the actual file columns.

    Returns
    -------
    (best_profile, confidence)
        best_profile : dict or None if nothing clears the threshold
        confidence   : float in [0, 1]

    Matching is case-insensitive and strips leading/trailing whitespace.
    """
    normalised = {c.strip().lower() for c in columns}
    best_score   = 0.0
    best_profile = None

    for profile in BROKER_PROFILES:
        fp_norm = {c.strip().lower() for c in profile["fingerprint"]}
        score   = sum(1 for c in fp_norm if c in normalised) / len(fp_norm)
        if score > best_score:
            best_score   = score
            best_profile = profile

    if best_score >= MIN_MATCH_THRESHOLD:
        return best_profile, round(best_score, 4)
    return None, round(best_score, 4)


# ─────────────────────────────────────────────────────────────────────────────
# REMAPPING
# ─────────────────────────────────────────────────────────────────────────────

CANONICAL_REQUIRED = ["instrument", "entered_at", "entry_price", "pnl_gross",
                      "size", "direction"]
CANONICAL_OPTIONAL = ["trade_id", "exited_at", "exit_price",
                      "fees", "commissions", "duration_str"]
ALL_CANONICAL      = CANONICAL_REQUIRED + CANONICAL_OPTIONAL


def remap_columns(df: pd.DataFrame, profile: dict) -> pd.DataFrame:
    """
    Apply broker column_map to df.

    Rules
    -----
    - Mapped columns      → renamed to canonical name
    - Unmapped columns    → kept as  raw__<original>  (no data lost)
    - Missing canonicals  → filled with NaN
    - direction values    → normalised to "Long" / "Short"
    - fees / commissions  → coerced to float, NaN → 0.0
    """
    col_map       = profile["column_map"]
    direction_map = profile.get("direction_map", {})

    # Build case-insensitive lookup: lowercase → actual column name in df
    col_lower = {c.strip().lower(): c for c in df.columns}

    rename: dict[str, str] = {}
    mapped_actual: set[str] = set()

    for src, tgt in col_map.items():
        actual = col_lower.get(src.strip().lower())
        if actual is not None:
            rename[actual] = tgt
            mapped_actual.add(actual)

    # Unmapped columns → raw__ prefix
    for col in df.columns:
        if col not in mapped_actual and col not in rename.values():
            rename[col] = f"raw__{col}"

    df = df.rename(columns=rename)

    # Normalise direction
    if "direction" in df.columns and direction_map:
        df["direction"] = (
            df["direction"]
            .astype(str)
            .str.strip()
            .map(lambda v: direction_map.get(v, direction_map.get(v.upper(), v)))
        )

    # Ensure all canonical columns exist
    for col in ALL_CANONICAL:
        if col not in df.columns:
            df[col] = np.nan

    # Coerce cost columns
    for cost_col in ("fees", "commissions"):
        df[cost_col] = pd.to_numeric(df[cost_col], errors="coerce").fillna(0.0)

    return df
