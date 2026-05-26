"""
core/etl.py
===========
Feature engineering on the canonical DataFrame produced by ingest.py.

Design principles
-----------------
- Zero instrument knowledge  — no hardcoded tickers, contract specs, or exchanges.
- Zero timezone assumptions  — timestamps are stored in whatever tz the broker
  exported; we only extract relative features (hour, weekday) from them.
- Session tagging is REPLACED by hour-of-day bins, which are universal.
- duration_seconds is computed from (exited_at - entered_at) when a raw
  duration_str column is absent or unparseable.

New columns added by enrich()
------------------------------
  entered_at_dt   : parsed datetime (tz-aware if source had tz, else naive)
  exited_at_dt    : parsed datetime (optional — NaT if absent)
  trade_date      : date of entry
  entry_hour      : integer 0–23 (local to whatever tz is in the timestamp)
  hour_bin        : string label  e.g. "06-08", "08-10" … in 2-hour buckets
  day_of_week     : 0=Mon … 4=Fri
  day_name        : "Monday" … "Friday"
  duration_seconds: float (NaN if neither duration_str nor exited_at available)
  duration_minutes: float
  net_pnl         : pnl_gross - fees - commissions
  outcome         : "win" | "loss" | "breakeven"
  trade_index     : chronological integer rank (1-based)
  pnl_per_minute  : net_pnl / duration_minutes  (efficiency proxy)
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# TIMESTAMP PARSING
# ─────────────────────────────────────────────────────────────────────────────

def _parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse entered_at / exited_at strings into datetime objects.
    Compatible with pandas 2.x and 3.x (infer_datetime_format removed in 3.0).
    utc=False preserves the original offset so hour-of-day features reflect
    the trader's local time.
    """
    for raw_col, dt_col in [("entered_at", "entered_at_dt"),
                             ("exited_at",  "exited_at_dt")]:
        if raw_col not in df.columns or df[raw_col].isna().all():
            df[dt_col] = pd.NaT
            continue
        try:
            df[dt_col] = pd.to_datetime(df[raw_col], utc=False, errors="coerce")
        except Exception:
            df[dt_col] = pd.NaT

    return df


# ─────────────────────────────────────────────────────────────────────────────
# DURATION PARSING
# ─────────────────────────────────────────────────────────────────────────────

_DURATION_RE = re.compile(
    r"(?P<h>\d+):(?P<m>\d+):(?P<s>\d+)(?:\.(?P<f>\d+))?"
)


def _parse_duration_str(s: str) -> float:
    """'HH:MM:SS[.fffffff]'  →  total seconds (float).  NaN on failure.

    NinjaTrader exports fractional seconds as a 7-digit field representing
    100-nanosecond ticks (i.e. value / 1e7 = fractional seconds).
    For all other formats the field is treated as plain decimal seconds
    (padded/truncated to the number of digits present).
    """
    try:
        m = _DURATION_RE.match(str(s))
        if not m:
            return np.nan
        h  = int(m.group("h"))
        mn = int(m.group("m"))
        sc = int(m.group("s"))
        frac_str = m.group("f") or "0"
        # 7-digit fractional part → NinjaTrader 100-ns ticks
        if len(frac_str) == 7:
            frac = int(frac_str) / 1e7
        else:
            frac = int(frac_str) / (10 ** len(frac_str))
        return h * 3600 + mn * 60 + sc + frac
    except Exception:
        return np.nan


def _compute_duration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Populate duration_seconds from duration_str if present, otherwise
    derive it from (exited_at_dt - entered_at_dt).
    """
    if "duration_str" in df.columns and not df["duration_str"].isna().all():
        df["duration_seconds"] = df["duration_str"].apply(_parse_duration_str)
        # Fill any NaN rows from timestamps if we have them
        mask = df["duration_seconds"].isna()
    else:
        df["duration_seconds"] = np.nan
        mask = pd.Series(True, index=df.index)

    # Fill from timestamp difference where needed
    if mask.any() and "entered_at_dt" in df.columns and "exited_at_dt" in df.columns:
        delta = (df.loc[mask, "exited_at_dt"] - df.loc[mask, "entered_at_dt"])
        df.loc[mask, "duration_seconds"] = delta.dt.total_seconds()

    df["duration_minutes"] = df["duration_seconds"] / 60.0
    return df


# ─────────────────────────────────────────────────────────────────────────────
# HOUR BINS  (universal session proxy)
# ─────────────────────────────────────────────────────────────────────────────

def _hour_bin(hour: int, bin_size: int = 2) -> str:
    """Map integer hour → zero-padded label, e.g. 9 → '08-10'."""
    lo = (hour // bin_size) * bin_size
    hi = lo + bin_size
    return f"{lo:02d}-{hi:02d}"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENRICHMENT FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take the canonical DataFrame from ingest.py and add all derived features.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ingest.IngestResult.df  (canonical column names).

    Returns
    -------
    pd.DataFrame
        Original columns preserved + new feature columns appended.
        Sorted chronologically by entered_at_dt.
    """
    df = df.copy()

    # ── 1. Parse timestamps ──────────────────────────────────────────────────
    df = _parse_timestamps(df)

    # ── 2. Sort chronologically ──────────────────────────────────────────────
    if "entered_at_dt" in df.columns and not df["entered_at_dt"].isna().all():
        try:
            sort_key = df["entered_at_dt"].dt.tz_convert("UTC")
        except TypeError:
            sort_key = df["entered_at_dt"]
        df = df.assign(_sort_key=sort_key).sort_values(
            "_sort_key", na_position="last"
        ).drop(columns=["_sort_key"]).reset_index(drop=True)

    # ── 3. Temporal features ─────────────────────────────────────────────────
    if "entered_at_dt" in df.columns and not df["entered_at_dt"].isna().all():
        ts = df["entered_at_dt"]
        # Normalise to tz-naive UTC for consistent .dt accessor in pandas 3.x
        try:
            ts_naive = ts.dt.tz_convert("UTC").dt.tz_localize(None)
        except TypeError:
            ts_naive = ts  # already tz-naive

        df["trade_date"]  = ts_naive.dt.date
        df["entry_hour"]  = ts_naive.dt.hour
        df["day_of_week"] = ts_naive.dt.dayofweek
        df["day_name"]    = ts_naive.dt.day_name()
        df["hour_bin"]    = df["entry_hour"].apply(_hour_bin)
    else:
        for col in ("trade_date", "entry_hour", "day_of_week",
                    "day_name", "hour_bin"):
            df[col] = np.nan

    # ── 4. Duration ──────────────────────────────────────────────────────────
    df = _compute_duration(df)

    # ── 5. Net PnL ───────────────────────────────────────────────────────────
    df["net_pnl"] = (
        pd.to_numeric(df["pnl_gross"],    errors="coerce").fillna(0)
        - pd.to_numeric(df["fees"],       errors="coerce").fillna(0)
        - pd.to_numeric(df["commissions"],errors="coerce").fillna(0)
    )

    # ── 6. Outcome ───────────────────────────────────────────────────────────
    df["outcome"] = np.where(
        df["net_pnl"] > 1e-9,  "win",
        np.where(df["net_pnl"] < -1e-9, "loss", "breakeven")
    )

    # ── 7. Chronological index ───────────────────────────────────────────────
    df["trade_index"] = np.arange(1, len(df) + 1)

    # ── 8. Efficiency metric ─────────────────────────────────────────────────
    df["pnl_per_minute"] = (
        df["net_pnl"] / df["duration_minutes"].replace(0, np.nan)
    )

    return df
