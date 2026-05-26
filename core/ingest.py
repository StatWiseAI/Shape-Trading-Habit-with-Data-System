"""
core/ingest.py
==============
File ingestion layer.

Responsibilities
----------------
1. Accept a file path (CSV or XLSX) and load it into a raw DataFrame.
2. Auto-detect the broker by fingerprinting the column names.
3. Remap columns to the canonical schema via the broker adapter.
4. Return a clean IngestResult so the rest of the pipeline never
   needs to know which broker the data came from.

Public API
----------
    result = ingest(path)
    result.df          # canonical DataFrame, ready for etl.py
    result.broker      # detected broker name (str) or "Unknown"
    result.confidence  # float  0–1
    result.warnings    # list[str]  — surface these to the user
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from core.adapters.brokers import detect_broker, remap_columns, ALL_CANONICAL


# ─────────────────────────────────────────────────────────────────────────────
# RESULT CONTAINER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IngestResult:
    df:         pd.DataFrame
    broker:     str
    confidence: float
    warnings:   list[str] = field(default_factory=list)

    def ok(self) -> bool:
        """True if the DataFrame has the minimum required canonical columns."""
        required = {"instrument", "entered_at", "entry_price",
                    "pnl_gross", "size", "direction"}
        present  = set(self.df.columns)
        return required.issubset(present)

    def missing_required(self) -> list[str]:
        required = ["instrument", "entered_at", "entry_price",
                    "pnl_gross", "size", "direction"]
        return [c for c in required if c not in self.df.columns
                or self.df[c].isna().all()]


# ─────────────────────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_csv(path: Path) -> pd.DataFrame:
    """
    Load CSV robustly:
    - Try UTF-8 first, fall back to latin-1 (handles Windows exports).
    - Skip leading blank rows (some broker exports have header metadata
      before the actual column row).
    """
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=encoding, skip_blank_lines=True)
            # Drop rows where every value is NaN (can appear after metadata rows)
            df = df.dropna(how="all").reset_index(drop=True)
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode file: {path}. "
                     "Try saving the file as UTF-8 CSV.")


def _load_xlsx(path: Path) -> pd.DataFrame:
    """
    Load the first non-empty sheet of an XLSX workbook.
    Skips fully-blank leading rows (same broker-metadata problem as CSV).
    Note: skip_blank_lines is not supported by ExcelFile.parse in pandas 2.x,
    so we drop all-NaN rows manually after loading.
    """
    xl = pd.ExcelFile(path, engine="openpyxl")
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        df = df.dropna(how="all").reset_index(drop=True)
        if len(df) > 0 and len(df.columns) > 2:
            return df
    raise ValueError(f"No usable sheet found in {path.name}.")


def _load_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_csv(path)
    elif suffix in (".xlsx", ".xls", ".xlsm"):
        return _load_xlsx(path)
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            "Please upload a .csv or .xlsx file."
        )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def ingest(path: str | Path) -> IngestResult:
    """
    Load a trade export file, detect the broker, and return a canonical
    DataFrame wrapped in an IngestResult.

    Parameters
    ----------
    path : str or Path
        Absolute or relative path to a .csv / .xlsx file.

    Returns
    -------
    IngestResult
        .df          — canonical DataFrame
        .broker      — detected broker name or "Unknown"
        .confidence  — match confidence in [0, 1]
        .warnings    — list of non-fatal issues to surface to the user

    Raises
    ------
    FileNotFoundError  if the file does not exist
    ValueError         if the file type is unsupported or unreadable
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    warnings: list[str] = []

    # 1. Load raw file
    raw_df = _load_file(path)

    # 2. Detect broker
    profile, confidence = detect_broker(raw_df.columns.tolist())

    if profile is None:
        broker = "Unknown"
        warnings.append(
            f"Broker could not be identified (best match confidence: "
            f"{confidence:.0%}). "
            "The file will be used as-is. "
            "If columns are missing, you may need to rename them to match "
            "the canonical schema (see README)."
        )
        # Best-effort: pass through without remapping
        df = raw_df.copy()
        # Ensure canonical columns exist even if empty
        import numpy as np
        for col in ALL_CANONICAL:
            if col not in df.columns:
                df[col] = np.nan
    else:
        broker = profile["broker"]
        df     = remap_columns(raw_df.copy(), profile)

        if confidence < 1.0:
            missing_fp = profile["fingerprint"] - set(raw_df.columns)
            if missing_fp:
                warnings.append(
                    f"Detected as {broker} ({confidence:.0%} confidence). "
                    f"Expected columns not found: {sorted(missing_fp)}. "
                    "Results may be incomplete."
                )

    # 3. Validate
    result = IngestResult(df=df, broker=broker,
                          confidence=confidence, warnings=warnings)

    missing = result.missing_required()
    if missing:
        warnings.append(
            f"Required columns missing or entirely empty after mapping: "
            f"{missing}. The pipeline will proceed but some metrics may be NaN."
        )

    # 4. Light type coercion — numeric columns that should be float
    for col in ("entry_price", "exit_price", "pnl_gross", "size",
                "fees", "commissions"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return result
