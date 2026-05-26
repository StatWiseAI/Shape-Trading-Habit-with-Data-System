"""
tests/test_pipeline.py
======================
Automated tests for all three pipeline layers:
  - core.adapters.brokers  (detection + remapping)
  - core.ingest            (file loading + broker dispatch)
  - core.etl               (enrichment)
  - core.descriptive       (stats correctness)

Run with:
    pytest tests/ -v
    pytest tests/ -v --tb=short      # compact tracebacks
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── path fixup so tests can be run from any working directory ────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.adapters.brokers import detect_broker, remap_columns, BROKER_PROFILES
from core.ingest            import ingest, IngestResult
from core.etl               import enrich, _parse_duration_str, _hour_bin
from core.descriptive       import (
    build_summary, descriptive_block, profit_factor,
    sharpe_ratio, max_drawdown, consecutive_runs, to_json,
)


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

NINJATRADER_ROWS = [
    {
        "Id": 1, "TradeDay": "04/16/2026 00:00:00 -05:00",
        "ContractName": "MESM6",
        "EnteredAt": "04/16/2026 10:05:01 +02:00",
        "ExitedAt":  "04/16/2026 13:08:35 +02:00",
        "EntryPrice": 5200.25, "ExitPrice": 5215.50,
        "PnL": 75.00, "Fees": 0.74, "Commissions": 0.50,
        "Size": 1, "Type": "Long",
        "TradeDuration": "03:03:34.5289460",
    },
    {
        "Id": 2, "TradeDay": "04/16/2026 00:00:00 -05:00",
        "ContractName": "MNQM6",
        "EnteredAt": "04/16/2026 14:30:02 +02:00",
        "ExitedAt":  "04/16/2026 17:16:08 +02:00",
        "EntryPrice": 17800.0, "ExitPrice": 17760.0,
        "PnL": -20.00, "Fees": 0.74, "Commissions": 0.50,
        "Size": 1, "Type": "Short",
        "TradeDuration": "02:46:06.2665630",
    },
    {
        "Id": 3, "TradeDay": "04/17/2026 00:00:00 -05:00",
        "ContractName": "MESM6",
        "EnteredAt": "04/17/2026 09:35:00 +02:00",
        "ExitedAt":  "04/17/2026 11:42:00 +02:00",
        "EntryPrice": 5190.0, "ExitPrice": 5195.0,
        "PnL": 25.00, "Fees": 0.74, "Commissions": 0.50,
        "Size": 1, "Type": "Long",
        "TradeDuration": "02:07:00.0000000",
    },
]


@pytest.fixture
def nt_df() -> pd.DataFrame:
    return pd.DataFrame(NINJATRADER_ROWS)


@pytest.fixture
def nt_csv(tmp_path, nt_df) -> Path:
    p = tmp_path / "nt_trades.csv"
    nt_df.to_csv(p, index=False)
    return p


@pytest.fixture
def nt_xlsx(tmp_path, nt_df) -> Path:
    p = tmp_path / "nt_trades.xlsx"
    nt_df.to_excel(p, index=False)
    return p


@pytest.fixture
def enriched_df(nt_csv) -> pd.DataFrame:
    result = ingest(nt_csv)
    return enrich(result.df)


# ─────────────────────────────────────────────────────────────────────────────
# 1. BROKER DETECTION
# ─────────────────────────────────────────────────────────────────────────────

class TestBrokerDetection:

    def test_detects_ninjatrader(self, nt_df):
        profile, confidence = detect_broker(nt_df.columns.tolist())
        assert profile is not None
        assert profile["broker"] == "NinjaTrader"
        assert confidence >= 0.9

    def test_unknown_broker_returns_none(self):
        profile, confidence = detect_broker(["foo", "bar", "baz"])
        assert profile is None
        assert confidence < 0.6

    def test_all_profiles_have_required_keys(self):
        required = {"broker", "fingerprint", "column_map", "direction_map"}
        for p in BROKER_PROFILES:
            assert required.issubset(p.keys()), f"Profile missing keys: {p['broker']}"

    def test_fingerprints_are_nonempty_sets(self):
        for p in BROKER_PROFILES:
            assert isinstance(p["fingerprint"], set)
            assert len(p["fingerprint"]) >= 4

    def test_partial_match_still_detects(self, nt_df):
        # Drop 2 columns — should still detect NinjaTrader
        cols_to_drop = ["TradeDay", "TradeDuration"]
        partial = nt_df.drop(columns=cols_to_drop)
        profile, confidence = detect_broker(partial.columns.tolist())
        assert profile is not None
        assert profile["broker"] == "NinjaTrader"


class TestColumnRemapping:

    def test_canonical_columns_present_after_remap(self, nt_df):
        profile, _ = detect_broker(nt_df.columns.tolist())
        remapped = remap_columns(nt_df.copy(), profile)
        for col in ["instrument", "entered_at", "entry_price",
                    "pnl_gross", "size", "direction"]:
            assert col in remapped.columns, f"Missing: {col}"

    def test_direction_normalised(self, nt_df):
        profile, _ = detect_broker(nt_df.columns.tolist())
        remapped = remap_columns(nt_df.copy(), profile)
        assert set(remapped["direction"].unique()).issubset({"Long", "Short"})

    def test_unmapped_columns_get_raw_prefix(self, nt_df):
        profile, _ = detect_broker(nt_df.columns.tolist())
        remapped = remap_columns(nt_df.copy(), profile)
        # TradeDay is not in column_map → should become raw__TradeDay
        assert "raw__TradeDay" in remapped.columns

    def test_fees_commissions_are_float(self, nt_df):
        profile, _ = detect_broker(nt_df.columns.tolist())
        remapped = remap_columns(nt_df.copy(), profile)
        assert remapped["fees"].dtype == float
        assert remapped["commissions"].dtype == float


# ─────────────────────────────────────────────────────────────────────────────
# 2. INGEST
# ─────────────────────────────────────────────────────────────────────────────

class TestIngest:

    def test_csv_ingest_returns_result(self, nt_csv):
        result = ingest(nt_csv)
        assert isinstance(result, IngestResult)
        assert result.ok()

    def test_xlsx_ingest_returns_result(self, nt_xlsx):
        result = ingest(nt_xlsx)
        assert isinstance(result, IngestResult)
        assert result.ok()

    def test_broker_identified(self, nt_csv):
        result = ingest(nt_csv)
        assert result.broker == "NinjaTrader"
        assert result.confidence >= 0.9

    def test_row_count_preserved(self, nt_csv, nt_df):
        result = ingest(nt_csv)
        assert len(result.df) == len(nt_df)

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            ingest("/nonexistent/path/trades.csv")

    def test_unsupported_extension_raises(self, tmp_path):
        p = tmp_path / "trades.parquet"
        p.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported file type"):
            ingest(p)

    def test_csv_with_bom_encoding(self, tmp_path, nt_df):
        p = tmp_path / "bom_trades.csv"
        nt_df.to_csv(p, index=False, encoding="utf-8-sig")
        result = ingest(p)
        assert result.ok()

    def test_warnings_list_is_list(self, nt_csv):
        result = ingest(nt_csv)
        assert isinstance(result.warnings, list)


# ─────────────────────────────────────────────────────────────────────────────
# 3. ETL — ENRICHMENT
# ─────────────────────────────────────────────────────────────────────────────

class TestETL:

    def test_net_pnl_computed_correctly(self, enriched_df):
        row = enriched_df.iloc[0]
        expected = round(row["pnl_gross"] - row["fees"] - row["commissions"], 6)
        assert abs(row["net_pnl"] - expected) < 1e-4

    def test_outcome_labels(self, enriched_df):
        assert set(enriched_df["outcome"].unique()).issubset(
            {"win", "loss", "breakeven"}
        )

    def test_trade_index_starts_at_one(self, enriched_df):
        assert enriched_df["trade_index"].min() == 1

    def test_trade_index_is_sequential(self, enriched_df):
        idx = enriched_df["trade_index"].tolist()
        assert idx == list(range(1, len(idx) + 1))

    def test_sorted_chronologically(self, enriched_df):
        dates = enriched_df["entered_at_dt"].dropna()
        assert dates.is_monotonic_increasing

    def test_duration_seconds_positive(self, enriched_df):
        durations = enriched_df["duration_seconds"].dropna()
        assert (durations > 0).all()

    def test_hour_bin_format(self, enriched_df):
        bins = enriched_df["hour_bin"].dropna().unique()
        import re
        pattern = re.compile(r"^\d{2}-\d{2}$")
        for b in bins:
            assert pattern.match(str(b)), f"Bad hour_bin format: {b}"

    def test_entry_hour_in_valid_range(self, enriched_df):
        hours = enriched_df["entry_hour"].dropna()
        assert ((hours >= 0) & (hours <= 23)).all()

    def test_pnl_per_minute_sign_matches_net_pnl(self, enriched_df):
        sub = enriched_df.dropna(subset=["pnl_per_minute"])
        signs_match = (
            (sub["net_pnl"] >= 0) == (sub["pnl_per_minute"] >= 0)
        )
        assert signs_match.all()

    def test_missing_exited_at_does_not_crash(self, nt_csv):
        result = ingest(nt_csv)
        result.df["exited_at"] = np.nan
        df = enrich(result.df)
        assert "net_pnl" in df.columns    # pipeline still runs

    def test_duration_str_parsing_ninjatrader_format(self):
        # 3h * 3600 + 3m * 60 + 34s + 5289460 ticks / 1e7 = 11014.528946 s
        assert abs(_parse_duration_str("03:03:34.5289460") - 11_014.52894) < 0.001

    def test_duration_str_parsing_fallback_no_frac(self):
        assert abs(_parse_duration_str("01:30:00") - 5400.0) < 0.001

    def test_duration_str_parsing_invalid_returns_nan(self):
        assert np.isnan(_parse_duration_str("not_a_duration"))


class TestHourBin:
    def test_bin_size_2(self):
        assert _hour_bin(0)  == "00-02"
        assert _hour_bin(1)  == "00-02"
        assert _hour_bin(8)  == "08-10"
        assert _hour_bin(9)  == "08-10"
        assert _hour_bin(23) == "22-24"

    def test_custom_bin_size(self):
        assert _hour_bin(0, bin_size=4)  == "00-04"
        assert _hour_bin(13, bin_size=4) == "12-16"


# ─────────────────────────────────────────────────────────────────────────────
# 4. DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

class TestDescriptiveHelpers:

    def test_profit_factor_basic(self):
        pnl = pd.Series([100, -50, 50, -25])
        pf  = profit_factor(pnl)
        # gross_profit=150, gross_loss=75 → 2.0
        assert abs(pf - 2.0) < 1e-4

    def test_profit_factor_no_losses(self):
        pnl = pd.Series([10, 20, 30])
        assert profit_factor(pnl) is None

    def test_profit_factor_all_losses(self):
        pnl = pd.Series([-10, -20])
        assert profit_factor(pnl) == 0.0

    def test_sharpe_positive_edge(self):
        pnl = pd.Series([10.0] * 100)   # zero variance → None
        assert sharpe_ratio(pnl) is None

    def test_sharpe_mixed(self):
        np.random.seed(42)
        pnl = pd.Series(np.random.normal(1, 5, 500))
        sr  = sharpe_ratio(pnl)
        assert sr is not None
        assert -50 < sr < 50   # sanity bounds

    def test_max_drawdown_all_wins(self):
        pnl = pd.Series([10, 20, 30])
        dd  = max_drawdown(pnl)
        assert dd["max_dd_usd"] == 0.0
        assert dd["max_dd_trades"] == 0

    def test_max_drawdown_known_sequence(self):
        # equity: 10, 5, 15, 8, 20 → max dd at trade 4: 15-8=7
        pnl = pd.Series([10, -5, 10, -7, 12])
        dd  = max_drawdown(pnl)
        assert dd["max_dd_usd"] < 0

    def test_consecutive_runs_basic(self):
        pnl  = pd.Series([10, 10, -5, -5, -5, 10])
        runs = consecutive_runs(pnl)
        assert runs["max_win_streak"]  == 2
        assert runs["max_loss_streak"] == 3

    def test_consecutive_runs_all_wins(self):
        pnl  = pd.Series([5, 10, 15])
        runs = consecutive_runs(pnl)
        assert runs["max_loss_streak"] == 0
        assert runs["max_win_streak"]  == 3


class TestDescriptiveBlock:

    def test_keys_present(self):
        pnl   = pd.Series([10, -5, 20, -10, 15])
        block = descriptive_block(pnl, "test")
        for key in ["n_trades", "win_rate", "expectancy_usd",
                    "profit_factor", "sharpe", "max_dd_usd"]:
            assert key in block, f"Missing key: {key}"

    def test_win_rate_correct(self):
        pnl   = pd.Series([10, -5, 20, -10, 15])  # 3 wins / 5
        block = descriptive_block(pnl)
        assert abs(block["win_rate"] - 0.6) < 1e-4

    def test_empty_series_returns_zero_trades(self):
        block = descriptive_block(pd.Series([], dtype=float))
        assert block["n_trades"] == 0

    def test_single_row_no_crash(self):
        block = descriptive_block(pd.Series([42.0]))
        assert block["n_trades"] == 1


class TestBuildSummary:

    def test_summary_json_serialisable(self, enriched_df):
        summary = build_summary(enriched_df)
        json_str = to_json(summary)
        parsed   = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_summary_has_required_sections(self, enriched_df):
        summary = build_summary(enriched_df)
        for section in ["meta", "overall", "by_instrument",
                        "by_direction", "duration_analysis",
                        "fee_analysis", "equity_curve"]:
            assert section in summary, f"Missing section: {section}"

    def test_equity_curve_length_matches_trades(self, enriched_df):
        summary = build_summary(enriched_df)
        assert len(summary["equity_curve"]) == len(enriched_df)

    def test_equity_curve_last_value_equals_total_pnl(self, enriched_df):
        summary = build_summary(enriched_df)
        total   = round(float(enriched_df["net_pnl"].sum()), 4)
        last    = summary["equity_curve"][-1]
        assert abs(last - total) < 0.01

    def test_no_nan_in_json_output(self, enriched_df):
        summary  = build_summary(enriched_df)
        json_str = to_json(summary)
        assert "NaN" not in json_str
        assert "Infinity" not in json_str

    def test_by_instrument_keys_match_instruments(self, enriched_df):
        summary     = build_summary(enriched_df)
        instruments = set(enriched_df["instrument"].dropna().unique())
        keys        = set(summary["by_instrument"].keys())
        assert instruments == keys

    def test_daily_pnl_sums_to_total(self, enriched_df):
        summary   = build_summary(enriched_df)
        daily_pnl = summary["daily_pnl"]
        if not daily_pnl:
            pytest.skip("No trade_date values available in fixture — skipping")
        daily_sum = round(sum(daily_pnl.values()), 4)
        total     = round(float(enriched_df["net_pnl"].sum()), 4)
        assert abs(daily_sum - total) < 0.01

    def test_duration_analysis_hold_ratio_present(self, enriched_df):
        summary = build_summary(enriched_df)
        da      = summary["duration_analysis"]
        assert "loss_to_win_hold_ratio" in da


# ─────────────────────────────────────────────────────────────────────────────
# 5. END-TO-END
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEnd:

    def test_full_pipeline_csv(self, nt_csv):
        result  = ingest(nt_csv)
        df      = enrich(result.df)
        summary = build_summary(df)
        assert summary["overall"]["n_trades"] == 3
        assert summary["overall"]["total_net_pnl"] is not None

    def test_full_pipeline_xlsx(self, nt_xlsx):
        result  = ingest(nt_xlsx)
        df      = enrich(result.df)
        summary = build_summary(df)
        assert summary["overall"]["n_trades"] == 3

    def test_pipeline_on_real_file(self):
        """Use the actual Topstep/NinjaTrader export if present."""
        real_file = Path("data/trades_export.csv")
        if not real_file.exists():
            pytest.skip("Real data file not present — skipping live test")
        result  = ingest(real_file)
        df      = enrich(result.df)
        summary = build_summary(df)
        assert summary["overall"]["n_trades"] > 0

    def test_summary_json_roundtrip(self, nt_csv, tmp_path):
        result   = ingest(nt_csv)
        df       = enrich(result.df)
        summary  = build_summary(df)
        out_path = tmp_path / "summary.json"
        out_path.write_text(to_json(summary))
        loaded   = json.loads(out_path.read_text())
        assert loaded["overall"]["n_trades"] == 3


# ─────────────────────────────────────────────────────────────────────────────
# 6. INFERENTIAL STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

class TestInferential:

    @pytest.fixture
    def inf_df(self, nt_csv):
        from core.inferential import run_all
        result = ingest(nt_csv)
        df = enrich(result.df)
        return df, run_all(df)

    def test_run_all_returns_dict(self, inf_df):
        _, r = inf_df
        assert isinstance(r, dict)

    def test_all_top_level_keys_present(self, inf_df):
        _, r = inf_df
        for k in ["bootstrap","normality","hold_time","mwu_by_instrument",
                  "kruskal_wallis","permutation","ljung_box",
                  "durbin_watson","runs_test","cusum"]:
            assert k in r, f"Missing key: {k}"

    def test_bootstrap_ci_has_required_fields(self, inf_df):
        _, r = inf_df
        b = r["bootstrap"]["expectancy"]
        # Small fixture may return note instead of CI
        if "note" in b:
            pytest.skip("fixture too small for bootstrap CI")
        for f in ["estimate", "ci_lo", "ci_hi", "significant"]:
            assert f in b

    def test_normality_rejects_for_synthetic_data(self):
        from core.inferential import normality_tests
        import pandas as pd, numpy as np
        pnl = pd.Series(np.concatenate([np.ones(20) * 100, np.ones(5) * (-500)]))
        res = normality_tests(pnl)
        assert res.get("shapiro_wilk", {}).get("reject_h0") is True

    def test_runs_test_keys(self, inf_df):
        _, r = inf_df
        rr = r["runs_test"]
        if "note" in rr:
            pytest.skip("fixture too small for runs test")
        for k in ["observed_runs", "expected_runs", "z_stat", "p_value", "reject_h0"]:
            assert k in rr

    def test_cusum_has_series(self, inf_df):
        _, r = inf_df
        c = r["cusum"]
        if "note" in c:
            pytest.skip("fixture too small for CUSUM")
        assert "cusum_pos" in c and "cusum_neg" in c
        assert len(c["cusum_pos"]) == len(c["cusum_neg"])

    def test_ljung_box_has_lag_results(self, inf_df):
        _, r = inf_df
        lb = r["ljung_box"]
        if "note" in lb:
            pytest.skip("insufficient data")
        assert "results" in lb
        assert len(lb["results"]) == 3

    def test_json_serialisable(self, inf_df):
        _, r = inf_df
        import json
        from core.descriptive import _json_safe
        json.dumps(_json_safe(r))
