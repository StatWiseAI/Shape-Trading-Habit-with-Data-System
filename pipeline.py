"""
pipeline.py
===========
CLI entry point for the trading analytics pipeline.

Usage
-----
    python pipeline.py --input trades_export.csv
    python pipeline.py --input trades_export.xlsx --out-dir results/
    python pipeline.py --input trades_export.csv --print-summary

Outputs (written to --out-dir, default: same folder as input)
--------------------------------------------------------------
    clean_trades.csv   enriched trade-level DataFrame
    summary.json       analytical payload  (feed to Claude or Streamlit)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── make sure the package root is on sys.path when run as a script ──────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.ingest      import ingest
from core.etl         import enrich
from core.descriptive import build_summary, to_json


# ─────────────────────────────────────────────────────────────────────────────
# CONSOLE DIGEST
# ─────────────────────────────────────────────────────────────────────────────

def _print_digest(summary: dict, broker: str, confidence: float,
                  warnings: list[str]) -> None:
    ov = summary.get("overall", {})
    da = summary.get("duration_analysis", {})
    fa = summary.get("fee_analysis", {})
    ra = summary.get("run_analysis", {})
    me = summary.get("meta", {})

    sep = "─" * 62

    print(f"\n{sep}")
    print(f"  Trading Analytics Pipeline — Summary")
    print(sep)
    print(f"  Broker detected : {broker}  (confidence {confidence:.0%})")
    print(f"  Period          : {me.get('data_from')} → {me.get('data_to')}")
    print(f"  Trading days    : {me.get('trading_days')}")
    print(f"  Instruments     : {', '.join(me.get('instruments', []))}")
    print(sep)

    def _fmt(v, prefix="$", suffix="", decimals=2):
        if v is None:
            return "n/a"
        return f"{prefix}{v:>{8}.{decimals}f}{suffix}"

    print(f"  Trades          : {ov.get('n_trades')}")
    print(f"  Total net PnL   : {_fmt(ov.get('total_net_pnl'))}")
    print(f"  Win rate        : {(ov.get('win_rate') or 0)*100:.1f}%")
    print(f"  Expectancy      : {_fmt(ov.get('expectancy_usd'))} / trade")
    print(f"  Profit factor   : {ov.get('profit_factor') or 'n/a'}")
    print(f"  Sharpe (trade)  : {ov.get('sharpe') or 'n/a'}")
    print(f"  Max drawdown    : {_fmt(ov.get('max_dd_usd'))}  "
          f"({ov.get('max_dd_pct') or 0:.1f}%)  "
          f"over {ov.get('max_dd_trades')} trades")
    print(sep)

    wins_hold  = (da.get("wins")   or {}).get("mean")
    losses_hold= (da.get("losses") or {}).get("mean")
    ratio      = da.get("loss_to_win_hold_ratio")
    print(f"  Hold — wins     : {wins_hold  or 'n/a'} min")
    print(f"  Hold — losses   : {losses_hold or 'n/a'} min")
    if ratio is not None:
        flag = "  ⚠  losses held longer" if ratio > 1.2 else ""
        print(f"  Loss/win ratio  : {ratio:.2f}x{flag}")

    print(sep)
    print(f"  Fee drag        : ${fa.get('total_cost') or 0:.2f}  "
          f"({fa.get('cost_pct_of_gross') or 0:.1f}% of gross)")
    print(f"  Max win streak  : {ra.get('max_win_streak')}")
    print(f"  Max loss streak : {ra.get('max_loss_streak')}")
    print(sep)

    if warnings:
        print("\n  ⚠  Warnings:")
        for w in warnings:
            print(f"     • {w}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Trading Analytics Pipeline — Steps 1 + 2a",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to raw trade export (.csv or .xlsx)",
    )
    parser.add_argument(
        "--out-dir", "-o", default=None,
        help="Output directory (default: same directory as input file)",
    )
    parser.add_argument(
        "--out-csv", default="clean_trades.csv",
        help="Filename for the enriched CSV",
    )
    parser.add_argument(
        "--out-json", default="summary.json",
        help="Filename for the JSON summary",
    )
    parser.add_argument(
        "--print-summary", action="store_true",
        help="Print full JSON summary to stdout",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    out_dir    = Path(args.out_dir) if args.out_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1a: Ingest ──────────────────────────────────────────────────────
    print(f"\n[1/3] Ingesting  {input_path.name} …")
    result = ingest(input_path)
    print(f"      Broker : {result.broker}  (confidence {result.confidence:.0%})")
    if result.warnings:
        for w in result.warnings:
            print(f"      ⚠  {w}")

    # ── Step 1b: Enrich ──────────────────────────────────────────────────────
    print("[2/3] Enriching …")
    df = enrich(result.df)
    csv_path = out_dir / args.out_csv
    df.to_csv(csv_path, index=False)
    print(f"      ✓ {csv_path}  ({len(df)} rows × {len(df.columns)} cols)")

    # ── Step 2a: Descriptive stats ───────────────────────────────────────────
    print("[3/3] Computing descriptive statistics …")
    summary = build_summary(df)
    json_path = out_dir / args.out_json
    json_path.write_text(to_json(summary))
    print(f"      ✓ {json_path}")

    # ── Console digest ───────────────────────────────────────────────────────
    _print_digest(summary, result.broker, result.confidence, result.warnings)

    if args.print_summary:
        print(to_json(summary))


if __name__ == "__main__":
    main()
