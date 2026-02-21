"""Backtest and walk-forward entry point."""

import argparse
import sys
import glob
import os

import pandas as pd

from src.utils import setup_logging
from src.config import MARKET_FILES
from src.data_loader import load_kalshi_csv, get_oil_prices
from src.backtester import BacktestEngine, BacktestConfig, calculate_pnl, calculate_metrics
from src.walk_forward import WalkForwardTester

logger = setup_logging()


def run_backtest(csv_path, expiry_str, start_date=None, end_date=None):
    """Run a single market backtest."""
    if not os.path.exists(csv_path):
        logger.error("File not found: %s", csv_path)
        return

    expiry_date = pd.to_datetime(expiry_str, utc=True)

    if start_date is None:
        start_date = (expiry_date - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = expiry_str

    logger.info("=" * 70)
    logger.info("SINGLE MARKET BACKTEST")
    logger.info("CSV: %s | Expiry: %s", csv_path, expiry_str)
    logger.info("=" * 70)

    kalshi_df = load_kalshi_csv(csv_path)
    oil_prices = get_oil_prices(start_date, end_date, extended_lookback=120)

    config = BacktestConfig()
    engine = BacktestEngine(config, kalshi_df, oil_prices, expiry_date)
    result = engine.run()

    if not result.trades:
        logger.warning("No trades generated")
        return

    pnl_df = calculate_pnl(result, oil_prices, expiry_date)
    metrics = calculate_metrics(pnl_df, result.kelly_sizer)

    logger.info("=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)
    logger.info("Net P&L: %+.2f cents ($%+.2f)", metrics['net_pnl_cents'], metrics['net_pnl_cents'] / 100)
    logger.info("Trades: %d | Win Rate: %.1f%%", metrics['num_trades'], metrics['win_rate'] * 100)
    logger.info("Sharpe: %.2f | Max DD: %.1f%%", metrics['sharpe'], metrics['max_drawdown_pct'])
    logger.info("Profit Factor: %.2f", metrics['profit_factor'])
    logger.info("Final Bankroll: $%.2f", metrics['final_bankroll'] / 100)
    logger.info("=" * 70)

    # Save results
    output_file = csv_path.replace('.csv', '_results.csv')
    pnl_df.to_csv(output_file, index=False)
    logger.info("Results saved to %s", output_file)


def run_walk_forward():
    """Run walk-forward validation across all markets."""
    logger.info("=" * 70)
    logger.info("WALK-FORWARD VALIDATION")
    logger.info("=" * 70)

    tester = WalkForwardTester(market_files=MARKET_FILES)
    optimal_params, training, validation, test = tester.run()

    if optimal_params is None:
        logger.error("Walk-forward failed — no valid parameters found")
        return

    logger.info("Optimal params: %s", optimal_params)


def run_backtest_all():
    """Run backtest on every CSV file individually."""
    csv_files = sorted(glob.glob('kalshi-oil-*.csv'))

    if not csv_files:
        logger.error("No kalshi-oil-*.csv files found in current directory")
        return

    # Map CSV files to market info from config
    market_map = {info[0]: info for info in MARKET_FILES}

    for csv_path in csv_files:
        if csv_path in market_map:
            _, expiry_str, start_date, end_date = market_map[csv_path]
            run_backtest(csv_path, expiry_str, start_date, end_date)
        else:
            logger.warning("Skipping %s — not in MARKET_FILES config", csv_path)


def main():
    parser = argparse.ArgumentParser(description='Kalshi Oil V2 Backtester')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # backtest command
    bt_parser = subparsers.add_parser('backtest', help='Run single market backtest')
    bt_parser.add_argument('--csv', required=True, help='Path to Kalshi CSV file')
    bt_parser.add_argument('--expiry', required=True, help='Expiry date (YYYY-MM-DD)')
    bt_parser.add_argument('--start', default=None, help='Start date (default: 30 days before expiry)')
    bt_parser.add_argument('--end', default=None, help='End date (default: expiry date)')

    # walk-forward command
    subparsers.add_parser('walk-forward', help='Run walk-forward validation')

    # backtest-all command
    subparsers.add_parser('backtest-all', help='Backtest all CSV files')

    args = parser.parse_args()

    if args.command == 'backtest':
        run_backtest(args.csv, args.expiry, args.start, args.end)
    elif args.command == 'walk-forward':
        run_walk_forward()
    elif args.command == 'backtest-all':
        run_backtest_all()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
