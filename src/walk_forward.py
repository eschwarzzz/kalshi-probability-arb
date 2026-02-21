"""Walk-forward validation framework for Kalshi oil backtesting."""

import logging
import sys
import io
import numpy as np
import pandas as pd
from itertools import product

from src.config import MARKET_FILES, FEE_RATE
from src.data_loader import load_kalshi_csv, get_oil_prices
from src.backtester import BacktestEngine, BacktestConfig, calculate_pnl, calculate_metrics

logger = logging.getLogger('kalshi_oil')


class WalkForwardTester:
    """Walk-forward validation framework.

    Splits markets chronologically:
    - Training: parameter optimization via grid search
    - Validation: parameter selection by Sharpe ratio
    - Test: final out-of-sample evaluation
    """

    def __init__(self, market_files=None, n_training=4, n_validation=3):
        self.market_files = market_files or MARKET_FILES
        self.n_training = n_training
        self.n_validation = n_validation

        self.training_results = []
        self.validation_results = []
        self.test_results = []
        self.optimal_params = {}

    def run(self):
        """Run full walk-forward test.

        Returns:
            (optimal_params, training_results, validation_results, test_results)
        """
        training_markets = self.market_files[:self.n_training]
        validation_markets = self.market_files[self.n_training:self.n_training + self.n_validation]
        test_markets = self.market_files[self.n_training + self.n_validation:]

        logger.info(f"{'='*80}")
        logger.info(f"WALK-FORWARD VALIDATION")
        logger.info(f"Training: {len(training_markets)} | Validation: {len(validation_markets)} | Test: {len(test_markets)}")
        logger.info(f"{'='*80}")

        # Phase 1: Parameter calibration
        logger.info("PHASE 1: PARAMETER CALIBRATION (Training Set)")
        param_grid = {
            'kelly_fraction': [0.20, 0.30],
            'base_edge_threshold': [0.10, 0.12],
        }
        all_param_results = self.calibrate_parameters(training_markets, param_grid)

        # Phase 2: Parameter selection
        logger.info("PHASE 2: PARAMETER SELECTION (Validation Set)")
        top_params = sorted(all_param_results, key=lambda x: x['sharpe'], reverse=True)[:3]

        best_val_sharpe = -np.inf
        best_params = None

        for i, param_result in enumerate(top_params, 1):
            params = param_result['params']
            logger.info(f"Candidate {i}: {params} (training Sharpe: {param_result['sharpe']:.2f})")

            val_perf = self._test_on_set(validation_markets, params, verbose=False, carry_bankroll=False)
            if val_perf:
                for r in val_perf['results']:
                    r.update(params)
                self.validation_results.extend(val_perf['results'])
                logger.info(f"  Validation Sharpe: {val_perf['sharpe']:.2f}, P&L: {val_perf['total_pnl']:+.2f}")
                if val_perf['sharpe'] > best_val_sharpe:
                    best_val_sharpe = val_perf['sharpe']
                    best_params = params.copy()

        if best_params is None:
            logger.error("No valid parameters found during validation!")
            return None, self.training_results, self.validation_results, self.test_results

        self.optimal_params = best_params
        logger.info(f"Selected params: {best_params} (val Sharpe: {best_val_sharpe:.2f})")

        # Phase 3: Out-of-sample test (bankroll compounds across test markets)
        logger.info("PHASE 3: OUT-OF-SAMPLE TEST")
        test_perf = self._test_on_set(test_markets, best_params, verbose=True, carry_bankroll=True)
        if test_perf:
            self.test_results = test_perf['results']

        self.generate_report()

        return self.optimal_params, self.training_results, self.validation_results, self.test_results

    def calibrate_parameters(self, markets, param_grid):
        """Grid search over parameter combinations on training set."""
        combinations = [
            dict(zip(param_grid.keys(), v))
            for v in product(*param_grid.values())
        ]

        logger.info(f"Testing {len(combinations)} parameter combinations...")
        all_results = []

        for i, params in enumerate(combinations, 1):
            logger.info(f"Combination {i}/{len(combinations)}: {params}")
            results = []

            for market_info in markets:
                result = self._run_single_backtest(market_info, params, verbose=False)
                if result:
                    results.append(result)

            if results:
                pnls = [r['net_pnl_cents'] for r in results]
                avg_pnl = np.mean(pnls)
                std_pnl = np.std(pnls) if len(pnls) > 1 else 1
                sharpe = (avg_pnl / std_pnl) * np.sqrt(len(pnls)) if std_pnl > 0 else 0

                summary = {
                    'params': params.copy(),
                    'sharpe': sharpe,
                    'avg_pnl': avg_pnl,
                    'total_trades': sum(r['num_trades'] for r in results),
                    'results': results,
                }
                all_results.append(summary)
                logger.info(f"  Avg P&L: {avg_pnl:+.2f} | Sharpe: {sharpe:.2f}")

        self.training_results = all_results
        return all_results

    def _test_on_set(self, markets, params, verbose=False, carry_bankroll=True, starting_bankroll=None):
        """Test parameters on a set of markets.

        Args:
            markets: List of market info tuples.
            params: Parameter dict (kelly_fraction, base_edge_threshold).
            verbose: Whether to print detailed output.
            carry_bankroll: If True, bankroll compounds across markets.
            starting_bankroll: Initial bankroll for the first market (None = default).
        """
        results = []
        bankroll = starting_bankroll

        for market_info in markets:
            result = self._run_single_backtest(market_info, params, verbose=verbose, bankroll=bankroll)
            if result:
                results.append(result)
                if carry_bankroll:
                    bankroll = result['final_bankroll']

        if not results:
            return None

        pnls = [r['net_pnl_cents'] for r in results]
        avg_pnl = np.mean(pnls)
        std_pnl = np.std(pnls) if len(pnls) > 1 else 1
        sharpe = (avg_pnl / std_pnl) * np.sqrt(len(pnls)) if std_pnl > 0 else 0

        return {
            'sharpe': sharpe,
            'avg_pnl': avg_pnl,
            'total_pnl': sum(pnls),
            'results': results,
            'final_bankroll': bankroll,
        }

    def _run_single_backtest(self, market_info, params, verbose=False, bankroll=None):
        """Run backtest on a single market.

        Args:
            market_info: Tuple of (csv_path, expiry_str, start_date, end_date).
            params: Parameter dict.
            verbose: Whether to print detailed output.
            bankroll: Starting bankroll in cents (None = default from config).
        """
        csv_path, expiry_str, start_date, end_date = market_info

        try:
            # Suppress output if not verbose
            if not verbose:
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()

            try:
                kalshi_df = load_kalshi_csv(csv_path)
                oil_prices = get_oil_prices(
                    start_date=(pd.to_datetime(start_date) - pd.Timedelta(days=120)).strftime('%Y-%m-%d'),
                    end_date=end_date,
                    extended_lookback=0,
                )
                expiry_date = pd.to_datetime(expiry_str, utc=True)

                config_kwargs = {
                    'kelly_fraction': params['kelly_fraction'],
                    'base_edge_threshold': params['base_edge_threshold'],
                }
                if bankroll is not None:
                    config_kwargs['initial_bankroll'] = int(bankroll)

                config = BacktestConfig(**config_kwargs)

                engine = BacktestEngine(config, kalshi_df, oil_prices, expiry_date)
                bt_result = engine.run()
            finally:
                if not verbose:
                    sys.stdout = old_stdout

            if not bt_result.trades:
                logger.info(f"  {csv_path}: No trades generated")
                return None

            # Apply execution constraints
            pnl_df = calculate_pnl(bt_result, oil_prices, expiry_date)
            if len(pnl_df) == 0:
                return None

            pnl_df = self.apply_execution_constraints(pnl_df, slippage_bps=25, partial_fill_rate=0.8)
            if len(pnl_df) == 0:
                return None

            metrics = calculate_metrics(pnl_df, bt_result.kelly_sizer)

            # Override final_bankroll with actual constrained P&L
            constrained_pnl = pnl_df['pnl_cents_net'].sum()
            metrics['final_bankroll'] = bt_result.kelly_sizer.bankroll + constrained_pnl

            result = {
                'market': csv_path,
                'expiry': expiry_str,
                **metrics,
            }

            logger.info(f"  {csv_path}: {metrics['num_trades']} trades, P&L: {metrics['net_pnl_cents']:+.2f}")
            return result

        except FileNotFoundError:
            logger.error(f"  File not found: {csv_path}")
            return None
        except Exception as e:
            logger.error(f"  Error on {csv_path}: {e}")
            return None

    def apply_execution_constraints(self, trades_df, slippage_bps=25, partial_fill_rate=0.8):
        """Apply slippage and partial fill constraints.

        Args:
            trades_df: DataFrame of trades from calculate_pnl.
            slippage_bps: Slippage in basis points of cents (25 = 0.25c).
            partial_fill_rate: Fraction of contracts that actually fill (0-1).
        """
        if len(trades_df) == 0:
            return trades_df

        trades_df = trades_df.copy()

        # Slippage: reduce P&L by slippage per contract (worse fill for sells)
        slippage_cents = slippage_bps / 100
        slippage_cost = slippage_cents * trades_df['num_contracts']
        trades_df['pnl_cents_gross'] = trades_df['pnl_cents_gross'] - slippage_cost
        trades_df['fees_cents'] = trades_df.apply(
            lambda r: r['pnl_cents_gross'] * FEE_RATE if r['pnl_cents_gross'] > 0 else 0.0, axis=1
        )
        trades_df['pnl_cents_net'] = trades_df['pnl_cents_gross'] - trades_df['fees_cents']

        # Partial fills: use per-trade fill_probability if available, otherwise use flat rate
        if 'fill_probability' in trades_df.columns:
            fill_rates = trades_df['fill_probability'].clip(0, 1)
        else:
            fill_rates = partial_fill_rate

        original_contracts = trades_df['num_contracts'].copy()
        filled_contracts = (original_contracts * fill_rates).apply(lambda x: int(round(x)))

        # Drop trades where partial fill rounds to 0 contracts
        trades_df['num_contracts'] = filled_contracts
        trades_df = trades_df[trades_df['num_contracts'] > 0].copy()

        if len(trades_df) == 0:
            return trades_df

        # Scale P&L proportionally to the reduced contract count
        scale = trades_df['num_contracts'] / original_contracts.loc[trades_df.index].replace(0, 1)
        trades_df['pnl_cents_gross'] = trades_df['pnl_cents_gross'] * scale
        trades_df['fees_cents'] = trades_df['fees_cents'] * scale
        trades_df['pnl_cents_net'] = trades_df['pnl_cents_gross'] - trades_df['fees_cents']

        return trades_df

    def generate_report(self):
        """Generate comprehensive walk-forward report and save CSVs."""
        logger.info(f"\n{'='*80}")
        logger.info("WALK-FORWARD RESULTS SUMMARY")
        logger.info(f"{'='*80}")

        if self.optimal_params:
            logger.info(f"Optimal params: {self.optimal_params}")

        # Save training results
        if self.training_results:
            training_rows = []
            for combo in self.training_results:
                for r in combo.get('results', []):
                    row = {**r, **combo['params'], 'combo_sharpe': combo['sharpe'], 'combo_avg_pnl': combo['avg_pnl']}
                    row.pop('results', None)
                    training_rows.append(row)
            if training_rows:
                train_df = pd.DataFrame(training_rows)
                train_df.to_csv('walk_forward_training.csv', index=False)
                logger.info(f"Saved training results to walk_forward_training.csv ({len(train_df)} rows)")

        # Save validation results
        if self.validation_results:
            val_rows = []
            for item in self.validation_results:
                if isinstance(item, dict):
                    row = {k: v for k, v in item.items() if k != 'results'}
                    val_rows.append(row)
            if val_rows:
                val_df = pd.DataFrame(val_rows)
                val_df.to_csv('walk_forward_validation.csv', index=False)
                logger.info(f"Saved validation results to walk_forward_validation.csv ({len(val_df)} rows)")

        # Save and report test results
        if self.test_results:
            test_df = pd.DataFrame(self.test_results)
            test_df.to_csv('walk_forward_test.csv', index=False)
            logger.info(f"Saved test results to walk_forward_test.csv ({len(test_df)} rows)")

            total_pnl = test_df['net_pnl_cents'].sum()
            avg_pnl = test_df['net_pnl_cents'].mean()
            std_pnl = test_df['net_pnl_cents'].std()
            sharpe = (avg_pnl / std_pnl) * np.sqrt(len(test_df)) if std_pnl > 0 else 0

            logger.info(f"\nOUT-OF-SAMPLE TEST RESULTS:")
            logger.info(f"Markets: {len(test_df)}")
            logger.info(f"Total P&L: {total_pnl:+.2f} cents (${total_pnl/100:+.2f})")
            logger.info(f"Sharpe: {sharpe:.2f}")
            logger.info(f"Win Rate: {test_df['win_rate'].mean():.1%}")

            profitable = (test_df['net_pnl_cents'] > 0).sum()
            logger.info(f"Profitable markets: {profitable}/{len(test_df)}")

            for _, row in test_df.iterrows():
                logger.info(f"  {row['market']}: P&L={row['net_pnl_cents']:+.2f}, Trades={row['num_trades']}, WR={row['win_rate']:.1%}")
