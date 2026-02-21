"""Event-driven backtest engine for Kalshi oil binary options."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import (
    BANKROLL,
    KELLY_FRACTION,
    MIN_EDGE_THRESHOLD,
    MAX_POSITIONS,
    STOP_LOSS_PER_STRIKE,
    FEE_RATE,
    EXIT_EDGE_THRESHOLD,
    PROFIT_TARGET,
    DEFAULT_SYNTHETIC_SPREAD,
    DEFAULT_SYNTHETIC_DEPTH,
    SYNTHETIC_NUM_LEVELS,
)
from src.orderbook import Orderbook, OrderbookFillSimulator
from src.position_manager import Position, PositionManager, StopLossMonitor, TradeRecord
from src.probability_model import calculate_garch_volatility, calculate_hybrid_probability, simulate_garch_price_paths
from src.strategy import KellyPositionSizer, calculate_dynamic_threshold, generate_signals
from src.utils import normalize_timestamp

logger = logging.getLogger('kalshi_oil')


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""

    initial_bankroll: int = BANKROLL
    kelly_fraction: float = KELLY_FRACTION
    base_edge_threshold: float = MIN_EDGE_THRESHOLD
    max_positions: int = MAX_POSITIONS
    stop_loss_per_strike: int = STOP_LOSS_PER_STRIKE
    fee_rate: float = FEE_RATE
    exit_edge_threshold: float = EXIT_EDGE_THRESHOLD
    profit_target: float = PROFIT_TARGET
    orderbook_mode: str = 'synthetic'
    synthetic_spread: int = DEFAULT_SYNTHETIC_SPREAD
    synthetic_depth: int = DEFAULT_SYNTHETIC_DEPTH
    synthetic_levels: int = SYNTHETIC_NUM_LEVELS


@dataclass
class BacktestResult:
    """Result of a backtest run."""

    trades: list = field(default_factory=list)
    kelly_sizer: KellyPositionSizer | None = None
    config: BacktestConfig | None = None
    metrics: dict = field(default_factory=dict)


class BacktestEngine:
    """Event-driven backtest engine.

    Iterates over timestamps in the Kalshi data, managing position entries
    and exits using the probability model, Kelly sizing, and orderbook
    simulation.
    """

    def __init__(self, config: BacktestConfig, kalshi_data: pd.DataFrame,
                 oil_prices: pd.Series, expiry_date):
        """Initialize the backtest engine.

        Args:
            config: BacktestConfig with all parameters.
            kalshi_data: Long-format DataFrame with timestamp, strike_range, price columns.
            oil_prices: Series of historical oil prices with datetime index.
            expiry_date: Expiry date as a pandas Timestamp (UTC-aware).
        """
        self.config = config
        self.kalshi_data = kalshi_data
        self.oil_prices = oil_prices
        self.expiry_date = expiry_date

        self.kelly_sizer = KellyPositionSizer(
            bankroll=config.initial_bankroll,
            kelly_fraction=config.kelly_fraction,
            min_edge=config.base_edge_threshold,
        )
        self.position_manager = PositionManager(
            max_positions=config.max_positions,
            exit_edge_threshold=config.exit_edge_threshold,
            profit_target=config.profit_target,
        )
        self.stop_loss_monitor = StopLossMonitor(
            max_loss_per_strike=config.stop_loss_per_strike,
        )
        self.fill_simulator = OrderbookFillSimulator()

    def run(self) -> BacktestResult:
        """Run the backtest over all timestamps.

        Returns:
            BacktestResult with trades, kelly_sizer, config, and metrics.
        """
        timestamps = sorted(self.kalshi_data['timestamp'].unique())
        all_trades: list[TradeRecord] = []

        logger.info(
            "Starting backtest: bankroll=%d, kelly=%.2f, edge_threshold=%.2f, max_positions=%d",
            self.config.initial_bankroll, self.config.kelly_fraction,
            self.config.base_edge_threshold, self.config.max_positions,
        )

        for timestamp in timestamps:
            current_spot = self._get_spot_price(timestamp)
            if current_spot is None:
                continue

            time_to_expiry = (self.expiry_date - timestamp).total_seconds() / (365.25 * 24 * 3600)
            if time_to_expiry <= 0:
                continue

            days_to_expiry = time_to_expiry * 365.25
            forecast_horizon = max(1, int(days_to_expiry))
            sigma_garch = calculate_garch_volatility(
                self.oil_prices, timestamp, forecast_horizon=forecast_horizon,
            )
            if sigma_garch is None or sigma_garch == 0:
                continue

            # Simulate GARCH price paths once per timestamp (shared across all strikes)
            _, terminal_prices = simulate_garch_price_paths(
                self.oil_prices, timestamp, forecast_horizon=forecast_horizon,
            )

            # Phase 1: check existing positions for exits
            exit_records = self._check_exits(
                timestamp, current_spot, time_to_expiry, sigma_garch, terminal_prices,
            )
            for record in exit_records:
                all_trades.append(record)
                self.stop_loss_monitor.update(record.strike_range, record.unrealized_pnl)
                self.kelly_sizer.update_bankroll(record.unrealized_pnl)

            # Phase 2: generate new signals and enter positions
            entry_records = self._generate_and_enter(
                timestamp, current_spot, time_to_expiry, days_to_expiry, sigma_garch,
                terminal_prices,
            )
            all_trades.extend(entry_records)

        # Settle any positions still open at expiry
        settlement_records = self._settle_open_positions()
        all_trades.extend(settlement_records)

        logger.info(
            "Backtest complete: %d trades (%d settled at expiry)",
            len(all_trades), len(settlement_records),
        )

        return BacktestResult(
            trades=all_trades,
            kelly_sizer=self.kelly_sizer,
            config=self.config,
            metrics={},
        )

    def _settle_open_positions(self):
        """Settle all remaining open positions at expiry using binary payout.

        Returns:
            List of TradeRecord for settlement exits.
        """
        if not self.position_manager.open_positions:
            return []

        expiry_naive = normalize_timestamp(self.expiry_date)
        available = self.oil_prices[self.oil_prices.index <= expiry_naive]
        if len(available) == 0:
            logger.error("No oil price data at expiry for settlement")
            return []
        final_price = available.iloc[-1].item()

        records = []
        for strike_range in list(self.position_manager.open_positions.keys()):
            position = self.position_manager.open_positions[strike_range]
            settled_in_range = position.lower_bound <= final_price < position.upper_bound
            # SELL settlement: if settled in range, buyer gets 100c; otherwise 0c
            settlement_price = 100.0 if settled_in_range else 0.0
            record = self.position_manager.exit(
                strike_range, settlement_price, self.expiry_date,
                reason=f"SETTLEMENT (final=${final_price:.2f})",
                spot_price=final_price,
            )
            if record:
                records.append(record)

        return records

    def _get_spot_price(self, timestamp):
        """Get the most recent oil price at or before *timestamp*.

        Returns:
            The spot price as a float, or None if no data is available.
        """
        try:
            ts_naive = normalize_timestamp(timestamp)
            available = self.oil_prices[self.oil_prices.index <= ts_naive]
            if len(available) == 0:
                return None
            return available.iloc[-1].item()
        except Exception:
            logger.debug("Could not get spot price for %s", timestamp)
            return None

    def _check_exits(self, timestamp, current_spot, time_to_expiry, sigma_garch,
                      terminal_prices=None):
        """Recalculate theoretical probs for open positions and check exit conditions.

        Returns:
            List of TradeRecord for any exits executed.
        """
        if not self.position_manager.open_positions:
            return []

        current_prices = {}
        theoretical_probs = {}

        market_at_ts = self.kalshi_data[self.kalshi_data['timestamp'] == timestamp]

        for strike_range in list(self.position_manager.open_positions.keys()):
            row = market_at_ts[market_at_ts['strike_range'] == strike_range]
            if len(row) == 0:
                continue

            current_price = row.iloc[0]['price']
            current_prices[strike_range] = current_price

            position = self.position_manager.open_positions[strike_range]
            theoretical_prob = calculate_hybrid_probability(
                oil_prices=self.oil_prices,
                current_spot=current_spot,
                lower_bound=position.lower_bound,
                upper_bound=position.upper_bound,
                timestamp=timestamp,
                time_to_expiry=time_to_expiry,
                sigma_garch=sigma_garch,
                strike_type=position.strike_type,
                terminal_prices=terminal_prices,
            )
            if theoretical_prob is not None:
                theoretical_probs[strike_range] = theoretical_prob

        return self.position_manager.check_exits(
            timestamp=timestamp,
            current_prices=current_prices,
            theoretical_probs=theoretical_probs,
            spot_price=current_spot,
        )

    def _generate_and_enter(self, timestamp, current_spot, time_to_expiry,
                            days_to_expiry, sigma_garch, terminal_prices=None):
        """Generate sell signals, simulate orderbook fills, and enter positions.

        Returns:
            List of TradeRecord for new entries.
        """
        available_slots = self.position_manager.available_slots
        if available_slots <= 0:
            return []

        market_snapshot = self.kalshi_data[self.kalshi_data['timestamp'] == timestamp].copy()

        signals = generate_signals(
            market_snapshot=market_snapshot,
            oil_prices=self.oil_prices,
            current_spot=current_spot,
            timestamp=timestamp,
            time_to_expiry=time_to_expiry,
            sigma_garch=sigma_garch,
            terminal_prices=terminal_prices,
        )

        dynamic_threshold = calculate_dynamic_threshold(days_to_expiry, self.config.base_edge_threshold)
        entry_records = []

        for signal in signals:
            if available_slots <= 0:
                break

            strike_range = signal['strike_range']

            # Skip blocked or already-held strikes
            if self.stop_loss_monitor.is_blocked(strike_range):
                continue
            if self.position_manager.has_position(strike_range):
                continue

            # Check edge vs dynamic threshold
            if signal['abs_edge'] < dynamic_threshold:
                continue

            # Simulate orderbook fill
            orderbook = self._get_orderbook(signal['market_price_cents'])
            fill_result = self.fill_simulator.simulate_sell_fill(
                orderbook=orderbook,
                theoretical_prob=signal['theoretical_prob'],
            )

            if fill_result.contracts_filled == 0:
                continue

            # Kelly sizing
            kelly_frac, num_contracts = self.kelly_sizer.calculate_position_size(
                fill_result=fill_result,
                theoretical_prob=signal['theoretical_prob'],
                strike_type=signal['strike_type'],
            )

            if kelly_frac <= 0 or num_contracts <= 0:
                continue

            # Enter position
            position = Position(
                strike_range=strike_range,
                entry_time=timestamp,
                entry_price=fill_result.avg_fill_price,
                num_contracts=num_contracts,
                edge=signal['edge'],
                strike_type=signal['strike_type'],
                lower_bound=signal['lower_bound'],
                upper_bound=signal['upper_bound'],
            )

            record = self.position_manager.enter(position)
            if record is not None:
                record.spot_price = current_spot
                record.theoretical_prob = signal['theoretical_prob']
                record.market_prob = signal['market_prob']
                record.kelly_fraction = kelly_frac
                record.days_to_expiry = days_to_expiry
                record.fill_probability = fill_result.fill_probability
                entry_records.append(record)
                available_slots -= 1

        return entry_records

    def _get_orderbook(self, market_price_cents):
        """Build an Orderbook from a CSV mid-price using synthetic mode.

        Args:
            market_price_cents: The mid-price from CSV data in cents.

        Returns:
            Orderbook instance.
        """
        return Orderbook.from_csv_price(
            price_cents=market_price_cents,
            spread=self.config.synthetic_spread,
            depth_per_level=self.config.synthetic_depth,
            num_levels=self.config.synthetic_levels,
        )


def calculate_pnl(backtest_result: BacktestResult, oil_prices: pd.Series,
                   expiry_date, fee_rate: float = FEE_RATE) -> pd.DataFrame:
    """Calculate P&L for all trades in a backtest result.

    Positions exited early use their realized P&L. Positions held to
    settlement are settled as binary payouts (100 cents if in range, else 0).
    Fees of *fee_rate* are applied to profitable trades only.

    Args:
        backtest_result: A completed BacktestResult.
        oil_prices: Historical oil price series.
        expiry_date: Expiry date as a pandas Timestamp.
        fee_rate: Fee rate applied to gross profits.

    Returns:
        DataFrame with one row per entry trade and columns for P&L, fees,
        exit_type, and settlement info.
    """
    trades = backtest_result.trades
    kelly_sizer = backtest_result.kelly_sizer

    if not trades:
        logger.warning("No trades to calculate P&L for")
        return pd.DataFrame()

    # Get final settlement price
    expiry_naive = normalize_timestamp(expiry_date)
    available = oil_prices[oil_prices.index <= expiry_naive]
    if len(available) == 0:
        logger.error("No oil price data available at expiry")
        return pd.DataFrame()
    final_price = available.iloc[-1].item()
    logger.info("Settlement price: $%.2f", final_price)

    # Pair each ENTER with its next EXIT on the same strike (handles re-entries)
    entry_trades = []
    exit_queues = {}  # strike_range -> list of EXIT records in order
    for t in trades:
        if t.action == 'ENTER':
            entry_trades.append(t)
        elif t.action == 'EXIT':
            exit_queues.setdefault(t.strike_range, []).append(t)

    rows = []
    for entry in entry_trades:
        strike_range = entry.strike_range
        num_contracts = entry.num_contracts
        entry_price = entry.price

        if strike_range in exit_queues and exit_queues[strike_range]:
            exit_record = exit_queues[strike_range].pop(0)
            pnl_gross = exit_record.unrealized_pnl
            exit_reason = exit_record.exit_reason
            if 'SETTLEMENT' in exit_reason:
                settlement = 1 if entry.lower_bound <= final_price < entry.upper_bound else 0
                exit_type = 'settlement'
            else:
                settlement = None
                exit_type = 'early_exit'
        else:
            # Fallback: should not happen now that run() settles all positions
            lower = entry.lower_bound
            upper = entry.upper_bound
            settlement = 1 if lower <= final_price < upper else 0

            premium = entry_price * num_contracts
            payout = -100 * settlement * num_contracts
            pnl_gross = premium + payout
            exit_type = 'settlement'
            exit_reason = ''

        # Fees on profits only
        fee = pnl_gross * fee_rate if pnl_gross > 0 else 0.0
        pnl_net = pnl_gross - fee

        rows.append({
            'timestamp': entry.timestamp,
            'strike_range': strike_range,
            'strike_type': entry.strike_type,
            'lower_bound': entry.lower_bound,
            'upper_bound': entry.upper_bound,
            'entry_price': entry_price,
            'num_contracts': num_contracts,
            'edge': entry.edge,
            'theoretical_prob': entry.theoretical_prob,
            'market_prob': entry.market_prob,
            'kelly_fraction': entry.kelly_fraction,
            'spot_price': entry.spot_price,
            'days_to_expiry': entry.days_to_expiry,
            'fill_probability': entry.fill_probability,
            'exit_type': exit_type,
            'exit_reason': exit_reason,
            'settlement': settlement,
            'final_price': final_price,
            'pnl_cents_gross': pnl_gross,
            'fees_cents': fee,
            'pnl_cents_net': pnl_net,
        })

    df = pd.DataFrame(rows)
    logger.info(
        "P&L calculated: %d entries, gross=%.2f, fees=%.2f, net=%.2f",
        len(df),
        df['pnl_cents_gross'].sum() if len(df) > 0 else 0,
        df['fees_cents'].sum() if len(df) > 0 else 0,
        df['pnl_cents_net'].sum() if len(df) > 0 else 0,
    )
    return df


def calculate_metrics(trades_df: pd.DataFrame, kelly_sizer: KellyPositionSizer) -> dict:
    """Compute summary performance metrics from a P&L DataFrame.

    Args:
        trades_df: DataFrame returned by calculate_pnl with pnl_cents_net column.
        kelly_sizer: The KellyPositionSizer used in the backtest (for bankroll info).

    Returns:
        Dict with keys: net_pnl_cents, num_trades, win_rate, sharpe,
        max_drawdown_pct, avg_win, avg_loss, profit_factor, final_bankroll.
    """
    if trades_df is None or len(trades_df) == 0:
        return {
            'net_pnl_cents': 0.0,
            'num_trades': 0,
            'win_rate': 0.0,
            'sharpe': 0.0,
            'max_drawdown_pct': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'final_bankroll': kelly_sizer.current_bankroll,
        }

    pnl = trades_df['pnl_cents_net']
    net_pnl = pnl.sum()
    num_trades = len(pnl)

    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    win_rate = len(wins) / num_trades if num_trades > 0 else 0.0

    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0

    # Profit factor: gross wins / abs(gross losses)
    total_wins = wins.sum() if len(wins) > 0 else 0.0
    total_losses = abs(losses.sum()) if len(losses) > 0 else 0.0
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0.0

    # Sharpe ratio (per-trade, annualized by sqrt of num_trades)
    if num_trades > 1 and pnl.std() > 0:
        sharpe = (pnl.mean() / pnl.std()) * np.sqrt(num_trades)
    else:
        sharpe = 0.0

    # Max drawdown as percentage of initial bankroll
    cumulative = pnl.cumsum()
    running_max = cumulative.cummax()
    drawdown = running_max - cumulative
    max_drawdown = drawdown.max() if len(drawdown) > 0 else 0.0
    max_drawdown_pct = max_drawdown / kelly_sizer.bankroll if kelly_sizer.bankroll > 0 else 0.0

    return {
        'net_pnl_cents': net_pnl,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'sharpe': sharpe,
        'max_drawdown_pct': max_drawdown_pct,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'final_bankroll': kelly_sizer.current_bankroll,
    }
