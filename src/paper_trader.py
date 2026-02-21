"""Live paper trading using Kalshi API with the same strategy as backtester."""

import asyncio
import csv
import logging
import os
import time as time_module
from datetime import datetime, timezone

import pandas as pd

from src.config import (
    BANKROLL, KALSHI_KEY_ID, KALSHI_PRIVATE_KEY_PATH,
    OIL_SERIES_TICKER, MAX_POSITIONS,
)
from src.kalshi_client import KalshiClient, KalshiWebSocket
from src.data_loader import get_oil_prices
from src.probability_model import calculate_garch_volatility, calculate_hybrid_probability
from src.orderbook import Orderbook, OrderbookFillSimulator
from src.strategy import KellyPositionSizer, calculate_dynamic_threshold
from src.position_manager import PositionManager, Position, StopLossMonitor
from src.utils import parse_strike_range

logger = logging.getLogger('kalshi_oil')

PAPER_TRADES_FILE = 'paper_trades.csv'


class PaperTrader:
    """Live paper trading using Kalshi API.

    Connects to Kalshi REST API for orderbook data, runs the probability
    model on each poll, and logs paper trades to CSV.
    """

    def __init__(self, bankroll=BANKROLL, poll_interval=60):
        self.client = KalshiClient(
            key_id=KALSHI_KEY_ID,
            private_key_path=KALSHI_PRIVATE_KEY_PATH,
        )
        self.ws = KalshiWebSocket(self.client)

        self.kelly_sizer = KellyPositionSizer(bankroll=bankroll)
        self.position_manager = PositionManager(max_positions=MAX_POSITIONS)
        self.stop_loss_monitor = StopLossMonitor()
        self.fill_simulator = OrderbookFillSimulator()

        self.poll_interval = poll_interval
        self._running = False
        self._active_markets = []
        self._oil_prices = None
        self._last_oil_fetch = None

    def start(self):
        """Start the paper trading loop."""
        if not self.client.is_authenticated:
            logger.error("Cannot start paper trading without API credentials.")
            logger.error("Set KALSHI_KEY_ID and KALSHI_PRIVATE_KEY_PATH in .env")
            return

        logger.info("=" * 70)
        logger.info("PAPER TRADING MODE")
        logger.info("Bankroll: $%.2f", self.kelly_sizer.current_bankroll / 100)
        logger.info("Poll interval: %ds", self.poll_interval)
        logger.info("=" * 70)

        self._running = True

        try:
            asyncio.run(self._run_loop())
        except KeyboardInterrupt:
            logger.info("Paper trading stopped by user")
        finally:
            self._running = False
            self._save_state()

    async def _run_loop(self):
        """Main async trading loop."""
        while self._running:
            try:
                self._refresh_oil_prices()
                self._discover_markets()

                if not self._active_markets:
                    logger.info("No active oil markets found. Waiting...")
                    await asyncio.sleep(self.poll_interval * 5)
                    continue

                for market in self._active_markets:
                    await self._process_market(market)

                await asyncio.sleep(self.poll_interval)

            except Exception as e:
                logger.error("Error in trading loop: %s", e)
                await asyncio.sleep(self.poll_interval)

    def _discover_markets(self):
        """Find active oil weekly markets."""
        markets = self.client.get_markets(
            series_ticker=OIL_SERIES_TICKER,
            status='open',
        )

        if markets:
            self._active_markets = markets
            logger.info("Found %d active oil markets", len(markets))
        else:
            self._active_markets = []

    def _refresh_oil_prices(self):
        """Fetch oil prices if stale (older than 1 hour)."""
        now = time_module.time()
        if self._last_oil_fetch and (now - self._last_oil_fetch) < 3600:
            return

        try:
            end_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            self._oil_prices = get_oil_prices(
                start_date=end_date,
                end_date=end_date,
                extended_lookback=120,
            )
            self._last_oil_fetch = now
            logger.info("Oil prices refreshed. Latest: $%.2f",
                        float(self._oil_prices.iloc[-1]))
        except Exception as e:
            logger.error("Failed to fetch oil prices: %s", e)

    async def _process_market(self, market):
        """Process a single market: check exits, generate signals, enter trades."""
        ticker = market.get('ticker', '')

        if self._oil_prices is None or len(self._oil_prices) == 0:
            return

        current_spot = float(self._oil_prices.iloc[-1])
        now = pd.Timestamp.now(tz='UTC')

        close_time = market.get('close_time', '')
        if close_time:
            expiry = pd.to_datetime(close_time, utc=True)
        else:
            return

        time_to_expiry = (expiry - now).total_seconds() / (365.25 * 24 * 3600)
        if time_to_expiry <= 0:
            return

        days_to_expiry = time_to_expiry * 365.25

        sigma_garch = calculate_garch_volatility(
            self._oil_prices, now,
            forecast_horizon=max(1, int(days_to_expiry)),
        )
        if sigma_garch is None:
            return

        ob_data = self.client.get_orderbook(ticker)
        if not ob_data:
            return

        orderbook = Orderbook.from_api_response(ob_data)
        if orderbook.mid_price is None:
            return

        market_price = orderbook.mid_price
        strike_str = market.get('title', ticker)

        try:
            lower, upper, strike_type = parse_strike_range(strike_str)
        except Exception:
            return

        theoretical_prob = calculate_hybrid_probability(
            oil_prices=self._oil_prices,
            current_spot=current_spot,
            lower_bound=lower,
            upper_bound=upper,
            timestamp=now,
            time_to_expiry=time_to_expiry,
            sigma_garch=sigma_garch,
            strike_type=strike_type,
        )

        if theoretical_prob is None:
            return

        # Check exits for open positions
        if self.position_manager.has_position(strike_str):
            current_prices = {strike_str: market_price}
            theoretical_probs = {strike_str: theoretical_prob}
            exits = self.position_manager.check_exits(
                now, current_prices, theoretical_probs, current_spot,
            )
            for exit_trade in exits:
                self._log_paper_trade(exit_trade)

        # Generate signal
        edge = theoretical_prob - (market_price / 100.0)

        if edge >= 0:
            return

        if self.position_manager.has_position(strike_str):
            return
        if self.stop_loss_monitor.is_blocked(strike_str):
            return

        dynamic_threshold = calculate_dynamic_threshold(days_to_expiry)
        if abs(edge) < dynamic_threshold:
            return

        fill_result = self.fill_simulator.simulate_sell_fill(
            orderbook, theoretical_prob,
        )
        if fill_result.contracts_filled == 0:
            return

        kelly_frac, num_contracts = self.kelly_sizer.calculate_position_size(
            fill_result, theoretical_prob, strike_type,
        )

        if num_contracts <= 0:
            return

        position = Position(
            strike_range=strike_str,
            entry_time=now,
            entry_price=fill_result.avg_fill_price,
            num_contracts=num_contracts,
            edge=edge,
            strike_type=strike_type,
            lower_bound=lower,
            upper_bound=upper,
        )

        record = self.position_manager.enter(position)
        if record:
            record.spot_price = current_spot
            record.theoretical_prob = theoretical_prob
            record.market_prob = market_price / 100.0
            record.kelly_fraction = kelly_frac
            record.days_to_expiry = days_to_expiry
            self._log_paper_trade(record)

    def _log_paper_trade(self, trade_record):
        """Log a paper trade to CSV."""
        file_exists = os.path.exists(PAPER_TRADES_FILE)

        row = {
            'timestamp': trade_record.timestamp,
            'strike_range': trade_record.strike_range,
            'action': trade_record.action,
            'price': trade_record.price,
            'num_contracts': trade_record.num_contracts,
            'spot_price': trade_record.spot_price,
            'edge': trade_record.edge,
            'theoretical_prob': trade_record.theoretical_prob,
            'market_prob': trade_record.market_prob,
            'kelly_fraction': trade_record.kelly_fraction,
            'exit_reason': trade_record.exit_reason,
            'unrealized_pnl': trade_record.unrealized_pnl,
        }

        with open(PAPER_TRADES_FILE, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        logger.info(
            "PAPER TRADE: %s %s @ %.1fc x%d",
            trade_record.action, trade_record.strike_range,
            trade_record.price, trade_record.num_contracts,
        )

    def _save_state(self):
        """Save current state for resumability."""
        logger.info("Final bankroll: $%.2f",
                     self.kelly_sizer.current_bankroll / 100)
        logger.info("Open positions: %d",
                     len(self.position_manager.open_positions))
        for strike, pos in self.position_manager.open_positions.items():
            logger.info("  %s: %d contracts @ %.1fc",
                        strike, pos.num_contracts, pos.entry_price)

    def stop(self):
        """Gracefully stop paper trading."""
        self._running = False
        logger.info("Stopping paper trader...")
