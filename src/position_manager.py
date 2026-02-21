import logging
from dataclasses import dataclass

logger = logging.getLogger('kalshi_oil')


@dataclass
class Position:
    """An open trading position."""
    strike_range: str
    entry_time: object  # pd.Timestamp
    entry_price: float  # cents
    num_contracts: int
    edge: float
    strike_type: str
    lower_bound: float = 0.0
    upper_bound: float = 0.0


@dataclass
class TradeRecord:
    """Record of an entry or exit trade."""
    timestamp: object
    strike_range: str
    action: str  # 'ENTER' or 'EXIT'
    price: float
    num_contracts: int
    spot_price: float = 0.0
    edge: float = 0.0
    theoretical_prob: float = 0.0
    market_prob: float = 0.0
    kelly_fraction: float = 0.0
    strike_type: str = 'standard'
    lower_bound: float = 0.0
    upper_bound: float = 0.0
    exit_reason: str = ''
    holding_period_hours: float = 0.0
    unrealized_pnl: float = 0.0
    days_to_expiry: float = 0.0
    fill_probability: float = 1.0


class StopLossMonitor:
    """Monitor cumulative P&L per strike and block trading when stop hit."""

    def __init__(self, max_loss_per_strike=500):
        self.strike_pnl = {}
        self.max_loss_per_strike = max_loss_per_strike
        self.blocked_strikes = set()

    def update(self, strike_range, pnl):
        if strike_range not in self.strike_pnl:
            self.strike_pnl[strike_range] = 0.0
        self.strike_pnl[strike_range] += pnl

        if self.strike_pnl[strike_range] < -self.max_loss_per_strike:
            if strike_range not in self.blocked_strikes:
                logger.warning(f"STOP LOSS HIT on {strike_range}: {self.strike_pnl[strike_range]:.2f} cents")
                self.blocked_strikes.add(strike_range)

    def is_blocked(self, strike_range):
        return strike_range in self.blocked_strikes


class PositionManager:
    """Manages open positions with active exit logic."""

    def __init__(self, max_positions=3, exit_edge_threshold=0.03,
                 profit_target=0.30, stop_loss_pct=0.50):
        self.open_positions = {}  # strike_range -> Position
        self.max_positions = max_positions
        self.exit_edge_threshold = exit_edge_threshold
        self.profit_target = profit_target
        self.stop_loss_pct = stop_loss_pct

    @property
    def available_slots(self):
        return self.max_positions - len(self.open_positions)

    def has_position(self, strike_range):
        return strike_range in self.open_positions

    def enter(self, position):
        """Enter a new position."""
        if position.strike_range in self.open_positions:
            logger.warning(f"Already have position in {position.strike_range}")
            return None

        self.open_positions[position.strike_range] = position

        record = TradeRecord(
            timestamp=position.entry_time,
            strike_range=position.strike_range,
            action='ENTER',
            price=position.entry_price,
            num_contracts=position.num_contracts,
            edge=position.edge,
            strike_type=position.strike_type,
            lower_bound=position.lower_bound,
            upper_bound=position.upper_bound,
        )

        logger.info(
            f"{position.entry_time} | {position.strike_range:15s} | "
            f"ENTER SELL | Price: {position.entry_price:.1f}c | "
            f"Edge: {position.edge:+.1%} | Contracts: {position.num_contracts}"
        )

        return record

    def exit(self, strike_range, exit_price, timestamp, reason, spot_price=0.0):
        """Exit a position."""
        if strike_range not in self.open_positions:
            return None

        position = self.open_positions[strike_range]

        # SELL position P&L: sold at entry, buy back at exit
        unrealized_pnl = (position.entry_price - exit_price) * position.num_contracts
        holding_hours = (timestamp - position.entry_time).total_seconds() / 3600

        record = TradeRecord(
            timestamp=timestamp,
            strike_range=strike_range,
            action='EXIT',
            price=exit_price,
            num_contracts=position.num_contracts,
            spot_price=spot_price,
            exit_reason=reason,
            holding_period_hours=holding_hours,
            unrealized_pnl=unrealized_pnl,
            strike_type=position.strike_type,
            lower_bound=position.lower_bound,
            upper_bound=position.upper_bound,
        )

        logger.info(
            f"{timestamp} | {strike_range:15s} | "
            f"EXIT: {reason} | P&L: {unrealized_pnl:+.2f}c | "
            f"Held: {holding_hours:.1f}h"
        )

        del self.open_positions[strike_range]
        return record

    def check_exits(self, timestamp, current_prices, theoretical_probs, spot_price=0.0):
        """Check all open positions for exit conditions.

        Args:
            timestamp: Current timestamp
            current_prices: dict of strike_range -> current market price in cents
            theoretical_probs: dict of strike_range -> current theoretical probability
            spot_price: Current oil spot price

        Returns:
            List of TradeRecord for exits
        """
        exits = []
        positions_to_exit = []

        for strike_range, position in self.open_positions.items():
            if strike_range not in current_prices:
                continue

            current_price = current_prices[strike_range]
            theoretical_prob = theoretical_probs.get(strike_range)

            if theoretical_prob is None:
                continue

            current_edge = theoretical_prob - (current_price / 100.0)

            # Unrealized P&L for SELL position
            unrealized_pnl = (position.entry_price - current_price) * position.num_contracts
            unrealized_pnl_pct = unrealized_pnl / (position.entry_price * position.num_contracts) if position.entry_price > 0 else 0

            should_exit = False
            exit_reason = ""

            # Exit 1: Edge reversed (should be buying now)
            if current_edge >= 0:
                should_exit = True
                exit_reason = f"EDGE REVERSED ({current_edge:+.1%})"

            # Exit 2: Edge decayed below threshold
            elif abs(current_edge) < self.exit_edge_threshold:
                should_exit = True
                exit_reason = f"EDGE DECAY ({abs(current_edge):.1%} < {self.exit_edge_threshold:.1%})"

            # Exit 3: Profit target hit
            elif unrealized_pnl_pct > self.profit_target:
                should_exit = True
                exit_reason = f"PROFIT TARGET ({unrealized_pnl_pct:.1%})"

            # Exit 4: Stop loss
            elif unrealized_pnl_pct < -self.stop_loss_pct:
                should_exit = True
                exit_reason = f"STOP LOSS ({unrealized_pnl_pct:.1%})"

            if should_exit:
                positions_to_exit.append((strike_range, current_price, exit_reason))

        for strike_range, exit_price, reason in positions_to_exit:
            record = self.exit(strike_range, exit_price, timestamp, reason, spot_price)
            if record:
                exits.append(record)

        return exits
