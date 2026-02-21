import logging
from src.config import KELLY_FRACTION, MIN_EDGE_THRESHOLD, BANKROLL

logger = logging.getLogger('kalshi_oil')


class KellyPositionSizer:
    """Kelly Criterion position sizing for binary prediction markets.

    Adapted for orderbook-aware fills: position size is capped by BOTH
    the Kelly fraction AND the available orderbook liquidity.
    """

    def __init__(self, bankroll=BANKROLL, kelly_fraction=KELLY_FRACTION,
                 min_edge=MIN_EDGE_THRESHOLD, max_position=0.20):
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge
        self.max_position = max_position
        self.current_bankroll = bankroll
        self.trade_history = []

    def calculate_kelly_fraction_sell(self, market_price_cents, theoretical_prob):
        """Calculate Kelly fraction for a SELL position.

        Edge = market_prob - theoretical_prob (we think market is overpriced).
        """
        market_prob = market_price_cents / 100.0
        edge = market_prob - theoretical_prob

        if edge <= 0:
            return 0.0

        # SELL Kelly for binary options:
        # win_prob = 1 - theoretical_prob, win_amount = entry_price
        # lose_prob = theoretical_prob, lose_amount = 100 - entry_price
        # f* = edge / (market_prob * (1 - market_prob))
        denom = market_prob * (1 - market_prob)
        kelly = edge / denom if denom > 0.01 else 0.0
        kelly_position = min(kelly * self.kelly_fraction, self.max_position)

        return max(0.0, kelly_position)

    def calculate_position_size(self, fill_result, theoretical_prob, strike_type='standard'):
        """Calculate position size capped by Kelly AND orderbook liquidity.

        Args:
            fill_result: FillResult from orderbook simulator
            theoretical_prob: Our model's probability
            strike_type: 'standard', 'or_above', or 'or_below'

        Returns:
            (kelly_frac, num_contracts) tuple
        """
        if fill_result.contracts_filled == 0:
            return 0.0, 0

        avg_price = fill_result.avg_fill_price
        kelly_frac = self.calculate_kelly_fraction_sell(avg_price, theoretical_prob)

        if kelly_frac <= 0:
            return 0.0, 0

        # Tail strike adjustment
        tail_adjustment = 0.2 if strike_type in ('or_above', 'or_below') else 1.0

        # Kelly-based contract limit
        risk_amount = self.current_bankroll * kelly_frac * tail_adjustment
        max_loss_per_contract = 100 - avg_price
        kelly_contracts = int(risk_amount / max_loss_per_contract) if max_loss_per_contract > 0 else 0

        # Cap by expected fill (probabilistically discounted liquidity)
        num_contracts = min(kelly_contracts, fill_result.expected_fill)

        return kelly_frac, max(0, num_contracts)

    def update_bankroll(self, pnl_cents):
        """Update bankroll after a trade settles."""
        self.current_bankroll += pnl_cents
        self.trade_history.append({
            'pnl': pnl_cents,
            'bankroll': self.current_bankroll
        })


def calculate_dynamic_threshold(time_to_expiry_days, base_threshold=MIN_EDGE_THRESHOLD):
    """Scale edge threshold by time to expiry."""
    if time_to_expiry_days > 5:
        return base_threshold * 1.5
    elif time_to_expiry_days > 2:
        return base_threshold * 1.2
    elif time_to_expiry_days > 1:
        return base_threshold
    else:
        return base_threshold * 0.8


def generate_signals(market_snapshot, oil_prices, current_spot, timestamp,
                     time_to_expiry, sigma_garch, terminal_prices=None):
    """Generate SELL signals from market snapshot.

    Args:
        market_snapshot: DataFrame with strike_range, price columns for one timestamp
        oil_prices: Historical oil price series
        current_spot: Current oil spot price
        timestamp: Current timestamp
        time_to_expiry: Years to expiry
        sigma_garch: GARCH volatility estimate
        terminal_prices: Array of MC simulated terminal prices (optional)

    Returns:
        List of signal dicts with strike_range, theoretical_prob, market_prob, edge
    """
    from src.probability_model import calculate_hybrid_probability
    from src.utils import parse_strike_range

    signals = []

    for _, row in market_snapshot.iterrows():
        strike_range = row['strike_range']
        market_price_cents = row['price']
        market_prob = market_price_cents / 100.0

        lower_bound, upper_bound, strike_type = parse_strike_range(strike_range)

        theoretical_prob = calculate_hybrid_probability(
            oil_prices=oil_prices,
            current_spot=current_spot,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            timestamp=timestamp,
            time_to_expiry=time_to_expiry,
            sigma_garch=sigma_garch,
            strike_type=strike_type,
            terminal_prices=terminal_prices,
        )

        if theoretical_prob is None:
            continue

        edge = theoretical_prob - market_prob

        # SELL signals only: edge < 0 means market overpriced
        if edge >= 0:
            continue

        signals.append({
            'strike_range': strike_range,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'strike_type': strike_type,
            'theoretical_prob': theoretical_prob,
            'market_price_cents': market_price_cents,
            'market_prob': market_prob,
            'edge': edge,
            'abs_edge': abs(edge),
        })

    # Sort by absolute edge descending (best opportunities first)
    signals.sort(key=lambda x: x['abs_edge'], reverse=True)

    return signals
