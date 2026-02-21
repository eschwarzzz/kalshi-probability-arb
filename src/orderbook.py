import logging
from dataclasses import dataclass

from src.config import BASE_FILL_PROBABILITY, FILL_DECAY_PER_LEVEL, WIDE_SPREAD_THRESHOLD, WIDE_SPREAD_PENALTY

logger = logging.getLogger('kalshi_oil')


@dataclass
class OrderbookLevel:
    """Single price level in the orderbook."""
    price_cents: float
    quantity: int


@dataclass
class FillResult:
    """Result of simulating a fill against the orderbook."""
    contracts_filled: int
    avg_fill_price: float
    total_value: float
    levels_consumed: int
    edge_at_fill: float
    fill_probability: float = 1.0
    expected_fill: int = 0


class Orderbook:
    """Orderbook representation for a binary market.

    Kalshi orderbooks only have bids (no asks):
    - YES bids: people wanting to buy YES contracts
    - NO bids: people wanting to buy NO contracts

    A YES bid at price X is equivalent to a NO ask at (100 - X).
    So to SELL YES contracts, we fill against YES bids (selling to buyers).
    """

    def __init__(self, yes_bids=None, no_bids=None):
        self.yes_bids = sorted(yes_bids or [], key=lambda x: x.price_cents)
        self.no_bids = sorted(no_bids or [], key=lambda x: x.price_cents)

    @property
    def best_yes_bid(self):
        """Highest YES bid price (best price to sell YES into)."""
        if self.yes_bids:
            return self.yes_bids[-1]
        return None

    @property
    def best_no_bid(self):
        """Highest NO bid price."""
        if self.no_bids:
            return self.no_bids[-1]
        return None

    @property
    def mid_price(self):
        """Mid price derived from best bids.

        Best YES bid = sell price for YES.
        Implied YES ask = 100 - best NO bid = buy price for YES.
        Mid = average of these two.
        """
        yes_bid = self.best_yes_bid.price_cents if self.best_yes_bid else None
        no_bid = self.best_no_bid.price_cents if self.best_no_bid else None

        if yes_bid is not None and no_bid is not None:
            implied_yes_ask = 100 - no_bid
            return (yes_bid + implied_yes_ask) / 2
        elif yes_bid is not None:
            return yes_bid
        elif no_bid is not None:
            return 100 - no_bid
        return None

    @property
    def spread(self):
        """Spread between best YES bid and implied YES ask."""
        yes_bid = self.best_yes_bid.price_cents if self.best_yes_bid else None
        no_bid = self.best_no_bid.price_cents if self.best_no_bid else None

        if yes_bid is not None and no_bid is not None:
            implied_yes_ask = 100 - no_bid
            return implied_yes_ask - yes_bid
        return None

    @classmethod
    def from_api_response(cls, response_dict):
        """Parse Kalshi API orderbook response into Orderbook.

        API returns: {"yes": [[price, qty], ...], "no": [[price, qty], ...]}
        Prices are in cents, sorted ascending.
        """
        yes_bids = []
        no_bids = []

        for price, qty in response_dict.get('yes', []):
            yes_bids.append(OrderbookLevel(
                price_cents=float(price),
                quantity=int(qty)
            ))

        for price, qty in response_dict.get('no', []):
            no_bids.append(OrderbookLevel(
                price_cents=float(price),
                quantity=int(qty)
            ))

        return cls(yes_bids=yes_bids, no_bids=no_bids)

    @classmethod
    def from_csv_price(cls, price_cents, spread=2, depth_per_level=100, num_levels=5,
                       depth_step=50):
        """Create synthetic orderbook from a CSV mid-price.

        Depth increases away from best bid — the most aggressive resting
        orders (highest price) are thinnest, with more contracts at lower
        prices where buyers are less aggressive.

        Args:
            price_cents: The mid-price from CSV (in cents, 0-100)
            spread: Total spread in cents (default 2)
            depth_per_level: Contracts at the best level (default 100)
            num_levels: Number of price levels on each side (default 5)
            depth_step: Additional contracts per level deeper (default 50)
        """
        half_spread = spread / 2

        # YES bids: below mid, stepping down by 1 cent
        yes_bids = []
        for i in range(num_levels):
            level_price = price_cents - half_spread - i
            quantity = depth_per_level + i * depth_step
            if level_price >= 1:
                yes_bids.append(OrderbookLevel(
                    price_cents=level_price,
                    quantity=quantity
                ))

        # NO bids: complement of implied YES asks above mid
        # Implied YES ask = 100 - NO bid
        # So NO bid at (100 - ask_price) where ask starts at mid + half_spread
        no_bids = []
        for i in range(num_levels):
            implied_ask = price_cents + half_spread + i
            quantity = depth_per_level + i * depth_step
            if implied_ask <= 99:
                no_bid_price = 100 - implied_ask
                if no_bid_price >= 1:
                    no_bids.append(OrderbookLevel(
                        price_cents=no_bid_price,
                        quantity=quantity
                    ))

        return cls(yes_bids=yes_bids, no_bids=no_bids)


class OrderbookFillSimulator:
    """Simulates filling orders against the orderbook.

    For SELL signals: we sell YES contracts into existing YES buy orders.
    Walk the YES bid side from highest price (best bid) downward.
    At each level, if bid_price / 100 > theoretical_prob, edge exists -> fill.
    """

    @staticmethod
    def simulate_sell_fill(orderbook, theoretical_prob, max_contracts=None,
                           base_fill_prob=BASE_FILL_PROBABILITY,
                           decay_per_level=FILL_DECAY_PER_LEVEL):
        """Simulate selling YES contracts into the orderbook.

        Walks YES bids from highest to lowest, filling at each level
        where the bid price implies a probability higher than our
        theoretical probability (i.e., the market is overpriced).

        Computes a per-level fill probability that decays with depth and
        penalizes wide spreads, then aggregates into a quantity-weighted
        average fill_probability on the result.

        Args:
            orderbook: Orderbook instance
            theoretical_prob: Our model's probability estimate (0-1)
            max_contracts: Maximum contracts to fill (None = unlimited by orderbook)
            base_fill_prob: Fill probability at the best level (default from config)
            decay_per_level: Multiplicative decay per subsequent level

        Returns:
            FillResult with fill details including fill_probability and expected_fill
        """
        if not orderbook.yes_bids:
            return FillResult(
                contracts_filled=0,
                avg_fill_price=0.0,
                total_value=0.0,
                levels_consumed=0,
                edge_at_fill=0.0,
                fill_probability=0.0,
                expected_fill=0,
            )

        # Check spread for wide-spread penalty
        spread = orderbook.spread
        spread_penalty = WIDE_SPREAD_PENALTY if (spread is not None and spread > WIDE_SPREAD_THRESHOLD) else 1.0

        contracts_filled = 0
        total_value = 0.0
        levels_consumed = 0
        weighted_prob_sum = 0.0  # sum of qty * level_prob for weighted average
        expected_fill_sum = 0.0  # sum of qty * level_prob for expected contracts

        # Walk from highest bid (best) downward
        level_idx = 0
        for level in reversed(orderbook.yes_bids):
            implied_prob = level.price_cents / 100.0

            # Only fill where we have edge (market overpriced vs our model)
            if implied_prob <= theoretical_prob:
                break  # No more edge at lower prices

            available = level.quantity
            if max_contracts is not None:
                remaining = max_contracts - contracts_filled
                if remaining <= 0:
                    break
                available = min(available, remaining)

            # Per-level fill probability
            level_prob = base_fill_prob * (decay_per_level ** level_idx) * spread_penalty

            contracts_filled += available
            total_value += available * level.price_cents
            weighted_prob_sum += available * level_prob
            expected_fill_sum += available * level_prob
            levels_consumed += 1
            level_idx += 1

        avg_fill_price = total_value / contracts_filled if contracts_filled > 0 else 0.0
        edge_at_fill = (avg_fill_price / 100.0 - theoretical_prob) if contracts_filled > 0 else 0.0
        fill_probability = weighted_prob_sum / contracts_filled if contracts_filled > 0 else 0.0
        expected_fill = int(expected_fill_sum)

        return FillResult(
            contracts_filled=contracts_filled,
            avg_fill_price=avg_fill_price,
            total_value=total_value,
            levels_consumed=levels_consumed,
            edge_at_fill=edge_at_fill,
            fill_probability=fill_probability,
            expected_fill=max(expected_fill, 1) if contracts_filled > 0 else 0,
        )
