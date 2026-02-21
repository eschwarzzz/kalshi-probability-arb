import numpy as np
from scipy.stats import norm
from arch import arch_model

from src.config import HOURS_PER_TRADING_DAY, TRADING_HOURS_PER_YEAR, EMPIRICAL_LOOKBACK_HOURS, GARCH_MIN_OBSERVATIONS
from src.utils import normalize_timestamp, setup_logging

logger = setup_logging()


def calculate_garch_volatility(oil_prices, timestamp, forecast_horizon=7):
    """Fit GARCH(1,1) to hourly returns and forecast annualized volatility.

    Uses Monte Carlo simulation to forecast variance over the horizon.
    Falls back to simple rolling standard deviation if GARCH fitting fails.

    Args:
        oil_prices: Series of hourly oil prices with datetime index.
        timestamp: Current timestamp to split historical data.
        forecast_horizon: Number of days to forecast variance over.

    Returns:
        Annualized volatility as a float, or None if insufficient data.
    """
    timestamp_naive = normalize_timestamp(timestamp)
    historical = oil_prices[oil_prices.index < timestamp_naive]
    if len(historical) < GARCH_MIN_OBSERVATIONS:
        return None
    returns = 100 * historical.pct_change().dropna()
    if len(returns) < GARCH_MIN_OBSERVATIONS:
        return None
    forecast_hours = max(1, forecast_horizon * HOURS_PER_TRADING_DAY)
    try:
        model = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
        fitted = model.fit(disp='off', show_warning=False)
        forecast = fitted.forecast(horizon=forecast_hours, method='simulation',
                                   simulations=1000, reindex=False)
        variance_forecast = forecast.variance.values[-1, :].mean()
        vol_annual = np.sqrt(variance_forecast) / 100 * np.sqrt(TRADING_HOURS_PER_YEAR)
        return vol_annual
    except Exception:
        returns_recent = returns.tail(forecast_hours)
        vol_hourly = returns_recent.std() / 100
        return vol_hourly * np.sqrt(TRADING_HOURS_PER_YEAR)


def simulate_garch_price_paths(oil_prices, timestamp, forecast_horizon=7,
                                n_simulations=5000):
    """Simulate price paths using GARCH(1,1) Monte Carlo.

    Fits GARCH to historical hourly returns, then simulates forward price
    paths over the forecast horizon. Returns terminal prices for each path.

    Args:
        oil_prices: Series of hourly oil prices with datetime index.
        timestamp: Current timestamp to split historical data.
        forecast_horizon: Number of days to forecast.
        n_simulations: Number of Monte Carlo paths to simulate.

    Returns:
        (current_spot, terminal_prices) tuple where terminal_prices is an
        array of simulated prices at expiry, or (current_spot, None) if
        insufficient data or GARCH fitting fails.
    """
    timestamp_naive = normalize_timestamp(timestamp)
    historical = oil_prices[oil_prices.index < timestamp_naive]
    if len(historical) < GARCH_MIN_OBSERVATIONS:
        return None, None

    current_spot = historical.iloc[-1].item()
    returns = 100 * historical.pct_change().dropna()
    if len(returns) < GARCH_MIN_OBSERVATIONS:
        return current_spot, None

    forecast_hours = max(1, forecast_horizon * HOURS_PER_TRADING_DAY)

    try:
        model = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
        fitted = model.fit(disp='off', show_warning=False)

        # Simulate return paths: shape (forecast_hours, n_simulations)
        forecast = fitted.forecast(horizon=forecast_hours, method='simulation',
                                   simulations=n_simulations, reindex=False)
        # forecast.simulations.values shape: (1, n_simulations, forecast_hours)
        simulated_returns = forecast.simulations.values[-1, :, :] / 100  # undo 100x scaling

        # Cumulate returns to get price paths
        cumulative_returns = simulated_returns.sum(axis=1)  # sum of hourly returns per path
        terminal_prices = current_spot * np.exp(cumulative_returns)

        return current_spot, terminal_prices
    except Exception:
        return current_spot, None


def calculate_empirical_probability(oil_prices, current_spot, lower_bound,
                                     upper_bound, timestamp, time_to_expiry,
                                     lookback_bars=EMPIRICAL_LOOKBACK_HOURS):
    """Calculate probability of price landing in a range using historical returns.

    Uses overlapping multi-bar returns matching the time horizon to expiry,
    so the empirical distribution reflects the correct magnitude of price moves.

    Args:
        oil_prices: Series of historical oil prices with datetime index.
        current_spot: Current oil price.
        lower_bound: Lower bound of the strike range.
        upper_bound: Upper bound of the strike range.
        timestamp: Current timestamp to split historical data.
        time_to_expiry: Time to expiry in years.
        lookback_bars: Number of hourly bars to look back.

    Returns:
        Probability as a float in [0, 1], or None if insufficient data.
    """
    timestamp_naive = normalize_timestamp(timestamp)
    historical = oil_prices[oil_prices.index < timestamp_naive]
    if len(historical) < lookback_bars:
        lookback_bars = len(historical)
    if lookback_bars < 100:
        return None
    recent = historical.tail(lookback_bars)

    # Match return horizon to time-to-expiry (in hourly bars, min 1, max 500)
    horizon_hours = max(1, min(int(time_to_expiry * TRADING_HOURS_PER_YEAR), 500))
    returns = np.log(recent / recent.shift(horizon_hours)).dropna()
    if len(returns) < 20:
        return None
    target_return_lower = np.log(lower_bound / current_spot) if lower_bound > 0 else -np.inf
    target_return_upper = np.log(upper_bound / current_spot) if upper_bound > 0 else -np.inf
    in_range = ((returns >= target_return_lower) & (returns < target_return_upper)).sum()
    count = int(in_range.iloc[0]) if hasattr(in_range, 'iloc') else int(in_range)
    prob = count / len(returns)
    return max(0.0, min(prob, 1.0))


def calculate_mean_reversion_adjustment(oil_prices, current_spot, timestamp, lookback_bars=60):
    """Compute a mean-reversion z-score adjustment factor.

    Returns a small adjustment based on how far the current spot deviates
    from the recent mean price, scaled by -0.05 per z-score unit.

    Args:
        oil_prices: Series of historical oil prices.
        current_spot: Current oil price.
        timestamp: Current timestamp to avoid look-ahead.
        lookback_bars: Number of recent bars to consider.

    Returns:
        Adjustment factor as a float. Positive when price is below mean
        (suggesting reversion upward), negative when above.
    """
    timestamp_naive = normalize_timestamp(timestamp)
    historical = oil_prices[oil_prices.index < timestamp_naive]
    recent = historical.tail(lookback_bars)
    if len(recent) < 20:
        return 0.0
    mean_price = float(recent.mean().iloc[0]) if hasattr(recent.mean(), 'iloc') else float(recent.mean())
    std_price = float(recent.std().iloc[0]) if hasattr(recent.std(), 'iloc') else float(recent.std())
    if std_price == 0:
        return 0.0
    z_score = (current_spot - mean_price) / std_price
    return -z_score * 0.05


def detect_price_regime(oil_prices, timestamp, lookback=10):
    """Detect whether recent price action is trending or sideways.

    Compares mean return to volatility to classify the regime.

    Args:
        oil_prices: Series of historical oil prices with datetime index.
        timestamp: Current timestamp to split historical data.
        lookback: Number of recent data points to analyze.

    Returns:
        One of 'UPTREND', 'DOWNTREND', or 'SIDEWAYS'.
    """
    timestamp_naive = normalize_timestamp(timestamp)
    recent = oil_prices[oil_prices.index < timestamp_naive].tail(lookback)
    if len(recent) < lookback:
        return 'SIDEWAYS'
    pct = recent.pct_change()
    returns = float(pct.mean().iloc[0]) if hasattr(pct.mean(), 'iloc') else float(pct.mean())
    volatility = float(pct.std().iloc[0]) if hasattr(pct.std(), 'iloc') else float(pct.std())
    if returns > volatility * 0.5:
        return 'UPTREND'
    elif returns < -volatility * 0.5:
        return 'DOWNTREND'
    else:
        return 'SIDEWAYS'


def calculate_upper_tail_probability(current_spot, barrier, time_to_expiry, sigma):
    """Calculate probability that price exceeds a barrier using lognormal model.

    Applies a proximity boost when spot is close to the barrier.

    Args:
        current_spot: Current oil price.
        barrier: The barrier price level.
        time_to_expiry: Time to expiry in years.
        sigma: Annualized volatility.

    Returns:
        Probability as a float in [0, 1].
    """
    if current_spot >= barrier:
        return 1.0
    if time_to_expiry <= 0 or sigma <= 0:
        return 0.0
    d = (np.log(barrier / current_spot) - 0.5 * sigma**2 * time_to_expiry) / (
        sigma * np.sqrt(time_to_expiry)
    )
    prob = 1 - norm.cdf(d)
    proximity_ratio = current_spot / barrier
    if proximity_ratio > 0.95:
        prob *= 1.5
    elif proximity_ratio > 0.90:
        prob *= 1.3
    return min(prob, 1.0)


def calculate_mc_probability(terminal_prices, lower_bound, upper_bound):
    """Calculate probability from Monte Carlo simulated terminal prices.

    Counts the fraction of simulated paths that land in [lower_bound, upper_bound).

    Args:
        terminal_prices: Array of simulated terminal prices.
        lower_bound: Lower bound of the strike range.
        upper_bound: Upper bound of the strike range.

    Returns:
        Probability as a float in [0, 1].
    """
    in_range = (terminal_prices >= lower_bound) & (terminal_prices < upper_bound)
    return float(in_range.mean())


def calculate_hybrid_probability(oil_prices, current_spot, lower_bound, upper_bound,
                                  timestamp, time_to_expiry, sigma_garch,
                                  strike_type='standard', terminal_prices=None):
    """Blend Monte Carlo and empirical probabilities 50/50 with mean-reversion adjustment.

    For 'or_above' strike types, uses MC paths to count tail probability with
    regime-based scaling. For standard strikes, averages MC path probability
    and empirical estimates, then applies a mean-reversion correction.

    Args:
        oil_prices: Series of historical oil prices with datetime index.
        current_spot: Current oil price.
        lower_bound: Lower bound of the strike range.
        upper_bound: Upper bound of the strike range.
        timestamp: Current timestamp to split historical data.
        time_to_expiry: Time to expiry in years.
        sigma_garch: GARCH-estimated annualized volatility (used as fallback).
        strike_type: One of 'standard', 'or_above', 'or_below'.
        terminal_prices: Array of simulated terminal prices from GARCH MC.

    Returns:
        Combined probability as a float in [0, 1], or None if no estimates
        are available.
    """
    if strike_type == 'or_above':
        if terminal_prices is not None:
            tail_prob = float((terminal_prices >= lower_bound).mean())
        else:
            tail_prob = calculate_upper_tail_probability(
                current_spot, lower_bound, time_to_expiry, sigma_garch
            )
        regime = detect_price_regime(oil_prices, timestamp)
        if regime == 'UPTREND':
            tail_prob *= 1.4
        elif regime == 'DOWNTREND':
            tail_prob *= 0.7
        return min(tail_prob, 1.0)

    # Monte Carlo probability from GARCH simulated paths
    if terminal_prices is not None:
        prob_mc = calculate_mc_probability(terminal_prices, lower_bound, upper_bound)
    elif sigma_garch and time_to_expiry > 0:
        # Fallback to lognormal CDF if no MC paths available
        denom = sigma_garch * np.sqrt(time_to_expiry)
        drift = 0.5 * sigma_garch**2 * time_to_expiry
        if lower_bound > 0:
            d_lower = (np.log(lower_bound / current_spot) - drift) / denom
        else:
            d_lower = -np.inf
        d_upper = (np.log(upper_bound / current_spot) - drift) / denom
        prob_mc = norm.cdf(d_upper) - norm.cdf(d_lower)
    else:
        prob_mc = None

    prob_empirical = calculate_empirical_probability(
        oil_prices, current_spot, lower_bound, upper_bound, timestamp,
        time_to_expiry=time_to_expiry, lookback_bars=EMPIRICAL_LOOKBACK_HOURS
    )
    mr_adjustment = calculate_mean_reversion_adjustment(
        oil_prices, current_spot, timestamp, lookback_bars=60
    )

    weights = []
    probs = []

    if prob_mc is not None:
        weights.append(0.5)
        probs.append(prob_mc)

    if prob_empirical is not None:
        weights.append(0.5)
        probs.append(prob_empirical)

    if len(probs) == 0:
        return None

    combined_prob = np.average(probs, weights=weights)

    if lower_bound <= current_spot < upper_bound:
        combined_prob = combined_prob * (1 + abs(mr_adjustment))
    else:
        combined_prob = combined_prob * (1 + mr_adjustment)

    return max(0.0, min(combined_prob, 1.0))
