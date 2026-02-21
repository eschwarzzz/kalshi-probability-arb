import logging

def parse_strike_range(strike_str):
    """Parse strike range string into (lower_bound, upper_bound, strike_type).

    Examples:
        "$62 to 62.99" -> (62.0, 63.0, 'standard')
        "$68.0 or above" -> (68.0, 99999.0, 'or_above')
        "$50.99 or below" -> (0.0, 51.0, 'or_below')
    """
    strike_str = strike_str.strip()

    if 'or above' in strike_str:
        lower = float(strike_str.replace('$', '').split(' or above')[0])
        return (lower, 99999.0, 'or_above')
    elif 'or below' in strike_str:
        upper_raw = float(strike_str.replace('$', '').split(' or below')[0])
        return (0.0, upper_raw + 0.01, 'or_below')
    else:
        parts = strike_str.replace('$', '').split(' to ')
        lower = float(parts[0])
        upper = float(parts[1]) + 0.01
        return (lower, upper, 'standard')

def normalize_timestamp(ts):
    """Remove timezone info from a timestamp if present."""
    if hasattr(ts, 'tz') and ts.tz is not None:
        return ts.tz_localize(None)
    return ts

def setup_logging(level=logging.INFO):
    """Configure logging for the application."""
    logger = logging.getLogger('kalshi_oil')
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
