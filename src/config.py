import os

try:
    from dotenv import load_dotenv  # type: ignore[import-untyped]
    load_dotenv()
except ImportError:
    pass

# Trading parameters
BANKROLL = 10000  # cents ($100)
KELLY_FRACTION = 0.25
MIN_EDGE_THRESHOLD = 0.10
MAX_POSITIONS = 3
STOP_LOSS_PER_STRIKE = 500  # cents ($5)
FEE_RATE = 0.07
EXIT_EDGE_THRESHOLD = 0.03
PROFIT_TARGET = 0.5

# Kalshi API
KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_WS_URL = "wss://api.elections.kalshi.com"
KALSHI_KEY_ID = os.getenv("KALSHI_KEY_ID", "")
KALSHI_PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH", "")
OIL_SERIES_TICKER = "KXWTIW"

# Model parameters
GARCH_LOOKBACK = 120  # days
EMPIRICAL_LOOKBACK = 252  # days
GARCH_FORECAST_HORIZON = 7

# Hourly data constants
HOURS_PER_TRADING_DAY = 23  # CL=F trades ~23h/day on CME
TRADING_HOURS_PER_YEAR = 252 * 23  # 5796 hourly bars per year
EMPIRICAL_LOOKBACK_HOURS = 252 * 23  # ~1 year of hourly bars
GARCH_MIN_OBSERVATIONS = 100  # minimum hourly bars for GARCH fit

# Orderbook simulation
DEFAULT_SYNTHETIC_SPREAD = 2  # cents
DEFAULT_SYNTHETIC_DEPTH = 100  # contracts at best level
SYNTHETIC_NUM_LEVELS = 5

# Fill probability model
BASE_FILL_PROBABILITY = 0.90
FILL_DECAY_PER_LEVEL = 0.85
WIDE_SPREAD_THRESHOLD = 4  # cents
WIDE_SPREAD_PENALTY = 0.90

# Market files for walk-forward
MARKET_FILES = [
    ('kalshi-oil-nov14.csv', '2025-11-14', '2025-11-01', '2025-11-14'),
    ('kalshi-oil-nov21.csv', '2025-11-21', '2025-11-01', '2025-11-21'),
    ('kalshi-oil-nov28.csv', '2025-11-28', '2025-11-01', '2025-11-28'),
    ('kalshi-oil-dec5.csv', '2025-12-05', '2025-11-01', '2025-12-05'),
    ('kalshi-oil-dec19.csv', '2025-12-19', '2025-11-01', '2025-12-19'),
    ('kalshi-oil-dec26.csv', '2025-12-26', '2025-11-01', '2025-12-26'),
    ('kalshi-oil-jan2.csv', '2026-01-02', '2025-12-01', '2026-01-02'),
    ('kalshi-oil-jan9.csv', '2026-01-09', '2025-12-01', '2026-01-09'),
    ('kalshi-oil-jan16.csv', '2026-01-16', '2025-12-01', '2026-01-16'),
    ('kalshi-oil-jan23.csv', '2026-01-23', '2025-12-01', '2026-01-23'),
    ('kalshi-oil-jan30.csv', '2026-01-30', '2025-12-01', '2026-01-30'),
    ('kalshi-oil-feb6.csv', '2026-02-06', '2026-01-01', '2026-02-06'),
    ('kalshi-oil-feb13.csv', '2026-02-13', '2026-01-01', '2026-02-13'),
]
