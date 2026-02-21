import pandas as pd
import yfinance as yf
import time
import glob
import os
from dataclasses import dataclass
from typing import Optional
from src.utils import setup_logging

logger = setup_logging()


@dataclass
class MarketDataPoint:
    """Single market data observation."""
    timestamp: pd.Timestamp
    strike_range: str
    price: float
    orderbook_snapshot: Optional[dict] = None


def load_kalshi_csv(csv_path):
    """Load Kalshi market data from CSV, melt wide to long format.

    Returns DataFrame with columns: timestamp, strike_range, price
    """
    logger.info(f"Loading data from {csv_path}...")

    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    value_vars = [col for col in df.columns if col != 'timestamp']

    df_long = df.melt(
        id_vars=['timestamp'],
        value_vars=value_vars,
        var_name='strike_range',
        value_name='price'
    )

    df_long = df_long.dropna(subset=['price'])

    logger.info(f"Parsed {len(df_long)} data points, {df_long['strike_range'].nunique()} strikes")
    logger.info(f"Date range: {df_long['timestamp'].min()} to {df_long['timestamp'].max()}")

    return df_long


def load_all_market_csvs(csv_dir='.', pattern='kalshi-oil-*.csv'):
    """Load all market CSV files from directory.

    Returns dict keyed by filename -> DataFrame
    """
    search_pattern = os.path.join(csv_dir, pattern)
    files = sorted(glob.glob(search_pattern))

    if not files:
        logger.warning(f"No files matching {search_pattern}")
        return {}

    result = {}
    for filepath in files:
        filename = os.path.basename(filepath)
        try:
            result[filename] = load_kalshi_csv(filepath)
        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")

    logger.info(f"Loaded {len(result)} market files")
    return result


def get_oil_prices(start_date, end_date, extended_lookback=120, max_retries=3):
    """Fetch WTI crude oil prices from Yahoo Finance with retry logic.

    Args:
        start_date: Start of market period
        end_date: End of market period
        extended_lookback: Extra days before start_date for GARCH calibration
        max_retries: Number of retry attempts

    Returns:
        pandas Series of closing prices
    """
    extended_start = pd.to_datetime(start_date) - pd.Timedelta(days=extended_lookback)

    for attempt in range(max_retries):
        try:
            logger.info(f"Fetching WTI oil prices (attempt {attempt + 1}/{max_retries})...")
            oil = yf.download(
                'CL=F',
                start=extended_start,
                end=end_date,
                interval='1h',
                progress=False,
                auto_adjust=True,
                timeout=60
            )

            if len(oil) > 0 and len(oil['Close']) > 0:
                oil_prices = oil['Close']
                if oil_prices.index.tz is not None:
                    oil_prices.index = oil_prices.index.tz_convert('UTC').tz_localize(None)
                logger.info(f"Fetched {len(oil_prices)} hourly bars of oil data")
                logger.info(f"Price range: ${oil_prices.min().item():.2f} - ${oil_prices.max().item():.2f}")
                return oil_prices
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)

    raise ValueError("Failed to download oil price data after all retries.")
