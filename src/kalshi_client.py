import json
import time as time_module
import base64
import logging
from pathlib import Path

import requests

from src.config import KALSHI_API_BASE, KALSHI_WS_URL, KALSHI_KEY_ID, KALSHI_PRIVATE_KEY_PATH

logger = logging.getLogger('kalshi_oil')


class KalshiClient:
    """REST client for Kalshi Trade API v2."""

    def __init__(self, key_id=None, private_key_path=None):
        self.key_id = key_id or KALSHI_KEY_ID
        self.private_key_path = private_key_path or KALSHI_PRIVATE_KEY_PATH
        self.base_url = KALSHI_API_BASE
        self._private_key = None
        self._session = requests.Session()

        if self.key_id and self.private_key_path:
            self._load_private_key()
        else:
            logger.warning("No Kalshi API keys configured — API calls will be unavailable")

    def _load_private_key(self):
        """Load RSA private key from file."""
        try:
            key_path = Path(self.private_key_path)
            if key_path.exists():
                from cryptography.hazmat.primitives import serialization
                with open(key_path, 'rb') as f:
                    self._private_key = serialization.load_pem_private_key(f.read(), password=None)
                logger.info("Kalshi API key loaded successfully")
            else:
                logger.warning(f"Private key file not found: {self.private_key_path}")
        except ImportError:
            logger.warning("cryptography package not installed — API auth unavailable")
        except Exception as e:
            logger.error(f"Failed to load private key: {e}")

    def _sign_request(self, method, path, timestamp_ms):
        """Create RSA-PSS signature for API request."""
        if not self._private_key:
            return None

        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        message = f"{timestamp_ms}{method}{path}"
        signature = self._private_key.sign(
            message.encode('utf-8'),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode('utf-8')

    def _get_headers(self, method, path):
        """Build authenticated request headers."""
        timestamp_ms = str(int(time_module.time() * 1000))
        signature = self._sign_request(method, path, timestamp_ms)

        if not signature:
            return {}

        return {
            'KALSHI-ACCESS-KEY': self.key_id,
            'KALSHI-ACCESS-SIGNATURE': signature,
            'KALSHI-ACCESS-TIMESTAMP': timestamp_ms,
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

    def _request(self, method, path, params=None):
        """Make authenticated API request."""
        if not self._private_key:
            logger.debug("No API key — skipping request")
            return None

        url = f"{self.base_url}{path}"
        headers = self._get_headers(method, path)

        try:
            if method == 'GET':
                resp = self._session.get(url, headers=headers, params=params, timeout=30)
            else:
                resp = self._session.request(method, url, headers=headers, json=params, timeout=30)

            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {method} {path} — {e}")
            return None

    def get_markets(self, series_ticker=None, event_ticker=None, status=None, limit=100):
        """Get list of markets with optional filters."""
        params = {'limit': limit}
        if series_ticker:
            params['series_ticker'] = series_ticker
        if event_ticker:
            params['event_ticker'] = event_ticker
        if status:
            params['status'] = status

        result = self._request('GET', '/markets', params=params)
        if result:
            return result.get('markets', [])
        return None

    def get_orderbook(self, ticker, depth=0):
        """Get orderbook for a specific market.

        Returns dict with 'yes' and 'no' bid arrays.
        Each bid is [price_cents, quantity].
        YES bid at price X = NO ask at (100 - X).
        """
        params = {'depth': depth} if depth > 0 else {}
        result = self._request('GET', f'/markets/{ticker}/orderbook', params=params)
        if result:
            return result.get('orderbook', result)
        return None

    def get_candlesticks(self, series_ticker, ticker, period_minutes=60):
        """Get OHLC candlestick data for a market.

        period_minutes: 1, 60, or 1440
        """
        params = {'period_interval': period_minutes}
        result = self._request('GET', f'/series/{series_ticker}/markets/{ticker}/candlesticks', params=params)
        if result:
            return result.get('candlesticks', [])
        return None

    def get_historical_markets(self, series_ticker=None):
        """Get historical (settled) markets."""
        params = {}
        if series_ticker:
            params['series_ticker'] = series_ticker

        result = self._request('GET', '/historical/markets', params=params)
        if result:
            return result.get('markets', [])
        return None

    @property
    def is_authenticated(self):
        """Check if API client has valid credentials."""
        return self._private_key is not None


class KalshiWebSocket:
    """WebSocket client for real-time Kalshi orderbook updates."""

    def __init__(self, client: KalshiClient):
        self.client = client
        self.ws_url = KALSHI_WS_URL
        self._ws = None
        self._orderbooks = {}  # ticker -> orderbook state
        self._snapshot_callback = None
        self._delta_callback = None
        self._running = False

    def on_snapshot(self, callback):
        """Register callback for orderbook snapshots."""
        self._snapshot_callback = callback

    def on_delta(self, callback):
        """Register callback for orderbook deltas."""
        self._delta_callback = callback

    async def connect(self, market_tickers):
        """Connect to WebSocket and subscribe to orderbook updates.

        Args:
            market_tickers: list of market ticker strings
        """
        try:
            import websockets
        except ImportError:
            logger.error("websockets package not installed")
            return

        if not self.client.is_authenticated:
            logger.error("Cannot connect WebSocket without API credentials")
            return

        self._running = True

        url = f"{self.ws_url}/trade-api/ws/v2"

        try:
            async with websockets.connect(url) as ws:
                self._ws = ws

                # Subscribe to orderbook channel
                subscribe_msg = {
                    "id": 1,
                    "cmd": "subscribe",
                    "params": {
                        "channels": ["orderbook_delta"],
                        "market_tickers": market_tickers
                    }
                }
                await ws.send(json.dumps(subscribe_msg))
                logger.info(f"Subscribed to orderbook updates for {len(market_tickers)} markets")

                # Process messages
                while self._running:
                    try:
                        message = await ws.recv()
                        data = json.loads(message)
                        self._handle_message(data)
                    except Exception as e:
                        if self._running:
                            logger.error(f"WebSocket message error: {e}")
                        break

        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")

    def _handle_message(self, data):
        """Process incoming WebSocket message."""
        msg_type = data.get('type', '')

        if msg_type == 'orderbook_snapshot':
            ticker = data.get('msg', {}).get('market_ticker', '')
            self._orderbooks[ticker] = data.get('msg', {})
            if self._snapshot_callback:
                self._snapshot_callback(ticker, data['msg'])

        elif msg_type == 'orderbook_delta':
            ticker = data.get('msg', {}).get('market_ticker', '')
            delta = data.get('msg', {})
            self._apply_delta(ticker, delta)
            if self._delta_callback:
                self._delta_callback(ticker, delta)

    def _apply_delta(self, ticker, delta):
        """Apply incremental delta to local orderbook state."""
        if ticker not in self._orderbooks:
            return

        ob = self._orderbooks[ticker]

        # Apply price level updates
        for side in ['yes', 'no']:
            if side in delta:
                for price, qty in delta[side]:
                    # Update or remove level
                    existing = ob.get(side, [])
                    updated = False
                    for i, (p, q) in enumerate(existing):
                        if p == price:
                            if qty == 0:
                                existing.pop(i)
                            else:
                                existing[i] = [price, qty]
                            updated = True
                            break
                    if not updated and qty > 0:
                        existing.append([price, qty])
                        existing.sort(key=lambda x: x[0])
                    ob[side] = existing

    def get_orderbook(self, ticker):
        """Get current local orderbook state for a market."""
        return self._orderbooks.get(ticker)

    async def disconnect(self):
        """Gracefully close WebSocket connection."""
        self._running = False
        if self._ws:
            await self._ws.close()
            logger.info("WebSocket disconnected")
