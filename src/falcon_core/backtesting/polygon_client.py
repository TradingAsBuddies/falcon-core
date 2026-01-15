"""
Polygon.io Client for Intraday Market Data

Fetches minute-level and aggregated bar data from Polygon.io API.
Requires a Polygon.io API key (free tier available).

Features:
- Minute bars (1m, 5m, 15m)
- Daily bars with extended history
- Pre/post market data support
- Rate limiting and caching
- Automatic pagination for large requests
"""

import os
import time
import logging
from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Optional, Union
import requests
import pandas as pd

logger = logging.getLogger(__name__)


class PolygonClient:
    """
    Client for Polygon.io market data API.

    Free tier limits:
    - 5 API calls/minute
    - 2 years of historical data
    - End-of-day data only (15-min delayed for intraday)

    Paid tiers provide real-time and more history.
    """

    BASE_URL = "https://api.polygon.io"

    # Rate limiting for free tier
    RATE_LIMIT_CALLS = 5
    RATE_LIMIT_PERIOD = 60  # seconds

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        rate_limit: bool = True,
    ):
        """
        Initialize Polygon client.

        Args:
            api_key: Polygon.io API key (or set POLYGON_API_KEY env var)
            cache_dir: Directory to cache responses
            rate_limit: Whether to enforce rate limiting
        """
        self.api_key = api_key or os.getenv('POLYGON_API_KEY')
        if not self.api_key:
            logger.warning(
                "No Polygon API key provided. Set POLYGON_API_KEY environment variable "
                "or pass api_key parameter. Get free key at https://polygon.io/"
            )

        self.cache_dir = cache_dir or os.path.expanduser('~/.cache/falcon/polygon')
        os.makedirs(self.cache_dir, exist_ok=True)

        self.rate_limit = rate_limit
        self._call_times: List[float] = []
        self._session = requests.Session()

    def _wait_for_rate_limit(self):
        """Wait if necessary to respect rate limits"""
        if not self.rate_limit:
            return

        now = time.time()
        # Remove calls older than the rate limit period
        self._call_times = [t for t in self._call_times if now - t < self.RATE_LIMIT_PERIOD]

        if len(self._call_times) >= self.RATE_LIMIT_CALLS:
            # Wait until oldest call expires
            wait_time = self.RATE_LIMIT_PERIOD - (now - self._call_times[0]) + 0.1
            if wait_time > 0:
                logger.debug(f"Rate limit reached, waiting {wait_time:.1f}s")
                time.sleep(wait_time)

        self._call_times.append(time.time())

    def _request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make API request with rate limiting"""
        if not self.api_key:
            raise ValueError("Polygon API key required. Set POLYGON_API_KEY env var.")

        self._wait_for_rate_limit()

        url = f"{self.BASE_URL}{endpoint}"
        params = params or {}
        params['apiKey'] = self.api_key

        try:
            response = self._session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Polygon API request failed: {e}")
            raise

    def get_aggregates(
        self,
        symbol: str,
        multiplier: int,
        timespan: str,
        from_date: Union[str, date, datetime],
        to_date: Union[str, date, datetime],
        adjusted: bool = True,
        sort: str = 'asc',
        limit: int = 50000,
    ) -> pd.DataFrame:
        """
        Get aggregate bars (OHLCV) for a symbol.

        Args:
            symbol: Stock ticker symbol
            multiplier: Size of the timespan multiplier (e.g., 5 for 5-minute bars)
            timespan: 'minute', 'hour', 'day', 'week', 'month', 'quarter', 'year'
            from_date: Start date
            to_date: End date
            adjusted: Whether to adjust for splits
            sort: 'asc' or 'desc'
            limit: Max results per request

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume, vwap, trades
        """
        # Format dates
        if isinstance(from_date, (date, datetime)):
            from_date = from_date.strftime('%Y-%m-%d')
        if isinstance(to_date, (date, datetime)):
            to_date = to_date.strftime('%Y-%m-%d')

        endpoint = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"

        params = {
            'adjusted': str(adjusted).lower(),
            'sort': sort,
            'limit': limit,
        }

        all_results = []

        while True:
            data = self._request(endpoint, params)

            if data.get('status') != 'OK':
                error = data.get('error', 'Unknown error')
                logger.warning(f"Polygon API error for {symbol}: {error}")
                break

            results = data.get('results', [])
            if not results:
                break

            all_results.extend(results)

            # Check for pagination
            if data.get('next_url'):
                # Parse next_url for cursor
                endpoint = data['next_url'].replace(self.BASE_URL, '')
                params = {}  # URL already has params
            else:
                break

        if not all_results:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(all_results)

        # Rename columns
        column_map = {
            't': 'timestamp',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            'vw': 'vwap',
            'n': 'trades',
        }
        df = df.rename(columns=column_map)

        # Convert timestamp from milliseconds to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')

        # Ensure required columns exist
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                df[col] = 0

        logger.info(f"Loaded {len(df)} bars for {symbol} ({multiplier}{timespan})")
        return df

    def get_intraday_bars(
        self,
        symbol: str,
        date: Union[str, date, datetime],
        interval: str = '5m',
        include_premarket: bool = False,
        include_afterhours: bool = False,
    ) -> pd.DataFrame:
        """
        Get intraday bars for a specific date.

        Args:
            symbol: Stock ticker
            date: Trading date
            interval: '1m', '5m', '15m', '30m', '1h'
            include_premarket: Include pre-market data (4:00 AM - 9:30 AM ET)
            include_afterhours: Include after-hours data (4:00 PM - 8:00 PM ET)

        Returns:
            DataFrame with OHLCV data
        """
        # Parse interval
        interval_map = {
            '1m': (1, 'minute'),
            '5m': (5, 'minute'),
            '15m': (15, 'minute'),
            '30m': (30, 'minute'),
            '1h': (1, 'hour'),
        }

        if interval not in interval_map:
            raise ValueError(f"Invalid interval: {interval}. Use: {list(interval_map.keys())}")

        multiplier, timespan = interval_map[interval]

        # Format date
        if isinstance(date, datetime):
            date = date.date()
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d').date()

        # Get data for the day
        df = self.get_aggregates(
            symbol=symbol,
            multiplier=multiplier,
            timespan=timespan,
            from_date=date,
            to_date=date,
        )

        if df.empty:
            return df

        # Filter by market hours if requested
        if not include_premarket or not include_afterhours:
            # Convert to Eastern time for filtering
            df_et = df.copy()
            if df_et.index.tz is None:
                df_et.index = df_et.index.tz_localize('UTC')
            df_et.index = df_et.index.tz_convert('America/New_York')

            mask = pd.Series(True, index=df_et.index)

            if not include_premarket:
                # Regular market starts at 9:30 AM ET
                mask &= df_et.index.time >= pd.Timestamp('09:30').time()

            if not include_afterhours:
                # Regular market ends at 4:00 PM ET
                mask &= df_et.index.time < pd.Timestamp('16:00').time()

            df = df[mask.values]

        return df

    def get_multi_day_intraday(
        self,
        symbol: str,
        start_date: Union[str, date, datetime],
        end_date: Union[str, date, datetime],
        interval: str = '5m',
    ) -> pd.DataFrame:
        """
        Get intraday data for multiple days.

        Args:
            symbol: Stock ticker
            start_date: Start date
            end_date: End date
            interval: Bar interval ('1m', '5m', '15m', '30m', '1h')

        Returns:
            DataFrame with OHLCV data for all days
        """
        interval_map = {
            '1m': (1, 'minute'),
            '5m': (5, 'minute'),
            '15m': (15, 'minute'),
            '30m': (30, 'minute'),
            '1h': (1, 'hour'),
        }

        if interval not in interval_map:
            raise ValueError(f"Invalid interval: {interval}")

        multiplier, timespan = interval_map[interval]

        return self.get_aggregates(
            symbol=symbol,
            multiplier=multiplier,
            timespan=timespan,
            from_date=start_date,
            to_date=end_date,
        )

    def get_previous_close(self, symbol: str) -> Dict[str, Any]:
        """Get previous day's close data"""
        endpoint = f"/v2/aggs/ticker/{symbol}/prev"
        data = self._request(endpoint)

        if data.get('status') == 'OK' and data.get('results'):
            return data['results'][0]
        return {}

    def get_ticker_details(self, symbol: str) -> Dict[str, Any]:
        """Get ticker details (name, market cap, etc.)"""
        endpoint = f"/v3/reference/tickers/{symbol}"
        data = self._request(endpoint)

        if data.get('status') == 'OK':
            return data.get('results', {})
        return {}

    def save_to_cache(self, df: pd.DataFrame, symbol: str, interval: str, date_str: str):
        """Save data to cache file"""
        filename = f"{symbol}_{interval}_{date_str}.parquet"
        filepath = os.path.join(self.cache_dir, filename)
        df.to_parquet(filepath)
        logger.debug(f"Cached {len(df)} bars to {filepath}")

    def load_from_cache(self, symbol: str, interval: str, date_str: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available"""
        filename = f"{symbol}_{interval}_{date_str}.parquet"
        filepath = os.path.join(self.cache_dir, filename)

        if os.path.exists(filepath):
            try:
                df = pd.read_parquet(filepath)
                logger.debug(f"Loaded {len(df)} bars from cache: {filepath}")
                return df
            except Exception as e:
                logger.warning(f"Failed to load cache file {filepath}: {e}")

        return None


def get_polygon_client(api_key: Optional[str] = None) -> PolygonClient:
    """Factory function to create Polygon client"""
    return PolygonClient(api_key=api_key)
