"""
Data Feed for Falcon Backtesting

Loads market data from various sources:
- Massive Flat Files (bulk historical via S3)
- Polygon.io API (for intraday data)
- Database (TimescaleDB/PostgreSQL)
- CSV files
- yfinance (for historical data)

Priority for intraday data:
1. Massive Flat Files (fastest, no rate limits, 20+ years history)
2. Polygon.io API (real-time, requires API key)
3. yfinance (free, but limited history)
4. Database (if populated)
5. CSV files (manual import)
"""

import os
import logging
from datetime import date, datetime, timedelta, time as dt_time
from typing import Any, Dict, List, Optional, Union
import pandas as pd

logger = logging.getLogger(__name__)


class DataFeed:
    """
    Market data feed for backtesting.

    Supports multiple data sources with a unified interface.
    Optimized for intraday trading strategy development.
    """

    # Market hours (Eastern Time)
    MARKET_OPEN = dt_time(9, 30)
    MARKET_CLOSE = dt_time(16, 0)
    PREMARKET_START = dt_time(4, 0)
    AFTERHOURS_END = dt_time(20, 0)

    def __init__(
        self,
        db_manager=None,
        cache_dir: Optional[str] = None,
        polygon_api_key: Optional[str] = None,
        massive_access_key: Optional[str] = None,
        massive_secret_key: Optional[str] = None,
    ):
        """
        Initialize data feed.

        Args:
            db_manager: DatabaseManager instance for DB queries
            cache_dir: Directory to cache downloaded data
            polygon_api_key: Polygon.io API key for intraday data
            massive_access_key: Massive S3 access key for flat files
            massive_secret_key: Massive S3 secret key for flat files
        """
        self.db = db_manager
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/falcon/market_data")
        self._cache: Dict[str, pd.DataFrame] = {}
        self._polygon_client = None
        self._flatfiles_client = None
        self._polygon_api_key = polygon_api_key or os.getenv('POLYGON_API_KEY')
        self._massive_access_key = massive_access_key or os.getenv('MASSIVE_ACCESS_KEY')
        self._massive_secret_key = massive_secret_key or os.getenv('MASSIVE_SECRET_KEY')

        # Ensure cache directory exists
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    @property
    def polygon(self):
        """Lazy-load Polygon client"""
        if self._polygon_client is None and self._polygon_api_key:
            try:
                from falcon_core.backtesting.polygon_client import PolygonClient
                self._polygon_client = PolygonClient(
                    api_key=self._polygon_api_key,
                    cache_dir=self.cache_dir,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Polygon client: {e}")
        return self._polygon_client

    @property
    def flatfiles(self):
        """Lazy-load Flat Files client (Massive S3)"""
        if self._flatfiles_client is None and self._massive_access_key and self._massive_secret_key:
            try:
                from falcon_core.backtesting.flatfiles_client import FlatFilesClient
                self._flatfiles_client = FlatFilesClient(
                    access_key=self._massive_access_key,
                    secret_key=self._massive_secret_key,
                    cache_dir=self.cache_dir,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Flat Files client: {e}")
        return self._flatfiles_client

    def get_historical_data(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime, None] = None,
        interval: str = "1d",
        source: str = "auto",
        market_hours_only: bool = True,
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data for a symbol.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date for data
            end_date: End date (default: today)
            interval: Data interval ('1m', '5m', '15m', '1h', '1d')
            source: Data source ('auto', 'flatfiles', 'polygon', 'database', 'yfinance', 'csv')
            market_hours_only: Filter to regular market hours (9:30 AM - 4:00 PM ET)

        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        # Parse dates - handle date, datetime, or string
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        elif isinstance(start_date, date) and not isinstance(start_date, datetime):
            start_date = datetime.combine(start_date, datetime.min.time())

        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
        elif isinstance(end_date, date) and not isinstance(end_date, datetime):
            end_date = datetime.combine(end_date, datetime.max.time().replace(microsecond=0))

        # Check cache first
        cache_key = f"{symbol}_{interval}_{start_date.date()}_{end_date.date()}_{market_hours_only}"
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        # Determine if intraday
        is_intraday = interval in ['1m', '5m', '15m', '30m', '1h']

        # Try sources in order
        data = None

        if source == "auto":
            if is_intraday:
                # For intraday: try Flat Files first (fastest), then Polygon API, then yfinance
                data = self._try_flatfiles(symbol, start_date, end_date, interval)
                if data is None or data.empty:
                    data = self._try_polygon(symbol, start_date, end_date, interval)
                if data is None or data.empty:
                    data = self._try_yfinance(symbol, start_date, end_date, interval)
            else:
                # For daily: try flat files, then database, then yfinance
                data = self._try_flatfiles(symbol, start_date, end_date, interval)
                if data is None or data.empty:
                    data = self._try_database(symbol, start_date, end_date, interval)
                if data is None or data.empty:
                    data = self._try_yfinance(symbol, start_date, end_date, interval)
        elif source == "flatfiles":
            data = self._try_flatfiles(symbol, start_date, end_date, interval)
        elif source == "polygon":
            data = self._try_polygon(symbol, start_date, end_date, interval)
        elif source == "database":
            data = self._try_database(symbol, start_date, end_date, interval)
        elif source == "yfinance":
            data = self._try_yfinance(symbol, start_date, end_date, interval)
        elif source == "csv":
            data = self._try_csv(symbol, start_date, end_date, interval)

        if data is None or data.empty:
            raise ValueError(f"No data found for {symbol} from {start_date} to {end_date}")

        # Filter to market hours if requested
        if is_intraday and market_hours_only:
            data = self._filter_market_hours(data)

        # Cache the result
        self._cache[cache_key] = data

        return data.copy()

    def _try_flatfiles(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str,
    ) -> Optional[pd.DataFrame]:
        """Try to load data from Massive Flat Files (S3)"""
        if self.flatfiles is None:
            logger.debug("Flat Files client not available")
            return None

        try:
            if interval == "1d":
                # Daily aggregates
                data = self.flatfiles.get_daily_bars(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                )
            else:
                # Minute aggregates (flat files are 1-minute, we'll resample if needed)
                data = self.flatfiles.get_multi_day_minute_bars(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                )

                # Resample if not 1-minute
                if not data.empty and interval != "1m":
                    data = self._resample_bars(data, interval)

            if not data.empty:
                logger.info(f"Loaded {len(data)} bars for {symbol} from Flat Files")
            return data

        except Exception as e:
            logger.warning(f"Flat Files query failed for {symbol}: {e}")
            return None

    def _resample_bars(self, data: pd.DataFrame, interval: str) -> pd.DataFrame:
        """Resample minute bars to a different interval"""
        interval_map = {
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '1H',
        }

        if interval not in interval_map:
            return data

        resampled = data.resample(interval_map[interval]).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }).dropna()

        logger.debug(f"Resampled {len(data)} 1m bars to {len(resampled)} {interval} bars")
        return resampled

    def _try_polygon(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str,
    ) -> Optional[pd.DataFrame]:
        """Try to load data from Polygon.io"""
        if self.polygon is None:
            logger.debug("Polygon client not available")
            return None

        try:
            data = self.polygon.get_multi_day_intraday(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
            )

            if not data.empty:
                logger.info(f"Loaded {len(data)} bars for {symbol} from Polygon.io")
            return data

        except Exception as e:
            logger.warning(f"Polygon query failed for {symbol}: {e}")
            return None

    def _filter_market_hours(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter data to regular market hours (9:30 AM - 4:00 PM ET)"""
        if data.empty:
            return data

        # Ensure timezone aware
        if data.index.tz is None:
            # Assume UTC if no timezone
            data.index = data.index.tz_localize('UTC')

        # Convert to Eastern time for filtering
        data_et = data.copy()
        data_et.index = data_et.index.tz_convert('America/New_York')

        # Filter to market hours
        mask = (
            (data_et.index.time >= self.MARKET_OPEN) &
            (data_et.index.time < self.MARKET_CLOSE)
        )

        # Handle both Series and ndarray masks
        if hasattr(mask, 'values'):
            filtered = data[mask.values]
        else:
            filtered = data[mask]
        logger.debug(f"Filtered {len(data)} -> {len(filtered)} bars (market hours only)")
        return filtered

    def _try_database(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str,
    ) -> Optional[pd.DataFrame]:
        """Try to load data from database"""
        if self.db is None:
            return None

        try:
            # Determine table based on interval
            if interval in ["1m", "5m", "15m"]:
                table = "minute_bars"
            else:
                table = "daily_bars"

            query = f"""
                SELECT date as timestamp, open, high, low, close, volume
                FROM {table}
                WHERE symbol = %s
                AND date >= %s
                AND date <= %s
                ORDER BY date
            """
            result = self.db.execute(
                query,
                (symbol, start_date, end_date),
                fetch='all'
            )

            if not result:
                return None

            # Convert to DataFrame
            data = pd.DataFrame(result)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data = data.set_index('timestamp')

            logger.debug(f"Loaded {len(data)} rows for {symbol} from database")
            return data

        except Exception as e:
            logger.warning(f"Database query failed for {symbol}: {e}")
            return None

    def _try_yfinance(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str,
    ) -> Optional[pd.DataFrame]:
        """Try to load data from yfinance"""
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance not installed, skipping")
            return None

        try:
            # Map interval to yfinance format
            interval_map = {
                "1m": "1m",
                "5m": "5m",
                "15m": "15m",
                "1h": "1h",
                "1d": "1d",
                "1wk": "1wk",
                "1mo": "1mo",
            }
            yf_interval = interval_map.get(interval, "1d")

            # yfinance has limits on intraday data
            if interval in ["1m", "5m", "15m"]:
                # Max 7 days for 1m, 60 days for 5m/15m
                max_days = 7 if interval == "1m" else 60
                if (end_date - start_date).days > max_days:
                    logger.warning(
                        f"Limiting {interval} data to {max_days} days"
                    )
                    start_date = end_date - timedelta(days=max_days)

            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date + timedelta(days=1),  # Include end date
                interval=yf_interval,
            )

            if data.empty:
                return None

            # Standardize column names
            data.columns = data.columns.str.lower()
            data = data[['open', 'high', 'low', 'close', 'volume']]

            logger.debug(f"Loaded {len(data)} rows for {symbol} from yfinance")
            return data

        except Exception as e:
            logger.warning(f"yfinance query failed for {symbol}: {e}")
            return None

    def _try_csv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str,
    ) -> Optional[pd.DataFrame]:
        """Try to load data from CSV file"""
        # Check common CSV locations
        possible_paths = [
            os.path.join(self.cache_dir, f"{symbol}_{interval}.csv"),
            os.path.join(self.cache_dir, f"{symbol}.csv"),
            f"/var/lib/falcon/market_data/{symbol}_{interval}.csv",
            f"/var/lib/falcon/market_data/{symbol}.csv",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                try:
                    data = pd.read_csv(path, parse_dates=['timestamp'])
                    data = data.set_index('timestamp')

                    # Filter to date range
                    data = data[
                        (data.index >= start_date) &
                        (data.index <= end_date)
                    ]

                    if not data.empty:
                        logger.debug(f"Loaded {len(data)} rows for {symbol} from {path}")
                        return data

                except Exception as e:
                    logger.warning(f"Failed to load CSV {path}: {e}")

        return None

    def get_multiple_symbols(
        self,
        symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime, None] = None,
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple symbols.

        Returns dict of symbol -> DataFrame
        """
        results = {}
        for symbol in symbols:
            try:
                data = self.get_historical_data(
                    symbol, start_date, end_date, interval
                )
                results[symbol] = data
            except Exception as e:
                logger.warning(f"Failed to get data for {symbol}: {e}")

        return results

    def get_intraday_data(
        self,
        symbol: str,
        date: Union[str, datetime],
        interval: str = "5m",
    ) -> pd.DataFrame:
        """
        Get intraday data for a specific date.

        Useful for strategy development and debugging.
        """
        if isinstance(date, str):
            date = datetime.strptime(date, "%Y-%m-%d")

        return self.get_historical_data(
            symbol,
            start_date=date,
            end_date=date,
            interval=interval,
        )

    def save_to_csv(
        self,
        data: pd.DataFrame,
        symbol: str,
        interval: str = "1d",
    ) -> str:
        """Save data to CSV file in cache directory"""
        filename = f"{symbol}_{interval}.csv"
        filepath = os.path.join(self.cache_dir, filename)

        # Reset index for CSV
        data_to_save = data.reset_index()
        data_to_save.columns = ['timestamp'] + list(data_to_save.columns[1:])
        data_to_save.to_csv(filepath, index=False)

        logger.info(f"Saved {len(data)} rows to {filepath}")
        return filepath

    def clear_cache(self):
        """Clear in-memory cache"""
        self._cache.clear()
        logger.info("Data feed cache cleared")

    def get_intraday_bars(
        self,
        symbol: str,
        trading_date: Union[str, datetime],
        interval: str = "5m",
        include_premarket: bool = False,
    ) -> pd.DataFrame:
        """
        Get intraday bars for a single trading day.

        Optimized for day trading strategy backtesting.

        Args:
            symbol: Stock ticker
            trading_date: The trading date
            interval: '1m', '5m', '15m'
            include_premarket: Include 4:00 AM - 9:30 AM data

        Returns:
            DataFrame with OHLCV data for the trading day
        """
        if isinstance(trading_date, str):
            trading_date = datetime.strptime(trading_date, "%Y-%m-%d")

        data = self.get_historical_data(
            symbol=symbol,
            start_date=trading_date,
            end_date=trading_date,
            interval=interval,
            market_hours_only=not include_premarket,
        )

        return data

    def get_multi_day_intraday(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "5m",
    ) -> pd.DataFrame:
        """
        Get intraday data spanning multiple days.

        Args:
            symbol: Stock ticker
            start_date: Start date
            end_date: End date
            interval: Bar interval

        Returns:
            DataFrame with multi-day intraday data
        """
        return self.get_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            market_hours_only=True,
        )

    def get_day2_data(
        self,
        symbol: str,
        day1_date: Union[str, datetime],
        interval: str = "5m",
    ) -> Dict[str, pd.DataFrame]:
        """
        Get data for Day 1 and Day 2 of a move (for Market Memory strategy).

        Args:
            symbol: Stock ticker
            day1_date: The "Day 1" date (big move day)
            interval: Bar interval

        Returns:
            Dict with 'day1' and 'day2' DataFrames
        """
        if isinstance(day1_date, str):
            day1_date = datetime.strptime(day1_date, "%Y-%m-%d")

        # Day 2 is the next trading day
        day2_date = day1_date + timedelta(days=1)
        # Skip weekends
        while day2_date.weekday() >= 5:
            day2_date += timedelta(days=1)

        day1_data = self.get_intraday_bars(symbol, day1_date, interval)
        day2_data = self.get_intraday_bars(symbol, day2_date, interval)

        return {
            'day1': day1_data,
            'day2': day2_data,
            'day1_date': day1_date,
            'day2_date': day2_date,
        }

    def get_gappers(
        self,
        symbols: List[str],
        trading_date: Union[str, datetime],
        min_gap_pct: float = 0.03,
    ) -> List[Dict[str, Any]]:
        """
        Find stocks that gapped up/down on a given date.

        Useful for finding candidates for intraday strategies.

        Args:
            symbols: List of symbols to check
            trading_date: The trading date
            min_gap_pct: Minimum gap percentage (e.g., 0.03 = 3%)

        Returns:
            List of dicts with symbol, gap_pct, prev_close, open_price
        """
        if isinstance(trading_date, str):
            trading_date = datetime.strptime(trading_date, "%Y-%m-%d")

        prev_date = trading_date - timedelta(days=1)
        while prev_date.weekday() >= 5:
            prev_date -= timedelta(days=1)

        gappers = []

        for symbol in symbols:
            try:
                # Get previous close
                prev_data = self.get_historical_data(
                    symbol, prev_date, prev_date, '1d'
                )
                if prev_data.empty:
                    continue
                prev_close = prev_data['close'].iloc[-1]

                # Get today's open
                today_data = self.get_intraday_bars(symbol, trading_date, '5m')
                if today_data.empty:
                    continue
                today_open = today_data['open'].iloc[0]

                # Calculate gap
                gap_pct = (today_open - prev_close) / prev_close

                if abs(gap_pct) >= min_gap_pct:
                    gappers.append({
                        'symbol': symbol,
                        'gap_pct': gap_pct,
                        'prev_close': prev_close,
                        'open_price': today_open,
                        'direction': 'up' if gap_pct > 0 else 'down',
                    })

            except Exception as e:
                logger.debug(f"Failed to check gap for {symbol}: {e}")

        # Sort by absolute gap size
        gappers.sort(key=lambda x: abs(x['gap_pct']), reverse=True)
        return gappers
