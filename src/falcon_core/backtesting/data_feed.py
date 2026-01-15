"""
Data Feed for Falcon Backtesting

Loads market data from various sources:
- Database (TimescaleDB/PostgreSQL)
- CSV files
- yfinance (for historical data)
- Polygon.io API
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import pandas as pd

logger = logging.getLogger(__name__)


class DataFeed:
    """
    Market data feed for backtesting.

    Supports multiple data sources with a unified interface.
    """

    def __init__(self, db_manager=None, cache_dir: Optional[str] = None):
        """
        Initialize data feed.

        Args:
            db_manager: DatabaseManager instance for DB queries
            cache_dir: Directory to cache downloaded data
        """
        self.db = db_manager
        self.cache_dir = cache_dir or "/var/cache/falcon/market_data"
        self._cache: Dict[str, pd.DataFrame] = {}

        # Ensure cache directory exists
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def get_historical_data(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime, None] = None,
        interval: str = "1d",
        source: str = "auto",
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data for a symbol.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date for data
            end_date: End date (default: today)
            interval: Data interval ('1m', '5m', '1h', '1d')
            source: Data source ('auto', 'database', 'yfinance', 'csv')

        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        # Parse dates
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")

        # Check cache first
        cache_key = f"{symbol}_{interval}_{start_date.date()}_{end_date.date()}"
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        # Try sources in order
        data = None

        if source == "auto":
            # Try database first, then yfinance
            data = self._try_database(symbol, start_date, end_date, interval)
            if data is None or data.empty:
                data = self._try_yfinance(symbol, start_date, end_date, interval)
        elif source == "database":
            data = self._try_database(symbol, start_date, end_date, interval)
        elif source == "yfinance":
            data = self._try_yfinance(symbol, start_date, end_date, interval)
        elif source == "csv":
            data = self._try_csv(symbol, start_date, end_date, interval)

        if data is None or data.empty:
            raise ValueError(f"No data found for {symbol} from {start_date} to {end_date}")

        # Cache the result
        self._cache[cache_key] = data

        return data.copy()

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
