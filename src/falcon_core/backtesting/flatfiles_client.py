"""
Polygon.io Flat Files Client (via Massive)

Downloads bulk historical market data from Polygon's Flat Files service
using S3-compatible access through Massive.

Features:
- Direct S3 download of CSV files
- Much faster than API for bulk historical data
- No rate limiting
- Supports minute, hour, and daily bars

Requirements:
- Active Massive subscription with Flat Files access
- S3 Access Key and Secret Key from Massive Dashboard
- boto3 library

Configuration:
- Endpoint: https://files.massive.com
- Bucket: flatfiles
"""

import os
import io
import logging
from datetime import datetime, date
from typing import List, Optional, Union
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Check for boto3 availability
try:
    import boto3
    from botocore.config import Config
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.warning("boto3 not installed. Install with: pip install boto3")


class FlatFilesClient:
    """
    Client for Polygon.io Flat Files via Massive S3.

    Flat Files are organized in the S3 bucket as:
    - us_stocks_sip/minute_aggs/YYYY/MM/YYYY-MM-DD.csv.gz
    - us_stocks_sip/day_aggs/YYYY/YYYY-MM.csv.gz

    Each file contains data for all symbols for that time period.
    """

    ENDPOINT = "https://files.massive.com"
    BUCKET = "flatfiles"

    # Data paths in the bucket (v1 format)
    MINUTE_AGGS_PATH = "us_stocks_sip/minute_aggs_v1"
    DAY_AGGS_PATH = "us_stocks_sip/day_aggs_v1"

    def __init__(
        self,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize Flat Files client.

        Args:
            access_key: Massive S3 Access Key (or set MASSIVE_ACCESS_KEY env var)
            secret_key: Massive S3 Secret Key (or set MASSIVE_SECRET_KEY env var)
            cache_dir: Directory to cache downloaded files
        """
        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required for Flat Files access. "
                "Install with: pip install boto3"
            )

        self.access_key = access_key or os.getenv('MASSIVE_ACCESS_KEY')
        self.secret_key = secret_key or os.getenv('MASSIVE_SECRET_KEY')

        if not self.access_key or not self.secret_key:
            logger.warning(
                "Massive credentials not configured. Set MASSIVE_ACCESS_KEY and "
                "MASSIVE_SECRET_KEY environment variables, or pass them directly."
            )

        self.cache_dir = Path(cache_dir or os.path.expanduser('~/.cache/falcon/flatfiles'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._client = None

    @property
    def client(self):
        """Lazy-initialize S3 client."""
        if self._client is None:
            if not self.access_key or not self.secret_key:
                raise ValueError(
                    "Massive credentials required. Set MASSIVE_ACCESS_KEY and "
                    "MASSIVE_SECRET_KEY environment variables."
                )

            self._client = boto3.client(
                's3',
                endpoint_url=self.ENDPOINT,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                config=Config(signature_version='s3v4'),
            )
        return self._client

    def _get_minute_aggs_key(self, trading_date: date) -> str:
        """Get S3 key for minute aggregates file."""
        return (
            f"{self.MINUTE_AGGS_PATH}/"
            f"{trading_date.year}/{trading_date.month:02d}/"
            f"{trading_date.strftime('%Y-%m-%d')}.csv.gz"
        )

    def _get_day_aggs_key(self, year: int, month: int) -> str:
        """Get S3 key for daily aggregates file."""
        return f"{self.DAY_AGGS_PATH}/{year}/{year}-{month:02d}.csv.gz"

    def _get_cache_path(self, s3_key: str) -> Path:
        """Get local cache path for an S3 key."""
        return self.cache_dir / s3_key.replace('/', '_')

    def _download_file(self, s3_key: str, use_cache: bool = True) -> pd.DataFrame:
        """
        Download and parse a flat file from S3.

        Args:
            s3_key: S3 object key
            use_cache: Whether to use local cache

        Returns:
            DataFrame with market data
        """
        cache_path = self._get_cache_path(s3_key)

        # Check cache first
        if use_cache and cache_path.exists():
            logger.debug(f"Loading from cache: {cache_path}")
            return pd.read_parquet(cache_path)

        # Download from S3
        logger.info(f"Downloading: s3://{self.BUCKET}/{s3_key}")
        try:
            response = self.client.get_object(Bucket=self.BUCKET, Key=s3_key)
            content = response['Body'].read()

            # Parse CSV (gzipped)
            df = pd.read_csv(
                io.BytesIO(content),
                compression='gzip',
            )

            # Save to cache as parquet (faster to read)
            if use_cache:
                df.to_parquet(cache_path)
                logger.debug(f"Cached to: {cache_path}")

            return df

        except self.client.exceptions.NoSuchKey:
            logger.warning(f"File not found: {s3_key}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to download {s3_key}: {e}")
            raise

    def get_minute_bars(
        self,
        symbol: str,
        trading_date: Union[str, date, datetime],
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Get minute bars for a symbol on a specific date.

        Args:
            symbol: Stock ticker symbol
            trading_date: Trading date
            use_cache: Whether to use local cache

        Returns:
            DataFrame with OHLCV data indexed by timestamp
        """
        if isinstance(trading_date, str):
            trading_date = datetime.strptime(trading_date, '%Y-%m-%d').date()
        elif isinstance(trading_date, datetime):
            trading_date = trading_date.date()

        s3_key = self._get_minute_aggs_key(trading_date)
        df = self._download_file(s3_key, use_cache)

        if df.empty:
            return df

        # Filter for symbol
        # Flat files have 'ticker' column
        ticker_col = 'ticker' if 'ticker' in df.columns else 'symbol'
        df = df[df[ticker_col] == symbol.upper()].copy()

        if df.empty:
            logger.warning(f"No data for {symbol} on {trading_date}")
            return df

        # Standardize column names
        column_map = {
            'window_start': 'timestamp',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            'vw': 'vwap',
            'n': 'trades',
        }
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

        # Convert timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        elif 'window_start' in df.columns:
            df['timestamp'] = pd.to_datetime(df['window_start'])
            df = df.set_index('timestamp')

        # Keep only standard columns
        keep_cols = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'trades']
        df = df[[c for c in keep_cols if c in df.columns]]

        return df.sort_index()

    def get_daily_bars(
        self,
        symbol: str,
        start_date: Union[str, date, datetime],
        end_date: Union[str, date, datetime],
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Get daily bars for a symbol over a date range.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date
            end_date: End date
            use_cache: Whether to use local cache

        Returns:
            DataFrame with OHLCV data indexed by date
        """
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        elif isinstance(start_date, datetime):
            start_date = start_date.date()

        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        elif isinstance(end_date, datetime):
            end_date = end_date.date()

        # Collect data from each month file
        all_data = []

        current = date(start_date.year, start_date.month, 1)
        end_month = date(end_date.year, end_date.month, 1)

        while current <= end_month:
            s3_key = self._get_day_aggs_key(current.year, current.month)
            df = self._download_file(s3_key, use_cache)

            if not df.empty:
                # Filter for symbol
                ticker_col = 'ticker' if 'ticker' in df.columns else 'symbol'
                month_data = df[df[ticker_col] == symbol.upper()].copy()
                if not month_data.empty:
                    all_data.append(month_data)

            # Next month
            if current.month == 12:
                current = date(current.year + 1, 1, 1)
            else:
                current = date(current.year, current.month + 1, 1)

        if not all_data:
            logger.warning(f"No daily data for {symbol}")
            return pd.DataFrame()

        df = pd.concat(all_data, ignore_index=True)

        # Standardize columns
        column_map = {
            'window_start': 'date',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            'vw': 'vwap',
            'n': 'trades',
        }
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

        # Convert date
        date_col = 'date' if 'date' in df.columns else 'window_start'
        df['date'] = pd.to_datetime(df[date_col]).dt.date
        df = df.set_index('date')

        # Filter date range
        df = df[(df.index >= start_date) & (df.index <= end_date)]

        # Keep only standard columns
        keep_cols = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'trades']
        df = df[[c for c in keep_cols if c in df.columns]]

        return df.sort_index()

    def get_multi_day_minute_bars(
        self,
        symbol: str,
        start_date: Union[str, date, datetime],
        end_date: Union[str, date, datetime],
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Get minute bars for multiple days.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date
            end_date: End date
            use_cache: Whether to use local cache

        Returns:
            DataFrame with OHLCV data indexed by timestamp
        """
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        elif isinstance(start_date, datetime):
            start_date = start_date.date()

        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        elif isinstance(end_date, datetime):
            end_date = end_date.date()

        all_data = []
        current = start_date

        while current <= end_date:
            # Skip weekends
            if current.weekday() < 5:
                df = self.get_minute_bars(symbol, current, use_cache)
                if not df.empty:
                    all_data.append(df)

            current += pd.Timedelta(days=1).to_pytimedelta()

        if not all_data:
            return pd.DataFrame()

        return pd.concat(all_data).sort_index()

    def list_available_dates(self, year: int, month: int) -> List[str]:
        """
        List available minute data files for a given month.

        Args:
            year: Year
            month: Month

        Returns:
            List of available date strings (YYYY-MM-DD)
        """
        prefix = f"{self.MINUTE_AGGS_PATH}/{year}/{month:02d}/"

        try:
            response = self.client.list_objects_v2(
                Bucket=self.BUCKET,
                Prefix=prefix,
            )

            dates = []
            for obj in response.get('Contents', []):
                # Extract date from filename like "2024-01-15.csv.gz"
                filename = obj['Key'].split('/')[-1]
                if filename.endswith('.csv.gz'):
                    date_str = filename.replace('.csv.gz', '')
                    dates.append(date_str)

            return sorted(dates)

        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []


def get_flatfiles_client(
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
) -> FlatFilesClient:
    """Factory function to create Flat Files client."""
    return FlatFilesClient(access_key=access_key, secret_key=secret_key)
