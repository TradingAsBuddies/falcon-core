#!/usr/bin/env python3
"""
Finviz Elite API Client with Rate Limiting

Centralized client for all Finviz Elite API requests with:
- Exponential backoff with jitter
- Request caching to avoid duplicate calls
- Thread-safe rate limiting
- Automatic retry on throttling (429) errors
"""

import os
import csv
import time
import random
import hashlib
import logging
import threading
from io import StringIO
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    # Minimum seconds between requests
    min_request_interval: float = 1.0

    # Maximum seconds between requests (for backoff ceiling)
    max_request_interval: float = 60.0

    # Initial backoff duration on throttle
    initial_backoff: float = 2.0

    # Backoff multiplier (exponential factor)
    backoff_multiplier: float = 2.0

    # Maximum retries before giving up
    max_retries: int = 5

    # Jitter factor (0.0 to 1.0) - randomization to prevent thundering herd
    jitter_factor: float = 0.25

    # Cache TTL in seconds (to avoid hitting same URL repeatedly)
    cache_ttl: int = 60


@dataclass
class CachedResponse:
    """Cached API response"""
    data: str
    timestamp: datetime
    url_hash: str


class FinvizRateLimiter:
    """Thread-safe rate limiter with exponential backoff"""

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self._lock = threading.Lock()
        self._last_request_time: Optional[datetime] = None
        self._consecutive_errors: int = 0
        self._current_backoff: float = self.config.initial_backoff

    def wait_if_needed(self) -> None:
        """Wait if we need to respect rate limits"""
        with self._lock:
            if self._last_request_time is None:
                return

            elapsed = (datetime.now() - self._last_request_time).total_seconds()

            # Calculate required wait time
            required_wait = self.config.min_request_interval

            # Add extra wait if we've had recent errors
            if self._consecutive_errors > 0:
                required_wait = min(
                    self._current_backoff,
                    self.config.max_request_interval
                )

            # Add jitter
            jitter = random.uniform(
                -self.config.jitter_factor * required_wait,
                self.config.jitter_factor * required_wait
            )
            required_wait = max(0, required_wait + jitter)

            if elapsed < required_wait:
                sleep_time = required_wait - elapsed
                logger.debug(f"[FINVIZ] Rate limiting: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)

    def record_request(self) -> None:
        """Record that a request was made"""
        with self._lock:
            self._last_request_time = datetime.now()

    def record_success(self) -> None:
        """Record successful request - reset error counters"""
        with self._lock:
            self._consecutive_errors = 0
            self._current_backoff = self.config.initial_backoff

    def record_error(self, is_throttle: bool = False) -> float:
        """
        Record failed request - increase backoff

        Args:
            is_throttle: True if this was a 429/rate limit error

        Returns:
            Current backoff duration in seconds
        """
        with self._lock:
            self._consecutive_errors += 1

            if is_throttle:
                # Exponential backoff for throttle errors
                self._current_backoff = min(
                    self._current_backoff * self.config.backoff_multiplier,
                    self.config.max_request_interval
                )

            return self._current_backoff

    def should_retry(self) -> bool:
        """Check if we should retry after an error"""
        with self._lock:
            return self._consecutive_errors < self.config.max_retries


class FinvizClient:
    """
    Centralized Finviz Elite API client

    Handles all requests to Finviz with proper rate limiting,
    caching, and exponential backoff on errors.
    """

    # Finviz Elite CSV export endpoint
    ELITE_BASE_URL = "https://elite.finviz.com/export.ashx"

    # Standard screener URL (for free tier fallback)
    SCREENER_BASE_URL = "https://finviz.com/screener.ashx"

    # Common column IDs for CSV export
    COLUMNS = {
        'ticker': 1,
        'company': 2,
        'sector': 3,
        'industry': 4,
        'country': 5,
        'market_cap': 6,
        'pe': 7,
        'price': 8,
        'change': 9,
        'volume': 10,
        'avg_volume': 64,
        'rsi': 65,
        'rel_volume': 66,
        'earnings_date': 68,
        'target_price': 69,
        'perf_5min': 93,
    }

    # Default columns for comprehensive data
    DEFAULT_COLUMNS = "1,2,3,4,6,7,8,9,10,65,66,93"

    # Columns for earnings-focused screening (includes earnings date)
    EARNINGS_COLUMNS = "1,2,3,4,6,7,8,9,10,64,65,66,68,69,93"

    def __init__(self, auth_key: Optional[str] = None,
                 rate_limit_config: Optional[RateLimitConfig] = None):
        """
        Initialize Finviz client

        Args:
            auth_key: Finviz Elite authentication key
            rate_limit_config: Custom rate limiting configuration
        """
        self.auth_key = auth_key or os.getenv('FINVIZ_AUTH_KEY', '')
        self.rate_limiter = FinvizRateLimiter(rate_limit_config)

        # Response cache
        self._cache: Dict[str, CachedResponse] = {}
        self._cache_lock = threading.Lock()

        # Session for connection pooling
        self._session = requests.Session()
        self._session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; FalconTradingBot/1.0)'
        })

        if not self.auth_key:
            logger.warning("[FINVIZ] No auth key configured - Elite features unavailable")

    def _get_cache_key(self, url: str, params: Dict) -> str:
        """Generate cache key from URL and params"""
        param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        full_url = f"{url}?{param_str}"
        return hashlib.md5(full_url.encode()).hexdigest()

    def _get_cached(self, cache_key: str) -> Optional[str]:
        """Get cached response if still valid"""
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached:
                age = (datetime.now() - cached.timestamp).total_seconds()
                if age < self.rate_limiter.config.cache_ttl:
                    logger.debug(f"[FINVIZ] Cache hit (age: {age:.1f}s)")
                    return cached.data
                else:
                    # Expired
                    del self._cache[cache_key]
        return None

    def _set_cached(self, cache_key: str, data: str) -> None:
        """Cache response data"""
        with self._cache_lock:
            self._cache[cache_key] = CachedResponse(
                data=data,
                timestamp=datetime.now(),
                url_hash=cache_key
            )

            # Clean old entries (keep cache bounded)
            if len(self._cache) > 100:
                oldest_key = min(
                    self._cache.keys(),
                    key=lambda k: self._cache[k].timestamp
                )
                del self._cache[oldest_key]

    def _make_request(self, url: str, params: Dict,
                      timeout: int = 30) -> Optional[str]:
        """
        Make HTTP request with rate limiting and retry logic

        Args:
            url: Request URL
            params: Query parameters
            timeout: Request timeout in seconds

        Returns:
            Response text or None on failure
        """
        cache_key = self._get_cache_key(url, params)

        # Check cache first
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        # Rate limit
        self.rate_limiter.wait_if_needed()

        while self.rate_limiter.should_retry():
            try:
                self.rate_limiter.record_request()

                response = self._session.get(
                    url,
                    params=params,
                    timeout=timeout
                )

                # Handle rate limiting (429)
                if response.status_code == 429:
                    backoff = self.rate_limiter.record_error(is_throttle=True)
                    logger.warning(f"[FINVIZ] Rate limited (429) - backing off {backoff:.1f}s")
                    time.sleep(backoff)
                    continue

                # Handle server errors (5xx)
                if response.status_code >= 500:
                    backoff = self.rate_limiter.record_error(is_throttle=False)
                    logger.warning(f"[FINVIZ] Server error ({response.status_code}) - retrying")
                    time.sleep(backoff)
                    continue

                response.raise_for_status()

                # Success
                self.rate_limiter.record_success()

                # Cache the response
                self._set_cached(cache_key, response.text)

                return response.text

            except requests.exceptions.Timeout:
                backoff = self.rate_limiter.record_error(is_throttle=False)
                logger.warning(f"[FINVIZ] Request timeout - retrying in {backoff:.1f}s")
                time.sleep(backoff)

            except requests.exceptions.ConnectionError:
                backoff = self.rate_limiter.record_error(is_throttle=False)
                logger.warning(f"[FINVIZ] Connection error - retrying in {backoff:.1f}s")
                time.sleep(backoff)

            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response else 'unknown'
                logger.error(f"[FINVIZ] HTTP error: {status}")
                self.rate_limiter.record_error(is_throttle=False)
                return None

            except Exception as e:
                logger.error(f"[FINVIZ] Unexpected error: {type(e).__name__}")
                self.rate_limiter.record_error(is_throttle=False)
                return None

        logger.error("[FINVIZ] Max retries exceeded")
        return None

    def fetch_csv(self, filters: Optional[str] = None,
                  columns: Optional[str] = None,
                  view: int = 152) -> Optional[str]:
        """
        Fetch CSV data from Finviz Elite

        Args:
            filters: Finviz filter string (e.g., "sh_avgvol_o750,sh_price_u20")
            columns: Column IDs to include (e.g., "1,2,3,8,9,93")
            view: View type (152 for custom)

        Returns:
            CSV data as string, or None on error
        """
        if not self.auth_key:
            logger.error("[FINVIZ] Auth key required for Elite CSV export")
            return None

        params = {
            'v': view,
            'c': columns or self.DEFAULT_COLUMNS,
            'auth': self.auth_key
        }

        if filters:
            params['f'] = filters

        logger.info(f"[FINVIZ] Fetching CSV with filters: {filters or 'none'}")

        return self._make_request(self.ELITE_BASE_URL, params)

    def parse_csv(self, csv_data: str) -> List[Dict[str, Any]]:
        """
        Parse CSV data into list of dictionaries

        Args:
            csv_data: CSV string from Finviz

        Returns:
            List of stock data dictionaries with normalized keys
        """
        stocks = []
        reader = csv.DictReader(StringIO(csv_data))

        for row in reader:
            # Normalize keys to snake_case
            stock = {
                'ticker': row.get('Ticker', ''),
                'company': row.get('Company', ''),
                'sector': row.get('Sector', ''),
                'industry': row.get('Industry', ''),
                'market_cap': row.get('Market Cap', ''),
                'pe_ratio': row.get('P/E', ''),
                'price': self._parse_float(row.get('Price', '0')),
                'change': row.get('Change', ''),
                'change_pct': self._parse_percent(row.get('Change', '0%')),
                'volume': self._parse_int(row.get('Volume', '0')),
                'avg_volume': self._parse_float(row.get('Average Volume', '0')),
                'rsi': self._parse_float(row.get('RSI (14)', row.get('Relative Strength Index (14)', '50'))),
                'rel_volume': self._parse_float(row.get('Relative Volume', '1')),
                'earnings_date': row.get('Earnings Date', ''),
                'target_price': self._parse_float(row.get('Target Price', '0')),
                'performance_5min': self._parse_percent(
                    row.get('Performance (5 Minutes)', '0%')
                ),
                'performance_5min_str': row.get('Performance (5 Minutes)', '0.00%'),
                '_raw': row,  # Keep original data
            }
            stocks.append(stock)

        return stocks

    def _parse_float(self, value: str) -> float:
        """Parse float from string, handling commas and errors"""
        try:
            return float(str(value).replace(',', '').replace('%', ''))
        except (ValueError, TypeError):
            return 0.0

    def _parse_int(self, value: str) -> int:
        """Parse integer from string, handling commas and errors"""
        try:
            return int(str(value).replace(',', ''))
        except (ValueError, TypeError):
            return 0

    def _parse_percent(self, value: str) -> float:
        """Parse percentage string to float"""
        try:
            return float(str(value).replace('%', '').replace(',', ''))
        except (ValueError, TypeError):
            return 0.0

    def get_stocks(self, filters: Optional[str] = None,
                   columns: Optional[str] = None,
                   sort_by_5min: bool = True,
                   limit: Optional[int] = None,
                   include_earnings: bool = False) -> List[Dict[str, Any]]:
        """
        Get stocks from Finviz Elite with optional filtering and sorting

        Args:
            filters: Finviz filter string
            columns: Column IDs to fetch
            sort_by_5min: Sort by 5-minute performance (absolute value)
            limit: Maximum number of stocks to return
            include_earnings: Include earnings date/target price columns

        Returns:
            List of stock dictionaries
        """
        # Auto-detect if we should include earnings columns
        if include_earnings or (filters and 'earnings' in filters.lower()):
            columns = columns or self.EARNINGS_COLUMNS

        csv_data = self.fetch_csv(filters=filters, columns=columns)

        if not csv_data:
            return []

        stocks = self.parse_csv(csv_data)
        logger.info(f"[FINVIZ] Parsed {len(stocks)} stocks")

        if sort_by_5min and stocks:
            # Randomize first to break alphabetical bias
            random.shuffle(stocks)

            # Sort by multiple criteria
            stocks.sort(
                key=lambda s: (
                    abs(s.get('performance_5min', 0)),
                    s.get('rel_volume', 0),
                    s.get('volume', 0)
                ),
                reverse=True
            )
            logger.debug("[FINVIZ] Sorted by 5-min performance + volume")

        if limit:
            stocks = stocks[:limit]

        return stocks

    def get_top_movers(self, filters: Optional[str] = None,
                       top_n: int = 20,
                       direction: str = 'both') -> List[Dict[str, Any]]:
        """
        Get top N 5-minute movers

        Args:
            filters: Optional Finviz filters
            top_n: Number of stocks to return
            direction: 'up' (gainers), 'down' (losers), or 'both'

        Returns:
            List of top moving stocks
        """
        stocks = self.get_stocks(filters=filters, sort_by_5min=False)

        if direction == 'up':
            stocks = [s for s in stocks if s.get('performance_5min', 0) > 0]
            stocks.sort(key=lambda x: x.get('performance_5min', 0), reverse=True)
        elif direction == 'down':
            stocks = [s for s in stocks if s.get('performance_5min', 0) < 0]
            stocks.sort(key=lambda x: x.get('performance_5min', 0))
        else:
            stocks.sort(
                key=lambda x: abs(x.get('performance_5min', 0)),
                reverse=True
            )

        return stocks[:top_n]

    def extract_filters_from_url(self, url: str) -> Dict[str, str]:
        """
        Extract filters and auth from a Finviz screener URL

        Args:
            url: Full Finviz screener URL

        Returns:
            Dictionary with 'filters' and optionally 'auth' keys
        """
        import urllib.parse

        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query)

        result = {
            'filters': params.get('f', [''])[0],
        }

        if params.get('auth'):
            result['auth'] = params['auth'][0]

        return result

    def clear_cache(self) -> None:
        """Clear the response cache"""
        with self._cache_lock:
            self._cache.clear()
            logger.debug("[FINVIZ] Cache cleared")


# Global client instance (lazy initialization)
_client: Optional[FinvizClient] = None
_client_lock = threading.Lock()


def get_finviz_client(auth_key: Optional[str] = None,
                      config: Optional[RateLimitConfig] = None) -> FinvizClient:
    """
    Get or create the global Finviz client instance

    Args:
        auth_key: Optional auth key (uses env var if not provided)
        config: Optional rate limit configuration

    Returns:
        FinvizClient instance
    """
    global _client

    with _client_lock:
        if _client is None:
            _client = FinvizClient(auth_key=auth_key, rate_limit_config=config)
        return _client


# Convenience function for backward compatibility
def fetch_finviz_stocks(filters: Optional[str] = None,
                        limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Convenience function to fetch stocks from Finviz

    Args:
        filters: Finviz filter string
        limit: Maximum stocks to return

    Returns:
        List of stock dictionaries
    """
    client = get_finviz_client()
    return client.get_stocks(filters=filters, limit=limit)


if __name__ == '__main__':
    """Test the Finviz client"""
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    auth_key = os.getenv('FINVIZ_AUTH_KEY')
    if not auth_key:
        print("ERROR: FINVIZ_AUTH_KEY not set in environment")
        print("Add it to your .env file: FINVIZ_AUTH_KEY=your_key")
        sys.exit(1)

    client = FinvizClient(auth_key=auth_key)

    print("=" * 70)
    print("FINVIZ CLIENT TEST - Rate Limited Requests")
    print("=" * 70)

    # Test 1: Basic fetch
    print("\n[TEST 1] Fetching stocks with filters...")
    filters = "sh_avgvol_o750,sh_price_u20,sh_relvol_o1.5"
    stocks = client.get_stocks(filters=filters, limit=20)

    print(f"Found {len(stocks)} stocks")
    for i, stock in enumerate(stocks[:5], 1):
        print(f"  {i}. {stock['ticker']:6s} ${stock['price']:>7.2f} "
              f"5min: {stock['performance_5min']:+6.2f}%")

    # Test 2: Top movers
    print("\n[TEST 2] Getting top gainers...")
    gainers = client.get_top_movers(filters=filters, top_n=5, direction='up')
    for stock in gainers:
        print(f"  {stock['ticker']:6s} {stock['performance_5min']:+6.2f}%")

    # Test 3: Cache test
    print("\n[TEST 3] Cache test (should be instant)...")
    start = time.time()
    stocks2 = client.get_stocks(filters=filters, limit=20)
    elapsed = time.time() - start
    print(f"  Fetched {len(stocks2)} stocks in {elapsed:.3f}s (cached)")

    print("\n" + "=" * 70)
    print("All tests completed successfully!")
