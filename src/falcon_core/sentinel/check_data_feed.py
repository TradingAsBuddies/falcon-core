"""Sentinel: Data feed — verify Massive flat files are the primary source."""

import logging
from datetime import datetime, timedelta
from falcon_core.sentinel.base import BaseSentinel, SentinelResult, SentinelStatus

logger = logging.getLogger(__name__)

PROBE_SYMBOL = "SPY"


class DataFeedSentinel(BaseSentinel):

    name = "data-feed"
    description = "Verify Massive flat files are reachable and return valid OHLCV (daily)"

    def _recent_trading_day(self) -> str:
        """Find a weekday in the recent past (skip weekends)."""
        dt = datetime.now() - timedelta(days=3)
        while dt.weekday() >= 5:
            dt -= timedelta(days=1)
        return dt.strftime("%Y-%m-%d")

    def check(self) -> SentinelResult:
        try:
            from falcon_core.backtesting.data_feed import DataFeed
        except ImportError as e:
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.FAIL,
                reason=f"Cannot import DataFeed: {e}",
            )

        feed = DataFeed()

        # 1. Check flat files client initialized
        if feed.flatfiles is None:
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.FAIL,
                reason="Flat files client not available (check MASSIVE_ACCESS_KEY / MASSIVE_SECRET_KEY and boto3)",
            )

        # 2. Try fetching daily data from flat files
        probe_date = self._recent_trading_day()
        try:
            data = feed.get_historical_data(
                PROBE_SYMBOL, probe_date, probe_date,
                interval="1d", source="flatfiles",
            )
        except Exception as e:
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.FAIL,
                reason=f"Flat files daily fetch failed for {PROBE_SYMBOL} on {probe_date}: {e}",
            )

        if data.empty:
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.WARN,
                reason=f"Flat files returned no daily data for {PROBE_SYMBOL} on {probe_date} (holiday?)",
                details={"probe_date": probe_date},
            )

        # 3. Validate OHLCV columns
        required_cols = {"open", "high", "low", "close", "volume"}
        missing_cols = required_cols - set(data.columns)
        if missing_cols:
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.FAIL,
                reason=f"Missing columns: {missing_cols}",
                details={"columns": list(data.columns)},
            )

        # 4. Sanity check values
        row = data.iloc[0]
        if row["close"] <= 0 or row["volume"] <= 0:
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.FAIL,
                reason=f"Invalid data: close={row['close']}, volume={row['volume']}",
            )

        return SentinelResult(
            name=self.name,
            status=SentinelStatus.PASS,
            reason=f"Flat files daily OK — {PROBE_SYMBOL} {probe_date}: close=${row['close']:.2f}, vol={int(row['volume']):,}",
            details={
                "probe_date": probe_date,
                "close": float(row["close"]),
                "volume": int(row["volume"]),
            },
        )


class DataFeedMinuteSentinel(BaseSentinel):

    name = "data-feed-minute"
    description = "Verify Massive flat files return valid minute bars"

    def _recent_trading_day(self) -> str:
        dt = datetime.now() - timedelta(days=3)
        while dt.weekday() >= 5:
            dt -= timedelta(days=1)
        return dt.strftime("%Y-%m-%d")

    def check(self) -> SentinelResult:
        try:
            from falcon_core.backtesting.data_feed import DataFeed
        except ImportError as e:
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.FAIL,
                reason=f"Cannot import DataFeed: {e}",
            )

        feed = DataFeed()

        if feed.flatfiles is None:
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.FAIL,
                reason="Flat files client not available",
            )

        probe_date = self._recent_trading_day()
        try:
            data = feed.get_historical_data(
                PROBE_SYMBOL, probe_date, probe_date,
                interval="1m", source="flatfiles",
            )
        except Exception as e:
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.FAIL,
                reason=f"Flat files minute fetch failed for {PROBE_SYMBOL} on {probe_date}: {e}",
            )

        if data.empty:
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.WARN,
                reason=f"No minute bars for {PROBE_SYMBOL} on {probe_date}",
                details={"probe_date": probe_date},
            )

        # Expect several hundred bars for a full trading day
        bar_count = len(data)
        if bar_count < 100:
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.WARN,
                reason=f"Only {bar_count} minute bars for {PROBE_SYMBOL} on {probe_date} (expected 390+)",
                details={"probe_date": probe_date, "bars": bar_count},
            )

        required_cols = {"open", "high", "low", "close", "volume"}
        missing_cols = required_cols - set(data.columns)
        if missing_cols:
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.FAIL,
                reason=f"Missing columns in minute data: {missing_cols}",
            )

        return SentinelResult(
            name=self.name,
            status=SentinelStatus.PASS,
            reason=f"Flat files minute OK — {PROBE_SYMBOL} {probe_date}: {bar_count} bars",
            details={"probe_date": probe_date, "bars": bar_count},
        )


class PolygonMinuteSentinel(BaseSentinel):

    name = "polygon-minute"
    description = "Verify Polygon API returns fresh minute bars for live trading"

    def check(self) -> SentinelResult:
        import os
        import requests

        api_key = os.getenv("MASSIVE_API_KEY", "") or os.getenv("POLYGON_API_KEY", "")
        if not api_key:
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.FAIL,
                reason="No Polygon API key (MASSIVE_API_KEY / POLYGON_API_KEY)",
            )

        # Fetch most recent minute bars
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{PROBE_SYMBOL}"
            f"/range/1/minute/{today}/{today}"
        )
        try:
            resp = requests.get(
                url,
                params={"adjusted": "true", "sort": "desc", "limit": 5, "apiKey": api_key},
                timeout=10,
            )
        except Exception as e:
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.FAIL,
                reason=f"Polygon API request failed: {e}",
            )

        if resp.status_code != 200:
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.FAIL,
                reason=f"Polygon API returned HTTP {resp.status_code}",
            )

        data = resp.json()
        status = data.get("status", "unknown")
        results = data.get("results", [])

        if not results:
            # Could be outside market hours
            weekday = now.weekday()
            hour = now.hour
            if weekday >= 5 or hour < 4 or hour >= 20:
                return SentinelResult(
                    name=self.name,
                    status=SentinelStatus.PASS,
                    reason=f"No minute bars (market closed), API status={status}",
                    details={"status": status},
                )
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.WARN,
                reason=f"No minute bars returned during market hours, API status={status}",
                details={"status": status},
            )

        # Check freshness of the latest bar
        latest_epoch_ms = results[0]["t"]
        latest_bar = datetime.utcfromtimestamp(latest_epoch_ms / 1000)
        utc_now = datetime.utcnow()
        delay = utc_now - latest_bar
        delay_minutes = delay.total_seconds() / 60

        price = results[0]["c"]
        volume = results[0]["v"]

        # On a DELAYED plan, expect ~15-20 min delay during market hours
        if delay_minutes > 30:
            # Could be outside market hours
            weekday = now.weekday()
            hour = now.hour
            if weekday >= 5 or hour < 9 or hour >= 17:
                return SentinelResult(
                    name=self.name,
                    status=SentinelStatus.PASS,
                    reason=f"Last bar {delay_minutes:.0f}m ago (market closed), price=${price:.2f}",
                    details={
                        "delay_minutes": round(delay_minutes, 1),
                        "latest_bar_utc": latest_bar.isoformat(),
                        "price": price,
                        "status": status,
                    },
                )
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.WARN,
                reason=f"Latest bar is {delay_minutes:.0f}m old during market hours — expected <25m",
                details={
                    "delay_minutes": round(delay_minutes, 1),
                    "latest_bar_utc": latest_bar.isoformat(),
                    "price": price,
                    "status": status,
                },
            )

        return SentinelResult(
            name=self.name,
            status=SentinelStatus.PASS,
            reason=f"Polygon {status} — {PROBE_SYMBOL} ${price:.2f}, {delay_minutes:.0f}m delay, vol={int(volume):,}",
            details={
                "delay_minutes": round(delay_minutes, 1),
                "latest_bar_utc": latest_bar.isoformat(),
                "price": price,
                "volume": int(volume),
                "status": status,
            },
        )
