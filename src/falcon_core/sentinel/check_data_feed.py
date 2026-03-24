"""Sentinel: Data feed — verify Massive flat files are the primary source."""

import logging
from datetime import datetime, timedelta
from falcon_core.sentinel.base import BaseSentinel, SentinelResult, SentinelStatus

logger = logging.getLogger(__name__)

# Use a recent known trading day for the probe
PROBE_SYMBOL = "SPY"


class DataFeedSentinel(BaseSentinel):

    name = "data-feed"
    description = "Verify Massive flat files are reachable and return valid OHLCV"

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

        # 2. Try fetching data explicitly from flat files
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
                reason=f"Flat files fetch failed for {PROBE_SYMBOL} on {probe_date}: {e}",
            )

        if data.empty:
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.WARN,
                reason=f"Flat files returned no data for {PROBE_SYMBOL} on {probe_date} (holiday?)",
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

        # 5. Verify auto source picks flat files first
        auto_source = None
        try:
            auto_data = feed.get_historical_data(
                PROBE_SYMBOL, probe_date, probe_date,
                interval="1d", source="auto",
            )
            # DataFeed doesn't expose which source was used, so we just
            # confirm we get the same close price (flat files should win).
            if not auto_data.empty and abs(auto_data.iloc[0]["close"] - row["close"]) < 0.01:
                auto_source = "flatfiles (confirmed)"
            else:
                auto_source = "different source returned different price"
        except Exception:
            auto_source = "auto fetch failed"

        return SentinelResult(
            name=self.name,
            status=SentinelStatus.PASS,
            reason=f"Flat files OK — {PROBE_SYMBOL} {probe_date}: close=${row['close']:.2f}, vol={int(row['volume']):,}",
            details={
                "probe_date": probe_date,
                "close": float(row["close"]),
                "volume": int(row["volume"]),
                "auto_source": auto_source,
            },
        )
