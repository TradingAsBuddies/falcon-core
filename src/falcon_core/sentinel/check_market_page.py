"""Sentinel: Market page — verify published prices match live Polygon data."""

import os
import logging
from datetime import datetime
from falcon_core.sentinel.base import BaseSentinel, SentinelResult, SentinelStatus

logger = logging.getLogger(__name__)

SYMBOLS = ['SPY', 'TQQQ', 'VIXY', 'BNO']
# Max acceptable difference between market page and direct Polygon query
MAX_PRICE_DRIFT = 0.50  # dollars
# Max acceptable staleness for minute bars during market hours
MAX_DELAY_MINUTES = 25


class MarketPageSentinel(BaseSentinel):

    name = "market-page"
    description = "Verify market page prices match live Polygon data and are current"

    def check(self) -> SentinelResult:
        import requests as http_requests

        api_key = os.environ.get('MASSIVE_API_KEY', '') or os.environ.get('POLYGON_API_KEY', '')
        if not api_key:
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.SKIP,
                reason="No Polygon API key available",
            )

        # 1. Fetch from the market page API
        dashboard_url = os.environ.get('FALCON_DASHBOARD_URL', 'http://falcon-dashboard:5000')
        try:
            page_resp = http_requests.get(f"{dashboard_url}/api/market", timeout=15)
            if page_resp.status_code != 200:
                return SentinelResult(
                    name=self.name,
                    status=SentinelStatus.FAIL,
                    reason=f"/api/market returned HTTP {page_resp.status_code}",
                )
            page_data = page_resp.json()
        except Exception as e:
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.FAIL,
                reason=f"Cannot reach /api/market: {e}",
            )

        page_indicators = page_data.get('indicators', {})
        if not page_indicators:
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.FAIL,
                reason="/api/market returned no indicators",
            )

        # 2. Fetch directly from Polygon for comparison
        now = datetime.utcnow()
        today = datetime.now().strftime('%Y-%m-%d')
        issues = []
        checked = []

        for sym in SYMBOLS:
            page = page_indicators.get(sym)
            if not page:
                issues.append(f"{sym}: missing from market page")
                continue

            page_price = page.get('current', 0)
            page_prev = page.get('prev_close', 0)

            # Get latest minute bar directly
            try:
                resp = http_requests.get(
                    f"https://api.polygon.io/v2/aggs/ticker/{sym}/range/1/minute/{today}/{today}",
                    params={'adjusted': 'true', 'sort': 'desc', 'limit': 1, 'apiKey': api_key},
                    timeout=5,
                )
                data = resp.json()
                results = data.get('results', [])
            except Exception as e:
                issues.append(f"{sym}: Polygon query failed: {e}")
                continue

            if not results:
                # Market may be closed — check if that's expected
                hour = now.hour
                weekday = now.weekday()
                if weekday >= 5 or hour < 13 or hour >= 21:  # UTC market hours ~13:30-20:00
                    checked.append(f"{sym}: no minute bars (market closed), page=${page_price:.2f}")
                    continue
                else:
                    issues.append(f"{sym}: no minute bars during market hours")
                    continue

            polygon_price = results[0]['c']
            bar_time = datetime.utcfromtimestamp(results[0]['t'] / 1000)
            delay_min = (now - bar_time).total_seconds() / 60

            # Check price drift
            drift = abs(page_price - polygon_price)
            if drift > MAX_PRICE_DRIFT:
                issues.append(
                    f"{sym}: price drift ${drift:.2f} "
                    f"(page=${page_price:.2f} vs polygon=${polygon_price:.2f})"
                )

            # Check staleness
            if delay_min > MAX_DELAY_MINUTES:
                hour = now.hour
                weekday = now.weekday()
                if weekday < 5 and 13 <= hour < 21:
                    issues.append(f"{sym}: data is {delay_min:.0f}m old (expected <{MAX_DELAY_MINUTES}m)")

            # Check prev_close is set
            if page_prev <= 0:
                issues.append(f"{sym}: prev_close is ${page_prev} (red-to-green line broken)")

            checked.append(
                f"{sym}: page=${page_price:.2f} polygon=${polygon_price:.2f} "
                f"drift=${drift:.2f} delay={delay_min:.0f}m prev=${page_prev:.2f}"
            )

        if issues:
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.WARN if len(issues) < len(SYMBOLS) else SentinelStatus.FAIL,
                reason=f"{len(issues)} issue(s) across {len(SYMBOLS)} indicators",
                details={"issues": issues, "checked": checked},
            )

        return SentinelResult(
            name=self.name,
            status=SentinelStatus.PASS,
            reason=f"All {len(SYMBOLS)} indicators match Polygon within ${MAX_PRICE_DRIFT}",
            details={"checked": checked},
        )
