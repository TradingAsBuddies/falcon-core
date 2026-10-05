"""Sentinels: is the platform actually producing anything?

Every other check here verifies an input — the database answers, the bars are
current, the roster parses, the engine runs. All ten passed 10/10 through two
long silences at the output end:

* **13 days with zero recommendations** (2026-09-18 to 09-30). Finviz renumbered
  its CSV columns, the liquidity floor rejected every row as $0 dollar volume,
  and each run still logged "success" with 0 stocks found.
* **24 days with zero orders** (2026-09-11 to 10-05). The orchestrator read the
  screener's file from a path that did not exist in its container and used a
  throwaway SQLite database, so it monitored an empty book while reporting
  itself healthy.

A green check beside a silent platform is worse than no check. These two look
at what came *out*: recommendations in profile_runs, and orders in the book.

Both are database-only, like the other sentinels: this normally runs inside the
trader container, where there is no systemctl and no podman.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Optional

from falcon_core.market_calendar import is_session, previous_session
from falcon_core.sentinel.base import (
    BaseSentinel,
    SentinelResult,
    SentinelStatus,
    sessions_since,
)

logger = logging.getLogger(__name__)

#: Sessions without a single recommendation before this is an outage. The
#: screener runs three times a session, so one empty session is a quiet market
#: or a bad morning; three in a row is the Finviz-column signature.
SCREEN_WARN_SESSIONS = 1
SCREEN_FAIL_SESSIONS = 3

#: Sessions without an order. Signals genuinely do not fire every day, so this
#: is deliberately slack -- but 24 days is not a quiet market.
ORDER_WARN_SESSIONS = 5
ORDER_FAIL_SESSIONS = 10


def _as_date(value) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.fromisoformat(str(value)).date()
    except (TypeError, ValueError):
        return None


def _newest_session(today: Optional[date] = None) -> date:
    """The session that should already have produced output.

    Today while the market is open or closed-for-the-day; otherwise the
    previous session, so a Sunday does not read as an outage.
    """
    today = today or date.today()
    return today if is_session(today) else previous_session(today)


def _db():
    from falcon_core import get_db_manager
    return get_db_manager()


class ScreenerOutputSentinel(BaseSentinel):
    """Has the screener produced any recommendation recently?"""

    name = "screener-output"
    description = "Verify the screener is still producing recommendations"

    def check(self) -> SentinelResult:
        try:
            db = _db()
        except Exception as e:
            return SentinelResult(name=self.name, status=SentinelStatus.FAIL,
                                  reason=f"Cannot connect to database: {e}")

        try:
            row = db.execute(
                "SELECT max(run_timestamp) AS newest FROM profile_runs "
                "WHERE recommendations_generated > 0",
                fetch="one",
            )
            runs = db.execute(
                "SELECT count(*) AS runs, "
                "coalesce(sum(recommendations_generated), 0) AS recs "
                "FROM profile_runs WHERE run_timestamp > %s",
                (datetime.now() - timedelta(days=2),),
                fetch="one",
            )
        except Exception as e:
            return SentinelResult(name=self.name, status=SentinelStatus.FAIL,
                                  reason=f"Cannot read profile_runs: {e}")

        newest = _as_date(row["newest"] if row else None)
        recent_runs = int(runs["runs"]) if runs else 0
        recent_recs = int(runs["recs"]) if runs else 0

        if newest is None:
            return SentinelResult(
                name=self.name, status=SentinelStatus.FAIL,
                reason="No screener run has ever produced a recommendation",
            )

        expected = _newest_session()
        behind = sessions_since(newest, expected)
        details = {
            "newest_recommendation": str(newest),
            "expected_session": str(expected),
            "sessions_behind": behind,
            "runs_last_2d": recent_runs,
            "recommendations_last_2d": recent_recs,
        }

        # The failure that actually happened: runs completing and reporting
        # success while producing nothing. Naming it beats "N sessions behind".
        empty_but_running = behind > SCREEN_WARN_SESSIONS and recent_runs > 0
        detail = (f"{recent_runs} run(s) in 2d produced {recent_recs} "
                  f"recommendation(s)")

        if behind > SCREEN_FAIL_SESSIONS:
            return SentinelResult(
                name=self.name, status=SentinelStatus.FAIL,
                reason=(f"No recommendations for {behind} sessions — newest "
                        f"{newest}, expected {expected} · {detail}"
                        + (" · runs are completing but screening nothing"
                           if empty_but_running else " · screener is not running")),
                details=details,
            )

        if behind > SCREEN_WARN_SESSIONS:
            return SentinelResult(
                name=self.name, status=SentinelStatus.WARN,
                reason=(f"No recommendations for {behind} sessions — newest "
                        f"{newest} · {detail}"),
                details=details,
            )

        return SentinelResult(
            name=self.name, status=SentinelStatus.PASS,
            reason=f"Recommendations current through {newest} · {detail}",
            details=details,
        )


class TradingActivitySentinel(BaseSentinel):
    """Has the trader placed an order recently, and is it meant to be trading?"""

    name = "trading-activity"
    description = "Verify the orchestrator is still placing orders"

    def check(self) -> SentinelResult:
        try:
            db = _db()
        except Exception as e:
            return SentinelResult(name=self.name, status=SentinelStatus.FAIL,
                                  reason=f"Cannot connect to database: {e}")

        try:
            row = db.execute("SELECT max(timestamp) AS newest FROM orders", fetch="one")
        except Exception as e:
            return SentinelResult(name=self.name, status=SentinelStatus.FAIL,
                                  reason=f"Cannot read orders: {e}")

        newest = _as_date(row["newest"] if row else None)

        # A deliberate halt is not an outage. Reported, not alarmed.
        halted, halt_reason = self._halt_state()

        if newest is None:
            if halted:
                return SentinelResult(name=self.name, status=SentinelStatus.SKIP,
                                      reason=f"No orders ever placed; trading halted ({halt_reason})")
            return SentinelResult(name=self.name, status=SentinelStatus.FAIL,
                                  reason="No order has ever been placed")

        expected = _newest_session()
        behind = sessions_since(newest, expected)
        details = {
            "newest_order": str(newest),
            "expected_session": str(expected),
            "sessions_behind": behind,
            "trading_halted": halted,
            "halt_reason": halt_reason,
        }

        if halted and behind > ORDER_WARN_SESSIONS:
            return SentinelResult(
                name=self.name, status=SentinelStatus.SKIP,
                reason=(f"No orders for {behind} sessions, but trading is "
                        f"halted on purpose ({halt_reason}) — newest {newest}"),
                details=details,
            )

        if behind > ORDER_FAIL_SESSIONS:
            return SentinelResult(
                name=self.name, status=SentinelStatus.FAIL,
                reason=(f"No orders for {behind} sessions — newest {newest}, "
                        f"expected {expected}. The orchestrator is not trading."),
                details=details,
            )

        if behind > ORDER_WARN_SESSIONS:
            return SentinelResult(
                name=self.name, status=SentinelStatus.WARN,
                reason=f"No orders for {behind} sessions — newest {newest}",
                details=details,
            )

        return SentinelResult(
            name=self.name, status=SentinelStatus.PASS,
            reason=f"Orders current through {newest} ({behind} session(s) behind)",
            details=details,
        )

    @staticmethod
    def _halt_state():
        """(halted, reason). falcon-trader may not be importable from here."""
        try:
            from falcon_trader.risk_limits import KillSwitch
        except Exception:
            return False, None
        try:
            switch = KillSwitch()
            return (not switch.is_trading_enabled()), switch.reason()
        except Exception as e:          # pragma: no cover - defensive
            logger.warning("Could not read the kill switch: %s", e)
            return False, None
