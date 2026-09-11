"""Sentinel: Database connectivity and schema health."""

import logging
from falcon_core.sentinel.base import BaseSentinel, SentinelResult, SentinelStatus

logger = logging.getLogger(__name__)

REQUIRED_TABLES = [
    "account",
    "orders",
    "positions",
    "strategy_roster",
    "backtest_runs",
    "daily_bars",
]


class DatabaseSentinel(BaseSentinel):

    name = "database"
    description = "Verify database connection and required tables exist"

    def check(self) -> SentinelResult:
        try:
            from falcon_core import get_db_manager
            db = get_db_manager()
        except Exception as e:
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.FAIL,
                reason=f"Cannot connect to database: {e}",
            )

        # Check connectivity with a simple query
        try:
            db.execute("SELECT 1", fetch="one")
        except Exception as e:
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.FAIL,
                reason=f"Database query failed: {e}",
            )

        # Check required tables
        try:
            result = db.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'",
                fetch="all",
            )
            existing = {r["tablename"] for r in result}
        except Exception as e:
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.FAIL,
                reason=f"Cannot list tables: {e}",
            )

        missing = [t for t in REQUIRED_TABLES if t not in existing]
        if missing:
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.WARN,
                reason=f"Missing tables: {', '.join(missing)}",
                details={"existing": sorted(existing), "missing": missing},
            )

        return SentinelResult(
            name=self.name,
            status=SentinelStatus.PASS,
            reason=f"Connected, {len(existing)} tables present",
            details={"table_count": len(existing)},
        )
