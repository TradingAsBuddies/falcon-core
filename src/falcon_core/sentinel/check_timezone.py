"""Sentinel: Timezone — verify ET handling is consistent."""

import logging
from datetime import datetime, timezone
from falcon_core.sentinel.base import BaseSentinel, SentinelResult, SentinelStatus

logger = logging.getLogger(__name__)


class TimezoneSentinel(BaseSentinel):

    name = "timezone"
    description = "Verify timezone-aware timestamps are consistent across the system"

    def check(self) -> SentinelResult:
        issues = []

        # 1. Check zoneinfo is available
        try:
            from zoneinfo import ZoneInfo
            et = ZoneInfo("America/New_York")
        except ImportError:
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.FAIL,
                reason="zoneinfo module not available (Python 3.9+ required)",
            )

        # 2. Verify now() in ET is sane
        now_et = datetime.now(et)
        now_utc = datetime.now(timezone.utc)
        offset_hours = (now_et.utcoffset().total_seconds()) / 3600

        # ET offset should be -5 (EST) or -4 (EDT)
        if offset_hours not in (-5.0, -4.0):
            issues.append(f"ET offset is {offset_hours}h, expected -4 or -5")

        # 3. Verify round-trip: ET → ISO string → parse → ET
        iso_str = now_et.isoformat()
        parsed = datetime.fromisoformat(iso_str)
        if parsed != now_et:
            issues.append(f"ISO round-trip failed: {now_et} != {parsed}")

        # 4. Check that naive datetime assumption works correctly
        # A naive datetime should be treated as market time (ET)
        naive = datetime(2026, 3, 24, 9, 30, 0)
        localized = naive.replace(tzinfo=et)
        if localized.tzname() not in ("EST", "EDT"):
            issues.append(f"Naive localization produced unexpected tz: {localized.tzname()}")

        # 5. Check DB timestamps are timezone-aware (if DB available)
        db_check = self._check_db_timestamps()
        if db_check:
            issues.append(db_check)

        if issues:
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.WARN,
                reason=f"{len(issues)} timezone issue(s)",
                details={"issues": issues, "current_et": now_et.isoformat()},
            )

        return SentinelResult(
            name=self.name,
            status=SentinelStatus.PASS,
            reason=f"ET offset={int(offset_hours)}h, ISO round-trip OK",
            details={"current_et": now_et.isoformat(), "offset_hours": offset_hours},
        )

    def _check_db_timestamps(self) -> str:
        """Spot-check that recent DB timestamps have timezone info."""
        try:
            from falcon_core import get_db_manager
            db = get_db_manager()
            row = db.execute(
                "SELECT timestamp FROM orders ORDER BY id DESC LIMIT 1",
                fetch="one",
            )
            if row and row["timestamp"]:
                ts = str(row["timestamp"])
                # If timestamp has offset info (e.g. -04:00), it's aware
                if "+" not in ts and "-" not in ts[-6:]:
                    return f"Latest order timestamp appears naive: {ts}"
        except Exception:
            pass  # DB not available is checked by database sentinel
        return ""
