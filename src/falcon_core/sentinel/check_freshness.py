"""Sentinels: freshness of ingested data, and of the scheduled pipeline.

These exist because the platform spent six months in a state where every
visible signal was green — containers healthy, timers firing, the data-feed
sentinel passing — while `daily_bars` sat frozen at 2026-02-20 and
`data-sync-daily` failed every single night.

Nothing caught it because the data-feed sentinel reads the S3 flat files, not
the table the pipeline writes into. Flat files were fine the whole time; the
ingest was not. These two checks look at the database instead, which is the
side that was actually broken.

Both are deliberately database-only. Sentinel normally runs inside the trader
container, where there is no systemctl and no podman, so unit state and image
drift cannot be observed from here. Those live host-side in
falcon-platform/scripts: `deploy-dev.sh status` reports failed units and
`build-images.sh status` reports image-vs-source drift.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Optional

from falcon_core.sentinel.base import BaseSentinel, SentinelResult, SentinelStatus

logger = logging.getLogger(__name__)

# Sessions behind the newest expected one before we complain. One absorbs a
# single market holiday without crying wolf.
DATA_STALE_WARN_SESSIONS = 1
DATA_STALE_FAIL_SESSIONS = 3

# Days since the last successful nightly sync recorded in sync_log.
SYNC_WARN_DAYS = 2
SYNC_FAIL_DAYS = 4

# Days since the newest strategy backtest. strategy-seed runs daily, but a
# missed cycle is less urgent than missed market data.
BACKTEST_WARN_DAYS = 4
BACKTEST_FAIL_DAYS = 8


def _previous_weekday(d: date) -> date:
    """The most recent weekday strictly before ``d``.

    The nightly sync loads *yesterday's* bars, so this — not today — is the
    newest session that should be present. Holidays are not modelled; the
    staleness thresholds absorb them instead.
    """
    d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _weekdays_between(start: date, end: date) -> int:
    """Count weekdays after ``start`` through ``end`` inclusive."""
    if end <= start:
        return 0
    count = 0
    cursor = start + timedelta(days=1)
    while cursor <= end:
        if cursor.weekday() < 5:
            count += 1
        cursor += timedelta(days=1)
    return count


def _as_date(value) -> Optional[date]:
    """Normalise a DB value to a date, tolerating datetime and ISO strings."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.fromisoformat(str(value)).date()
    except ValueError:
        return None


class DataFreshnessSentinel(BaseSentinel):

    name = "data-freshness"
    description = "Verify ingested market data in the database is current"

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

        try:
            daily_row = db.execute(
                "SELECT max(date) AS newest FROM daily_bars", fetch="one"
            )
            minute_row = db.execute(
                "SELECT max(timestamp) AS newest FROM minute_bars", fetch="one"
            )
        except Exception as e:
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.FAIL,
                reason=f"Cannot read bar tables: {e}",
            )

        newest_daily = _as_date(daily_row["newest"] if daily_row else None)
        newest_minute = _as_date(minute_row["newest"] if minute_row else None)

        if newest_daily is None:
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.FAIL,
                reason="daily_bars is empty — nothing has ever been ingested",
            )

        expected = _previous_weekday(date.today())
        behind = _weekdays_between(newest_daily, expected)

        details = {
            "daily_newest": str(newest_daily),
            "expected_newest": str(expected),
            "sessions_behind": behind,
            "minute_newest": str(newest_minute) if newest_minute else None,
        }

        # minute_bars is reported but deliberately does not drive status: its
        # only reader is data_feed's third-priority database fallback, behind
        # flat files and Polygon, so it is not maintained on this deployment.
        minute_note = (
            f" · minute_bars {newest_minute} (fallback path, not maintained)"
            if newest_minute else ""
        )

        if behind > DATA_STALE_FAIL_SESSIONS:
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.FAIL,
                reason=(
                    f"daily_bars {behind} sessions behind — newest {newest_daily}, "
                    f"expected {expected}{minute_note}"
                ),
                details=details,
            )

        if behind > DATA_STALE_WARN_SESSIONS:
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.WARN,
                reason=(
                    f"daily_bars {behind} sessions behind — newest {newest_daily}, "
                    f"expected {expected}{minute_note}"
                ),
                details=details,
            )

        return SentinelResult(
            name=self.name,
            status=SentinelStatus.PASS,
            reason=f"daily_bars current through {newest_daily}{minute_note}",
            details=details,
        )


class PipelineFreshnessSentinel(BaseSentinel):

    name = "pipeline-freshness"
    description = "Verify the scheduled jobs are still producing results"

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

        problems = []
        details = {}
        worst = SentinelStatus.PASS

        def escalate(status: SentinelStatus):
            nonlocal worst
            order = {
                SentinelStatus.PASS: 0,
                SentinelStatus.WARN: 1,
                SentinelStatus.FAIL: 2,
            }
            if order[status] > order[worst]:
                worst = status

        # ── Nightly ingest: last run that actually succeeded ──
        # Deliberately filtered to status='success'. The failure mode this
        # catches wrote a fresh error row every night, so "a recent sync_log
        # entry" would have looked healthy throughout.
        try:
            row = db.execute(
                """SELECT max(completed_at) AS last_ok
                   FROM sync_log
                   WHERE sync_type = 'daily' AND status = 'success'""",
                fetch="one",
            )
            last_ok = _as_date(row["last_ok"] if row else None)
            details["last_successful_daily_sync"] = str(last_ok) if last_ok else None

            if last_ok is None:
                problems.append("no successful daily sync on record")
                escalate(SentinelStatus.FAIL)
            else:
                age = (date.today() - last_ok).days
                details["daily_sync_age_days"] = age
                if age >= SYNC_FAIL_DAYS:
                    problems.append(f"last successful daily sync was {age}d ago")
                    escalate(SentinelStatus.FAIL)
                elif age >= SYNC_WARN_DAYS:
                    problems.append(f"last successful daily sync was {age}d ago")
                    escalate(SentinelStatus.WARN)
        except Exception as e:
            problems.append(f"sync_log unreadable: {e}")
            escalate(SentinelStatus.WARN)

        # ── Strategy backtests: newest roster metric write ──
        try:
            row = db.execute(
                "SELECT max(last_backtest_at) AS newest FROM strategy_roster",
                fetch="one",
            )
            newest = _as_date(row["newest"] if row else None)
            details["last_backtest_at"] = str(newest) if newest else None

            if newest is None:
                problems.append("no strategy has ever been backtested")
                escalate(SentinelStatus.WARN)
            else:
                age = (date.today() - newest).days
                details["backtest_age_days"] = age
                if age >= BACKTEST_FAIL_DAYS:
                    problems.append(f"newest backtest was {age}d ago")
                    escalate(SentinelStatus.FAIL)
                elif age >= BACKTEST_WARN_DAYS:
                    problems.append(f"newest backtest was {age}d ago")
                    escalate(SentinelStatus.WARN)
        except Exception as e:
            problems.append(f"strategy_roster unreadable: {e}")
            escalate(SentinelStatus.WARN)

        if problems:
            return SentinelResult(
                name=self.name,
                status=worst,
                reason="; ".join(problems),
                details=details,
            )

        return SentinelResult(
            name=self.name,
            status=SentinelStatus.PASS,
            reason=(
                f"daily sync OK {details.get('last_successful_daily_sync')} · "
                f"backtests OK {details.get('last_backtest_at')}"
            ),
            details=details,
        )
