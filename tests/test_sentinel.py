"""Tests for the sentinel framework.

Stdlib only — no pandas, no database, no network. The individual checks all
reach for live infrastructure and are exercised on the deployment host; what
is testable here is the framework around them, and specifically the date
selection that made them report FAIL for a week after Labor Day
(falcon-core#20).
"""

import datetime as dt

import pytest

from falcon_core.market_calendar import is_session
from falcon_core.sentinel.base import (
    BaseSentinel,
    sessions_since,
    SentinelResult,
    SentinelRunner,
    SentinelStatus,
    probe_session,
)


# ── probe_session ───────────────────────────────────────────────────────

def test_probe_session_returns_a_real_session():
    """Whatever it returns must be a day the market was actually open."""
    day = dt.date.fromisoformat(probe_session())
    assert is_session(day)


@pytest.mark.parametrize("lag", [1, 2, 3, 5, 10])
def test_probe_session_returns_a_session_for_any_lag(lag):
    day = dt.date.fromisoformat(probe_session(lag_days=lag))
    assert is_session(day)


def test_probe_session_never_returns_the_future():
    today = dt.date(2026, 9, 11)
    day = dt.date.fromisoformat(probe_session(today=today))
    assert day < today


def test_probe_session_skips_labor_day_2026():
    """The regression this helper exists for.

    2026-09-10 minus the default 3-day lag lands on Monday 2026-09-07, which
    is Labor Day. The old implementation skipped only weekends, returned it,
    and the flat-file fetch then failed because the S3 key does not exist.
    """
    labor_day = dt.date(2026, 9, 7)
    assert not is_session(labor_day), "2026-09-07 should be Labor Day"

    got = dt.date.fromisoformat(probe_session(today=dt.date(2026, 9, 10)))
    assert got != labor_day
    assert got == dt.date(2026, 9, 4), "should fall back to the Friday"


def test_probe_session_skips_weekends():
    # 2026-09-14 is a Monday; minus 3 days is Friday 09-11, a session.
    got = dt.date.fromisoformat(probe_session(today=dt.date(2026, 9, 14)))
    assert got == dt.date(2026, 9, 11)

    # 2026-09-12 is a Saturday; minus 3 days is Wednesday 09-09.
    got = dt.date.fromisoformat(probe_session(today=dt.date(2026, 9, 12)))
    assert got == dt.date(2026, 9, 9)


def test_probe_session_keeps_a_target_that_is_itself_a_session():
    """previous_session is strictly-before; a valid target must not be stepped over."""
    # 2026-09-11 minus 3 days = Tuesday 2026-09-08, a normal session.
    got = dt.date.fromisoformat(probe_session(today=dt.date(2026, 9, 11)))
    assert got == dt.date(2026, 9, 8)


def test_probe_session_handles_thanksgiving_week():
    """Thanksgiving 2026-11-26 is a Thursday closure."""
    thanksgiving = dt.date(2026, 11, 26)
    assert not is_session(thanksgiving)
    got = dt.date.fromisoformat(probe_session(today=dt.date(2026, 11, 29)))
    assert is_session(got)
    assert got != thanksgiving


# ── runner / status plumbing ────────────────────────────────────────────

class _Stub(BaseSentinel):
    name = "stub"
    description = "fixed result"

    def __init__(self, status):
        self._status = status

    def check(self):
        return SentinelResult(name=self.name, status=self._status, reason="stub")


class _Exploding(BaseSentinel):
    name = "exploding"
    description = "raises"

    def check(self):
        raise RuntimeError("boom")


def test_runner_reports_each_status():
    for status in (SentinelStatus.PASS, SentinelStatus.WARN,
                   SentinelStatus.FAIL, SentinelStatus.SKIP):
        result = SentinelRunner([_Stub(status)]).run_all()[0]
        assert result.status is status


def test_a_raising_sentinel_becomes_fail_not_a_crash():
    """One broken check must not take down the whole run."""
    results = SentinelRunner([_Exploding(), _Stub(SentinelStatus.PASS)]).run_all()
    assert results[0].status is SentinelStatus.FAIL
    assert "boom" in results[0].reason
    assert results[1].status is SentinelStatus.PASS


def test_unknown_sentinel_name_skips():
    result = SentinelRunner([_Stub(SentinelStatus.PASS)]).run_one("nope")
    assert result.status is SentinelStatus.SKIP


def test_builtin_sentinels_all_load():
    """Every shipped check must import — a typo here is invisible until runtime."""
    names = SentinelRunner().sentinel_names
    for expected in (
        "database", "data-feed", "data-feed-minute", "polygon-minute",
        "strategy-roster", "backtest-engine", "timezone", "market-page",
        "data-freshness", "pipeline-freshness",
    ):
        assert expected in names, f"{expected} missing from {names}"


def test_result_serializes():
    d = SentinelResult(name="x", status=SentinelStatus.SKIP, reason="r").to_dict()
    assert d["status"] == "skip"
    assert d["name"] == "x"


# ── sessions_since ──────────────────────────────────────────────────────

def test_sessions_since_weekend_is_one_not_three():
    """The regression: a Monday must not read as a multi-day outage.

    The nightly pipelines record no_data when the market was shut, so the
    newest success is always the last session. (today - day).days counted
    Saturday and Sunday as missed work and warned every Monday.
    """
    # Friday 2026-09-11 -> Monday 2026-09-14
    assert sessions_since(dt.date(2026, 9, 11), dt.date(2026, 9, 14)) == 1
    assert (dt.date(2026, 9, 14) - dt.date(2026, 9, 11)).days == 3  # the old answer


def test_sessions_since_holiday_week():
    """Labor Day 2026-09-07 must not count as a missed session."""
    # Friday 09-04 -> Tuesday 09-08, with Monday a holiday: one session (09-08).
    assert sessions_since(dt.date(2026, 9, 4), dt.date(2026, 9, 8)) == 1


def test_sessions_since_same_session_is_zero():
    assert sessions_since(dt.date(2026, 9, 11), dt.date(2026, 9, 11)) == 0


def test_sessions_since_non_session_day_adds_nothing():
    # Friday -> Saturday: no session has passed.
    assert sessions_since(dt.date(2026, 9, 11), dt.date(2026, 9, 12)) == 0


def test_sessions_since_consecutive_sessions():
    assert sessions_since(dt.date(2026, 9, 10), dt.date(2026, 9, 11)) == 1


def test_sessions_since_handles_none_and_future():
    assert sessions_since(None, dt.date(2026, 9, 14)) == 0
    assert sessions_since(dt.date(2026, 9, 20), dt.date(2026, 9, 14)) == 0
