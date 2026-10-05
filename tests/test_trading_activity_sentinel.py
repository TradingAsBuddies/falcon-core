"""Output-side sentinels: recommendations produced, and orders placed.

Both exist because the sentinel suite reported 10/10 through two long
silences: 13 days of zero recommendations (Finviz renumbered its CSV columns)
and 24 days of zero orders (the orchestrator read a path that did not exist in
its container and used a throwaway database). Every existing check verifies an
input; these check what came out.
"""

import datetime as dt

import pytest

from falcon_core.sentinel import check_trading_activity as activity
from falcon_core.sentinel.base import SentinelStatus


class _FakeDB:
    """Answers the aggregate queries these sentinels make."""

    def __init__(self, newest=None, runs=0, recs=0, fail=None):
        self.newest = newest
        self.runs = runs
        self.recs = recs
        self.fail = fail

    def execute(self, query, params=None, fetch=None):
        if self.fail:
            raise RuntimeError(self.fail)
        if "count(*)" in query:
            return {"runs": self.runs, "recs": self.recs}
        return {"newest": self.newest}


# 2026-10-05 is a Monday; 10-02 the previous Friday.
MONDAY = dt.date(2026, 10, 5)


@pytest.fixture
def frozen(monkeypatch):
    monkeypatch.setattr(activity, "_newest_session", lambda today=None: MONDAY)


def _screener(monkeypatch, db):
    monkeypatch.setattr(activity, "_db", lambda: db)
    return activity.ScreenerOutputSentinel().check()


def _trading(monkeypatch, db, halted=False, reason=None):
    monkeypatch.setattr(activity, "_db", lambda: db)
    monkeypatch.setattr(activity.TradingActivitySentinel, "_halt_state",
                        staticmethod(lambda: (halted, reason)))
    return activity.TradingActivitySentinel().check()


# ── screener output ─────────────────────────────────────────────────────

def test_todays_recommendations_pass(frozen, monkeypatch):
    db = _FakeDB(newest=dt.datetime(2026, 10, 5, 8, 0), runs=4, recs=14)
    result = _screener(monkeypatch, db)
    assert result.status is SentinelStatus.PASS
    assert "14 recommendation(s)" in result.reason


def test_the_thirteen_day_silence_fails(frozen, monkeypatch):
    """The real case: runs completing, reporting success, screening nothing."""
    db = _FakeDB(newest=dt.datetime(2026, 9, 17, 8, 0), runs=7, recs=0)
    result = _screener(monkeypatch, db)
    assert result.status is SentinelStatus.FAIL
    assert "runs are completing but screening nothing" in result.reason
    assert result.details["recommendations_last_2d"] == 0


def test_a_screener_that_is_not_running_says_so(frozen, monkeypatch):
    db = _FakeDB(newest=dt.datetime(2026, 9, 17, 8, 0), runs=0, recs=0)
    assert "screener is not running" in _screener(monkeypatch, db).reason


def test_one_quiet_session_is_not_an_alarm(frozen, monkeypatch):
    """Friday produced; Monday has not yet. A quiet morning is not an outage."""
    db = _FakeDB(newest=dt.datetime(2026, 10, 2, 8, 0), runs=4, recs=0)
    assert _screener(monkeypatch, db).status is SentinelStatus.PASS


def test_two_sessions_warn(frozen, monkeypatch):
    db = _FakeDB(newest=dt.datetime(2026, 10, 1, 8, 0), runs=4, recs=0)
    assert _screener(monkeypatch, db).status is SentinelStatus.WARN


def test_never_produced_anything_fails(frozen, monkeypatch):
    assert _screener(monkeypatch, _FakeDB(newest=None)).status is SentinelStatus.FAIL


def test_unreadable_table_fails_loudly(frozen, monkeypatch):
    result = _screener(monkeypatch, _FakeDB(fail="relation does not exist"))
    assert result.status is SentinelStatus.FAIL
    assert "relation does not exist" in result.reason


# ── trading activity ────────────────────────────────────────────────────

def test_orders_today_pass(frozen, monkeypatch):
    db = _FakeDB(newest=dt.datetime(2026, 10, 5, 14, 0))
    assert _trading(monkeypatch, db).status is SentinelStatus.PASS


def test_the_twenty_four_day_silence_fails(frozen, monkeypatch):
    """2026-09-11 to 10-05: 16 sessions with no order and 10/10 passing."""
    db = _FakeDB(newest=dt.datetime(2026, 9, 11, 14, 22))
    result = _trading(monkeypatch, db)
    assert result.status is SentinelStatus.FAIL
    assert "The orchestrator is not trading." in result.reason
    assert result.details["sessions_behind"] > 10


def test_five_quiet_sessions_still_pass(frozen, monkeypatch):
    """Signals do not fire every day; the warn threshold is *more* than five."""
    db = _FakeDB(newest=dt.datetime(2026, 9, 28, 14, 0))  # 5 sessions back
    result = _trading(monkeypatch, db)
    assert result.details["sessions_behind"] == 5
    assert result.status is SentinelStatus.PASS


def test_six_quiet_sessions_warn(frozen, monkeypatch):
    db = _FakeDB(newest=dt.datetime(2026, 9, 25, 14, 0))
    result = _trading(monkeypatch, db)
    assert result.details["sessions_behind"] == 6
    assert result.status is SentinelStatus.WARN


def test_yesterdays_order_passes(frozen, monkeypatch):
    db = _FakeDB(newest=dt.datetime(2026, 10, 2, 15, 0))
    assert _trading(monkeypatch, db).status is SentinelStatus.PASS


def test_a_deliberate_halt_is_not_an_outage(frozen, monkeypatch):
    """Halting on purpose must not raise an alarm about not trading."""
    db = _FakeDB(newest=dt.datetime(2026, 9, 11, 14, 22))
    result = _trading(monkeypatch, db, halted=True, reason="Halt file present")
    assert result.status is SentinelStatus.SKIP
    assert "halted on purpose" in result.reason


def test_a_halt_does_not_mask_a_recent_book(frozen, monkeypatch):
    db = _FakeDB(newest=dt.datetime(2026, 10, 5, 14, 0))
    assert _trading(monkeypatch, db, halted=True, reason="x").status is SentinelStatus.PASS


def test_no_orders_ever_fails(frozen, monkeypatch):
    assert _trading(monkeypatch, _FakeDB(newest=None)).status is SentinelStatus.FAIL


def test_no_orders_ever_while_halted_skips(frozen, monkeypatch):
    result = _trading(monkeypatch, _FakeDB(newest=None), halted=True, reason="env")
    assert result.status is SentinelStatus.SKIP


def test_unreadable_orders_fails(frozen, monkeypatch):
    assert _trading(monkeypatch, _FakeDB(fail="boom")).status is SentinelStatus.FAIL


def test_halt_state_is_safe_without_falcon_trader():
    """falcon-core must not require falcon-trader to run its sentinels."""
    halted, reason = activity.TradingActivitySentinel._halt_state()
    assert halted in (True, False)


# ── the session the check measures against ──────────────────────────────

def test_newest_session_on_a_trading_day_is_today():
    assert activity._newest_session(dt.date(2026, 10, 5)) == dt.date(2026, 10, 5)


def test_newest_session_on_a_sunday_is_the_friday():
    """A weekend must not read as an outage."""
    assert activity._newest_session(dt.date(2026, 10, 4)) == dt.date(2026, 10, 2)


def test_newest_session_after_a_holiday():
    """2026-09-07 was Labor Day; the session before it is the Friday."""
    assert activity._newest_session(dt.date(2026, 9, 7)) == dt.date(2026, 9, 4)


@pytest.mark.parametrize("value,expected", [
    (dt.datetime(2026, 10, 5, 8, 0), dt.date(2026, 10, 5)),
    (dt.date(2026, 10, 5), dt.date(2026, 10, 5)),
    ("2026-10-05T08:00:00", dt.date(2026, 10, 5)),
    ("2026-10-05", dt.date(2026, 10, 5)),
    (None, None),
    ("not a date", None),
    (42, None),
])
def test_date_coercion(value, expected):
    assert activity._as_date(value) == expected
