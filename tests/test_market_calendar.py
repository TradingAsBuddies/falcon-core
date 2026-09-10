"""Tests for falcon_core.market_calendar (falcon-core#20).

Deliberately stdlib-only: no pandas, no exchange_calendars, no network. The
module these cover exists precisely so that the platform does not depend on a
third-party calendar at import time, and the tests hold that line.
"""

import datetime as dt

import pytest

from falcon_core.market_calendar import (
    EARLY_CLOSE,
    MARKET_CLOSE,
    MarketCalendarError,
    is_early_close,
    is_rth,
    is_session,
    next_session,
    previous_session,
    session_close,
    sessions_between,
)


# --------------------------------------------------------------------------
# the bug that started this: Labor Day 2026-09-07
# --------------------------------------------------------------------------

def test_labor_day_2026_is_not_a_session():
    """The sentinel FAIL in the 2026-09-10 review was Labor Day."""
    assert is_session(dt.date(2026, 9, 7)) is False


def test_previous_session_skips_labor_day():
    """Tuesday's previous session is the preceding Friday, not Monday."""
    assert previous_session(dt.date(2026, 9, 8)) == dt.date(2026, 9, 4)


def test_scheduler_naive_subtraction_would_have_been_wrong():
    """Guards the exact defect: today-1day then skip weekends."""
    naive = dt.date(2026, 9, 8) - dt.timedelta(days=1)
    while naive.weekday() >= 5:
        naive -= dt.timedelta(days=1)
    assert naive == dt.date(2026, 9, 7)      # what the old code produced
    assert is_session(naive) is False        # and it was a holiday


# --------------------------------------------------------------------------
# weekends and ordinary sessions
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "day, expected",
    [
        (dt.date(2026, 9, 10), True),   # Thursday
        (dt.date(2026, 9, 11), True),   # Friday
        (dt.date(2026, 9, 12), False),  # Saturday
        (dt.date(2026, 9, 13), False),  # Sunday
        (dt.date(2026, 9, 14), True),   # Monday
    ],
)
def test_weekday_sessions(day, expected):
    assert is_session(day) is expected


# --------------------------------------------------------------------------
# the full NYSE holiday set
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "day, name",
    [
        (dt.date(2026, 1, 1), "New Year's Day"),
        (dt.date(2026, 1, 19), "MLK Jr. Day"),
        (dt.date(2026, 2, 16), "Washington's Birthday"),
        (dt.date(2026, 4, 3), "Good Friday"),
        (dt.date(2026, 5, 25), "Memorial Day"),
        (dt.date(2026, 6, 19), "Juneteenth"),
        (dt.date(2026, 7, 3), "Independence Day (observed)"),
        (dt.date(2026, 9, 7), "Labor Day"),
        (dt.date(2026, 11, 26), "Thanksgiving"),
        (dt.date(2026, 12, 25), "Christmas"),
    ],
)
def test_2026_holidays(day, name):
    assert is_session(day) is False, f"{name} on {day} should be closed"


@pytest.mark.parametrize(
    "day, name",
    [
        (dt.date(2025, 1, 1), "New Year's Day"),
        (dt.date(2025, 1, 20), "MLK Jr. Day"),
        (dt.date(2025, 4, 18), "Good Friday"),
        (dt.date(2025, 5, 26), "Memorial Day"),
        (dt.date(2025, 6, 19), "Juneteenth"),
        (dt.date(2025, 7, 4), "Independence Day"),
        (dt.date(2025, 9, 1), "Labor Day"),
        (dt.date(2025, 11, 27), "Thanksgiving"),
        (dt.date(2025, 12, 25), "Christmas"),
    ],
)
def test_2025_holidays(day, name):
    assert is_session(day) is False, f"{name} on {day} should be closed"


def test_juneteenth_not_observed_before_2022():
    """June 19 2020 was a Friday and a normal trading day."""
    assert is_session(dt.date(2020, 6, 19)) is True


def test_saturday_new_year_is_not_observed_on_december_31():
    """2022-01-01 was a Saturday; the NYSE traded Friday 2021-12-31."""
    assert is_session(dt.date(2021, 12, 31)) is True


def test_sunday_holiday_observed_on_monday():
    """2022-12-25 was a Sunday; Monday the 26th was the observed holiday."""
    assert is_session(dt.date(2022, 12, 26)) is False


# --------------------------------------------------------------------------
# early closes
# --------------------------------------------------------------------------

def test_day_after_thanksgiving_is_early_close():
    black_friday = dt.date(2026, 11, 27)
    assert is_session(black_friday) is True
    assert is_early_close(black_friday) is True
    assert session_close(black_friday) == EARLY_CLOSE


def test_ordinary_session_closes_at_four():
    assert session_close(dt.date(2026, 9, 10)) == MARKET_CLOSE


def test_christmas_eve_early_close_when_midweek():
    """2026-12-24 is a Thursday -- a 13:00 close."""
    assert is_early_close(dt.date(2026, 12, 24)) is True


def test_session_close_on_a_holiday_raises():
    with pytest.raises(MarketCalendarError):
        session_close(dt.date(2026, 9, 7))


# --------------------------------------------------------------------------
# is_rth -- the guard falcon-trader#23 needs
# --------------------------------------------------------------------------

def test_the_1650_fill_is_rejected():
    """2026-09-08 16:50 ET produced a live SELL burst. It must not be RTH."""
    assert is_rth(dt.datetime(2026, 9, 8, 16, 50, 7)) is False


@pytest.mark.parametrize(
    "moment, expected",
    [
        (dt.datetime(2026, 9, 8, 9, 29, 59), False),  # one second early
        (dt.datetime(2026, 9, 8, 9, 30, 0), True),    # the open
        (dt.datetime(2026, 9, 8, 12, 0, 0), True),    # midday
        (dt.datetime(2026, 9, 8, 15, 59, 59), True),  # one second to go
        (dt.datetime(2026, 9, 8, 16, 0, 0), False),   # the close is exclusive
        (dt.datetime(2026, 9, 8, 4, 0, 0), False),    # premarket
        (dt.datetime(2026, 9, 8, 19, 0, 0), False),   # after hours
    ],
)
def test_rth_boundaries(moment, expected):
    assert is_rth(moment) is expected


def test_rth_false_all_day_on_labor_day():
    """Covers the second test the issue asked for: 10:05 on a holiday."""
    assert is_rth(dt.datetime(2026, 9, 7, 10, 5)) is False


def test_rth_respects_early_close():
    black_friday = dt.date(2026, 11, 27)
    assert is_rth(dt.datetime.combine(black_friday, dt.time(12, 59))) is True
    assert is_rth(dt.datetime.combine(black_friday, dt.time(13, 0))) is False


def test_naive_datetime_is_treated_as_eastern():
    """A naive 10:00 must not be reinterpreted as 10:00 UTC (= 06:00 ET)."""
    assert is_rth(dt.datetime(2026, 9, 8, 10, 0)) is True


def test_aware_utc_datetime_is_converted():
    """14:00 UTC on a September session is 10:00 ET -- inside RTH."""
    utc_noon = dt.datetime(2026, 9, 8, 14, 0, tzinfo=dt.timezone.utc)
    assert is_rth(utc_noon) is True

    utc_late = dt.datetime(2026, 9, 8, 21, 0, tzinfo=dt.timezone.utc)  # 17:00 ET
    assert is_rth(utc_late) is False


# --------------------------------------------------------------------------
# navigation
# --------------------------------------------------------------------------

def test_next_session_skips_the_weekend():
    assert next_session(dt.date(2026, 9, 11)) == dt.date(2026, 9, 14)


def test_next_session_skips_a_holiday_weekend():
    """Friday before Labor Day -> the following Tuesday."""
    assert next_session(dt.date(2026, 9, 4)) == dt.date(2026, 9, 8)


def test_sessions_between_excludes_holiday_and_weekend():
    got = sessions_between(dt.date(2026, 9, 4), dt.date(2026, 9, 9))
    assert got == [dt.date(2026, 9, 4), dt.date(2026, 9, 8), dt.date(2026, 9, 9)]


def test_sessions_between_reversed_range_is_empty():
    assert sessions_between(dt.date(2026, 9, 9), dt.date(2026, 9, 4)) == []


def test_accepts_iso_strings_and_datetimes():
    assert is_session("2026-09-07") is False
    assert is_session("2026-09-08") is True
    assert is_session(dt.datetime(2026, 9, 7, 10, 0)) is False


# --------------------------------------------------------------------------
# cross-check against exchange_calendars when it is available
# --------------------------------------------------------------------------

def test_agrees_with_exchange_calendars_if_installed():
    """Skipped in this environment; meaningful in a full CI image."""
    ec = pytest.importorskip("exchange_calendars")
    xnys = ec.get_calendar("XNYS")
    day = dt.date(2026, 1, 2)
    while day < dt.date(2027, 1, 1):
        expected = xnys.is_session(day.isoformat())
        assert is_session(day) is bool(expected), f"disagreement on {day}"
        day += dt.timedelta(days=1)
