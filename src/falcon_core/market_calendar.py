"""NYSE/NASDAQ trading calendar.

Answers four questions the rest of the platform kept getting wrong:

    is_session(date)        -- is the US equity market open at all that day?
    previous_session(date)  -- the session strictly before `date`
    next_session(date)      -- the session strictly after `date`
    is_rth(timestamp)       -- is that instant inside regular trading hours?

Why this is hand-rolled instead of wrapping ``exchange_calendars``:
falcon-core's ``install_requires`` carries neither ``exchange_calendars`` nor
``pandas_market_calendars``, and this module is imported on paths that run at
container start. A missing third-party calendar would turn "the holiday logic is
wrong" into "the platform does not boot". So the NYSE rules are computed here from
stdlib only, and ``exchange_calendars`` is used *if it happens to be installed*
(see ``_external_calendar``) purely as a cross-check hook.

The rules are computed, not tabulated, so this does not expire at the end of a
hardcoded year. Holidays observed by the NYSE (Rule 7.2):

    New Year's Day, Martin Luther King Jr. Day, Washington's Birthday,
    Good Friday, Memorial Day, Juneteenth (from 2022), Independence Day,
    Labor Day, Thanksgiving Day, Christmas Day.

Saturday holidays are observed the preceding Friday, Sunday holidays the
following Monday -- except that New Year's Day falling on a Saturday is *not*
observed in the prior year (NYSE does not close December 31).

Early closes (13:00 ET): July 3rd when Independence Day is observed on a
weekday, the Friday after Thanksgiving, and Christmas Eve when it falls
Monday-Thursday.
"""

from __future__ import annotations

import datetime as _dt
from functools import lru_cache
from typing import Optional
from zoneinfo import ZoneInfo

__all__ = [
    "EASTERN",
    "MARKET_OPEN",
    "MARKET_CLOSE",
    "EARLY_CLOSE",
    "is_session",
    "previous_session",
    "next_session",
    "session_close",
    "is_early_close",
    "is_rth",
    "sessions_between",
    "MarketCalendarError",
]

EASTERN = ZoneInfo("America/New_York")

MARKET_OPEN = _dt.time(9, 30)
MARKET_CLOSE = _dt.time(16, 0)
EARLY_CLOSE = _dt.time(13, 0)

#: Juneteenth became an NYSE holiday in 2022.
_JUNETEENTH_FROM = 2022

#: Guard against an unbounded scan if the rules were ever broken.
_MAX_SCAN_DAYS = 30


class MarketCalendarError(RuntimeError):
    """Raised when no trading session can be found within a sane window."""


# --------------------------------------------------------------------------
# holiday computation
# --------------------------------------------------------------------------

def _nth_weekday(year: int, month: int, weekday: int, n: int) -> _dt.date:
    """The n-th `weekday` of `month` (Monday=0). n is 1-based."""
    first = _dt.date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + _dt.timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> _dt.date:
    """The last `weekday` of `month` (Monday=0)."""
    if month == 12:
        nxt = _dt.date(year + 1, 1, 1)
    else:
        nxt = _dt.date(year, month + 1, 1)
    last = nxt - _dt.timedelta(days=1)
    return last - _dt.timedelta(days=(last.weekday() - weekday) % 7)


def _easter(year: int) -> _dt.date:
    """Gregorian Easter Sunday (Anonymous Gregorian / Meeus algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 22 * m) // 451
    month, day = divmod(h + m - 7 * n + 114, 31)
    return _dt.date(year, month, day + 1)


def _observed(day: _dt.date, *, shift_saturday_back: bool = True) -> _dt.date:
    """Apply the weekend-observance shift to a fixed-date holiday."""
    if day.weekday() == 5:  # Saturday
        return day - _dt.timedelta(days=1) if shift_saturday_back else day
    if day.weekday() == 6:  # Sunday
        return day + _dt.timedelta(days=1)
    return day


@lru_cache(maxsize=64)
def _holidays(year: int) -> frozenset:
    """The set of dates on which the NYSE is fully closed in `year`."""
    days = set()

    # New Year's Day. A Saturday New Year is NOT observed on Dec 31 of the
    # prior year -- the NYSE stays open. So no backward shift here.
    days.add(_observed(_dt.date(year, 1, 1), shift_saturday_back=False))

    # Martin Luther King Jr. Day -- 3rd Monday in January (from 1998).
    if year >= 1998:
        days.add(_nth_weekday(year, 1, 0, 3))

    # Washington's Birthday -- 3rd Monday in February.
    days.add(_nth_weekday(year, 2, 0, 3))

    # Good Friday -- the Friday before Easter.
    days.add(_easter(year) - _dt.timedelta(days=2))

    # Memorial Day -- last Monday in May.
    days.add(_last_weekday(year, 5, 0))

    # Juneteenth -- June 19, from 2022.
    if year >= _JUNETEENTH_FROM:
        days.add(_observed(_dt.date(year, 6, 19)))

    # Independence Day -- July 4.
    days.add(_observed(_dt.date(year, 7, 4)))

    # Labor Day -- 1st Monday in September.
    days.add(_nth_weekday(year, 9, 0, 1))

    # Thanksgiving -- 4th Thursday in November.
    days.add(_nth_weekday(year, 11, 3, 4))

    # Christmas Day -- December 25.
    days.add(_observed(_dt.date(year, 12, 25)))

    return frozenset(days)


@lru_cache(maxsize=64)
def _early_closes(year: int) -> frozenset:
    """Dates with a 13:00 ET close."""
    days = set()

    # Day after Thanksgiving.
    days.add(_nth_weekday(year, 11, 3, 4) + _dt.timedelta(days=1))

    # July 3, when Independence Day is observed on a weekday and July 3 is a
    # weekday that is not itself the observed holiday.
    july_3 = _dt.date(year, 7, 3)
    if july_3.weekday() < 5 and july_3 not in _holidays(year):
        days.add(july_3)

    # Christmas Eve, when it falls Monday-Thursday.
    dec_24 = _dt.date(year, 12, 24)
    if dec_24.weekday() <= 3 and dec_24 not in _holidays(year):
        days.add(dec_24)

    return frozenset(d for d in days if d.weekday() < 5 and d not in _holidays(year))


def _external_calendar():
    """Return an ``exchange_calendars`` XNYS instance, or None.

    Present so a deployment that *does* carry the dependency can be
    cross-checked against these rules in tests. Never required at runtime.
    """
    try:  # pragma: no cover - depends on optional dependency
        import exchange_calendars  # type: ignore

        return exchange_calendars.get_calendar("XNYS")
    except Exception:
        return None


# --------------------------------------------------------------------------
# coercion
# --------------------------------------------------------------------------

def _as_date(value) -> _dt.date:
    """Coerce date / datetime / ISO string to a ``date``."""
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    if isinstance(value, str):
        return _dt.date.fromisoformat(value[:10])
    raise TypeError(f"cannot interpret {value!r} as a date")


def _as_eastern(value) -> _dt.datetime:
    """Coerce to a timezone-aware Eastern datetime.

    A naive datetime is *assumed to already be Eastern* rather than UTC.
    Every caller in this platform builds naive timestamps from Eastern-facing
    sources (Polygon aggregates, the scheduler's own ``datetime.now(TIMEZONE)``),
    so assuming UTC here would silently shift every RTH check by 4-5 hours.
    """
    if isinstance(value, str):
        value = _dt.datetime.fromisoformat(value)
    if not isinstance(value, _dt.datetime):
        raise TypeError(f"cannot interpret {value!r} as a datetime")
    if value.tzinfo is None:
        return value.replace(tzinfo=EASTERN)
    return value.astimezone(EASTERN)


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def is_session(value) -> bool:
    """True when the US equity market holds a regular session on that date."""
    day = _as_date(value)
    if day.weekday() >= 5:
        return False
    return day not in _holidays(day.year)


def is_early_close(value) -> bool:
    """True when that session ends at 13:00 ET instead of 16:00 ET."""
    day = _as_date(value)
    return is_session(day) and day in _early_closes(day.year)


def session_close(value) -> _dt.time:
    """The closing time for that date's session.

    Raises ``MarketCalendarError`` if the date is not a session at all --
    asking for the close of a holiday is a caller bug worth surfacing.
    """
    day = _as_date(value)
    if not is_session(day):
        raise MarketCalendarError(f"{day} is not a trading session")
    return EARLY_CLOSE if is_early_close(day) else MARKET_CLOSE


def previous_session(value) -> _dt.date:
    """The most recent session strictly before `value`."""
    day = _as_date(value) - _dt.timedelta(days=1)
    for _ in range(_MAX_SCAN_DAYS):
        if is_session(day):
            return day
        day -= _dt.timedelta(days=1)
    raise MarketCalendarError(
        f"no trading session in the {_MAX_SCAN_DAYS} days before {value}"
    )


def next_session(value) -> _dt.date:
    """The next session strictly after `value`."""
    day = _as_date(value) + _dt.timedelta(days=1)
    for _ in range(_MAX_SCAN_DAYS):
        if is_session(day):
            return day
        day += _dt.timedelta(days=1)
    raise MarketCalendarError(
        f"no trading session in the {_MAX_SCAN_DAYS} days after {value}"
    )


def sessions_between(start, end) -> list:
    """Every session in the inclusive range [start, end]."""
    first, last = _as_date(start), _as_date(end)
    if first > last:
        return []
    out, day = [], first
    while day <= last:
        if is_session(day):
            out.append(day)
        day += _dt.timedelta(days=1)
    return out


def is_rth(value: Optional[object] = None) -> bool:
    """True when `value` falls inside regular trading hours.

    RTH is 09:30 ET up to but not including the close (16:00, or 13:00 on an
    early-close day). Pre- and post-market do not count; that is the whole
    point -- see falcon-trader#23, where a SELL burst "filled" at 16:50.

    Passing no argument checks the current instant.
    """
    moment = _dt.datetime.now(EASTERN) if value is None else _as_eastern(value)
    if not is_session(moment.date()):
        return False
    return MARKET_OPEN <= moment.time() < session_close(moment.date())
