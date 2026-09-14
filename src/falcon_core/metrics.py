"""Risk-metric arithmetic, as pure functions (falcon-core#21).

Split out of ``backtesting.engine`` because the Sharpe calculation has now been
wrong twice, in two different ways, and neither was caught by a test:

1. Originally: an *annualised* return divided by a per-trade volatility scaled
   by ``sqrt(252)`` -- two incompatible time bases.
2. Then my own fix: per-*trade* returns scaled by ``sqrt(bars_per_year)``,
   which on 1-minute bars is ``sqrt(98280)`` ~ 313x. Worse than what it
   replaced.

Both survived because the engine needs pandas to import, pandas is not present
in every environment this is developed in, and so nothing exercised the
arithmetic directly. These functions take plain numbers and dates, so they can
be tested anywhere.

The fix for both is the same idea: **the returns series and the annualisation
factor must describe the same unit of time.** Build a daily equity curve, take
daily returns, scale by ``sqrt(252)``.
"""

from __future__ import annotations

import datetime as _dt
import math
import statistics as _statistics
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "MIN_RETURN_OBSERVATIONS",
    "daily_equity_curve",
    "simple_returns",
    "sharpe_ratio",
    "annualized_volatility",
    "max_drawdown",
]

TRADING_DAYS_PER_YEAR = 252

#: Below this many daily observations the standard deviation is dominated by
#: noise and the resulting Sharpe is not a measurement of anything. Callers
#: should treat a result computed from fewer as unreliable rather than acting
#: on it -- four trades once produced a reported Sharpe of 25.98, and the
#: advisor then optimised toward exactly that.
MIN_RETURN_OBSERVATIONS = 20


def _as_date(value) -> _dt.date:
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    if isinstance(value, str):
        return _dt.date.fromisoformat(value[:10])
    raise TypeError(f"cannot interpret {value!r} as a date")


def daily_equity_curve(
    initial_capital: float,
    exits: Iterable[Tuple[object, float]],
    session_dates: Optional[Sequence] = None,
) -> List[Tuple[_dt.date, float]]:
    """Equity per *session*, not per trade.

    `exits` is ``(exit_time, pnl_dollar)`` pairs. `session_dates` is the set of
    dates the backtest actually covered; supplying it makes days with no fill
    appear as flat days, which is what stops a strategy that trades rarely from
    looking artificially low-volatility.

    Weekends and holidays must not be in `session_dates` -- pass the dates
    present in the bar data, not a calendar range.
    """
    realized: Dict[_dt.date, float] = {}
    for exit_time, pnl in exits:
        day = _as_date(exit_time)
        realized[day] = realized.get(day, 0.0) + float(pnl)

    if session_dates:
        days = sorted({_as_date(d) for d in session_dates})
    else:
        days = sorted(realized)

    # A fill dated outside the supplied sessions still has to land somewhere,
    # or its P&L silently vanishes from the curve.
    for day in realized:
        if day not in days:
            days.append(day)
    days.sort()

    curve, equity = [], float(initial_capital)
    for day in days:
        equity += realized.get(day, 0.0)
        curve.append((day, equity))
    return curve


def simple_returns(equity: Sequence[float]) -> List[float]:
    """Period-over-period simple returns. Zero-equity steps are skipped."""
    out = []
    for prev, cur in zip(equity, equity[1:]):
        if prev:
            out.append((cur - prev) / prev)
    return out


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def _stdev(xs: Sequence[float]) -> float:
    """Sample standard deviation, matching pandas' default ddof=1.

    Delegates to `statistics.stdev`, which carries exact fractions
    internally. The hand-rolled form, sqrt(sum((x - mean)**2)/(n-1)), is not
    exact for a constant series: the computed mean of [0.001] * 50 is
    0.0010000000000000007, so every (x - m) is a non-zero residual and the
    result is ~6.6e-19 rather than 0.0, defeating any `sd == 0.0` guard.
    """
    if len(xs) < 2:
        return 0.0
    return _statistics.stdev(xs)


def annualized_volatility(
    daily_returns: Sequence[float],
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised volatility from *daily* returns."""
    if len(daily_returns) < 2:
        return 0.0
    return _stdev(daily_returns) * math.sqrt(periods_per_year)


def sharpe_ratio(
    daily_returns: Sequence[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    min_observations: int = MIN_RETURN_OBSERVATIONS,
) -> Tuple[float, bool]:
    """Annualised Sharpe from daily returns.

    Returns ``(sharpe, reliable)``. `reliable` is False when there were fewer
    than `min_observations` daily returns, or no variation in them -- the
    caller should not let a promotion gate act on an unreliable figure.

    `risk_free_rate` is an annual rate; it is converted to a per-period rate
    before subtracting, because subtracting an *annual* rate from a *daily*
    return is the same class of unit error this module exists to prevent.
    """
    if len(daily_returns) < 2:
        return 0.0, False

    per_period_rf = risk_free_rate / periods_per_year
    excess = [r - per_period_rf for r in daily_returns]

    sd = _stdev(excess)
    # Relative, not `sd == 0.0`. An exact comparison only catches variance that
    # cancels perfectly in floating point; a series constant to within rounding
    # still yields a tiny sd, and mean/sd then reports an astronomically
    # confident Sharpe on what is actually a flat equity curve.
    scale = max((abs(x) for x in excess), default=0.0)
    if sd <= scale * 1e-12:
        return 0.0, False

    sharpe = (_mean(excess) / sd) * math.sqrt(periods_per_year)
    return sharpe, len(daily_returns) >= min_observations


def max_drawdown(equity: Sequence[float]) -> float:
    """Largest peak-to-trough decline, as a positive fraction."""
    peak, worst = None, 0.0
    for value in equity:
        if peak is None or value > peak:
            peak = value
        if peak:
            worst = max(worst, (peak - value) / peak)
    return worst
