"""Signal validation: Eastern windows, price bounds, and visible counts.

Backtests reported 0 trades and 0.00% return for every strategy. On SPY,
atr_breakout generated 78 signals and validation rejected all 78:

    48  outside the trade window   (UTC bar stamp vs Eastern parameter)
    30  above max_price            (SPY is $770; the default cap was $500)

Both silent. backtest_runs.signals_count was read from an attribute no result
carried, so it was 0 on all 1,475 rows ever written, which made "saw nothing"
and "rejected everything" the same row.
"""

import datetime as dt
from zoneinfo import ZoneInfo

import pytest

pd = pytest.importorskip("pandas")

from falcon_core.backtesting.strategies.base import (
    EASTERN,
    Signal,
    SignalType,
    StrategyParams,
    eastern_time_of,
)

UTC = dt.timezone.utc


# ── the Eastern conversion ──────────────────────────────────────────────

@pytest.mark.parametrize("utc_hour,expected_et", [
    (13, 9),    # 13:35 UTC = 09:35 ET, the open
    (19, 15),   # 19:45 UTC = 15:45 ET, late session
    (14, 10),
])
def test_a_utc_stamp_reads_as_its_eastern_hour(utc_hour, expected_et):
    moment = dt.datetime(2026, 10, 15, utc_hour, 45, tzinfo=UTC)
    assert eastern_time_of(moment).hour == expected_et


def test_a_naive_stamp_is_assumed_eastern():
    """The platform's convention for naive market data."""
    assert eastern_time_of(dt.datetime(2026, 10, 15, 10, 30)) == dt.time(10, 30)


def test_an_eastern_stamp_is_unchanged():
    moment = dt.datetime(2026, 10, 15, 10, 30, tzinfo=EASTERN)
    assert eastern_time_of(moment) == dt.time(10, 30)


def test_winter_uses_est():
    """19:45 UTC is 14:45 EST in December, not 15:45."""
    assert eastern_time_of(dt.datetime(2026, 12, 15, 19, 45, tzinfo=UTC)).hour == 14


# ── validate_signal ─────────────────────────────────────────────────────

class _Strategy:
    """The base class's validate_signal, with nothing else attached."""

    def __init__(self, **params):
        from falcon_core.backtesting.strategies.base import BaseStrategy
        self.params = StrategyParams(**params)
        self.name = "test"
        self._validate = BaseStrategy.validate_signal.__get__(self)

    def validate(self, signal, positions=0):
        return self._validate(signal, pd.DataFrame(), positions)


def _signal(when, price=770.11, kind=SignalType.LONG, confidence=0.75):
    return Signal(symbol="SPY", signal_type=kind, timestamp=when,
                  price=price, confidence=confidence)


def test_an_afternoon_signal_is_accepted():
    """The regression: 19:45 UTC is 15:45 ET and was compared as 19:45."""
    strategy = _Strategy(trade_start_time="09:35", trade_end_time="15:50")
    assert strategy.validate(_signal(dt.datetime(2026, 10, 15, 19, 45, tzinfo=UTC))) is True


def test_a_signal_after_the_window_is_still_refused():
    strategy = _Strategy(trade_start_time="09:35", trade_end_time="15:30")
    # 19:45 UTC = 15:45 ET, past a 15:30 cutoff.
    assert strategy.validate(_signal(dt.datetime(2026, 10, 15, 19, 45, tzinfo=UTC))) is False


def test_a_premarket_signal_is_refused():
    strategy = _Strategy(trade_start_time="09:35", trade_end_time="15:30")
    # 12:00 UTC = 08:00 ET.
    assert strategy.validate(_signal(dt.datetime(2026, 10, 15, 12, 0, tzinfo=UTC))) is False


def test_an_exit_is_let_through_outside_the_window():
    """Only new entries are time-gated; a position must be closable."""
    strategy = _Strategy(trade_start_time="09:35", trade_end_time="15:30")
    late = dt.datetime(2026, 10, 15, 19, 55, tzinfo=UTC)
    assert strategy.validate(_signal(late, kind=SignalType.EXIT_LONG)) is True


# ── the price band ──────────────────────────────────────────────────────

def test_a_high_priced_symbol_is_no_longer_excluded_by_default():
    """SPY $770, MSFT $527, NVDA: the $500 default dropped all of them."""
    strategy = _Strategy()
    assert strategy.params.max_price is None
    mid_session = dt.datetime(2026, 10, 15, 15, 0, tzinfo=UTC)   # 11:00 ET
    assert strategy.validate(_signal(mid_session, price=770.11)) is True


def test_an_explicit_cap_is_still_honoured():
    strategy = _Strategy(max_price=500.0)
    mid_session = dt.datetime(2026, 10, 15, 15, 0, tzinfo=UTC)
    assert strategy.validate(_signal(mid_session, price=770.11)) is False
    assert strategy.validate(_signal(mid_session, price=499.0)) is True


def test_the_minimum_price_is_unchanged():
    strategy = _Strategy()
    mid_session = dt.datetime(2026, 10, 15, 15, 0, tzinfo=UTC)
    assert strategy.validate(_signal(mid_session, price=0.50)) is False


def test_confidence_and_position_limits_still_apply():
    strategy = _Strategy(max_positions=1)
    mid_session = dt.datetime(2026, 10, 15, 15, 0, tzinfo=UTC)
    assert strategy.validate(_signal(mid_session, confidence=0.40)) is False
    assert strategy.validate(_signal(mid_session), positions=1) is False


# ── the counts that make rejection visible ──────────────────────────────

def test_the_result_carries_generated_and_rejected_counts():
    from falcon_core.backtesting.engine import BacktestResult
    assert hasattr(BacktestResult, "__dataclass_fields__")
    fields = BacktestResult.__dataclass_fields__
    assert "signals_count" in fields
    assert "signals_rejected" in fields
