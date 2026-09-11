"""Tests for falcon_core.metrics (falcon-core#21).

Stdlib only. The Sharpe calculation has been wrong twice and neither error was
caught, because the engine needs pandas to import and nothing exercised the
arithmetic on its own. That is the whole reason this module exists separately.
"""

import datetime as dt
import math

import pytest

from falcon_core.metrics import (
    MIN_RETURN_OBSERVATIONS,
    TRADING_DAYS_PER_YEAR,
    annualized_volatility,
    daily_equity_curve,
    max_drawdown,
    sharpe_ratio,
    simple_returns,
)


def sessions(n, start=dt.date(2026, 1, 5)):
    """n consecutive weekdays."""
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


# --------------------------------------------------------------------------
# the regression this module was written for
# --------------------------------------------------------------------------

def test_annualisation_factor_is_daily_not_bar_frequency():
    """The bug: per-trade returns scaled by sqrt(bars_per_year).

    On 1-minute bars that is sqrt(252*390) ~ 313x, against a correct sqrt(252)
    ~ 15.9x -- a ~20x inflation of every Sharpe.
    """
    daily = [0.001] * 30 + [-0.0005] * 30
    correct, _ = sharpe_ratio(daily)
    wrong, _ = sharpe_ratio(daily, periods_per_year=252 * 390)

    assert math.isclose(wrong / correct, math.sqrt(390), rel_tol=1e-9)
    assert abs(wrong) > abs(correct) * 19


def test_sharpe_is_scale_invariant():
    """Scaling every return leaves Sharpe unchanged -- mean and standard
    deviation scale together. A Sharpe that moves with position size is
    measuring leverage, not edge."""
    base = [0.01, -0.005, 0.02, -0.01] * 10
    s1, _ = sharpe_ratio(base)
    s2, _ = sharpe_ratio([r * 2 for r in base])
    s3, _ = sharpe_ratio([r * 0.1 for r in base])
    assert math.isclose(s2, s1, rel_tol=1e-9)
    assert math.isclose(s3, s1, rel_tol=1e-9)


def test_sharpe_responds_to_consistency_not_magnitude():
    """Same mean, lower dispersion -> higher Sharpe."""
    noisy = [0.03, -0.01] * 30
    steady = [0.011, 0.009] * 30
    assert sharpe_ratio(steady)[0] > sharpe_ratio(noisy)[0]


def test_known_sharpe_value():
    """Hand-computable: mean 0.001, sample sd of the alternating series."""
    returns = [0.002, 0.000] * 30          # mean 0.001
    s, reliable = sharpe_ratio(returns)
    sd = math.sqrt(sum((r - 0.001) ** 2 for r in returns) / (len(returns) - 1))
    assert math.isclose(s, (0.001 / sd) * math.sqrt(252), rel_tol=1e-12)
    assert reliable is True


# --------------------------------------------------------------------------
# reliability gate
# --------------------------------------------------------------------------

def test_few_observations_flagged_unreliable():
    """Four trades once produced a reported Sharpe of 25.98."""
    _, reliable = sharpe_ratio([0.01, -0.002, 0.03, 0.001])
    assert reliable is False


def test_enough_observations_flagged_reliable():
    returns = [0.01, -0.01] * (MIN_RETURN_OBSERVATIONS // 2)
    _, reliable = sharpe_ratio(returns)
    assert reliable is True


def test_zero_variance_is_unreliable_not_infinite():
    s, reliable = sharpe_ratio([0.001] * 50)
    assert s == 0.0
    assert reliable is False


def test_empty_and_single_return():
    assert sharpe_ratio([]) == (0.0, False)
    assert sharpe_ratio([0.01]) == (0.0, False)


def test_risk_free_rate_is_converted_to_per_period():
    """Subtracting an annual rate from a daily return is the same unit error."""
    returns = [0.001] * 30 + [0.002] * 30
    with_rf, _ = sharpe_ratio(returns, risk_free_rate=0.05)
    without, _ = sharpe_ratio(returns, risk_free_rate=0.0)
    # 5% annual is ~0.0002/day: it shifts the mean slightly, not catastrophically.
    assert with_rf < without
    assert abs(with_rf - without) < abs(without) * 0.5


# --------------------------------------------------------------------------
# daily equity curve
# --------------------------------------------------------------------------

def test_curve_is_per_session_not_per_trade():
    """Three fills on one day are one equity point, not three."""
    day = dt.date(2026, 1, 5)
    exits = [
        (dt.datetime.combine(day, dt.time(10, 0)), 100.0),
        (dt.datetime.combine(day, dt.time(11, 0)), -40.0),
        (dt.datetime.combine(day, dt.time(14, 0)), 10.0),
    ]
    curve = daily_equity_curve(10_000.0, exits, [day])
    assert curve == [(day, 10_070.0)]


def test_flat_days_are_included_when_sessions_supplied():
    """A strategy that trades rarely must not look artificially calm."""
    days = sessions(5)
    exits = [(days[0], 100.0), (days[4], -50.0)]
    curve = daily_equity_curve(1000.0, exits, days)
    assert [v for _, v in curve] == [1100.0, 1100.0, 1100.0, 1100.0, 1050.0]


def test_without_sessions_only_trade_days_appear():
    days = sessions(5)
    curve = daily_equity_curve(1000.0, [(days[0], 100.0), (days[4], -50.0)])
    assert len(curve) == 2


def test_fill_outside_supplied_sessions_is_not_lost():
    days = sessions(3)
    stray = days[-1] + dt.timedelta(days=7)
    curve = daily_equity_curve(1000.0, [(stray, 250.0)], days)
    assert curve[-1] == (stray, 1250.0)
    assert len(curve) == 4


def test_curve_accepts_dates_datetimes_and_strings():
    d = dt.date(2026, 1, 5)
    for form in (d, dt.datetime(2026, 1, 5, 12), "2026-01-05"):
        assert daily_equity_curve(100.0, [(form, 10.0)]) == [(d, 110.0)]


def test_empty_exits_with_sessions_is_flat():
    days = sessions(3)
    assert [v for _, v in daily_equity_curve(500.0, [], days)] == [500.0] * 3


# --------------------------------------------------------------------------
# returns, volatility, drawdown
# --------------------------------------------------------------------------

def test_simple_returns():
    assert simple_returns([100.0, 110.0, 99.0]) == pytest.approx([0.1, -0.1])


def test_simple_returns_skips_zero_equity():
    assert simple_returns([0.0, 50.0, 100.0]) == pytest.approx([1.0])


def test_annualized_volatility_uses_daily_scaling():
    returns = [0.01, -0.01] * 30
    sd = math.sqrt(sum(r * r for r in returns) / (len(returns) - 1))
    assert annualized_volatility(returns) == pytest.approx(sd * math.sqrt(252))


def test_max_drawdown():
    assert max_drawdown([100.0, 120.0, 90.0, 110.0]) == pytest.approx(0.25)
    assert max_drawdown([100.0, 110.0, 120.0]) == 0.0


def test_trading_days_constant():
    assert TRADING_DAYS_PER_YEAR == 252


# --------------------------------------------------------------------------
# the CLI import bug: the class name referenced must actually exist
# --------------------------------------------------------------------------

def test_feedback_loop_cli_imports_a_class_that_exists():
    """feedback_loop_cli imported `BacktestScheduler`; the class is
    `FeedbackLoopScheduler`, so every real run raised ImportError.

    My test for that CLI passed because it only exercised the holiday path,
    which returns *before* the import. Checked statically here so it needs
    neither pandas nor a database.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "falcon_core"
    cli = ast.parse((root / "feedback_loop_cli.py").read_text())
    sched = ast.parse((root / "backtesting" / "scheduler.py").read_text())

    wanted = {
        alias.name
        for node in ast.walk(cli)
        if isinstance(node, ast.ImportFrom)
        and node.module == "falcon_core.backtesting.scheduler"
        for alias in node.names
    }
    assert wanted, "expected feedback_loop_cli to import from the scheduler"

    defined = {
        n.name for n in ast.walk(sched)
        if isinstance(n, (ast.ClassDef, ast.FunctionDef))
    }
    missing = wanted - defined
    assert not missing, f"feedback_loop_cli imports names that do not exist: {missing}"
