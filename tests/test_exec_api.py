"""Unit tests for the backtest execution API core.

These test the PURE `run_backtest_request` / validation / sweep-aggregation logic with fully mocked
collaborators, so they run on any machine — no falcon-core image, flat files, Flask, or pandas needed.
"""
import sys
import types
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Optional

import pytest

from falcon_core.backtesting.exec_api import (
    BacktestExecDeps,
    BacktestRequestError,
    run_backtest_request,
    _as_slippage_list,
    _validate,
    _sweep_aggregate,
)


# --- a stand-in BacktestRunRecord so tests don't depend on results_api's real one -------------
@dataclass
class FakeRecord:
    strategy_name: str = ""
    symbol: str = ""
    trading_date: date = None
    interval: str = "1m"
    total_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    signals_count: int = 0
    parameters: Dict = None


class FakeResult:
    def __init__(self, **kw):
        self._d = {
            "total_trades": 4, "win_rate": 0.5, "total_return": 0.012,
            "sharpe_ratio": 1.1, "profit_factor": 1.3, "expectancy": 2.5,
            "max_drawdown": 0.03,
        }
        self._d.update(kw)

    def to_dict(self):
        return dict(self._d)


def _deps(stored, *, bars=("bar",) * 10, run_result=None, raise_symbols=()):
    """Build injected deps that record what got stored."""
    def load_bars(symbol, start, end, interval):
        if symbol in raise_symbols:
            raise RuntimeError("flat files unavailable")
        return list(bars)

    def resolve_strategy(name):
        return object()

    class _Eng:
        def __init__(self, commission, slippage):
            self.commission, self.slippage = commission, slippage

        def run(self, strategy, data, symbol):
            return run_result or FakeResult()

    def store_run(record):
        stored.append(record)
        return len(stored)  # fake incrementing id

    def get_run(run_id):
        return {"id": run_id, "strategy_name": "x"} if 1 <= run_id <= len(stored) else None

    return BacktestExecDeps(
        load_bars=load_bars,
        resolve_strategy=resolve_strategy,
        make_engine=lambda c, s: _Eng(c, s),
        store_run=store_run,
        get_run=get_run,
    )


BASE = {"strategy": "vwap_reclaim_long", "symbols": ["IQST"],
        "start": "2026-07-01", "end": "2026-07-16"}


# ------------------------------------------------------------------ validation
def test_validate_requires_strategy():
    with pytest.raises(BacktestRequestError):
        _validate({**BASE, "strategy": None})


def test_validate_requires_symbols():
    with pytest.raises(BacktestRequestError):
        _validate({**BASE, "symbols": []})


def test_validate_rejects_bad_interval():
    with pytest.raises(BacktestRequestError):
        _validate({**BASE, "interval": "1d"})


def test_validate_rejects_end_before_start():
    with pytest.raises(BacktestRequestError):
        _validate({**BASE, "start": "2026-07-16", "end": "2026-07-01"})


def test_validate_uppercases_symbols_and_defaults():
    out = _validate({**BASE, "symbols": ["iqst", "atai"]})
    assert out["symbols"] == ["IQST", "ATAI"]
    assert out["interval"] == "1m"
    assert out["slippage_bps"] == [3.0]


# ------------------------------------------------------------------ slippage list
def test_slippage_scalar_becomes_list():
    assert _as_slippage_list(10) == [10.0]


def test_slippage_list_passthrough():
    assert _as_slippage_list([3, 10, 25]) == [3.0, 10.0, 25.0]


def test_slippage_negative_rejected():
    with pytest.raises(BacktestRequestError):
        _as_slippage_list([-1])


# ------------------------------------------------------------------ execution + sweep
def test_run_sweeps_three_slippage_levels():
    stored = []
    deps = _deps(stored)
    out = run_backtest_request({**BASE, "slippage_bps": [3, 10, 25]}, deps, record_cls=FakeRecord)
    # one symbol x three slippage levels = three runs
    assert len(out["runs"]) == 3
    assert len(stored) == 3
    assert {r["slippage_bps"] for r in out["runs"]} == {3.0, 10.0, 25.0}


def test_schema_loss_trap_carries_R_and_slippage_in_parameters():
    stored = []
    deps = _deps(stored)
    run_backtest_request({**BASE, "slippage_bps": [10]}, deps, record_cls=FakeRecord)
    rec = stored[0]
    # expectancy/track/slippage must live in parameters JSON, NOT be mapped onto total_return
    assert rec.parameters["track"] == "intraday"
    assert rec.parameters["slippage_bps"] == 10.0
    assert rec.parameters["expectancy"] == 2.5
    assert rec.total_return == 0.012  # real return preserved, not overwritten by R


def test_slippage_converted_bps_to_fraction():
    captured = {}
    stored = []
    deps = _deps(stored)
    orig = deps.make_engine
    deps.make_engine = lambda c, s: captured.setdefault("slippage", s) or orig(c, s)
    run_backtest_request({**BASE, "slippage_bps": [25]}, deps, record_cls=FakeRecord)
    assert captured["slippage"] == pytest.approx(0.0025)  # 25 bps -> 0.0025


def test_bad_symbol_recorded_as_error_not_crash():
    stored = []
    deps = _deps(stored, raise_symbols=("BADX",))
    out = run_backtest_request({**BASE, "symbols": ["IQST", "BADX"], "slippage_bps": [3]},
                               deps, record_cls=FakeRecord)
    assert len(out["runs"]) == 1  # only IQST ran
    assert any(e["symbol"] == "BADX" for e in out["errors"])


def test_sweep_aggregate_groups_by_slippage():
    runs = [
        {"slippage_bps": 3.0, "summary": {"total_return": 0.02, "win_rate": 0.6, "total_trades": 4}},
        {"slippage_bps": 3.0, "summary": {"total_return": 0.00, "win_rate": 0.4, "total_trades": 2}},
        {"slippage_bps": 25.0, "summary": {"total_return": -0.01, "win_rate": 0.3, "total_trades": 3}},
    ]
    agg = _sweep_aggregate(runs)
    assert [a["slippage_bps"] for a in agg] == [3.0, 25.0]  # sorted
    assert agg[0]["symbols"] == 2
    assert agg[0]["avg_total_return"] == pytest.approx(0.01)
    assert agg[0]["total_trades"] == 6
