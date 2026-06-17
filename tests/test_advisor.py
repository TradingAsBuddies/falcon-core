#!/usr/bin/env python3
"""
Verification tests for the AI Strategy Advisor (issue #6).

Covers:
  - corrected MODEL_COSTS flow through CostTracker.estimate_cost
  - data-sufficiency gate (<30 trades -> None, zero spend, no api_usage row)
  - differ-in-kind: _build_prompt embeds real n_trades/expectancy and the
    regime-classification instruction (no live Claude call required)

Run inside the falcon-core container so falcon_core imports resolve:
  podman run --rm --network systemd-falcon \
      --env-file ~/.config/falcon/falcon.env \
      -v $PWD:/work -w /work localhost/falcon-core:latest \
      python -m pytest tests/test_advisor.py -v
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from falcon_core.backtesting.advisor import (  # noqa: E402
    CostTracker,
    StrategyAdvisor,
    MODEL_COSTS,
    DEFAULT_MODEL,
)


class FakeDB:
    """Minimal db_manager stand-in that records writes and serves budget rows."""

    def __init__(self):
        self.calls = []
        self.api_usage_rows = []
        self.budget = {
            'strategy_name': None,
            'monthly_budget_usd': 1.00,
            'total_spent_usd': 0.0,
            'current_month_spent_usd': 0.0,
            'months_active': 0,
            'max_months': 4,
            'consecutive_no_improvement': 0,
            'last_improvement_at': None,
            'status': 'active',
        }

    def execute(self, query, params=None, fetch=None):
        self.calls.append((query, params, fetch))
        q = " ".join(query.split())
        if q.startswith("INSERT INTO api_usage"):
            self.api_usage_rows.append(params)
            return 1
        if "FROM strategy_advisor_budget WHERE strategy_name" in q and fetch == 'one':
            b = dict(self.budget)
            b['strategy_name'] = params[0] if params else None
            return b
        if q.startswith("INSERT INTO strategy_proposals"):
            return 4242
        if q.startswith("SELECT") and fetch == 'all':
            return []
        if q.startswith("SELECT") and fetch == 'one':
            return None
        return 1


# ---------------------------------------------------------------------------
# MODEL_COSTS / estimate_cost
# ---------------------------------------------------------------------------

def test_model_costs_are_current_ids():
    assert set(MODEL_COSTS) == {'claude-opus-4-8', 'claude-haiku-4-5'}
    assert MODEL_COSTS['claude-opus-4-8'] == {'input': 5.00, 'output': 25.00}
    assert MODEL_COSTS['claude-haiku-4-5'] == {'input': 1.00, 'output': 5.00}
    assert DEFAULT_MODEL == 'claude-haiku-4-5'


def test_estimate_cost_opus():
    ct = CostTracker(FakeDB())
    cost = ct.estimate_cost('claude-opus-4-8', 4000, 2000)
    # 4000/1e6*5 + 2000/1e6*25 = 0.02 + 0.05 = 0.07
    assert cost == pytest.approx(0.07, abs=1e-6)


def test_estimate_cost_haiku():
    ct = CostTracker(FakeDB())
    cost = ct.estimate_cost('claude-haiku-4-5', 4000, 2000)
    # 4000/1e6*1 + 2000/1e6*5 = 0.004 + 0.010 = 0.014
    assert cost == pytest.approx(0.014, abs=1e-6)


def test_can_spend_uses_corrected_costs():
    ct = CostTracker(FakeDB())
    ok, _ = ct.can_spend('x', ct.estimate_cost('claude-haiku-4-5', 4000, 2000))
    assert ok is True  # 0.014 well under $1 monthly budget


# ---------------------------------------------------------------------------
# Data-sufficiency gate
# ---------------------------------------------------------------------------

def test_gate_blocks_under_30_trades_no_spend():
    db = FakeDB()
    adv = StrategyAdvisor(db, model='claude-haiku-4-5')
    # bella_fade-like: 0 signals, 0 trades -> must hit the gate
    result = adv.analyze_and_propose(
        strategy_name='bella_fade',
        strategy_code='x = 1\n',
        backtest_results={
            'n_trades': 0, 'n_signals': 0, 'win_rate': None,
            'expectancy': None, 'per_symbol': [],
        },
        historical_proposals=[],
    )
    assert result is None
    # No api_usage row was written, no client constructed
    assert db.api_usage_rows == []
    assert adv._client is None


def test_gate_sums_per_symbol_when_n_trades_missing():
    db = FakeDB()
    adv = StrategyAdvisor(db, model='claude-haiku-4-5')
    result = adv.analyze_and_propose(
        strategy_name='offside_scalp',
        strategy_code='x = 1\n',
        backtest_results={
            'n_signals': 40,
            'per_symbol': [
                {'symbol': 'AAPL', 'n_trades': 5},
                {'symbol': 'TSLA', 'n_trades': 6},
            ],  # sums to 11 < 30
        },
        historical_proposals=[],
    )
    assert result is None
    assert db.api_usage_rows == []


# ---------------------------------------------------------------------------
# Differ-in-kind: prompt content (no live Claude call)
# ---------------------------------------------------------------------------

OFFSIDE = {
    'n_trades': 19, 'n_signals': 31, 'win_rate': 0.32,
    'expectancy': -0.73, 'mean_R': -0.73, 'total_return': -0.05,
    'interval': '5m', 'date_span': {'start': '2026-03-01', 'end': '2026-06-01'},
    'per_symbol': [
        {'symbol': 'NVDA', 'n_trades': 11, 'win_rate': 0.27, 'total_return': -0.04},
        {'symbol': 'AMD', 'n_trades': 8, 'win_rate': 0.38, 'total_return': -0.01},
    ],
    'data_source': 'flatfiles',
}

BELLA = {
    'n_trades': 0, 'n_signals': 0, 'win_rate': None,
    'expectancy': None, 'mean_R': None, 'total_return': None,
    'interval': '5m', 'date_span': {'start': '2026-03-01', 'end': '2026-06-01'},
    'per_symbol': [], 'data_source': 'flatfiles',
}


def test_prompt_cites_real_metrics_and_regime_instruction():
    adv = StrategyAdvisor(FakeDB(), model='claude-haiku-4-5')
    prompt = adv._build_prompt('offside_scalp', 'x = 1\n', OFFSIDE, [])

    # cites the actual metrics by number
    assert '19' in prompt          # n_trades
    assert '-0.73' in prompt       # expectancy
    assert 'NVDA' in prompt        # weak symbol from per_symbol

    # carver constraints + regime classification instruction present
    for needle in [
        'NO-SIGNAL', 'NO-CONVERSION', 'LOSING-EDGE',
        '20bps', 'volatility-targeted', 'ATR',
        'OVERFIT', 'expectancy', 'min-trades floor'.lower(),
    ]:
        assert needle.lower() in prompt.lower(), f"missing: {needle}"


def test_prompt_differs_in_kind_between_strategies():
    adv = StrategyAdvisor(FakeDB(), model='claude-haiku-4-5')
    p_off = adv._build_prompt('offside_scalp', 'x = 1\n', OFFSIDE, [])
    p_bella = adv._build_prompt('bella_fade', 'x = 1\n', BELLA, [])

    # The two prompts carry materially different evidence -> the model is
    # steered to different KINDS of fix (losing-edge exit/filter vs no-signal
    # entry). Identical inputs would defeat differ-in-kind.
    assert p_off != p_bella
    # offside's real expectancy -0.73 appears in its data block; bella's data
    # block has no such number. (Note: the static instruction text uses -0.73R
    # as an EXAMPLE in both, so we assert on the dumped metrics, not raw substr.)
    assert '"expectancy": -0.73' in p_off
    assert '"expectancy": -0.73' not in p_bella
    assert '"expectancy": null' in p_bella
    assert '"n_trades": 19' in p_off
    # bella_fade is a 0/0 strategy -> would also fail the <30 gate upstream
    assert (BELLA['n_trades'] or 0) < 30


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
