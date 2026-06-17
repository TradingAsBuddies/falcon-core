# falcon-core — Component Spec

_Spec v1 · 2026-06-14 · living doc._

## Purpose
Shared library + base container image for the Falcon platform. Everything else depends on it.

## Responsibilities
- **Database access** — `db_manager.py`: `DatabaseManager` (PostgreSQL/SQLite, `%s` params,
  dict rows, `init_schema`). Single source of DB truth for all services.
- **Backtesting primitives** — `backtesting/`:
  - `data_feed.py` `DataFeed` — unified historical data (Massive/Polygon flat files → Polygon
    REST → DB → yfinance → CSV), the **polygon.io backtest data** source.
  - `strategies/base.py` — the `BaseStrategy` / `Signal` / `StrategyParams` contract.
  - `engine.py` `SimpleBacktestEngine`; `flatfiles_client.py`, `polygon_client.py`.
  - `results_api.py` — `BacktestResultsStore` + `create_api_routes` (registers `/api/backtests/*`).
- **Data sync** — `data_sync.py` / `data_sync_cli.py`: ingest daily/minute bars to PostgreSQL.
- **Strategy seeding/migration** — `strategy_seeder.py`, `migrate_strategies.py`.

## Interfaces
- Console scripts: `falcon-backtest`, `falcon-data-sync`, `falcon-strategy-seed`,
  `falcon-migrate-strategies`.
- Python package `falcon_core` imported by falcon-trader / -strategies / -screener.

## Dependencies
PostgreSQL, Massive flat files (`MASSIVE_ACCESS_KEY/SECRET`), Polygon (`POLYGON_API_KEY`),
`bt`, `ffn`, `pandas`, `yfinance`, `lxml`, `anthropic`.

## Deployment
Base image `localhost/falcon-core:latest` — `FROM registry.fedoraproject.org/fedora:42`,
deps via `dnf` (incl. `python3-lxml`) + `pip install .[full]`. Other services build FROM or
reuse this image.

## AI Strategy Advisor (`backtesting/advisor.py`, `advisor_cli.py`)
Reviews REAL backtest results and proposes ONE actionable, cost-aware, non-overfit
improvement per strategy. Cost-bounded by `CostTracker` (per-strategy monthly budget,
pre-flight `can_spend` gate, auto-retire after N no-improvement cycles — keep intact).
Carver "Systematic Trading" lens: vol-targeted sizing, cost speed-limit, out-of-sample
validation, no curve-fitting to a tiny window.

### Data source (the core fix)
- The advisor MUST read per-trade metrics from the **populated `backtest_runs` table**
  via `BacktestResultsStore` (`get_strategy_summary`, `get_recent_backtests`), NOT the
  3 averaged scalars in `strategy_roster.backtest_*` (those are NULL for any strategy
  not seeded by `strategy_seeder.py` → advisor sees a stub).
- `backtest_runs` columns available: `total_trades`, `signals_count`, `win_rate`,
  `sharpe_ratio`, `total_return`, `max_drawdown`, `interval`, `trading_date`, `symbol`.
  Expectancy / mean-R derived from `total_return / total_trades` (R-units, not $).
- `strategy_roster` is queried ONLY to fetch `strategy_code`. Skip strategies with zero
  `backtest_runs` rows rather than analyzing a stub.
- Backtests/verification run on `DataFeed(source="flatfiles")` (Massive/Polygon flat
  files) + `SimpleBacktestEngine` — NEVER the Polygon REST API.

### Model + pricing
- `MODEL_COSTS` / `DEFAULT_MODEL` use current bare IDs: `claude-opus-4-8`
  ($5 in / $25 out per 1M) and `claude-haiku-4-5` ($1 / $5). Default = `claude-haiku-4-5`.
  Date-suffixed IDs and the old $15/$75 Opus price are wrong (3x mis-budget). Match the
  rest of the repo (`config.py`, `codify.py` use bare `claude-opus-4-8`).
- API key: `CLAUDE_API_KEY` OR `ANTHROPIC_API_KEY` (match `codify.py`).

### Behavior contract
- **Data-sufficiency gate** (highest leverage): reject (return None, log reason) BEFORE
  any Claude call when the strategy's real backtest has `total_trades < 30` across the
  suite. Tuning to <30 trades is curve-fitting noise.
- The prompt instructs the model to: (a) cite the supplied per-trade metrics; (b) classify
  the failure as **no-signal** vs **no-conversion** vs **losing-edge** (from
  signals_count vs total_trades vs win_rate); (c) respect the ~20bps round-trip cost
  (0.1% commission + 0.1% slippage); (d) emit ONE concrete old→new parameter or single
  logic edit + a numeric pass bar in expectancy/R AND a min-trades floor; (e) use
  vol-targeted/ATR sizing, never fixed-share or martingale; (f) not overfit to the
  in-sample 1-3 day window — require multi-day / out-of-sample validation.
- Proposals MUST be VERIFIED: `cmd_run` calls `backtest_proposal()` after
  `analyze_and_propose()`, then `record_improvement(improved=<proposed beats current>)`
  to close the budget loop. No proposal is "accepted" on the LLM's `expected_improvement`
  text alone.

### Acceptance bar
- `estimate_cost('claude-opus-4-8', 4000, 2000)` ≈ $0.07;
  `estimate_cost('claude-haiku-4-5', 4000, 2000)` ≈ $0.014.
- Feeding a losing strategy (offside_scalp) and a no-signal strategy (bella_fade) yields
  proposals that differ in KIND (entry/threshold vs exit/filter/edge) and each cites the
  real `total_trades` + expectancy. A generic "adjust the lookback" for both = FAIL.
- A 0-to-6-trade strategy spends ZERO tokens (gated before the Claude call).

## Status / notes
- lxml baked in (foreman ANSWER-016) so `yfinance.get_earnings_dates` works.
- DB config via `DATABASE_URL`. `init_schema()` must run before results/app_config use.
- Related: `[[reference_falcon_server]]`, falcon-platform HANDOFF.md.
