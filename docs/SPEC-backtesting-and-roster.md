# Spec — Backtesting Engine & Strategy Roster (falcon-core)

**Status:** current-as-of 2026-07-20. Authoritative description of the shipped backtesting system in `falcon_core.backtesting`. Supersedes ADR-0004 as the *as-built* record (ADR-0004 remains a *proposed* future direction — reconcile via falcon-platform#6).

## 1. Packages & entry points

```
falcon_core/backtesting/
  engine.py          BacktestEngine (ABC), SimpleBacktestEngine, BTBacktestEngine, create_engine()
  data_feed.py       DataFeed.get_historical_data() — source selection
  flatfiles_client.py / polygon_client.py   source clients (lazy-loaded)
  strategies/base.py BaseStrategy, StrategyParams, Signal, SignalType
  strategy_loader.py load from FALCON_STRATEGY_DIR + strategy_roster (AST-validated exec)
  results_api.py     BacktestResultsStore, Flask/FastAPI /api/backtests/*
  optimizer.py       ParameterOptimizer, FeedbackLoop
  scheduler.py       FeedbackLoopScheduler (daily 11:30 ET)
  advisor.py         StrategyAdvisor (Claude proposals), CostTracker
  proposal_reviewer.py ProposalReviewer (apply proposals)
```

Console scripts (`setup.py`, 7 total): `falcon-backtest`, `falcon-sentinel`, `falcon-advisor`, `falcon-data-sync`, `falcon-strategy-seed`, `falcon-migrate-strategies`, `falcon-proposal-review`.

## 2. Data feed contract

`DataFeed.get_historical_data(symbol, start, end, interval)` selects the first available source:

- **Intraday** (`1m,5m,15m,30m,1h`): Massive flat files (S3) → Polygon API → DB (`minute_bars`) → yfinance.
- **Daily+** (`1d,1wk,1mo`): flat files → DB (`daily_bars`) → yfinance.
- CSV fallback: `/var/lib/falcon/market_data/{symbol}_{interval}.csv`.
- Credentials: `POLYGON_API_KEY`, `MASSIVE_ACCESS_KEY`, `MASSIVE_SECRET_KEY`.
- Intraday bars resampled from 1-minute; market-hours filter 09:30–16:00 ET.

## 3. Engine contract

`BacktestEngine.run(strategy, data, symbol) -> BacktestResult`.

- `SimpleBacktestEngine` — complete, default. `strategy.run()` → `list[Signal]` → `_simulate_trades()` → `_calculate_metrics()`. Multiplicative slippage + additive commission; long and short.
- `BTBacktestEngine` — optional `bt`/`ffn` backend; **falls back to Simple if `bt` absent**. (Note: its `_extract_metrics` approximates trade count as `len(signals)//2`.)
- `create_engine(engine_type="auto")` — tries `bt`, falls back.
- `BacktestResult` — returns, Sharpe/Sortino, drawdown, win_rate, profit_factor, expectancy, equity_curve, trades, signals; `.to_dict()` / `.summary()`.

## 4. Persistence

| Sink | Written by |
|------|-----------|
| `backtest_runs`, `feedback_results`, `parameter_history` | `results_api.BacktestResultsStore` |
| `strategy_roster.backtest_*` aggregates | `strategy_seeder` |
| `~/.local/share/falcon/feedback_results/feedback_<date>.json` | `scheduler` |
| `~/.local/share/falcon/backtest_results.db` (SQLite fallback) | `results_api` when no `DATABASE_URL` |

## 5. strategy_roster schema (authoritative table)

`id, strategy_name (unique), status ('backtest'|'paper_trading'|'live'|'review'), promoted_at, demoted_at, review_notes, symbols (JSON), interval, params (JSON), last_backtest_at, backtest_sharpe, backtest_win_rate, backtest_profit_factor, backtest_total_return, paper_sharpe, paper_win_rate, paper_profit_factor, strategy_code, strategy_source, created_at, updated_at`.

Created in `db_manager._create_strategy_rotation_tables`; `strategy_code`/`strategy_source` added by `_migrate_strategy_roster_v2`.

## 6. Sentinels (`falcon-sentinel`)

8 checks: `database`, `data-feed` (flat files daily), `data-feed-minute`, `polygon-minute` (freshness), `strategy-roster` (AST-validates every rostered strategy), `backtest-engine` (end-to-end load+run of a `status='backtest'` strategy), `timezone`, `market-page`. Exit non-zero on any FAIL; `--json`, `--name`, `--list`.

## 7. Known gaps (tracked)

- **falcon-core#8** — roster strategies are never executed by the live `TradeExecutor` (hardcoded engines).
- **falcon-core#9** — `paper_trading → live` unimplemented; `paper_*` metrics never computed.
- **falcon-core#10** — execution-path `positions`/`orders` inserts + `active_strategies`/`strategy_performance`/`agent_memory` tables aren't created by `init_schema()`.
- **falcon-core#11** — default seed references strategy files that don't exist; most seeded strategies backtest to 0 signals.
- **falcon-core#12** — README documents 5 of 7 CLIs; license statement conflict.
- **falcon-core#13** — deployed `falcon-core` image is stale and missing `falcon-proposal-review`.
