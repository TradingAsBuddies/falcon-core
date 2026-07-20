# falcon-core — Handoff

**Role:** shared library, pip-installed into the trader/dashboard images. Owns the backtesting engine, the `strategy_roster` lifecycle, the data feed/sync, sentinels, and the AI advisor/proposal-review CLIs.

**Canonical platform handoff:** `FALCON_HANDOFF.md` in the `davdunc/falcon` repo (mirrored to Google Drive ▸ Falcon and Notion "Falcon — Project Handoff"). **Cross-cutting audit:** `falcon-platform/docs/GAP-ANALYSIS-2026-07.md`.

## What it does
- **Backtesting** — see `docs/SPEC-backtesting-and-roster.md`. `falcon-backtest run/optimize/list-strategies/feedback`; `SimpleBacktestEngine`; data from Massive flat files → Polygon → DB → yfinance.
- **Roster lifecycle** — `falcon-strategy-seed` (seed + backtest + auto-rotate), `falcon-advisor` (Claude proposals), `falcon-proposal-review` (apply), `falcon-migrate-strategies`.
- **Data sync** — `falcon-data-sync {daily,minute,backfill,status}` → `daily_bars`/`minute_bars`.
- **Sentinels** — `falcon-sentinel` (8 checks).

## Runtime state (2026-07-20)
- Backtest pipeline verified working end-to-end: `strategy-seed` runs, pulls flat files, backtests, writes `strategy_roster` metrics.
- **7 of 9 seeded strategies produce 0 signals** — they're empty shells (issue #11). Only `atr_breakout` / `opening_range_breakout` have real metrics.
- `daily_bars`/`minute_bars` stale since Feb 2026 (data-sync was failing; fixed — backfill pending, falcon-platform#4).
- Deployed `falcon-core:latest` image is ~4 months old and **missing** the `falcon-proposal-review` script (issue #13).

## Open issues (this repo)
- #8 roster strategies never executed by the live trader (roster ↔ executor decoupled) — **the key "submit" blocker**.
- #9 `paper_trading → live` unimplemented; `paper_*` metrics never computed.
- #10 execution-path schema mismatch; `active_strategies`/`strategy_performance`/`agent_memory` never created.
- #11 seeded strategies reference absent source files.
- #12 README CLIs 5/7, license conflict.
- #13 image rebuild/publish; missing console script.

## Gotchas
- DB access is always via `get_db_manager()` → `DATABASE_URL`. Code that hardcodes SQLite / ignores `DATABASE_URL` is a bug.
- `bt`, `anthropic`, `yfinance`, `boto3`, `psycopg2` are soft deps guarded by try/except — absence silently degrades behavior.
