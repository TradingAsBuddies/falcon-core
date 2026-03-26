# Falcon Core

Shared libraries for the Falcon Trading Platform.

## Installation

```bash
pip install git+https://github.com/TradingAsBuddies/falcon-core.git
```

With extras:
```bash
# PostgreSQL support
pip install "falcon-core[postgresql] @ git+https://github.com/TradingAsBuddies/falcon-core.git"

# Backtesting (bt engine, Massive flat files, yfinance)
pip install "falcon-core[backtesting] @ git+https://github.com/TradingAsBuddies/falcon-core.git"

# Everything
pip install "falcon-core[full] @ git+https://github.com/TradingAsBuddies/falcon-core.git"
```

## Components

### DatabaseManager
Database abstraction supporting SQLite and PostgreSQL.

```python
from falcon_core import get_db_manager

db = get_db_manager()
db.init_schema()
```

### FinvizClient
Rate-limited Finviz Elite API client.

```python
from falcon_core import get_finviz_client

client = get_finviz_client()
stocks = client.get_stocks(filters="sh_avgvol_o750,sh_price_u20", limit=30)
```

### Resilient HTTP Client
All API requests use `http_get` / `http_post` with incremental backoff.

```python
from falcon_core.http import http_get

# External API (500ms initial delay, 30s max, 3 retries)
resp = http_get("https://api.polygon.io/v2/...", params={...})

# Internal container-to-container (50ms initial, 2s max)
resp = http_get("http://falcon-dashboard:5000/api/market", profile="internal")
```

Retries on 429 (rate limit), 500, 502, 503, 504. Respects `Retry-After` headers.
Every retry is logged — no silent failures.

### Sentinel Health Checks
Per-subsystem health checks for the platform.

```bash
# CLI
falcon-sentinel
falcon-sentinel --name data-feed
falcon-sentinel --json

# Python
from falcon_core.sentinel import SentinelRunner
runner = SentinelRunner()
results = runner.run_all()
```

Built-in sentinels:
- **database** — connection and schema validation
- **data-feed** — Massive flat files daily bars
- **data-feed-minute** — Massive flat files minute bars
- **polygon-minute** — Polygon API freshness and delay
- **strategy-roster** — AST validation of all rostered strategies
- **backtest-engine** — end-to-end strategy load + backtest
- **timezone** — ET handling consistency
- **market-page** — published prices match live Polygon data

### Backtesting
Event-driven backtesting with data from Massive flat files, Polygon, or yfinance.

```python
from falcon_core import get_backtest_engine, get_data_feed

feed = get_data_feed()
data = feed.get_historical_data("SPY", "2025-01-01", "2025-12-31", interval="5m")

engine = get_backtest_engine()
result = engine.run(strategy, data, "SPY")
```

Strategies are loaded from the database (`strategy_roster` table) — never from files in this repo.

## CLI Commands

| Command | Description |
|---------|-------------|
| `falcon-backtest` | Run backtests |
| `falcon-sentinel` | Health checks |
| `falcon-advisor` | AI strategy advisor |
| `falcon-data-sync` | Sync market data |
| `falcon-strategy-seed` | Seed strategy roster |

## Configuration

Environment variables:
- `DATABASE_URL` — PostgreSQL connection string (preferred)
- `POLYGON_API_KEY` — Polygon.io API key
- `MASSIVE_ACCESS_KEY` / `MASSIVE_SECRET_KEY` — Massive S3 keys for flat files
- `FINVIZ_AUTH_KEY` — Finviz Elite authentication key
- `CLAUDE_API_KEY` — Claude API key (for AI advisor)

## License

MIT
