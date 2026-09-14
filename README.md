# Falcon Core

Shared libraries for the Falcon Trading Platform.

## Installation

```bash
pip install git+https://github.com/TradingAsBuddies/falcon-core.git
```

For PostgreSQL support:
```bash
pip install git+https://github.com/TradingAsBuddies/falcon-core.git#egg=falcon-core[postgresql]
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
Rate-limited Finviz Elite API client with exponential backoff.

```python
from falcon_core import get_finviz_client

client = get_finviz_client()
stocks = client.get_stocks(filters="sh_avgvol_o750,sh_price_u20", limit=30)
```

### Configuration
Environment variables:
- `DB_TYPE` - `sqlite` or `postgresql`
- `DB_PATH` - Path to SQLite database (used when `DB_TYPE=sqlite`)
- `DATABASE_URL` - Full PostgreSQL connection string (e.g. `postgresql://user:pass@host:5432/dbname`); takes precedence over the individual `DB_*` variables below
- `DB_HOST` - PostgreSQL host (used when `DB_TYPE=postgresql` and `DATABASE_URL` is not set)
- `DB_PORT` - PostgreSQL port (default: `5432`)
- `DB_NAME` - PostgreSQL database name
- `DB_USER` - PostgreSQL username
- `DB_PASSWORD` - PostgreSQL password
- `FINVIZ_AUTH_KEY` - Finviz Elite authentication key

## License

MIT
