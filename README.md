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
- `DB_PATH` - Path to SQLite database
- `FINVIZ_AUTH_KEY` - Finviz Elite authentication key

## License

MIT
