"""Price columns must hold the thousandth the system computes in.

positions.entry_price and orders.price were DECIMAL(15,2), so PostgreSQL
rounded every write to cents: a $1.345 stop stored as $1.35, which is 0.37%
away on that name. The operator's convention is thousandths below $52, so
storage has to carry one.

Widening a numeric scale is the only retype this class performs, because it
cannot lose data — every value representable at (15,2) is representable at
(15,3). Narrowing is not done.
"""

import pytest

from falcon_core.db_manager import DatabaseManager


class _FakeDB(DatabaseManager):
    """DatabaseManager with the schema query stubbed and writes recorded."""

    def __init__(self, columns, db_type="postgresql"):
        self.db_type = db_type
        self._columns = columns
        self.statements = []

    def execute(self, query, params=None, fetch=None):
        if "information_schema" in query:
            return self._columns
        self.statements.append(query)
        return None


def _column(table, name, scale):
    return {"table_name": table, "column_name": name, "numeric_scale": scale}


AT_CENTS = [
    _column("positions", "entry_price", 2),
    _column("positions", "current_price", 2),
    _column("positions", "stop_loss", 2),
    _column("positions", "profit_target", 2),
    _column("orders", "price", 2),
]


# ── which columns need widening ─────────────────────────────────────────

def test_every_price_column_at_cents_is_identified():
    """The live schema on 2026-10-05."""
    pending = _FakeDB(AT_CENTS)._price_columns_to_widen()
    assert {(t, c) for t, c, _ in pending} == {
        ("positions", "entry_price"), ("positions", "current_price"),
        ("positions", "stop_loss"), ("positions", "profit_target"),
        ("orders", "price"),
    }


def test_already_widened_columns_are_left_alone():
    """Idempotent: this runs at every container start."""
    widened = [_column(t, c, 3) for t, c, _ in
               [(t, c, 3) for t, cols in DatabaseManager._PRICE_COLUMNS.items() for c in cols]]
    assert _FakeDB(widened)._price_columns_to_widen() == []


def test_a_wider_column_is_not_narrowed():
    """A column already at four decimals must not be reduced to three."""
    assert _FakeDB([_column("orders", "price", 4)])._price_columns_to_widen() == []


def test_money_totals_are_not_touched():
    """cash and pnl are paid in cents; a third decimal there would be fiction."""
    rows = [_column("account", "cash", 2), _column("orders", "pnl", 2),
            _column("performance", "total_value", 2)]
    assert _FakeDB(rows)._price_columns_to_widen() == []


def test_unrelated_numeric_columns_are_not_touched():
    assert _FakeDB([_column("positions", "quantity", 4)])._price_columns_to_widen() == []


def test_sqlite_has_no_scale_to_widen():
    """REAL carries no declared scale."""
    assert _FakeDB(AT_CENTS, db_type="sqlite")._price_columns_to_widen() == []


def test_a_failed_inspection_widens_nothing():
    class _Broken(_FakeDB):
        def execute(self, query, params=None, fetch=None):
            raise RuntimeError("no information_schema")

    assert _Broken(AT_CENTS)._price_columns_to_widen() == []


# ── the statements it issues ────────────────────────────────────────────

def test_it_issues_one_widening_alter_per_column():
    db = _FakeDB(AT_CENTS)
    db._migrate_price_scale()
    assert len(db.statements) == 5
    for statement in db.statements:
        assert "ALTER COLUMN" in statement
        assert "TYPE DECIMAL(15,3)" in statement
        assert "DROP" not in statement.upper()


def test_a_failing_alter_does_not_stop_the_others():
    """One locked table must not leave the rest at cents."""
    class _PartlyBroken(_FakeDB):
        def execute(self, query, params=None, fetch=None):
            if "information_schema" in query:
                return self._columns
            self.statements.append(query)
            if "orders" in query:
                raise RuntimeError("lock timeout")
            return None

    db = _PartlyBroken(AT_CENTS)
    db._migrate_price_scale()          # must not raise
    assert len(db.statements) == 5


def test_nothing_to_do_issues_nothing():
    db = _FakeDB([])
    db._migrate_price_scale()
    assert db.statements == []


# ── what new installs get ───────────────────────────────────────────────

def test_the_target_scale_is_thousandths():
    assert DatabaseManager.PRICE_SCALE == 3


def test_a_fresh_sqlite_database_takes_prices_at_the_thousandth(tmp_path):
    """End to end on a real database: a $1.345 stop round-trips."""
    db = DatabaseManager({"db_type": "sqlite", "db_path": str(tmp_path / "t.db")})
    db.init_schema()
    db.execute(
        "INSERT INTO positions (symbol, quantity, entry_price, entry_date, "
        "last_updated, stop_loss) VALUES (%s, %s, %s, %s, %s, %s)",
        ("CDXS", 1588, 1.345, "2026-10-05T10:00:00", "2026-10-05T10:00:00", 1.279),
    )
    row = db.execute("SELECT entry_price, stop_loss FROM positions WHERE symbol = %s",
                     ("CDXS",), fetch="one")
    assert float(row["entry_price"]) == 1.345
    assert float(row["stop_loss"]) == 1.279
