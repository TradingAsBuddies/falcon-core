"""Tests for the additive trading-column migration (falcon-trader#28).

Runs against real SQLite. The bug these cover is that `positions.current_price`
and `account.initial_balance` were written and read by falcon-trader but
declared nowhere in falcon-core's schema -- they existed only in the deployed
database, added out of band, so a fresh deploy did not match production.
"""

import datetime as dt

import pytest

from falcon_core.db_manager import DatabaseManager


def mgr(tmp_path, name="t.db"):
    return DatabaseManager({'db_type': 'sqlite', 'db_path': str(tmp_path / name)})


def columns(db, table):
    return db._existing_columns(table)


# --------------------------------------------------------------------------
# the pre-existing-table case, which is what production actually is
# --------------------------------------------------------------------------

def test_migration_adds_columns_to_an_existing_old_table(tmp_path):
    """CREATE TABLE IF NOT EXISTS cannot alter a table that already exists."""
    db = mgr(tmp_path)
    db.execute(
        "CREATE TABLE positions (symbol TEXT PRIMARY KEY, quantity REAL, "
        "entry_price REAL, entry_date TEXT, last_updated TEXT)"
    )
    db.execute(
        "CREATE TABLE account (id INTEGER PRIMARY KEY, cash REAL, "
        "last_updated TEXT)"
    )
    assert "current_price" not in columns(db, "positions")
    assert "initial_balance" not in columns(db, "account")

    db._migrate_trading_columns()

    assert {"current_price", "stop_loss", "profit_target", "strategy"} <= columns(db, "positions")
    assert "initial_balance" in columns(db, "account")


def test_migration_preserves_existing_rows(tmp_path):
    db = mgr(tmp_path)
    db.execute(
        "CREATE TABLE positions (symbol TEXT PRIMARY KEY, quantity REAL, "
        "entry_price REAL, entry_date TEXT, last_updated TEXT)"
    )
    now = dt.datetime.now().isoformat()
    db.execute(
        "INSERT INTO positions VALUES (%s, %s, %s, %s, %s)",
        ("CDXS", 1588, 1.45, now, now),
    )
    db._migrate_trading_columns()

    row = db.execute(
        "SELECT symbol, quantity, entry_price, current_price FROM positions "
        "WHERE symbol = 'CDXS'", fetch='one',
    )
    assert row['quantity'] == 1588
    assert row['entry_price'] == pytest.approx(1.45)
    assert row['current_price'] is None


def test_migration_is_idempotent(tmp_path):
    db = mgr(tmp_path)
    db.execute(
        "CREATE TABLE account (id INTEGER PRIMARY KEY, cash REAL, "
        "last_updated TEXT)"
    )
    db._migrate_trading_columns()
    before = columns(db, "account")
    db._migrate_trading_columns()
    db._migrate_trading_columns()
    assert columns(db, "account") == before


def test_missing_table_is_skipped_not_fatal(tmp_path):
    db = mgr(tmp_path)
    db._migrate_trading_columns()          # no tables at all
    assert columns(db, "positions") == set()


# --------------------------------------------------------------------------
# the queries that were failing
# --------------------------------------------------------------------------

def test_initial_balance_select_works_after_migration(tmp_path):
    """`SELECT initial_balance FROM account` 500'd /api/account."""
    db = mgr(tmp_path)
    db.execute(
        "CREATE TABLE account (id INTEGER PRIMARY KEY, cash REAL, "
        "last_updated TEXT)"
    )
    db.execute(
        "INSERT INTO account VALUES (1, 7358.48, %s)",
        (dt.datetime.now().isoformat(),),
    )
    db._migrate_trading_columns()
    db.execute("UPDATE account SET initial_balance = %s WHERE id = 1", (10000.0,))

    row = db.execute(
        "SELECT initial_balance FROM account ORDER BY id LIMIT 1", fetch='one',
    )
    assert row['initial_balance'] == pytest.approx(10000.0)


def test_positions_mark_query_works_after_migration(tmp_path):
    """The /api/positions query reads stop_loss, current_price, last_updated."""
    db = mgr(tmp_path)
    db.execute(
        "CREATE TABLE positions (symbol TEXT PRIMARY KEY, quantity REAL, "
        "entry_price REAL, entry_date TEXT, last_updated TEXT)"
    )
    now = dt.datetime.now().isoformat()
    db.execute("INSERT INTO positions VALUES (%s,%s,%s,%s,%s)",
               ("INDV", 48, 34.32, now, now))
    db._migrate_trading_columns()
    db.execute(
        "UPDATE positions SET current_price = %s WHERE symbol = %s", (35.10, "INDV"),
    )
    row = db.execute(
        "SELECT stop_loss, current_price, last_updated FROM positions "
        "WHERE symbol = %s", ("INDV",), fetch='one',
    )
    assert row['current_price'] == pytest.approx(35.10)
    assert row['stop_loss'] is None


def test_fresh_schema_already_has_the_columns(tmp_path):
    """A brand-new database must not need the migration at all."""
    db = mgr(tmp_path, "fresh.db")
    db._create_trading_tables()
    assert {"current_price", "stop_loss", "profit_target", "strategy"} <= columns(db, "positions")
    assert "initial_balance" in columns(db, "account")
