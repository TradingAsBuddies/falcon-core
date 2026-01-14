#!/usr/bin/env python3
"""
Database Abstraction Layer for Falcon Trading Platform
Supports SQLite (default) and PostgreSQL
"""

import os
import sqlite3
import datetime
from typing import Any, Dict, List, Optional, Tuple
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Unified database interface supporting SQLite and PostgreSQL"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize database manager

        Args:
            config: Database configuration dict with keys:
                - db_type: 'sqlite' or 'postgresql' (default: from env or 'sqlite')
                - db_path: Path for SQLite (default: from env or /var/lib/falcon/paper_trading.db)
                - db_host: PostgreSQL host
                - db_port: PostgreSQL port
                - db_name: PostgreSQL database name
                - db_user: PostgreSQL username
                - db_password: PostgreSQL password
        """
        self.config = config or self._load_config_from_env()
        self.db_type = self.config.get('db_type', 'sqlite').lower()

        # Import PostgreSQL driver only if needed
        self.psycopg2 = None
        self.pool = None

        if self.db_type == 'postgresql':
            try:
                import psycopg2
                from psycopg2 import pool
                self.psycopg2 = psycopg2
                self._init_postgres_pool()
            except ImportError:
                raise ImportError(
                    "psycopg2 is required for PostgreSQL support. "
                    "Install with: pip install psycopg2-binary"
                )
        elif self.db_type == 'sqlite':
            self.db_path = self.config.get('db_path', '/var/lib/falcon/paper_trading.db')
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")

        logger.info(f"Database manager initialized: {self.db_type}")

    def _load_config_from_env(self) -> Dict[str, Any]:
        """Load database configuration from environment variables"""
        return {
            'db_type': os.getenv('DB_TYPE', 'sqlite'),
            'db_path': os.getenv('DB_PATH', '/var/lib/falcon/paper_trading.db'),
            'db_host': os.getenv('DB_HOST', 'localhost'),
            'db_port': int(os.getenv('DB_PORT', '5432')),
            'db_name': os.getenv('DB_NAME', 'falcon'),
            'db_user': os.getenv('DB_USER', 'falcon'),
            'db_password': os.getenv('DB_PASSWORD', ''),
        }

    def _init_postgres_pool(self):
        """Initialize PostgreSQL connection pool"""
        from psycopg2 import pool

        self.pool = pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            host=self.config['db_host'],
            port=self.config['db_port'],
            database=self.config['db_name'],
            user=self.config['db_user'],
            password=self.config['db_password']
        )
        logger.info("PostgreSQL connection pool initialized")

    @contextmanager
    def get_connection(self):
        """
        Get database connection as context manager

        Usage:
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM account")
        """
        if self.db_type == 'sqlite':
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            try:
                yield conn
            finally:
                conn.close()
        else:  # postgresql
            conn = self.pool.getconn()
            try:
                yield conn
            finally:
                self.pool.putconn(conn)

    def execute(self, query: str, params: Optional[Tuple] = None,
                fetch: str = 'none') -> Any:
        """
        Execute a query

        Args:
            query: SQL query (use %s for placeholders in both SQLite and PostgreSQL)
            params: Query parameters
            fetch: 'none', 'one', 'all'

        Returns:
            Query results based on fetch parameter
            For fetch='one'/'all': Returns dict-like rows for both SQLite and PostgreSQL
        """
        # Convert %s placeholders to ? for SQLite
        if self.db_type == 'sqlite':
            query = query.replace('%s', '?')

        with self.get_connection() as conn:
            # Use RealDictCursor for PostgreSQL to get dict-like rows
            if self.db_type == 'postgresql':
                from psycopg2.extras import RealDictCursor
                cursor = conn.cursor(cursor_factory=RealDictCursor)
            else:
                cursor = conn.cursor()

            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            result = None
            if fetch == 'one':
                result = cursor.fetchone()
            elif fetch == 'all':
                result = cursor.fetchall()
            elif fetch == 'none':
                conn.commit()
                result = cursor.lastrowid if self.db_type == 'sqlite' else cursor.rowcount

            return result

    def executemany(self, query: str, params_list: List[Tuple]) -> int:
        """Execute a query multiple times with different parameters"""
        if self.db_type == 'sqlite':
            query = query.replace('%s', '?')

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            conn.commit()
            return cursor.rowcount

    def init_schema(self):
        """Initialize database schema (all tables)"""
        logger.info("Initializing database schema...")
        self._create_trading_tables()
        self._create_youtube_strategy_tables()
        self._create_screener_tables()
        logger.info("Database schema initialized successfully")

    def _create_trading_tables(self):
        """Create paper trading tables"""

        # Account table
        if self.db_type == 'sqlite':
            account_sql = '''
                CREATE TABLE IF NOT EXISTS account (
                    id INTEGER PRIMARY KEY,
                    cash REAL NOT NULL,
                    last_updated TEXT NOT NULL
                )
            '''
        else:  # postgresql
            account_sql = '''
                CREATE TABLE IF NOT EXISTS account (
                    id SERIAL PRIMARY KEY,
                    cash DECIMAL(15,2) NOT NULL,
                    last_updated TIMESTAMP NOT NULL
                )
            '''

        # Positions table
        if self.db_type == 'sqlite':
            positions_sql = '''
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    quantity REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_date TEXT NOT NULL,
                    last_updated TEXT NOT NULL
                )
            '''
        else:  # postgresql
            positions_sql = '''
                CREATE TABLE IF NOT EXISTS positions (
                    symbol VARCHAR(20) PRIMARY KEY,
                    quantity DECIMAL(15,4) NOT NULL,
                    entry_price DECIMAL(15,2) NOT NULL,
                    entry_date TIMESTAMP NOT NULL,
                    last_updated TIMESTAMP NOT NULL
                )
            '''

        # Orders table
        if self.db_type == 'sqlite':
            orders_sql = '''
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    pnl REAL DEFAULT 0
                )
            '''
        else:  # postgresql
            orders_sql = '''
                CREATE TABLE IF NOT EXISTS orders (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    side VARCHAR(10) NOT NULL,
                    quantity DECIMAL(15,4) NOT NULL,
                    price DECIMAL(15,2) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    pnl DECIMAL(15,2) DEFAULT 0
                )
            '''

        # Performance table
        if self.db_type == 'sqlite':
            performance_sql = '''
                CREATE TABLE IF NOT EXISTS performance (
                    timestamp TEXT PRIMARY KEY,
                    total_value REAL NOT NULL,
                    cash REAL NOT NULL,
                    positions_value REAL NOT NULL
                )
            '''
        else:  # postgresql
            performance_sql = '''
                CREATE TABLE IF NOT EXISTS performance (
                    timestamp TIMESTAMP PRIMARY KEY,
                    total_value DECIMAL(15,2) NOT NULL,
                    cash DECIMAL(15,2) NOT NULL,
                    positions_value DECIMAL(15,2) NOT NULL
                )
            '''

        # Execute table creation
        self.execute(account_sql)
        self.execute(positions_sql)
        self.execute(orders_sql)
        self.execute(performance_sql)

        logger.info("Trading tables created")

    def _create_youtube_strategy_tables(self):
        """Create YouTube strategy tables"""

        # Strategies table
        if self.db_type == 'sqlite':
            strategies_sql = '''
                CREATE TABLE IF NOT EXISTS youtube_strategies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    creator TEXT NOT NULL,
                    youtube_url TEXT UNIQUE NOT NULL,
                    video_id TEXT NOT NULL,
                    description TEXT,
                    strategy_overview TEXT,
                    trading_style TEXT,
                    instruments TEXT,
                    entry_rules TEXT,
                    exit_rules TEXT,
                    risk_management TEXT,
                    strategy_code TEXT,
                    tags TEXT,
                    performance_metrics TEXT,
                    pros TEXT,
                    cons TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            '''
        else:  # postgresql
            strategies_sql = '''
                CREATE TABLE IF NOT EXISTS youtube_strategies (
                    id SERIAL PRIMARY KEY,
                    title VARCHAR(500) NOT NULL,
                    creator VARCHAR(200) NOT NULL,
                    youtube_url VARCHAR(500) UNIQUE NOT NULL,
                    video_id VARCHAR(20) NOT NULL,
                    description TEXT,
                    strategy_overview TEXT,
                    trading_style VARCHAR(100),
                    instruments TEXT,
                    entry_rules TEXT,
                    exit_rules TEXT,
                    risk_management TEXT,
                    strategy_code TEXT,
                    tags TEXT,
                    performance_metrics TEXT,
                    pros TEXT,
                    cons TEXT,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
            '''

        # Backtests table
        if self.db_type == 'sqlite':
            backtests_sql = '''
                CREATE TABLE IF NOT EXISTS strategy_backtests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id INTEGER NOT NULL,
                    ticker TEXT NOT NULL,
                    start_date TEXT,
                    end_date TEXT,
                    total_return REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    total_trades INTEGER,
                    backtest_data TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (strategy_id) REFERENCES youtube_strategies(id)
                )
            '''
        else:  # postgresql
            backtests_sql = '''
                CREATE TABLE IF NOT EXISTS strategy_backtests (
                    id SERIAL PRIMARY KEY,
                    strategy_id INTEGER NOT NULL,
                    ticker VARCHAR(20) NOT NULL,
                    start_date DATE,
                    end_date DATE,
                    total_return DECIMAL(10,4),
                    sharpe_ratio DECIMAL(10,4),
                    max_drawdown DECIMAL(10,4),
                    win_rate DECIMAL(5,4),
                    total_trades INTEGER,
                    backtest_data TEXT,
                    created_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (strategy_id) REFERENCES youtube_strategies(id)
                        ON DELETE CASCADE
                )
            '''

        self.execute(strategies_sql)
        self.execute(backtests_sql)

        logger.info("YouTube strategy tables created")

    def _create_screener_tables(self):
        """Create screener profile tables for multi-profile support"""

        # Screener profiles table
        if self.db_type == 'sqlite':
            profiles_sql = '''
                CREATE TABLE IF NOT EXISTS screener_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    theme TEXT NOT NULL,
                    finviz_url TEXT,
                    finviz_filters TEXT,
                    sector_focus TEXT,
                    schedule TEXT,
                    enabled INTEGER DEFAULT 1,
                    weights TEXT,
                    performance_score REAL DEFAULT 0.5,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            '''
        else:  # postgresql
            profiles_sql = '''
                CREATE TABLE IF NOT EXISTS screener_profiles (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100) UNIQUE NOT NULL,
                    description TEXT,
                    theme VARCHAR(50) NOT NULL,
                    finviz_url TEXT,
                    finviz_filters JSONB,
                    sector_focus JSONB,
                    schedule JSONB,
                    enabled BOOLEAN DEFAULT TRUE,
                    weights JSONB,
                    performance_score DECIMAL(5,4) DEFAULT 0.5,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
            '''

        # Profile runs table (track each screening execution)
        if self.db_type == 'sqlite':
            runs_sql = '''
                CREATE TABLE IF NOT EXISTS profile_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_id INTEGER NOT NULL,
                    run_type TEXT NOT NULL,
                    stocks_found INTEGER DEFAULT 0,
                    recommendations_generated INTEGER DEFAULT 0,
                    run_timestamp TEXT NOT NULL,
                    ai_agent_used TEXT,
                    run_data TEXT,
                    FOREIGN KEY (profile_id) REFERENCES screener_profiles(id) ON DELETE CASCADE
                )
            '''
        else:  # postgresql
            runs_sql = '''
                CREATE TABLE IF NOT EXISTS profile_runs (
                    id SERIAL PRIMARY KEY,
                    profile_id INTEGER NOT NULL,
                    run_type VARCHAR(20) NOT NULL,
                    stocks_found INTEGER DEFAULT 0,
                    recommendations_generated INTEGER DEFAULT 0,
                    run_timestamp TIMESTAMP NOT NULL,
                    ai_agent_used VARCHAR(50),
                    run_data JSONB,
                    FOREIGN KEY (profile_id) REFERENCES screener_profiles(id) ON DELETE CASCADE
                )
            '''

        # Profile performance table (track attribution-based outcomes)
        if self.db_type == 'sqlite':
            performance_sql = '''
                CREATE TABLE IF NOT EXISTS profile_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_id INTEGER NOT NULL,
                    date TEXT NOT NULL,
                    stocks_recommended INTEGER DEFAULT 0,
                    stocks_profitable INTEGER DEFAULT 0,
                    avg_return_pct REAL DEFAULT 0,
                    attribution_breakdown TEXT,
                    weight_adjustments TEXT,
                    calculated_at TEXT NOT NULL,
                    FOREIGN KEY (profile_id) REFERENCES screener_profiles(id) ON DELETE CASCADE,
                    UNIQUE(profile_id, date)
                )
            '''
        else:  # postgresql
            performance_sql = '''
                CREATE TABLE IF NOT EXISTS profile_performance (
                    id SERIAL PRIMARY KEY,
                    profile_id INTEGER NOT NULL,
                    date DATE NOT NULL,
                    stocks_recommended INTEGER DEFAULT 0,
                    stocks_profitable INTEGER DEFAULT 0,
                    avg_return_pct DECIMAL(10,4) DEFAULT 0,
                    attribution_breakdown JSONB,
                    weight_adjustments JSONB,
                    calculated_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (profile_id) REFERENCES screener_profiles(id) ON DELETE CASCADE,
                    UNIQUE(profile_id, date)
                )
            '''

        # Create indexes for performance
        if self.db_type == 'sqlite':
            runs_index_sql = '''
                CREATE INDEX IF NOT EXISTS idx_profile_runs_profile_id
                ON profile_runs(profile_id)
            '''
            perf_index_sql = '''
                CREATE INDEX IF NOT EXISTS idx_profile_performance_profile_date
                ON profile_performance(profile_id, date)
            '''
        else:
            runs_index_sql = '''
                CREATE INDEX IF NOT EXISTS idx_profile_runs_profile_id
                ON profile_runs(profile_id)
            '''
            perf_index_sql = '''
                CREATE INDEX IF NOT EXISTS idx_profile_performance_profile_date
                ON profile_performance(profile_id, date)
            '''

        self.execute(profiles_sql)
        self.execute(runs_sql)
        self.execute(performance_sql)
        self.execute(runs_index_sql)
        self.execute(perf_index_sql)

        logger.info("Screener profile tables created")

    def init_account(self, initial_balance: float = 10000.0):
        """Initialize account with starting balance"""
        count = self.execute('SELECT COUNT(*) FROM account', fetch='one')
        count_value = count[0] if count else 0

        if count_value == 0:
            timestamp = datetime.datetime.now()
            if self.db_type == 'sqlite':
                timestamp = timestamp.isoformat()

            self.execute(
                'INSERT INTO account (id, cash, last_updated) VALUES (%s, %s, %s)',
                (1, initial_balance, timestamp)
            )
            logger.info(f"Account initialized with ${initial_balance:,.2f}")
        else:
            logger.info("Account already exists")

    def close(self):
        """Close database connections"""
        if self.db_type == 'postgresql' and self.pool:
            self.pool.closeall()
            logger.info("Database connections closed")


def get_db_manager(config: Optional[Dict[str, Any]] = None) -> DatabaseManager:
    """
    Factory function to get database manager instance

    Args:
        config: Optional configuration dict. If None, loads from environment.

    Returns:
        DatabaseManager instance
    """
    return DatabaseManager(config)


if __name__ == '__main__':
    # Test/initialization script
    import sys

    print("Falcon Database Manager - Initialization")
    print("=" * 50)

    # Check command line args
    reset = '--reset' in sys.argv
    db_type = None

    for arg in sys.argv[1:]:
        if arg.startswith('--type='):
            db_type = arg.split('=')[1]

    # Override environment if specified
    if db_type:
        os.environ['DB_TYPE'] = db_type

    # Create database manager
    db = get_db_manager()

    print(f"Database Type: {db.db_type}")
    if db.db_type == 'sqlite':
        print(f"Database Path: {db.db_path}")
    else:
        print(f"Database Host: {db.config['db_host']}")
        print(f"Database Name: {db.config['db_name']}")

    if reset:
        response = input("\nThis will DELETE all existing data. Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            sys.exit(0)

        # Drop all tables for reset
        if db.db_type == 'sqlite':
            import os
            if os.path.exists(db.db_path):
                os.remove(db.db_path)
                print(f"[RESET] Deleted database: {db.db_path}")
        else:
            # For PostgreSQL, drop tables individually
            tables = ['profile_performance', 'profile_runs', 'screener_profiles',
                     'strategy_backtests', 'youtube_strategies',
                     'performance', 'orders', 'positions', 'account']
            for table in tables:
                try:
                    db.execute(f'DROP TABLE IF EXISTS {table} CASCADE')
                    print(f"[RESET] Dropped table: {table}")
                except Exception as e:
                    print(f"[RESET] Could not drop {table}: {e}")

    # Initialize schema
    print("\n[INIT] Creating database schema...")
    db.init_schema()

    # Initialize account
    print("[INIT] Initializing account...")
    db.init_account(initial_balance=10000.0)

    # Verify
    print("\n[VERIFY] Checking database...")
    result = db.execute('SELECT cash FROM account WHERE id = %s', (1,), fetch='one')
    if result:
        print(f"  Account balance: ${result[0]:,.2f}")

    print("\n[SUCCESS] Database initialization complete!")
    print(f"Database type: {db.db_type}")

    db.close()
