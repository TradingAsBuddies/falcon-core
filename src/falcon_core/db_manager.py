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
from urllib.parse import urlparse
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
        """Load database configuration from environment variables.

        DATABASE_URL takes precedence over individual DB_* variables.
        """
        database_url = os.getenv('DATABASE_URL')
        if database_url:
            parsed = urlparse(database_url)
            scheme = 'postgresql' if parsed.scheme in ('postgres', 'postgresql') else parsed.scheme
            return {
                'db_type': scheme.split('+')[0],
                'db_host': parsed.hostname or 'localhost',
                'db_port': parsed.port or 5432,
                'db_name': (parsed.path or '/falcon').lstrip('/'),
                'db_user': parsed.username or 'falcon',
                'db_password': parsed.password or '',
            }

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
        self._create_market_data_tables()
        self._create_strategy_rotation_tables()
        self._migrate_strategy_roster_v2()
        self._create_advisor_tables()
        self._create_backtest_results_tables()
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

    def _create_market_data_tables(self):
        """Create market data tables for Polygon flat files sync"""

        if self.db_type == 'sqlite':
            daily_bars_sql = '''
                CREATE TABLE IF NOT EXISTS daily_bars (
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    vwap REAL,
                    trades INTEGER,
                    PRIMARY KEY (symbol, date)
                )
            '''
            minute_bars_sql = '''
                CREATE TABLE IF NOT EXISTS minute_bars (
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    vwap REAL,
                    trades INTEGER,
                    PRIMARY KEY (symbol, timestamp)
                )
            '''
            sync_log_sql = '''
                CREATE TABLE IF NOT EXISTS sync_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sync_type TEXT NOT NULL,
                    sync_date TEXT NOT NULL,
                    file_key TEXT NOT NULL,
                    rows_loaded INTEGER DEFAULT 0,
                    duration_seconds REAL,
                    status TEXT DEFAULT 'success',
                    error_message TEXT,
                    completed_at TEXT,
                    UNIQUE(sync_type, sync_date)
                )
            '''
            daily_index_sql = '''
                CREATE INDEX IF NOT EXISTS idx_daily_bars_date ON daily_bars(date)
            '''
            minute_index_sql = '''
                CREATE INDEX IF NOT EXISTS idx_minute_bars_timestamp ON minute_bars(timestamp)
            '''
        else:  # postgresql
            daily_bars_sql = '''
                CREATE TABLE IF NOT EXISTS daily_bars (
                    symbol VARCHAR(10) NOT NULL,
                    date DATE NOT NULL,
                    open DECIMAL(12,4),
                    high DECIMAL(12,4),
                    low DECIMAL(12,4),
                    close DECIMAL(12,4),
                    volume BIGINT,
                    vwap DECIMAL(12,4),
                    trades INTEGER,
                    PRIMARY KEY (symbol, date)
                )
            '''
            minute_bars_sql = '''
                CREATE TABLE IF NOT EXISTS minute_bars (
                    symbol VARCHAR(10) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    open DECIMAL(12,4),
                    high DECIMAL(12,4),
                    low DECIMAL(12,4),
                    close DECIMAL(12,4),
                    volume BIGINT,
                    vwap DECIMAL(12,4),
                    trades INTEGER,
                    PRIMARY KEY (symbol, timestamp)
                )
            '''
            sync_log_sql = '''
                CREATE TABLE IF NOT EXISTS sync_log (
                    id SERIAL PRIMARY KEY,
                    sync_type VARCHAR(20) NOT NULL,
                    sync_date DATE NOT NULL,
                    file_key VARCHAR(200) NOT NULL,
                    rows_loaded INTEGER DEFAULT 0,
                    duration_seconds DECIMAL(10,2),
                    status VARCHAR(20) DEFAULT 'success',
                    error_message TEXT,
                    completed_at TIMESTAMP,
                    UNIQUE(sync_type, sync_date)
                )
            '''
            daily_index_sql = '''
                CREATE INDEX IF NOT EXISTS idx_daily_bars_date ON daily_bars(date)
            '''
            minute_index_sql = '''
                CREATE INDEX IF NOT EXISTS idx_minute_bars_timestamp ON minute_bars(timestamp)
            '''

        self.execute(daily_bars_sql)
        self.execute(minute_bars_sql)
        self.execute(sync_log_sql)
        self.execute(daily_index_sql)
        self.execute(minute_index_sql)

        logger.info("Market data tables created")

    def _create_strategy_rotation_tables(self):
        """Create strategy rotation/roster tables for strategy lifecycle management"""

        if self.db_type == 'sqlite':
            roster_sql = '''
                CREATE TABLE IF NOT EXISTS strategy_roster (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT UNIQUE NOT NULL,
                    status TEXT DEFAULT 'backtest',
                    promoted_at TEXT,
                    demoted_at TEXT,
                    review_notes TEXT,
                    symbols TEXT DEFAULT '[]',
                    interval TEXT DEFAULT '5m',
                    params TEXT DEFAULT '{}',
                    last_backtest_at TEXT,
                    backtest_sharpe REAL,
                    backtest_win_rate REAL,
                    backtest_profit_factor REAL,
                    backtest_total_return REAL,
                    paper_sharpe REAL,
                    paper_win_rate REAL,
                    paper_profit_factor REAL,
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now'))
                )
            '''
            rotation_log_sql = '''
                CREATE TABLE IF NOT EXISTS strategy_rotation_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    from_status TEXT,
                    to_status TEXT NOT NULL,
                    reason TEXT,
                    metrics TEXT,
                    rotated_at TEXT DEFAULT (datetime('now'))
                )
            '''
        else:  # postgresql
            roster_sql = '''
                CREATE TABLE IF NOT EXISTS strategy_roster (
                    id SERIAL PRIMARY KEY,
                    strategy_name VARCHAR(50) UNIQUE NOT NULL,
                    status VARCHAR(20) DEFAULT 'backtest',
                    promoted_at TIMESTAMP,
                    demoted_at TIMESTAMP,
                    review_notes TEXT,
                    symbols JSONB DEFAULT '[]',
                    interval VARCHAR(10) DEFAULT '5m',
                    params JSONB DEFAULT '{}',
                    last_backtest_at TIMESTAMP,
                    backtest_sharpe DECIMAL(10,4),
                    backtest_win_rate DECIMAL(5,4),
                    backtest_profit_factor DECIMAL(10,4),
                    backtest_total_return DECIMAL(10,4),
                    paper_sharpe DECIMAL(10,4),
                    paper_win_rate DECIMAL(5,4),
                    paper_profit_factor DECIMAL(10,4),
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            '''
            rotation_log_sql = '''
                CREATE TABLE IF NOT EXISTS strategy_rotation_log (
                    id SERIAL PRIMARY KEY,
                    strategy_name VARCHAR(50) NOT NULL,
                    from_status VARCHAR(20),
                    to_status VARCHAR(20) NOT NULL,
                    reason TEXT,
                    metrics JSONB,
                    rotated_at TIMESTAMP DEFAULT NOW()
                )
            '''

        roster_index_sql = '''
            CREATE INDEX IF NOT EXISTS idx_strategy_roster_status
            ON strategy_roster(status)
        '''
        rotation_log_index_sql = '''
            CREATE INDEX IF NOT EXISTS idx_rotation_log_strategy
            ON strategy_rotation_log(strategy_name)
        '''

        self.execute(roster_sql)
        self.execute(rotation_log_sql)
        self.execute(roster_index_sql)
        self.execute(rotation_log_index_sql)

        logger.info("Strategy rotation tables created")

    def _migrate_strategy_roster_v2(self):
        """Add strategy_code and strategy_source columns to strategy_roster if missing."""
        try:
            if self.db_type == 'sqlite':
                # Check columns via PRAGMA
                cols = self.execute(
                    "PRAGMA table_info(strategy_roster)", fetch='all'
                )
                if cols is None:
                    return
                col_names = {
                    (c['name'] if isinstance(c, dict) else c[1]) for c in cols
                }
                if 'strategy_code' not in col_names:
                    self.execute(
                        'ALTER TABLE strategy_roster ADD COLUMN strategy_code TEXT'
                    )
                    logger.info("Added strategy_code column to strategy_roster")
                if 'strategy_source' not in col_names:
                    self.execute(
                        "ALTER TABLE strategy_roster ADD COLUMN "
                        "strategy_source TEXT DEFAULT 'manual'"
                    )
                    logger.info("Added strategy_source column to strategy_roster")
            else:  # postgresql
                # Check via information_schema
                row = self.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'strategy_roster' "
                    "AND column_name = 'strategy_code'",
                    fetch='one'
                )
                if not row:
                    self.execute(
                        'ALTER TABLE strategy_roster ADD COLUMN strategy_code TEXT'
                    )
                    logger.info("Added strategy_code column to strategy_roster")
                row = self.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'strategy_roster' "
                    "AND column_name = 'strategy_source'",
                    fetch='one'
                )
                if not row:
                    self.execute(
                        "ALTER TABLE strategy_roster ADD COLUMN "
                        "strategy_source VARCHAR(20) DEFAULT 'manual'"
                    )
                    logger.info("Added strategy_source column to strategy_roster")
        except Exception as e:
            logger.debug(f"strategy_roster migration check: {e}")

    def _create_advisor_tables(self):
        """Create AI advisor and cost tracking tables."""

        if self.db_type == 'sqlite':
            api_usage_sql = '''
                CREATE TABLE IF NOT EXISTS api_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    service TEXT NOT NULL,
                    model TEXT,
                    strategy_name TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    cost_usd REAL,
                    request_type TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            '''
            budget_sql = '''
                CREATE TABLE IF NOT EXISTS strategy_advisor_budget (
                    strategy_name TEXT PRIMARY KEY,
                    monthly_budget_usd REAL DEFAULT 1.00,
                    total_spent_usd REAL DEFAULT 0,
                    current_month_spent_usd REAL DEFAULT 0,
                    months_active INTEGER DEFAULT 0,
                    max_months INTEGER DEFAULT 4,
                    consecutive_no_improvement INTEGER DEFAULT 0,
                    last_improvement_at TEXT,
                    status TEXT DEFAULT 'active',
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now'))
                )
            '''
            proposals_sql = '''
                CREATE TABLE IF NOT EXISTS strategy_proposals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    proposal_type TEXT,
                    current_code TEXT,
                    proposed_code TEXT,
                    analysis_summary TEXT,
                    change_description TEXT,
                    expected_improvement TEXT,
                    current_sharpe REAL,
                    proposed_sharpe REAL,
                    current_win_rate REAL,
                    proposed_win_rate REAL,
                    current_total_return REAL,
                    proposed_total_return REAL,
                    status TEXT DEFAULT 'pending',
                    reviewed_by TEXT,
                    reviewed_at TEXT,
                    review_notes TEXT,
                    api_cost_usd REAL,
                    created_at TEXT DEFAULT (datetime('now')),
                    applied_at TEXT
                )
            '''
        else:  # postgresql
            api_usage_sql = '''
                CREATE TABLE IF NOT EXISTS api_usage (
                    id SERIAL PRIMARY KEY,
                    service VARCHAR(50) NOT NULL,
                    model VARCHAR(50),
                    strategy_name VARCHAR(50),
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    cost_usd DECIMAL(10,6),
                    request_type VARCHAR(50),
                    created_at TIMESTAMP DEFAULT NOW()
                )
            '''
            budget_sql = '''
                CREATE TABLE IF NOT EXISTS strategy_advisor_budget (
                    strategy_name VARCHAR(50) PRIMARY KEY,
                    monthly_budget_usd DECIMAL(6,2) DEFAULT 1.00,
                    total_spent_usd DECIMAL(10,4) DEFAULT 0,
                    current_month_spent_usd DECIMAL(10,4) DEFAULT 0,
                    months_active INTEGER DEFAULT 0,
                    max_months INTEGER DEFAULT 4,
                    consecutive_no_improvement INTEGER DEFAULT 0,
                    last_improvement_at TIMESTAMP,
                    status VARCHAR(20) DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            '''
            proposals_sql = '''
                CREATE TABLE IF NOT EXISTS strategy_proposals (
                    id SERIAL PRIMARY KEY,
                    strategy_name VARCHAR(50) NOT NULL,
                    proposal_type VARCHAR(20),
                    current_code TEXT,
                    proposed_code TEXT,
                    analysis_summary TEXT,
                    change_description TEXT,
                    expected_improvement TEXT,
                    current_sharpe DECIMAL(10,4),
                    proposed_sharpe DECIMAL(10,4),
                    current_win_rate DECIMAL(5,4),
                    proposed_win_rate DECIMAL(5,4),
                    current_total_return DECIMAL(10,4),
                    proposed_total_return DECIMAL(10,4),
                    status VARCHAR(20) DEFAULT 'pending',
                    reviewed_by VARCHAR(100),
                    reviewed_at TIMESTAMP,
                    review_notes TEXT,
                    api_cost_usd DECIMAL(10,6),
                    created_at TIMESTAMP DEFAULT NOW(),
                    applied_at TIMESTAMP
                )
            '''

        usage_index_sql = '''
            CREATE INDEX IF NOT EXISTS idx_api_usage_service
            ON api_usage(service)
        '''
        proposals_index_sql = '''
            CREATE INDEX IF NOT EXISTS idx_strategy_proposals_status
            ON strategy_proposals(status)
        '''

        self.execute(api_usage_sql)
        self.execute(budget_sql)
        self.execute(proposals_sql)
        self.execute(usage_index_sql)
        self.execute(proposals_index_sql)

        logger.info("Advisor tables created")

    def _create_backtest_results_tables(self):
        """Create tables for storing backtest results (replaces separate SQLite DB)."""

        if self.db_type == 'sqlite':
            backtest_runs_sql = '''
                CREATE TABLE IF NOT EXISTS backtest_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name VARCHAR(100) NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    trading_date DATE NOT NULL,
                    interval VARCHAR(10) DEFAULT '5m',
                    total_return REAL DEFAULT 0,
                    max_drawdown REAL DEFAULT 0,
                    sharpe_ratio REAL DEFAULT 0,
                    win_rate REAL DEFAULT 0,
                    total_trades INTEGER DEFAULT 0,
                    signals_count INTEGER DEFAULT 0,
                    parameters TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            '''
            feedback_results_sql = '''
                CREATE TABLE IF NOT EXISTS feedback_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_date DATE NOT NULL,
                    strategy_name VARCHAR(100) NOT NULL,
                    symbols_tested INTEGER DEFAULT 0,
                    total_trades INTEGER DEFAULT 0,
                    avg_return REAL DEFAULT 0,
                    avg_win_rate REAL DEFAULT 0,
                    avg_sharpe REAL DEFAULT 0,
                    adjustments_recommended BOOLEAN DEFAULT FALSE,
                    adjustments_applied BOOLEAN DEFAULT FALSE,
                    adjustment_details TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            '''
            parameter_history_sql = '''
                CREATE TABLE IF NOT EXISTS parameter_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name VARCHAR(100) NOT NULL,
                    parameter_name VARCHAR(100) NOT NULL,
                    old_value REAL,
                    new_value REAL,
                    reason TEXT,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            '''
        else:
            backtest_runs_sql = '''
                CREATE TABLE IF NOT EXISTS backtest_runs (
                    id SERIAL PRIMARY KEY,
                    strategy_name VARCHAR(100) NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    trading_date DATE NOT NULL,
                    interval VARCHAR(10) DEFAULT '5m',
                    total_return DECIMAL(10,6) DEFAULT 0,
                    max_drawdown DECIMAL(10,6) DEFAULT 0,
                    sharpe_ratio DECIMAL(10,4) DEFAULT 0,
                    win_rate DECIMAL(10,6) DEFAULT 0,
                    total_trades INTEGER DEFAULT 0,
                    signals_count INTEGER DEFAULT 0,
                    parameters TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            '''
            feedback_results_sql = '''
                CREATE TABLE IF NOT EXISTS feedback_results (
                    id SERIAL PRIMARY KEY,
                    run_date DATE NOT NULL,
                    strategy_name VARCHAR(100) NOT NULL,
                    symbols_tested INTEGER DEFAULT 0,
                    total_trades INTEGER DEFAULT 0,
                    avg_return DECIMAL(10,6) DEFAULT 0,
                    avg_win_rate DECIMAL(10,6) DEFAULT 0,
                    avg_sharpe DECIMAL(10,4) DEFAULT 0,
                    adjustments_recommended BOOLEAN DEFAULT FALSE,
                    adjustments_applied BOOLEAN DEFAULT FALSE,
                    adjustment_details TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            '''
            parameter_history_sql = '''
                CREATE TABLE IF NOT EXISTS parameter_history (
                    id SERIAL PRIMARY KEY,
                    strategy_name VARCHAR(100) NOT NULL,
                    parameter_name VARCHAR(100) NOT NULL,
                    old_value DECIMAL(10,6),
                    new_value DECIMAL(10,6),
                    reason TEXT,
                    applied_at TIMESTAMP DEFAULT NOW()
                )
            '''

        self.execute(backtest_runs_sql)
        self.execute(feedback_results_sql)
        self.execute(parameter_history_sql)

        # Indexes
        self.execute('CREATE INDEX IF NOT EXISTS idx_backtest_runs_date ON backtest_runs(trading_date)')
        self.execute('CREATE INDEX IF NOT EXISTS idx_backtest_runs_strategy ON backtest_runs(strategy_name)')
        self.execute('CREATE INDEX IF NOT EXISTS idx_feedback_results_date ON feedback_results(run_date)')
        self.execute('CREATE INDEX IF NOT EXISTS idx_parameter_history_strategy ON parameter_history(strategy_name)')

        logger.info("Backtest results tables created")

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
            tables = ['strategy_rotation_log', 'strategy_roster',
                     'sync_log', 'minute_bars', 'daily_bars',
                     'profile_performance', 'profile_runs', 'screener_profiles',
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
