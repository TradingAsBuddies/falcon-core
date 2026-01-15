"""
Backtest Results API

Stores and serves backtest results for the Falcon web portal.
Provides endpoints for:
- Storing daily feedback loop results
- Querying historical performance
- Strategy analytics dashboard
- Parameter adjustment history

Database tables:
- backtest_runs: Individual backtest executions
- feedback_results: Daily feedback loop summaries
- parameter_history: Track of parameter changes over time
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class BacktestRunRecord:
    """Record of a single backtest run"""
    id: Optional[int] = None
    strategy_name: str = ""
    symbol: str = ""
    trading_date: date = None
    interval: str = "5m"
    total_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    signals_count: int = 0
    parameters: Dict = None
    created_at: datetime = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['trading_date'] = self.trading_date.isoformat() if self.trading_date else None
        d['created_at'] = self.created_at.isoformat() if self.created_at else None
        return d


@dataclass
class FeedbackResultRecord:
    """Record of daily feedback loop execution"""
    id: Optional[int] = None
    run_date: date = None
    strategy_name: str = ""
    symbols_tested: int = 0
    total_trades: int = 0
    avg_return: float = 0.0
    avg_win_rate: float = 0.0
    avg_sharpe: float = 0.0
    adjustments_recommended: bool = False
    adjustments_applied: bool = False
    adjustment_details: Dict = None
    created_at: datetime = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['run_date'] = self.run_date.isoformat() if self.run_date else None
        d['created_at'] = self.created_at.isoformat() if self.created_at else None
        return d


class BacktestResultsStore:
    """
    Database storage for backtest results.

    Supports both SQLite (local) and PostgreSQL (production).
    """

    # SQL for creating tables
    CREATE_TABLES_SQL = """
    -- Backtest run records
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
    );

    -- Feedback loop results
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
    );

    -- Parameter change history
    CREATE TABLE IF NOT EXISTS parameter_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        strategy_name VARCHAR(100) NOT NULL,
        parameter_name VARCHAR(100) NOT NULL,
        old_value REAL,
        new_value REAL,
        reason TEXT,
        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Indexes for faster queries
    CREATE INDEX IF NOT EXISTS idx_backtest_runs_date ON backtest_runs(trading_date);
    CREATE INDEX IF NOT EXISTS idx_backtest_runs_strategy ON backtest_runs(strategy_name);
    CREATE INDEX IF NOT EXISTS idx_feedback_results_date ON feedback_results(run_date);
    CREATE INDEX IF NOT EXISTS idx_parameter_history_strategy ON parameter_history(strategy_name);
    """

    def __init__(self, db_path: str = None):
        """
        Initialize the results store.

        Args:
            db_path: Path to SQLite database (or PostgreSQL connection string)
        """
        import os

        self.db_path = db_path or os.path.expanduser(
            "~/.local/share/falcon/backtest_results.db"
        )
        self._conn = None
        self._ensure_tables()

    def _get_connection(self):
        """Get database connection"""
        if self._conn is None:
            import sqlite3
            import os

            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_tables(self):
        """Create tables if they don't exist"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Execute each statement separately
        for statement in self.CREATE_TABLES_SQL.split(';'):
            statement = statement.strip()
            if statement:
                try:
                    cursor.execute(statement)
                except Exception as e:
                    logger.warning(f"Table creation warning: {e}")

        conn.commit()

    def store_backtest_run(self, record: BacktestRunRecord) -> int:
        """Store a single backtest run"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO backtest_runs
            (strategy_name, symbol, trading_date, interval, total_return,
             max_drawdown, sharpe_ratio, win_rate, total_trades, signals_count,
             parameters, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.strategy_name,
            record.symbol,
            record.trading_date.isoformat() if record.trading_date else None,
            record.interval,
            record.total_return,
            record.max_drawdown,
            record.sharpe_ratio,
            record.win_rate,
            record.total_trades,
            record.signals_count,
            json.dumps(record.parameters) if record.parameters else None,
            datetime.now().isoformat(),
        ))

        conn.commit()
        return cursor.lastrowid

    def store_feedback_result(self, record: FeedbackResultRecord) -> int:
        """Store feedback loop result"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO feedback_results
            (run_date, strategy_name, symbols_tested, total_trades,
             avg_return, avg_win_rate, avg_sharpe, adjustments_recommended,
             adjustments_applied, adjustment_details, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.run_date.isoformat() if record.run_date else None,
            record.strategy_name,
            record.symbols_tested,
            record.total_trades,
            record.avg_return,
            record.avg_win_rate,
            record.avg_sharpe,
            1 if record.adjustments_recommended else 0,
            1 if record.adjustments_applied else 0,
            json.dumps(record.adjustment_details) if record.adjustment_details else None,
            datetime.now().isoformat(),
        ))

        conn.commit()
        return cursor.lastrowid

    def store_parameter_change(
        self,
        strategy_name: str,
        parameter_name: str,
        old_value: float,
        new_value: float,
        reason: str = None,
    ) -> int:
        """Record a parameter change"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO parameter_history
            (strategy_name, parameter_name, old_value, new_value, reason, applied_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            strategy_name,
            parameter_name,
            old_value,
            new_value,
            reason,
            datetime.now().isoformat(),
        ))

        conn.commit()
        return cursor.lastrowid

    # Query methods for API endpoints

    def get_recent_backtests(
        self,
        strategy_name: str = None,
        symbol: str = None,
        days: int = 30,
        limit: int = 100,
    ) -> List[Dict]:
        """Get recent backtest runs"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cutoff = (datetime.now() - timedelta(days=days)).date().isoformat()

        query = """
            SELECT * FROM backtest_runs
            WHERE trading_date >= ?
        """
        params = [cutoff]

        if strategy_name:
            query += " AND strategy_name = ?"
            params.append(strategy_name)

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        query += " ORDER BY trading_date DESC, created_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        return [dict(row) for row in rows]

    def get_feedback_history(
        self,
        strategy_name: str = None,
        days: int = 30,
    ) -> List[Dict]:
        """Get feedback loop history"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cutoff = (datetime.now() - timedelta(days=days)).date().isoformat()

        query = "SELECT * FROM feedback_results WHERE run_date >= ?"
        params = [cutoff]

        if strategy_name:
            query += " AND strategy_name = ?"
            params.append(strategy_name)

        query += " ORDER BY run_date DESC"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        return [dict(row) for row in rows]

    def get_parameter_history(
        self,
        strategy_name: str,
        parameter_name: str = None,
        limit: int = 50,
    ) -> List[Dict]:
        """Get parameter change history"""
        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM parameter_history WHERE strategy_name = ?"
        params = [strategy_name]

        if parameter_name:
            query += " AND parameter_name = ?"
            params.append(parameter_name)

        query += " ORDER BY applied_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        return [dict(row) for row in rows]

    def get_strategy_summary(self, strategy_name: str, days: int = 30) -> Dict:
        """Get summary statistics for a strategy"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cutoff = (datetime.now() - timedelta(days=days)).date().isoformat()

        cursor.execute("""
            SELECT
                COUNT(*) as total_runs,
                COUNT(DISTINCT symbol) as symbols_tested,
                AVG(total_return) as avg_return,
                AVG(win_rate) as avg_win_rate,
                AVG(sharpe_ratio) as avg_sharpe,
                SUM(total_trades) as total_trades,
                MAX(total_return) as best_return,
                MIN(total_return) as worst_return,
                AVG(max_drawdown) as avg_drawdown
            FROM backtest_runs
            WHERE strategy_name = ? AND trading_date >= ?
        """, (strategy_name, cutoff))

        row = cursor.fetchone()
        return dict(row) if row else {}

    def get_all_strategies_summary(self, days: int = 30) -> List[Dict]:
        """Get summary for all strategies"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cutoff = (datetime.now() - timedelta(days=days)).date().isoformat()

        cursor.execute("""
            SELECT
                strategy_name,
                COUNT(*) as total_runs,
                COUNT(DISTINCT symbol) as symbols_tested,
                AVG(total_return) as avg_return,
                AVG(win_rate) as avg_win_rate,
                AVG(sharpe_ratio) as avg_sharpe,
                SUM(total_trades) as total_trades
            FROM backtest_runs
            WHERE trading_date >= ?
            GROUP BY strategy_name
            ORDER BY avg_return DESC
        """, (cutoff,))

        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_daily_performance(
        self,
        strategy_name: str = None,
        days: int = 30,
    ) -> List[Dict]:
        """Get daily performance aggregates"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cutoff = (datetime.now() - timedelta(days=days)).date().isoformat()

        query = """
            SELECT
                trading_date,
                strategy_name,
                COUNT(*) as runs,
                AVG(total_return) as avg_return,
                AVG(win_rate) as avg_win_rate,
                SUM(total_trades) as total_trades
            FROM backtest_runs
            WHERE trading_date >= ?
        """
        params = [cutoff]

        if strategy_name:
            query += " AND strategy_name = ?"
            params.append(strategy_name)

        query += " GROUP BY trading_date, strategy_name ORDER BY trading_date DESC"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        return [dict(row) for row in rows]


# Flask/FastAPI-compatible API routes
def create_api_routes(app, results_store: BacktestResultsStore):
    """
    Create API routes for the web portal.

    Works with Flask or FastAPI.
    """
    from flask import jsonify, request

    @app.route('/api/backtests/recent', methods=['GET'])
    def get_recent_backtests():
        strategy = request.args.get('strategy')
        symbol = request.args.get('symbol')
        days = int(request.args.get('days', 30))
        results = results_store.get_recent_backtests(strategy, symbol, days)
        return jsonify(results)

    @app.route('/api/backtests/summary', methods=['GET'])
    def get_strategies_summary():
        days = int(request.args.get('days', 30))
        results = results_store.get_all_strategies_summary(days)
        return jsonify(results)

    @app.route('/api/backtests/strategy/<strategy_name>', methods=['GET'])
    def get_strategy_summary(strategy_name):
        days = int(request.args.get('days', 30))
        result = results_store.get_strategy_summary(strategy_name, days)
        return jsonify(result)

    @app.route('/api/backtests/daily', methods=['GET'])
    def get_daily_performance():
        strategy = request.args.get('strategy')
        days = int(request.args.get('days', 30))
        results = results_store.get_daily_performance(strategy, days)
        return jsonify(results)

    @app.route('/api/feedback/history', methods=['GET'])
    def get_feedback_history():
        strategy = request.args.get('strategy')
        days = int(request.args.get('days', 30))
        results = results_store.get_feedback_history(strategy, days)
        return jsonify(results)

    @app.route('/api/parameters/history/<strategy_name>', methods=['GET'])
    def get_parameter_history(strategy_name):
        param = request.args.get('parameter')
        results = results_store.get_parameter_history(strategy_name, param)
        return jsonify(results)

    logger.info("Registered backtest API routes")
