"""
Data Sync Pipeline — Polygon Flat Files → PostgreSQL

Downloads bulk market data from Polygon's Flat Files (via Massive S3)
and bulk-loads into PostgreSQL using COPY for performance.

Supports daily and minute bars with idempotent upsert logic.
"""

import io
import logging
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataSyncPipeline:
    """Pipeline to sync Polygon flat files into PostgreSQL."""

    # Column mapping from flat file format to database columns
    # Flat files use full names (ticker, open, close, volume, transactions)
    # and short names (o, h, l, c, v, vw, n) depending on the file version
    COLUMN_MAP = {
        'ticker': 'symbol',
        'window_start': 'date',
        'transactions': 'trades',
        # v1 short-form aliases
        'o': 'open',
        'h': 'high',
        'l': 'low',
        'c': 'close',
        'v': 'volume',
        'vw': 'vwap',
        'n': 'trades',
    }

    DAILY_COLUMNS = ['symbol', 'date', 'open', 'high', 'low', 'close',
                     'volume', 'trades']
    DAILY_COLUMNS_OPTIONAL = ['vwap']
    MINUTE_COLUMNS = ['symbol', 'timestamp', 'open', 'high', 'low', 'close',
                      'volume', 'trades']
    MINUTE_COLUMNS_OPTIONAL = ['vwap']

    def __init__(self, db, flat_client):
        """
        Args:
            db: DatabaseManager instance (must be postgresql)
            flat_client: FlatFilesClient instance
        """
        self.db = db
        self.flat_client = flat_client

        if db.db_type != 'postgresql':
            raise ValueError("DataSyncPipeline requires PostgreSQL (COPY not supported on SQLite)")

    def sync_daily(self, target_date: date) -> Dict[str, Any]:
        """
        Sync daily bars for a single date.

        Downloads the per-day flat file for target_date (all symbols)
        and bulk-loads into daily_bars via COPY + upsert.

        Args:
            target_date: The trading date to sync

        Returns:
            Dict with keys: date, rows_loaded, duration_seconds, status
        """
        sync_type = 'daily'

        # Check if already synced
        if self._already_synced(sync_type, target_date):
            logger.info(f"Daily bars for {target_date} already synced, skipping")
            return {'date': str(target_date), 'rows_loaded': 0,
                    'duration_seconds': 0, 'status': 'skipped'}

        start_time = time.time()
        s3_key = self.flat_client._get_day_aggs_key(target_date)

        try:
            # Today's file may still be accumulating — skip cache
            use_cache = target_date < date.today()

            df = self.flat_client._download_file(s3_key, use_cache=use_cache)
            if df.empty:
                logger.info(f"No trading data for {target_date} (weekend/holiday?)")
                return self._record_sync(sync_type, target_date, s3_key, 0,
                                         time.time() - start_time, 'no_data')

            # Standardize columns
            df = df.rename(columns=self.COLUMN_MAP)

            # Parse date — may be nanosecond epoch or ISO string
            if 'date' in df.columns:
                raw = df['date']
                if pd.api.types.is_numeric_dtype(raw):
                    df['date'] = pd.to_datetime(raw, unit='ns').dt.date
                else:
                    df['date'] = pd.to_datetime(raw).dt.date

            # Add optional columns if missing (NULL-filled)
            for col in self.DAILY_COLUMNS_OPTIONAL:
                if col not in df.columns:
                    df[col] = None

            # Select columns present in the table
            use_cols = self.DAILY_COLUMNS + [c for c in self.DAILY_COLUMNS_OPTIONAL if c in df.columns]
            df = df[use_cols].dropna(subset=['symbol'])

            rows = self._bulk_load_daily(df)
            duration = time.time() - start_time

            logger.info(f"Synced {rows} daily bars for {target_date} in {duration:.1f}s")
            return self._record_sync(sync_type, target_date, s3_key, rows,
                                     duration, 'success')

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Failed to sync daily bars for {target_date}: {e}")
            return self._record_sync(sync_type, target_date, s3_key, 0,
                                     duration, 'error', str(e))

    def sync_minute(self, target_date: date) -> Dict[str, Any]:
        """
        Sync minute bars for a single date.

        Args:
            target_date: The trading date to sync

        Returns:
            Dict with keys: date, rows_loaded, duration_seconds, status
        """
        sync_type = 'minute'

        if self._already_synced(sync_type, target_date):
            logger.info(f"Minute bars for {target_date} already synced, skipping")
            return {'date': str(target_date), 'rows_loaded': 0,
                    'duration_seconds': 0, 'status': 'skipped'}

        start_time = time.time()
        s3_key = self.flat_client._get_minute_aggs_key(target_date)

        try:
            today = date.today()
            use_cache = target_date < today

            df = self.flat_client._download_file(s3_key, use_cache=use_cache)
            if df.empty:
                logger.warning(f"No data in {s3_key}")
                return self._record_sync(sync_type, target_date, s3_key, 0,
                                         time.time() - start_time, 'no_data')

            # Standardize columns (override date->timestamp for minute data)
            col_map = dict(self.COLUMN_MAP)
            col_map['window_start'] = 'timestamp'
            df = df.rename(columns=col_map)

            # Parse timestamp — may be nanosecond epoch or ISO string
            if 'timestamp' in df.columns:
                raw = df['timestamp']
                if pd.api.types.is_numeric_dtype(raw):
                    df['timestamp'] = pd.to_datetime(raw, unit='ns')
                else:
                    df['timestamp'] = pd.to_datetime(raw)

            # Add optional columns if missing
            for col in self.MINUTE_COLUMNS_OPTIONAL:
                if col not in df.columns:
                    df[col] = None

            # Select columns present in the table
            use_cols = self.MINUTE_COLUMNS + [c for c in self.MINUTE_COLUMNS_OPTIONAL if c in df.columns]
            df = df[use_cols].dropna(subset=['symbol'])

            rows = self._bulk_load_minute(df)
            duration = time.time() - start_time

            logger.info(f"Synced {rows} minute bars for {target_date} in {duration:.1f}s")
            return self._record_sync(sync_type, target_date, s3_key, rows,
                                     duration, 'success')

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Failed to sync minute bars for {target_date}: {e}")
            return self._record_sync(sync_type, target_date, s3_key, 0,
                                     duration, 'error', str(e))

    def backfill_daily(self, start: date, end: date) -> List[Dict[str, Any]]:
        """
        Backfill daily bars for a date range.

        Downloads per-day flat files and loads each into PostgreSQL.

        Args:
            start: Start date (inclusive)
            end: End date (inclusive)

        Returns:
            List of sync result dicts
        """
        results = []
        current = start

        while current <= end:
            # Skip weekends
            if current.weekday() < 5:
                result = self.sync_daily(current)
                results.append(result)

            current += timedelta(days=1)

        return results

    def backfill_minute(self, start: date, end: date) -> List[Dict[str, Any]]:
        """
        Backfill minute bars for a date range.

        Args:
            start: Start date (inclusive)
            end: End date (inclusive)

        Returns:
            List of sync result dicts
        """
        results = []
        current = start

        while current <= end:
            # Skip weekends
            if current.weekday() < 5:
                result = self.sync_minute(current)
                results.append(result)

            current += timedelta(days=1)

        return results

    def _bulk_load_daily(self, df: pd.DataFrame) -> int:
        """Bulk load daily bars using COPY + upsert staging table."""
        columns = [c for c in df.columns if c in
                   self.DAILY_COLUMNS + self.DAILY_COLUMNS_OPTIONAL]
        return self._bulk_load(df, 'daily_bars', columns,
                               conflict_cols=['symbol', 'date'])

    def _bulk_load_minute(self, df: pd.DataFrame) -> int:
        """Bulk load minute bars using COPY + upsert staging table."""
        columns = [c for c in df.columns if c in
                   self.MINUTE_COLUMNS + self.MINUTE_COLUMNS_OPTIONAL]
        return self._bulk_load(df, 'minute_bars', columns,
                               conflict_cols=['symbol', 'timestamp'])

    def _bulk_load(self, df: pd.DataFrame, table: str, columns: List[str],
                   conflict_cols: List[str]) -> int:
        """
        Bulk load a DataFrame into a table using PostgreSQL COPY via staging table.

        1. CREATE TEMP TABLE _staging (LIKE target)
        2. COPY into staging from CSV buffer
        3. INSERT INTO target SELECT FROM staging ON CONFLICT DO UPDATE
        """
        if df.empty:
            return 0

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            try:
                # Create staging table (structure only, no constraints/indexes)
                cursor.execute(f"CREATE TEMP TABLE _staging (LIKE {table})")

                # Write DataFrame to CSV buffer
                buf = io.StringIO()
                df[columns].to_csv(buf, index=False, header=True)
                buf.seek(0)

                # COPY into staging
                cursor.copy_expert(
                    f"COPY _staging ({', '.join(columns)}) FROM STDIN WITH CSV HEADER",
                    buf,
                )

                # Upsert from staging into target
                update_cols = [c for c in columns if c not in conflict_cols]
                update_set = ', '.join(f"{c} = EXCLUDED.{c}" for c in update_cols)
                conflict_key = ', '.join(conflict_cols)

                cursor.execute(f"""
                    INSERT INTO {table} ({', '.join(columns)})
                    SELECT {', '.join(columns)} FROM _staging
                    ON CONFLICT ({conflict_key}) DO UPDATE SET {update_set}
                """)

                rows = cursor.rowcount
                conn.commit()
                return rows

            except Exception:
                conn.rollback()
                raise
            finally:
                cursor.execute("DROP TABLE IF EXISTS _staging")
                conn.commit()

    def _already_synced(self, sync_type: str, sync_date: date) -> bool:
        """Check if a date has already been successfully synced."""
        result = self.db.execute(
            "SELECT id FROM sync_log WHERE sync_type = %s AND sync_date = %s AND status = %s",
            (sync_type, sync_date, 'success'),
            fetch='one',
        )
        return result is not None

    def _record_sync(self, sync_type: str, sync_date: date, file_key: str,
                     rows_loaded: int, duration: float, status: str,
                     error_message: Optional[str] = None) -> Dict[str, Any]:
        """Record sync attempt in sync_log."""
        try:
            self.db.execute(
                """INSERT INTO sync_log (sync_type, sync_date, file_key, rows_loaded,
                   duration_seconds, status, error_message, completed_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (sync_type, sync_date) DO UPDATE SET
                   file_key = EXCLUDED.file_key,
                   rows_loaded = EXCLUDED.rows_loaded,
                   duration_seconds = EXCLUDED.duration_seconds,
                   status = EXCLUDED.status,
                   error_message = EXCLUDED.error_message,
                   completed_at = EXCLUDED.completed_at""",
                (sync_type, sync_date, file_key, rows_loaded,
                 round(duration, 2), status, error_message, datetime.now()),
            )
        except Exception as e:
            logger.error(f"Failed to record sync log: {e}")

        return {
            'date': str(sync_date),
            'rows_loaded': rows_loaded,
            'duration_seconds': round(duration, 2),
            'status': status,
            'error': error_message,
        }

    def get_sync_status(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get recent sync log entries."""
        cutoff = date.today() - timedelta(days=days)
        rows = self.db.execute(
            """SELECT sync_type, sync_date, file_key, rows_loaded,
                      duration_seconds, status, error_message, completed_at
               FROM sync_log
               WHERE sync_date >= %s
               ORDER BY sync_date DESC, sync_type""",
            (cutoff,),
            fetch='all',
        )
        return [dict(r) for r in rows] if rows else []
