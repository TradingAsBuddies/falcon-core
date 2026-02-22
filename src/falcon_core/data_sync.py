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
    COLUMN_MAP = {
        'ticker': 'symbol',
        'window_start': 'date',
        'o': 'open',
        'h': 'high',
        'l': 'low',
        'c': 'close',
        'v': 'volume',
        'vw': 'vwap',
        'n': 'trades',
    }

    DAILY_COLUMNS = ['symbol', 'date', 'open', 'high', 'low', 'close',
                     'volume', 'vwap', 'trades']
    MINUTE_COLUMNS = ['symbol', 'timestamp', 'open', 'high', 'low', 'close',
                      'volume', 'vwap', 'trades']

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

        Downloads the monthly file containing target_date, filters to that date,
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
        s3_key = self.flat_client._get_day_aggs_key(target_date.year, target_date.month)

        try:
            # Current month's file may still be accumulating — skip cache
            today = date.today()
            use_cache = not (target_date.year == today.year and target_date.month == today.month)

            df = self.flat_client._download_file(s3_key, use_cache=use_cache)
            if df.empty:
                logger.warning(f"No data in {s3_key}")
                return self._record_sync(sync_type, target_date, s3_key, 0,
                                         time.time() - start_time, 'no_data')

            # Standardize columns
            df = df.rename(columns=self.COLUMN_MAP)

            # Parse date and filter to target
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.date
            df = df[df['date'] == target_date]

            if df.empty:
                logger.info(f"No trading data for {target_date} (weekend/holiday?)")
                return self._record_sync(sync_type, target_date, s3_key, 0,
                                         time.time() - start_time, 'no_data')

            # Keep only needed columns, drop rows with missing symbol
            df = df[self.DAILY_COLUMNS].dropna(subset=['symbol'])

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

            # Standardize columns
            col_map = dict(self.COLUMN_MAP)
            col_map['window_start'] = 'timestamp'
            df = df.rename(columns=col_map)

            # Parse timestamp
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Keep only needed columns
            df = df[self.MINUTE_COLUMNS].dropna(subset=['symbol'])

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

        Iterates through each trading day in the range, downloading month files
        as needed and loading all dates from each file.

        Args:
            start: Start date (inclusive)
            end: End date (inclusive)

        Returns:
            List of sync result dicts
        """
        results = []
        # Process month by month for efficiency
        current_month = date(start.year, start.month, 1)
        end_month = date(end.year, end.month, 1)

        while current_month <= end_month:
            month_results = self._backfill_daily_month(current_month, start, end)
            results.extend(month_results)

            # Next month
            if current_month.month == 12:
                current_month = date(current_month.year + 1, 1, 1)
            else:
                current_month = date(current_month.year, current_month.month + 1, 1)

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

    def _backfill_daily_month(self, month_start: date, range_start: date,
                              range_end: date) -> List[Dict[str, Any]]:
        """Backfill all daily bars from a single month file within the given range."""
        start_time = time.time()
        s3_key = self.flat_client._get_day_aggs_key(month_start.year, month_start.month)

        try:
            today = date.today()
            use_cache = not (month_start.year == today.year and month_start.month == today.month)

            df = self.flat_client._download_file(s3_key, use_cache=use_cache)
            if df.empty:
                logger.warning(f"No data in {s3_key}")
                return []

            df = df.rename(columns=self.COLUMN_MAP)

            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.date

            # Filter to requested range
            df = df[(df['date'] >= range_start) & (df['date'] <= range_end)]
            if df.empty:
                return []

            df = df[self.DAILY_COLUMNS].dropna(subset=['symbol'])

            # Group by date and load each day
            results = []
            for day, day_df in df.groupby('date'):
                if self._already_synced('daily', day):
                    results.append({'date': str(day), 'rows_loaded': 0,
                                    'duration_seconds': 0, 'status': 'skipped'})
                    continue

                day_start = time.time()
                rows = self._bulk_load_daily(day_df)
                duration = time.time() - day_start

                result = self._record_sync('daily', day, s3_key, rows,
                                           duration, 'success')
                results.append(result)
                logger.info(f"Backfilled {rows} daily bars for {day}")

            return results

        except Exception as e:
            logger.error(f"Failed to backfill month {month_start}: {e}")
            return [{'date': str(month_start), 'rows_loaded': 0,
                     'duration_seconds': time.time() - start_time,
                     'status': 'error', 'error': str(e)}]

    def _bulk_load_daily(self, df: pd.DataFrame) -> int:
        """Bulk load daily bars using COPY + upsert staging table."""
        return self._bulk_load(df, 'daily_bars', self.DAILY_COLUMNS,
                               conflict_cols=['symbol', 'date'])

    def _bulk_load_minute(self, df: pd.DataFrame) -> int:
        """Bulk load minute bars using COPY + upsert staging table."""
        return self._bulk_load(df, 'minute_bars', self.MINUTE_COLUMNS,
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
                # Create staging table
                cursor.execute(f"CREATE TEMP TABLE _staging (LIKE {table} INCLUDING NOTHING)")

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
