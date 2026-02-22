#!/usr/bin/env python3
"""
Falcon Data Sync CLI

Syncs Polygon flat files (daily/minute bars) into PostgreSQL.

Usage:
    falcon-data-sync daily [--date YYYY-MM-DD]
    falcon-data-sync minute [--date YYYY-MM-DD]
    falcon-data-sync backfill --from YYYY-MM-DD [--to YYYY-MM-DD] [--type daily|minute|all]
    falcon-data-sync status [--days 7]
"""

import argparse
import logging
import sys
from datetime import date, datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _get_pipeline():
    """Create DataSyncPipeline from environment config."""
    from falcon_core.db_manager import get_db_manager
    from falcon_core.backtesting.flatfiles_client import FlatFilesClient

    db = get_db_manager()
    db.init_schema()
    flat_client = FlatFilesClient()

    from falcon_core.data_sync import DataSyncPipeline
    return DataSyncPipeline(db, flat_client)


def _parse_date(date_str: str) -> date:
    return datetime.strptime(date_str, '%Y-%m-%d').date()


def cmd_daily(args):
    """Sync daily bars for a single date."""
    target = _parse_date(args.date) if args.date else date.today() - timedelta(days=1)

    print(f"Syncing daily bars for {target}...")
    pipeline = _get_pipeline()
    result = pipeline.sync_daily(target)

    _print_result(result)
    return 0 if result['status'] in ('success', 'skipped', 'no_data') else 1


def cmd_minute(args):
    """Sync minute bars for a single date."""
    target = _parse_date(args.date) if args.date else date.today() - timedelta(days=1)

    print(f"Syncing minute bars for {target}...")
    pipeline = _get_pipeline()
    result = pipeline.sync_minute(target)

    _print_result(result)
    return 0 if result['status'] in ('success', 'skipped', 'no_data') else 1


def cmd_backfill(args):
    """Backfill bars for a date range."""
    start = _parse_date(args.start)
    end = _parse_date(args.end) if args.end else date.today() - timedelta(days=1)
    sync_type = args.type

    print(f"Backfilling {sync_type} bars from {start} to {end}...")
    pipeline = _get_pipeline()

    results = []
    if sync_type in ('daily', 'all'):
        print("\n--- Daily Bars ---")
        daily_results = pipeline.backfill_daily(start, end)
        results.extend(daily_results)
        _print_backfill_summary(daily_results, 'daily')

    if sync_type in ('minute', 'all'):
        print("\n--- Minute Bars ---")
        minute_results = pipeline.backfill_minute(start, end)
        results.extend(minute_results)
        _print_backfill_summary(minute_results, 'minute')

    errors = [r for r in results if r.get('status') == 'error']
    return 1 if errors else 0


def cmd_status(args):
    """Show recent sync log entries."""
    pipeline = _get_pipeline()
    entries = pipeline.get_sync_status(days=args.days)

    if not entries:
        print(f"No sync activity in the last {args.days} days.")
        return 0

    print(f"\nSync Log (last {args.days} days):")
    print(f"{'Type':<8} {'Date':<12} {'Rows':>8} {'Duration':>10} {'Status':<10}")
    print("-" * 55)

    for entry in entries:
        print(f"{entry['sync_type']:<8} {str(entry['sync_date']):<12} "
              f"{entry['rows_loaded']:>8} {entry['duration_seconds']:>8.1f}s "
              f"{entry['status']:<10}")
        if entry.get('error_message'):
            print(f"  ERROR: {entry['error_message'][:80]}")

    return 0


def _print_result(result):
    """Print a single sync result."""
    status = result['status']
    if status == 'success':
        print(f"  OK: {result['rows_loaded']} rows loaded in {result['duration_seconds']:.1f}s")
    elif status == 'skipped':
        print(f"  Skipped (already synced)")
    elif status == 'no_data':
        print(f"  No data available (weekend/holiday?)")
    else:
        print(f"  FAILED: {result.get('error', 'unknown error')}")


def _print_backfill_summary(results, sync_type):
    """Print backfill summary."""
    total_rows = sum(r.get('rows_loaded', 0) for r in results)
    success = sum(1 for r in results if r.get('status') == 'success')
    skipped = sum(1 for r in results if r.get('status') == 'skipped')
    errors = sum(1 for r in results if r.get('status') == 'error')
    no_data = sum(1 for r in results if r.get('status') == 'no_data')

    print(f"\n{sync_type.title()} backfill complete:")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Success: {success}, Skipped: {skipped}, No data: {no_data}, Errors: {errors}")

    if errors:
        print("  Failed dates:")
        for r in results:
            if r.get('status') == 'error':
                print(f"    {r['date']}: {r.get('error', 'unknown')[:60]}")


def main():
    parser = argparse.ArgumentParser(
        description='Falcon Data Sync — Polygon Flat Files to PostgreSQL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # daily
    daily_parser = subparsers.add_parser('daily', help='Sync daily bars for one date')
    daily_parser.add_argument('--date', help='Target date (YYYY-MM-DD, default: yesterday)')
    daily_parser.set_defaults(func=cmd_daily)

    # minute
    minute_parser = subparsers.add_parser('minute', help='Sync minute bars for one date')
    minute_parser.add_argument('--date', help='Target date (YYYY-MM-DD, default: yesterday)')
    minute_parser.set_defaults(func=cmd_minute)

    # backfill
    bf_parser = subparsers.add_parser('backfill', help='Backfill bars for a date range')
    bf_parser.add_argument('--from', dest='start', required=True, help='Start date (YYYY-MM-DD)')
    bf_parser.add_argument('--to', dest='end', help='End date (YYYY-MM-DD, default: yesterday)')
    bf_parser.add_argument('--type', default='daily', choices=['daily', 'minute', 'all'],
                           help='Bar type to backfill (default: daily)')
    bf_parser.set_defaults(func=cmd_backfill)

    # status
    status_parser = subparsers.add_parser('status', help='Show recent sync log')
    status_parser.add_argument('--days', type=int, default=7, help='Days to show (default: 7)')
    status_parser.set_defaults(func=cmd_status)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
