#!/usr/bin/env python3
"""
Strategy Migration Tool

One-time migration to move hardcoded .py strategy files into the database
(strategy_roster.strategy_code), enabling the plugin-based loading system.

Usage:
    falcon-migrate-strategies              # dry run — show what would be migrated
    falcon-migrate-strategies --apply      # store code in DB
    falcon-migrate-strategies --verify     # compare DB-loaded vs file-loaded classes
"""

import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Strategy files relative to the strategies package directory
STRATEGY_FILES = [
    'one_candle_rule.py',
    'atr_breakout.py',
    'market_memory.py',
    'vwap_bounce.py',
    'opening_range_breakout.py',
    'red_to_green.py',
    'volatility_squeeze.py',
    'microstructure_momentum.py',
    'gap_fill_fade.py',
]


def _get_strategies_dir() -> Path:
    """Get the path to the strategies package directory."""
    import falcon_core.backtesting.strategies as pkg
    return Path(pkg.__file__).parent


def _get_db():
    """Get database manager instance."""
    # Load falcon.env if available
    env_paths = [
        '/etc/falcon/falcon.env',
        os.path.expanduser('~/.config/falcon/falcon.env'),
    ]
    if not os.getenv('DATABASE_URL'):
        for env_path in env_paths:
            if os.path.exists(env_path):
                try:
                    from dotenv import load_dotenv
                    load_dotenv(env_path, override=False)
                except ImportError:
                    pass
                break

    from falcon_core.db_manager import get_db_manager
    db = get_db_manager()
    db.init_schema()
    return db


def dry_run():
    """Show what would be migrated without making changes."""
    strategies_dir = _get_strategies_dir()
    print("Strategy Migration — Dry Run")
    print("=" * 60)

    found = 0
    for filename in STRATEGY_FILES:
        filepath = strategies_dir / filename
        if filepath.exists():
            code = filepath.read_text()
            lines = len(code.splitlines())
            size = len(code)
            strategy_name = filepath.stem
            print(f"  {strategy_name:<35} {lines:>4} lines  {size:>6} bytes")
            found += 1
        else:
            print(f"  {filename:<35} [NOT FOUND]")

    print(f"\n{found}/{len(STRATEGY_FILES)} strategy files found")
    print("\nRun with --apply to store in database")


def apply_migration():
    """Store strategy file contents in the database."""
    from falcon_core.backtesting.strategy_loader import validate_strategy_code

    strategies_dir = _get_strategies_dir()
    db = _get_db()

    print("Strategy Migration — Apply")
    print("=" * 60)

    migrated = 0
    errors = 0

    for filename in STRATEGY_FILES:
        filepath = strategies_dir / filename
        strategy_name = filepath.stem

        if not filepath.exists():
            print(f"  {strategy_name}: [SKIP] file not found")
            continue

        code = filepath.read_text()

        # Validate the code
        is_valid, error = validate_strategy_code(code)
        if not is_valid:
            print(f"  {strategy_name}: [ERROR] validation failed: {error}")
            errors += 1
            continue

        # Check if strategy exists in roster
        row = db.execute(
            'SELECT id, strategy_code FROM strategy_roster WHERE strategy_name = %s',
            (strategy_name,), fetch='one'
        )

        if not row:
            print(f"  {strategy_name}: [SKIP] not in roster (run falcon-strategy-seed first)")
            continue

        existing_code = row['strategy_code'] if isinstance(row, dict) else row[1]
        if existing_code:
            print(f"  {strategy_name}: [SKIP] already has code in DB")
            continue

        # Store the code
        from datetime import datetime
        now = datetime.now().isoformat()
        db.execute(
            '''UPDATE strategy_roster
               SET strategy_code = %s, strategy_source = %s, updated_at = %s
               WHERE strategy_name = %s''',
            (code, 'migrated', now, strategy_name)
        )

        print(f"  {strategy_name}: [OK] migrated ({len(code)} bytes)")
        migrated += 1

    print(f"\n{migrated} strategies migrated, {errors} errors")
    db.close()


def verify_migration():
    """Verify DB-loaded strategies match file-loaded strategies."""
    from falcon_core.backtesting.strategy_loader import (
        load_strategies_from_db,
        load_strategy_from_code,
        validate_strategy_code,
    )

    strategies_dir = _get_strategies_dir()
    db = _get_db()

    print("Strategy Migration — Verify")
    print("=" * 60)

    passed = 0
    failed = 0

    # Load from DB
    db_strategies = load_strategies_from_db(db)

    for filename in STRATEGY_FILES:
        filepath = strategies_dir / filename
        strategy_name = filepath.stem

        if not filepath.exists():
            print(f"  {strategy_name}: [SKIP] file not found")
            continue

        # Load from file
        file_code = filepath.read_text()
        is_valid, _ = validate_strategy_code(file_code)
        if not is_valid:
            print(f"  {strategy_name}: [SKIP] file validation failed")
            continue

        file_cls = load_strategy_from_code(file_code, strategy_name)
        if file_cls is None:
            print(f"  {strategy_name}: [SKIP] could not load from file")
            continue

        # Check DB version
        if strategy_name not in db_strategies:
            print(f"  {strategy_name}: [FAIL] not loaded from DB")
            failed += 1
            continue

        db_cls = db_strategies[strategy_name]

        # Compare key attributes
        file_instance = file_cls()
        db_instance = db_cls()

        checks = []
        if file_instance.name != db_instance.name:
            checks.append(f"name mismatch: {file_instance.name} vs {db_instance.name}")
        if file_instance.version != db_instance.version:
            checks.append(f"version mismatch: {file_instance.version} vs {db_instance.version}")

        if checks:
            print(f"  {strategy_name}: [FAIL] {'; '.join(checks)}")
            failed += 1
        else:
            print(f"  {strategy_name}: [OK] name={file_instance.name}, version={file_instance.version}")
            passed += 1

    print(f"\n{passed} passed, {failed} failed")
    db.close()

    return 0 if failed == 0 else 1


def main():
    parser = argparse.ArgumentParser(
        description='Migrate hardcoded strategy files into the database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  falcon-migrate-strategies              Dry run — show what would be migrated
  falcon-migrate-strategies --apply      Store strategy code in database
  falcon-migrate-strategies --verify     Compare DB-loaded vs file-loaded classes
        """
    )
    parser.add_argument('--apply', action='store_true',
                        help='Apply migration (store code in DB)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify DB-loaded strategies match file-loaded')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if args.verify:
        sys.exit(verify_migration())
    elif args.apply:
        apply_migration()
    else:
        dry_run()


if __name__ == '__main__':
    main()
