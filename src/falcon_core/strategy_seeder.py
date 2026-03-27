#!/usr/bin/env python3
"""
Strategy Seeder & Rotation Manager

Seeds all registered strategies into the strategy_roster table and manages
the rotation lifecycle: backtest -> paper_trading -> review -> backtest.

Usage:
    falcon-strategy-seed                        # Seed all strategies into roster
    falcon-strategy-seed --backtest             # Seed + run initial backtests
    falcon-strategy-seed --status               # Show roster with metrics
    falcon-strategy-seed --promote <name>       # Promote to paper_trading
    falcon-strategy-seed --demote <name> --reason "..."  # Demote to review
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Standard falcon.env locations
_ENV_FILE_PATHS = [
    '/etc/falcon/falcon.env',
    os.path.expanduser('~/.config/falcon/falcon.env'),
]


def _load_falcon_env():
    """Load falcon.env into os.environ if not already configured."""
    # Skip if DATABASE_URL is already set (env is already configured)
    if os.getenv('DATABASE_URL'):
        return

    for env_path in _ENV_FILE_PATHS:
        if os.path.exists(env_path):
            try:
                from dotenv import load_dotenv
                load_dotenv(env_path, override=False)
                logger.info(f"Loaded environment from {env_path}")
                return
            except ImportError:
                # Fallback: parse manually
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        if '=' in line:
                            key, _, value = line.partition('=')
                            key = key.strip()
                            value = value.strip()
                            if not os.getenv(key):
                                os.environ[key] = value
                logger.info(f"Loaded environment from {env_path} (manual parse)")
                return

    logger.warning("No falcon.env found. Set DATABASE_URL or create /etc/falcon/falcon.env")


def get_db():
    """Get database manager instance."""
    _load_falcon_env()
    from falcon_core.db_manager import get_db_manager
    db = get_db_manager()
    db.init_schema()
    return db


def seed_strategies(db) -> List[str]:
    """Seed all registered strategies into the strategy_roster table.

    Returns list of newly seeded strategy names.
    """
    from falcon_core.backtesting.strategies import get_available_strategies

    strategies = get_available_strategies(db=db)
    seeded = []

    for name, cls in strategies.items():
        strategy = cls()
        params = strategy.params

        # Determine default symbols and interval from scheduler defaults
        default_config = _get_default_config(name)
        symbols = json.dumps(default_config.get('symbols', []))
        interval = default_config.get('interval', getattr(params, 'recommended_interval', '5m'))

        # Check if already exists
        existing = db.execute(
            'SELECT id FROM strategy_roster WHERE strategy_name = %s',
            (name,), fetch='one'
        )

        if existing:
            logger.info(f"  {name}: already in roster (skipped)")
            continue

        # Serialize params
        params_json = json.dumps(params.to_dict())
        now = datetime.now().isoformat()

        db.execute(
            '''INSERT INTO strategy_roster
               (strategy_name, status, symbols, interval, params, created_at, updated_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s)''',
            (name, 'backtest', symbols, interval, params_json, now, now)
        )

        # Log the seeding
        db.execute(
            '''INSERT INTO strategy_rotation_log
               (strategy_name, from_status, to_status, reason, rotated_at)
               VALUES (%s, %s, %s, %s, %s)''',
            (name, None, 'backtest', 'Initial seed', now)
        )

        seeded.append(name)
        logger.info(f"  {name}: seeded (status=backtest)")

    return seeded


def _get_default_config(strategy_name: str) -> Dict[str, Any]:
    """Get default scheduler config for a strategy."""
    defaults = {
        'one_candle_rule': {'symbols': ['SPY', 'QQQ', 'AMD'], 'interval': '1m'},
        'atr_breakout': {'symbols': ['AMD', 'NVDA', 'TSLA', 'META', 'AAPL'], 'interval': '5m'},
        'market_memory': {'symbols': ['AMD', 'NVDA', 'TSLA'], 'interval': '5m'},
        'vwap_bounce': {'symbols': ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA'], 'interval': '5m'},
        'opening_range_breakout': {'symbols': ['AMD', 'TSLA', 'NVDA', 'META'], 'interval': '5m'},
        'red_to_green': {'symbols': ['AMD', 'NVDA', 'TSLA'], 'interval': '1m'},
        'volatility_squeeze': {'symbols': ['SPY', 'QQQ', 'IWM', 'AAPL'], 'interval': '5m'},
        'microstructure_momentum': {'symbols': ['SPY', 'QQQ', 'AMD'], 'interval': '1m'},
        'gap_fill_fade': {'symbols': ['TSLA', 'NVDA', 'AMD', 'META'], 'interval': '5m'},
    }
    return defaults.get(strategy_name, {'symbols': [], 'interval': '5m'})


def run_backtests(db) -> Dict[str, Any]:
    """Run backtests for all strategies in roster with status='backtest'.

    Returns dict of strategy_name -> metrics.
    """
    from falcon_core.backtesting.strategies import get_available_strategies
    from datetime import timedelta, date as date_type

    try:
        from falcon_core.backtesting.data_feed import DataFeed
        from falcon_core.backtesting.engine import SimpleBacktestEngine
        from falcon_core.backtesting.results_api import BacktestResultsStore, BacktestRunRecord
    except ImportError as e:
        logger.error(f"Backtesting dependencies not available: {e}")
        logger.error("Install with: pip install falcon-core[backtesting]")
        return {}

    strategies = get_available_strategies(db=db)
    feed = DataFeed(db_manager=db)
    results_store = BacktestResultsStore(db_manager=db)
    results = {}

    # Default date range: last 30 trading days (~6 weeks calendar)
    # Each day downloads a full minute-aggs file from flat files, so keep initial seed manageable
    end_date = date_type.today() - timedelta(days=1)  # Yesterday (data lag)
    start_date = end_date - timedelta(days=42)  # ~30 trading days

    # Get all strategies in backtest status
    rows = db.execute(
        'SELECT strategy_name, symbols, interval, params FROM strategy_roster WHERE status = %s',
        ('backtest',), fetch='all'
    )

    if not rows:
        logger.info("No strategies in 'backtest' status to run")
        return results

    for row in rows:
        name = row['strategy_name'] if isinstance(row, dict) else row[0]
        symbols_raw = row['symbols'] if isinstance(row, dict) else row[1]
        interval = row['interval'] if isinstance(row, dict) else row[2]

        if isinstance(symbols_raw, str):
            symbols = json.loads(symbols_raw)
        else:
            symbols = symbols_raw or []

        if name not in strategies:
            logger.warning(f"  {name}: strategy class not found, skipping")
            continue

        StrategyClass = strategies[name]
        logger.info(f"  {name}: backtesting on {symbols} ({interval})...")

        all_returns = []
        all_win_rates = []
        all_sharpes = []
        total_trades = 0

        for symbol in symbols:
            try:
                data = feed.get_historical_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval,
                    market_hours_only=True,
                )

                if data is None or data.empty:
                    logger.warning(f"    {symbol}: no data available")
                    continue

                strategy = StrategyClass()
                engine = SimpleBacktestEngine(initial_capital=25000)
                result = engine.run(strategy, data, symbol=symbol)

                all_returns.append(result.total_return)
                all_win_rates.append(result.win_rate)
                if hasattr(result, 'sharpe_ratio') and result.sharpe_ratio is not None:
                    all_sharpes.append(result.sharpe_ratio)
                total_trades += result.total_trades

                logger.info(
                    f"    {symbol}: {result.total_return:.2%} return, "
                    f"{result.total_trades} trades, {result.win_rate:.1%} win rate"
                )

                # Store per-symbol result in backtest_runs table
                try:
                    record = BacktestRunRecord(
                        strategy_name=name,
                        symbol=symbol,
                        trading_date=end_date,
                        interval=interval,
                        total_return=result.total_return,
                        max_drawdown=getattr(result, 'max_drawdown', 0.0) or 0.0,
                        sharpe_ratio=getattr(result, 'sharpe_ratio', 0.0) or 0.0,
                        win_rate=result.win_rate,
                        total_trades=result.total_trades,
                        signals_count=getattr(result, 'signals_count', 0) or 0,
                        parameters=strategy.params.to_dict() if hasattr(strategy.params, 'to_dict') else None,
                    )
                    results_store.store_backtest_run(record)
                except Exception as store_err:
                    logger.warning(f"    {symbol}: failed to store backtest result: {store_err}")

            except Exception as e:
                logger.error(f"    {symbol}: backtest failed: {e}")

        # Aggregate metrics
        if all_returns:
            avg_return = float(sum(all_returns) / len(all_returns))
            avg_win_rate = float(sum(all_win_rates) / len(all_win_rates))
            avg_sharpe = float(sum(all_sharpes) / len(all_sharpes)) if all_sharpes else 0.0
            # Profit factor approximation from win rate and R/R
            profit_factor = float((avg_win_rate * 2) / (1 - avg_win_rate)) if avg_win_rate < 1 else 10.0

            now = datetime.now().isoformat()
            db.execute(
                '''UPDATE strategy_roster SET
                   last_backtest_at = %s,
                   backtest_sharpe = %s,
                   backtest_win_rate = %s,
                   backtest_profit_factor = %s,
                   backtest_total_return = %s,
                   updated_at = %s
                   WHERE strategy_name = %s''',
                (now, avg_sharpe, avg_win_rate, profit_factor, avg_return, now, name)
            )

            results[name] = {
                'avg_return': avg_return,
                'avg_win_rate': avg_win_rate,
                'avg_sharpe': avg_sharpe,
                'profit_factor': profit_factor,
                'total_trades': total_trades,
                'symbols_tested': len(all_returns),
            }

            logger.info(
                f"  {name}: avg return {avg_return:.2%}, "
                f"win rate {avg_win_rate:.1%}, sharpe {avg_sharpe:.2f}"
            )
        else:
            logger.warning(f"  {name}: no successful backtests")
            results[name] = {'error': 'No data available'}

    return results


def show_status(db):
    """Display the strategy roster with metrics."""
    rows = db.execute(
        '''SELECT strategy_name, status, backtest_sharpe, backtest_win_rate,
                  backtest_profit_factor, backtest_total_return,
                  paper_sharpe, paper_win_rate, paper_profit_factor,
                  last_backtest_at, promoted_at, demoted_at, review_notes
           FROM strategy_roster ORDER BY strategy_name''',
        fetch='all'
    )

    if not rows:
        print("No strategies in roster. Run 'falcon-strategy-seed' to seed.")
        return

    # Header
    print(f"\n{'Strategy':<30} {'Status':<15} {'Sharpe':>8} {'Win%':>8} {'PF':>8} {'Return':>10}")
    print("-" * 85)

    for row in rows:
        if isinstance(row, dict):
            name = row['strategy_name']
            status = row['status']
            sharpe = row['backtest_sharpe']
            win_rate = row['backtest_win_rate']
            pf = row['backtest_profit_factor']
            ret = row['backtest_total_return']
        else:
            name, status, sharpe, win_rate, pf, ret = row[0], row[1], row[2], row[3], row[4], row[5]

        sharpe_str = f"{sharpe:.2f}" if sharpe is not None else "---"
        wr_str = f"{win_rate:.1%}" if win_rate is not None else "---"
        pf_str = f"{pf:.2f}" if pf is not None else "---"
        ret_str = f"{ret:.2%}" if ret is not None else "---"

        print(f"{name:<30} {status:<15} {sharpe_str:>8} {wr_str:>8} {pf_str:>8} {ret_str:>10}")

    print()


def promote_strategy(db, strategy_name: str):
    """Promote a strategy from backtest to paper_trading."""
    row = db.execute(
        'SELECT status FROM strategy_roster WHERE strategy_name = %s',
        (strategy_name,), fetch='one'
    )

    if not row:
        print(f"Strategy '{strategy_name}' not found in roster.")
        sys.exit(1)

    current_status = row['status'] if isinstance(row, dict) else row[0]

    if current_status == 'paper_trading':
        print(f"Strategy '{strategy_name}' is already in paper_trading.")
        return

    if current_status not in ('backtest', 'review'):
        print(f"Cannot promote from status '{current_status}'. Must be 'backtest' or 'review'.")
        sys.exit(1)

    now = datetime.now().isoformat()

    db.execute(
        '''UPDATE strategy_roster SET
           status = %s, promoted_at = %s, updated_at = %s
           WHERE strategy_name = %s''',
        ('paper_trading', now, now, strategy_name)
    )

    db.execute(
        '''INSERT INTO strategy_rotation_log
           (strategy_name, from_status, to_status, reason, rotated_at)
           VALUES (%s, %s, %s, %s, %s)''',
        (strategy_name, current_status, 'paper_trading',
         f'Promoted from {current_status}', now)
    )

    print(f"Promoted '{strategy_name}': {current_status} -> paper_trading")


def demote_strategy(db, strategy_name: str, reason: str = ""):
    """Demote a strategy to review status."""
    row = db.execute(
        'SELECT status FROM strategy_roster WHERE strategy_name = %s',
        (strategy_name,), fetch='one'
    )

    if not row:
        print(f"Strategy '{strategy_name}' not found in roster.")
        sys.exit(1)

    current_status = row['status'] if isinstance(row, dict) else row[0]

    if current_status == 'review':
        print(f"Strategy '{strategy_name}' is already in review.")
        return

    now = datetime.now().isoformat()

    db.execute(
        '''UPDATE strategy_roster SET
           status = %s, demoted_at = %s, review_notes = %s, updated_at = %s
           WHERE strategy_name = %s''',
        ('review', now, reason, now, strategy_name)
    )

    db.execute(
        '''INSERT INTO strategy_rotation_log
           (strategy_name, from_status, to_status, reason, rotated_at)
           VALUES (%s, %s, %s, %s, %s)''',
        (strategy_name, current_status, 'review', reason or 'Demoted for review', now)
    )

    print(f"Demoted '{strategy_name}': {current_status} -> review (reason: {reason or 'N/A'})")


# ── Promotion / Demotion Gates ──────────────────────────────────────
#
# Lifecycle:  backtest → paper_trading → live → (demote) → review → backtest
#
# Promotion thresholds are intentionally conservative — false promotions
# cost real money, false demotions just delay by one cycle.

PROMOTION_GATES = {
    # backtest → paper_trading
    'to_paper': {
        'min_sharpe': 0.0,           # Must be non-negative
        'min_win_rate': 0.20,        # At least 20% win rate
        'min_trades': 5,             # At least 5 trades in backtest
        'max_days_since_backtest': 7,  # Backtested in last 7 days
    },
    # paper_trading → live  (future — requires paper trading metrics)
    'to_live': {
        'min_paper_sharpe': 0.5,
        'min_paper_win_rate': 0.30,
        'min_paper_trades': 10,
    },
}

DEMOTION_RULES = {
    'zero_trade_cycles': 3,          # Retire after N backtest cycles with 0 trades
    'negative_sharpe_cycles': 5,     # Demote after N cycles with negative Sharpe
    'max_review_days': 30,           # Retire if stuck in review for this long
}


def auto_rotate(db, dry_run: bool = False) -> Dict[str, Any]:
    """
    Apply automated promotion and demotion gates to all strategies.

    Returns dict with rotation actions taken.
    """
    from datetime import timedelta

    now = datetime.now()
    now_str = now.isoformat()
    actions = {'promoted': [], 'demoted': [], 'retired': [], 'unchanged': []}

    rows = db.execute(
        '''SELECT strategy_name, status, backtest_sharpe, backtest_win_rate,
                  backtest_total_return, backtest_profit_factor,
                  paper_sharpe, paper_win_rate, paper_profit_factor,
                  last_backtest_at, promoted_at, demoted_at
           FROM strategy_roster ORDER BY strategy_name''',
        fetch='all',
    )

    if not rows:
        return actions

    # Count consecutive zero-trade cycles per strategy from backtest_runs
    zero_trade_counts = {}
    cycle_rows = db.execute(
        '''SELECT strategy_name,
                  count(*) FILTER (WHERE total_trades = 0) as zero_cycles,
                  count(*) as total_cycles
           FROM (
               SELECT strategy_name, sum(total_trades) as total_trades, trading_date
               FROM backtest_runs
               GROUP BY strategy_name, trading_date
               ORDER BY trading_date DESC
           ) sub
           GROUP BY strategy_name''',
        fetch='all',
    )
    for cr in (cycle_rows or []):
        name = cr['strategy_name'] if isinstance(cr, dict) else cr[0]
        zero = cr['zero_cycles'] if isinstance(cr, dict) else cr[1]
        total = cr['total_cycles'] if isinstance(cr, dict) else cr[2]
        zero_trade_counts[name] = {'zero': zero, 'total': total}

    for row in rows:
        name = row['strategy_name']
        status = row['status']
        sharpe = float(row['backtest_sharpe'] or 0)
        win_rate = float(row['backtest_win_rate'] or 0)
        total_return = float(row['backtest_total_return'] or 0)
        last_bt = row['last_backtest_at']
        paper_sharpe = float(row['paper_sharpe'] or 0) if row.get('paper_sharpe') else None

        # ── Backtest → Paper Trading ──
        if status == 'backtest':
            gates = PROMOTION_GATES['to_paper']

            # Check freshness
            if last_bt:
                days_since = (now - last_bt).days if isinstance(last_bt, datetime) else 999
            else:
                days_since = 999

            # Check zero-trade retirement
            ztc = zero_trade_counts.get(name, {})
            if ztc.get('total', 0) >= DEMOTION_RULES['zero_trade_cycles'] and ztc.get('zero', 0) == ztc.get('total', 0):
                reason = f"Retired: {ztc['zero']}/{ztc['total']} backtest cycles produced zero trades"
                if not dry_run:
                    _rotate(db, name, 'backtest', 'retired', reason, now_str)
                actions['retired'].append({'name': name, 'reason': reason})
                logger.info(f"  {name}: RETIRED — {reason}")
                continue

            # Check promotion
            # Need backtest_runs data to count trades
            trade_count_row = db.execute(
                '''SELECT COALESCE(sum(total_trades), 0) as trades
                   FROM backtest_runs
                   WHERE strategy_name = %s
                     AND created_at > %s''',
                (name, (now - timedelta(days=gates['max_days_since_backtest'])).isoformat()),
                fetch='one',
            )
            recent_trades = int(trade_count_row['trades']) if trade_count_row else 0

            if (sharpe >= gates['min_sharpe'] and
                win_rate >= gates['min_win_rate'] and
                recent_trades >= gates['min_trades'] and
                days_since <= gates['max_days_since_backtest']):

                reason = (
                    f"Auto-promoted: sharpe={sharpe:+.2f}, wr={win_rate:.1%}, "
                    f"{recent_trades} trades in last {gates['max_days_since_backtest']}d"
                )
                if not dry_run:
                    _rotate(db, name, 'backtest', 'paper_trading', reason, now_str)
                actions['promoted'].append({'name': name, 'reason': reason})
                logger.info(f"  {name}: PROMOTED to paper_trading — {reason}")
            else:
                actions['unchanged'].append({
                    'name': name, 'status': status,
                    'note': f"sharpe={sharpe:+.2f} wr={win_rate:.1%} trades={recent_trades} days={days_since}"
                })

        # ── Paper Trading → Live (future) ──
        elif status == 'paper_trading':
            # For now, just check if paper metrics exist and are positive
            if paper_sharpe is not None:
                gates = PROMOTION_GATES['to_live']
                if (paper_sharpe >= gates['min_paper_sharpe'] and
                    float(row.get('paper_win_rate') or 0) >= gates['min_paper_win_rate']):
                    reason = f"Ready for live: paper_sharpe={paper_sharpe:+.2f}"
                    actions['promoted'].append({'name': name, 'reason': reason, 'to': 'live'})
                    logger.info(f"  {name}: READY for live — {reason}")
                    # Don't auto-promote to live — that requires human approval
            actions['unchanged'].append({'name': name, 'status': status})

        # ── Review → Backtest or Retire ──
        elif status == 'review':
            demoted_at = row.get('demoted_at')
            if demoted_at and isinstance(demoted_at, datetime):
                days_in_review = (now - demoted_at).days
                if days_in_review > DEMOTION_RULES['max_review_days']:
                    reason = f"Retired: in review for {days_in_review} days (max {DEMOTION_RULES['max_review_days']})"
                    if not dry_run:
                        _rotate(db, name, 'review', 'retired', reason, now_str)
                    actions['retired'].append({'name': name, 'reason': reason})
                    logger.info(f"  {name}: RETIRED — {reason}")
                    continue
            actions['unchanged'].append({'name': name, 'status': status})

        # ── Retired — no action ──
        elif status == 'retired':
            actions['unchanged'].append({'name': name, 'status': status})

    # Store rotation report in agent_memory
    _store_rotation_report(db, actions, now_str)

    return actions


def _rotate(db, strategy_name: str, from_status: str, to_status: str,
            reason: str, timestamp: str):
    """Apply a status rotation."""
    update_fields = {'status': to_status, 'updated_at': timestamp}

    if to_status == 'paper_trading':
        update_fields['promoted_at'] = timestamp
    elif to_status in ('review', 'retired'):
        update_fields['demoted_at'] = timestamp
        update_fields['review_notes'] = reason

    set_clause = ', '.join(f'{k} = %s' for k in update_fields)
    values = list(update_fields.values()) + [strategy_name]

    db.execute(
        f'UPDATE strategy_roster SET {set_clause} WHERE strategy_name = %s',
        tuple(values),
    )

    db.execute(
        '''INSERT INTO strategy_rotation_log
           (strategy_name, from_status, to_status, reason, rotated_at)
           VALUES (%s, %s, %s, %s, %s)''',
        (strategy_name, from_status, to_status, reason, timestamp),
    )


def _store_rotation_report(db, actions: dict, timestamp: str):
    """Store rotation actions in agent_memory for RAG."""
    import json as _json
    promoted = actions.get('promoted', [])
    demoted = actions.get('demoted', [])
    retired = actions.get('retired', [])

    if not promoted and not demoted and not retired:
        return  # Nothing interesting to record

    lines = []
    for a in promoted:
        lines.append(f"PROMOTED {a['name']}: {a['reason']}")
    for a in demoted:
        lines.append(f"DEMOTED {a['name']}: {a['reason']}")
    for a in retired:
        lines.append(f"RETIRED {a['name']}: {a['reason']}")

    content = f"Strategy rotation: {len(promoted)} promoted, {len(demoted)} demoted, {len(retired)} retired\n\n" + "\n".join(lines)

    try:
        db.execute(
            """INSERT INTO agent_memory
               (agent_name, category, content, metadata, created_at, updated_at)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            ('strategy-rotation', 'rotation-report', content,
             _json.dumps(actions, default=str), timestamp, timestamp),
        )
    except Exception as e:
        logger.warning(f"Could not store rotation report: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Falcon Strategy Seeder & Rotation Manager',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  falcon-strategy-seed                          Seed all strategies into roster
  falcon-strategy-seed --backtest               Seed + run backtests
  falcon-strategy-seed --status                 Show roster with metrics
  falcon-strategy-seed --promote vwap_bounce    Promote to paper trading
  falcon-strategy-seed --demote vwap_bounce --reason "Low win rate"
        """
    )
    parser.add_argument('--backtest', action='store_true',
                        help='Seed strategies and run initial backtests')
    parser.add_argument('--rotate', action='store_true',
                        help='Apply automated promotion/demotion gates')
    parser.add_argument('--rotate-dry-run', action='store_true',
                        help='Show what rotation would do without applying')
    parser.add_argument('--status', action='store_true',
                        help='Show strategy roster with metrics')
    parser.add_argument('--promote', metavar='STRATEGY',
                        help='Promote strategy to paper_trading')
    parser.add_argument('--demote', metavar='STRATEGY',
                        help='Demote strategy to review')
    parser.add_argument('--reason', default='',
                        help='Reason for demotion (used with --demote)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    db = get_db()

    try:
        if args.status:
            show_status(db)
        elif args.rotate or args.rotate_dry_run:
            print("Running automated rotation gates...")
            actions = auto_rotate(db, dry_run=args.rotate_dry_run)
            mode = " (DRY RUN)" if args.rotate_dry_run else ""
            print(f"\nRotation{mode}:")
            print(f"  Promoted: {len(actions['promoted'])}")
            print(f"  Demoted:  {len(actions.get('demoted', []))}")
            print(f"  Retired:  {len(actions['retired'])}")
            for a in actions['promoted']:
                print(f"  [+] {a['name']}: {a['reason']}")
            for a in actions.get('demoted', []):
                print(f"  [-] {a['name']}: {a['reason']}")
            for a in actions['retired']:
                print(f"  [x] {a['name']}: {a['reason']}")
            if not actions['promoted'] and not actions.get('demoted') and not actions['retired']:
                print("  No rotations needed.")
            print()
            show_status(db)
        elif args.promote:
            promote_strategy(db, args.promote)
        elif args.demote:
            demote_strategy(db, args.demote, args.reason)
        else:
            # Default: seed strategies
            print("Seeding strategies into roster...")
            seeded = seed_strategies(db)
            if seeded:
                print(f"\nSeeded {len(seeded)} new strategies: {', '.join(seeded)}")
            else:
                print("\nAll strategies already in roster.")

            if args.backtest:
                print("\nRunning initial backtests...")
                results = run_backtests(db)
                if results:
                    print(f"\nBacktest complete for {len(results)} strategies.")
                else:
                    print("\nNo backtests completed (check data availability).")

                # Auto-rotate after backtests
                print("\nApplying rotation gates...")
                actions = auto_rotate(db)
                promoted = actions.get('promoted', [])
                retired = actions.get('retired', [])
                if promoted or retired:
                    for a in promoted:
                        print(f"  [+] {a['name']}: {a['reason']}")
                    for a in retired:
                        print(f"  [x] {a['name']}: {a['reason']}")
                else:
                    print("  No rotations needed.")

                show_status(db)

            # Always show status at end
            if not args.backtest:
                show_status(db)
    finally:
        db.close()


if __name__ == '__main__':
    main()
