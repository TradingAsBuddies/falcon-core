#!/usr/bin/env python3
"""
Falcon AI Advisor CLI

Manages AI-powered strategy analysis, proposals, and budgets.

Usage:
    falcon-advisor run                     # Analyze all eligible strategies
    falcon-advisor run --strategy NAME     # Analyze one strategy
    falcon-advisor budget                  # Show budget status
    falcon-advisor reset-monthly           # Reset monthly counters (cron)
    falcon-advisor proposals               # List pending proposals
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

logger = logging.getLogger(__name__)

# Standard falcon.env locations
_ENV_FILE_PATHS = [
    '/etc/falcon/falcon.env',
    os.path.expanduser('~/.config/falcon/falcon.env'),
]


def _load_falcon_env():
    """Load falcon.env into os.environ if not already configured."""
    if os.getenv('DATABASE_URL'):
        return
    for env_path in _ENV_FILE_PATHS:
        if os.path.exists(env_path):
            try:
                from dotenv import load_dotenv
                load_dotenv(env_path, override=False)
            except ImportError:
                pass
            return


def _get_db():
    """Get database manager instance."""
    _load_falcon_env()
    from falcon_core.db_manager import get_db_manager
    db = get_db_manager()
    db.init_schema()
    return db


def _coerce(value, default=None):
    """Coerce DB Decimal/None to float, leaving non-numerics alone."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _build_backtest_results(store, name):
    """Build a rich backtest_results dict from the populated backtest_runs table.

    Pulls aggregates from get_strategy_summary(days=90) and per-symbol/per-day
    detail from get_recent_backtests(days=90). Returns None if there is no
    backtest_runs data for this strategy (caller skips it — never a stub).
    """
    summary = store.get_strategy_summary(name, days=90) or {}
    recent = store.get_recent_backtests(strategy_name=name, days=90, limit=500) or []

    total_trades = summary.get('total_trades')
    if (not summary or total_trades in (None, 0)) and not recent:
        return None

    # Per-symbol rollup from recent rows
    per_symbol_map = {}
    n_signals_total = 0
    intervals = set()
    trading_dates = []
    for r in recent:
        sym = r.get('symbol')
        nt = int(r.get('total_trades') or 0)
        sc = int(r.get('signals_count') or 0)
        wr = _coerce(r.get('win_rate'))
        tr = _coerce(r.get('total_return'))
        n_signals_total += sc
        if r.get('interval'):
            intervals.add(r.get('interval'))
        if r.get('trading_date'):
            trading_dates.append(str(r.get('trading_date')))
        acc = per_symbol_map.setdefault(
            sym, {'symbol': sym, 'n_trades': 0, 'signals_count': 0,
                  '_wr_sum': 0.0, '_wr_n': 0, '_ret_sum': 0.0, '_ret_n': 0}
        )
        acc['n_trades'] += nt
        acc['signals_count'] += sc
        if wr is not None:
            acc['_wr_sum'] += wr
            acc['_wr_n'] += 1
        if tr is not None:
            acc['_ret_sum'] += tr
            acc['_ret_n'] += 1

    per_symbol = []
    for sym, acc in per_symbol_map.items():
        per_symbol.append({
            'symbol': sym,
            'n_trades': acc['n_trades'],
            'signals_count': acc['signals_count'],
            'win_rate': (acc['_wr_sum'] / acc['_wr_n']) if acc['_wr_n'] else None,
            'total_return': (acc['_ret_sum'] / acc['_ret_n']) if acc['_ret_n'] else None,
        })

    if total_trades in (None, 0):
        total_trades = sum(p['n_trades'] for p in per_symbol)

    avg_return = _coerce(summary.get('avg_return'))
    win_rate = _coerce(summary.get('avg_win_rate'))
    expectancy = (avg_return / total_trades) if (total_trades and avg_return is not None) else None
    total_R = (avg_return * len(per_symbol)) if avg_return is not None else None

    date_span = None
    if trading_dates:
        date_span = {'start': min(trading_dates), 'end': max(trading_dates)}

    return {
        'n_trades': total_trades,
        'n_signals': n_signals_total,
        'win_rate': win_rate,
        'mean_R': expectancy,
        'expectancy': expectancy,
        'total_return': avg_return,
        'total_R': total_R,
        'max_drawdown': _coerce(summary.get('avg_drawdown')),
        'sharpe_ratio': _coerce(summary.get('avg_sharpe')),
        'interval': sorted(intervals)[0] if intervals else None,
        'symbols_tested': summary.get('symbols_tested') or len(per_symbol),
        'date_span': date_span,
        'per_symbol': per_symbol,
        'data_source': 'flatfiles',
    }


def cmd_run(args):
    """Analyze strategies and create proposals."""
    from falcon_core.backtesting.advisor import StrategyAdvisor
    from falcon_core.backtesting.results_api import BacktestResultsStore

    db = _get_db()
    advisor = StrategyAdvisor(db, model=args.model)
    store = BacktestResultsStore(db_manager=db)

    # Roster is used ONLY to fetch strategy code (metrics come from backtest_runs).
    if args.strategy:
        rows = db.execute(
            '''SELECT strategy_name, strategy_code, symbols, interval
               FROM strategy_roster
               WHERE strategy_name = %s AND strategy_code IS NOT NULL''',
            (args.strategy,), fetch='all'
        )
    else:
        rows = db.execute(
            '''SELECT strategy_name, strategy_code, symbols, interval
               FROM strategy_roster
               WHERE strategy_code IS NOT NULL''',
            fetch='all'
        )

    if not rows:
        print("No strategies with code found. Run falcon-migrate-strategies --apply first.")
        return

    proposals_created = 0

    for row in rows:
        name = row['strategy_name'] if isinstance(row, dict) else row[0]
        code = row['strategy_code'] if isinstance(row, dict) else row[1]

        if not code:
            continue

        # Pull REAL metrics from the populated backtest_runs table.
        backtest_results = _build_backtest_results(store, name)
        if backtest_results is None:
            print(f"\n{name}: skipped: no backtest_runs data")
            continue

        print(f"\nAnalyzing: {name} "
              f"(trades={backtest_results['n_trades']}, "
              f"signals={backtest_results['n_signals']}, "
              f"win_rate={backtest_results['win_rate']})...")
        proposal = advisor.analyze_and_propose(
            strategy_name=name,
            strategy_code=code,
            backtest_results=backtest_results,
        )

        if not proposal:
            print(f"  No proposal generated (budget, data-sufficiency, or validation gate)")
            continue

        print(f"  Proposal created (cost: ${proposal['api_cost_usd']:.4f})")
        print(f"  Type: {proposal['proposal_type']}")
        print(f"  Change: {proposal['change_description'][:80]}...")
        proposals_created += 1

        # Close the loop: verify the proposal via flat-file backtest and record
        # whether it actually beat the current code (drives budget retirement).
        cmp = advisor.backtest_proposal(proposal['id'])
        if cmp:
            cur = cmp.get('current', {}) or {}
            prop = cmp.get('proposed', {}) or {}
            improved = (
                prop.get('total_return') is not None
                and cur.get('total_return') is not None
                and prop['total_return'] > cur['total_return']
            )
            advisor.cost_tracker.record_improvement(name, improved)

            def _fmt(v):
                return f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"

            print(f"  Verified (flatfiles): "
                  f"sharpe {_fmt(cur.get('sharpe'))}->{_fmt(prop.get('sharpe'))}  "
                  f"win_rate {_fmt(cur.get('win_rate'))}->{_fmt(prop.get('win_rate'))}  "
                  f"return {_fmt(cur.get('total_return'))}->{_fmt(prop.get('total_return'))}  "
                  f"=> {'IMPROVED' if improved else 'no improvement'}")
        else:
            print("  Verification skipped (could not load/backtest proposal)")

    print(f"\n{proposals_created} proposals created")
    db.close()


def cmd_budget(args):
    """Show budget status for all strategies."""
    from falcon_core.backtesting.advisor import CostTracker

    db = _get_db()
    tracker = CostTracker(db)
    budgets = tracker.get_all_budgets()

    if not budgets:
        print("No advisor budgets configured yet.")
        return

    print(f"\n{'Strategy':<30} {'Status':<10} {'Month $':>10} {'Budget $':>10} {'Total $':>10} {'No-Imp':>8}")
    print("-" * 85)

    for b in budgets:
        if isinstance(b, dict):
            name = b['strategy_name']
            status = b['status']
            month_spent = float(b.get('current_month_spent_usd', 0))
            budget = float(b.get('monthly_budget_usd', 1.0))
            total = float(b.get('total_spent_usd', 0))
            no_imp = int(b.get('consecutive_no_improvement', 0))
        else:
            name, status = b[0], b[8]
            month_spent = float(b[3] or 0)
            budget = float(b[1] or 1.0)
            total = float(b[2] or 0)
            no_imp = int(b[6] or 0)

        print(
            f"{name:<30} {status:<10} "
            f"${month_spent:>9.4f} ${budget:>9.2f} ${total:>9.4f} {no_imp:>8}"
        )

    print()
    db.close()


def cmd_reset_monthly(args):
    """Reset monthly budget counters."""
    from falcon_core.backtesting.advisor import CostTracker

    db = _get_db()
    tracker = CostTracker(db)
    tracker.reset_monthly_budgets()
    print("Monthly budgets reset for all active strategies")
    db.close()


def cmd_proposals(args):
    """List proposals."""
    db = _get_db()

    status_filter = args.status or 'pending'
    rows = db.execute(
        '''SELECT id, strategy_name, proposal_type, change_description,
                  status, api_cost_usd, created_at,
                  current_sharpe, proposed_sharpe,
                  current_win_rate, proposed_win_rate
           FROM strategy_proposals
           WHERE status = %s
           ORDER BY created_at DESC''',
        (status_filter,), fetch='all'
    )

    if not rows:
        print(f"No {status_filter} proposals.")
        return

    print(f"\n{status_filter.upper()} Proposals:")
    print(f"{'ID':>4} {'Strategy':<25} {'Type':<14} {'Change':<40} {'Cost':>8}")
    print("-" * 95)

    for row in rows:
        if isinstance(row, dict):
            pid = row['id']
            name = row['strategy_name']
            ptype = row['proposal_type']
            change = row['change_description'] or ''
            cost = float(row.get('api_cost_usd', 0) or 0)
        else:
            pid, name, ptype, change = row[0], row[1], row[2], row[3] or ''
            cost = float(row[5] or 0)

        change_short = change[:38] + '..' if len(change) > 40 else change
        print(f"{pid:>4} {name:<25} {ptype:<14} {change_short:<40} ${cost:>7.4f}")

    print()
    db.close()


def main():
    parser = argparse.ArgumentParser(
        description='Falcon AI Strategy Advisor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  falcon-advisor run                     Analyze all eligible strategies
  falcon-advisor run --strategy NAME     Analyze one strategy
  falcon-advisor budget                  Show budget status
  falcon-advisor reset-monthly           Reset monthly counters
  falcon-advisor proposals               List pending proposals
  falcon-advisor proposals --status all  List all proposals
        """
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # run command
    run_parser = subparsers.add_parser('run', help='Analyze and propose improvements')
    run_parser.add_argument('--strategy', help='Analyze specific strategy')
    run_parser.add_argument('--model', default=None, help='Override Claude model')
    run_parser.set_defaults(func=cmd_run)

    # budget command
    budget_parser = subparsers.add_parser('budget', help='Show budget status')
    budget_parser.set_defaults(func=cmd_budget)

    # reset-monthly command
    reset_parser = subparsers.add_parser('reset-monthly', help='Reset monthly budgets')
    reset_parser.set_defaults(func=cmd_reset_monthly)

    # proposals command
    prop_parser = subparsers.add_parser('proposals', help='List proposals')
    prop_parser.add_argument('--status', default='pending',
                             help='Filter by status (pending/approved/rejected/all)')
    prop_parser.set_defaults(func=cmd_proposals)

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if getattr(args, 'verbose', False) else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == '__main__':
    sys.exit(main() or 0)
