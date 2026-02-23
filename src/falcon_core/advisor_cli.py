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


def cmd_run(args):
    """Analyze strategies and create proposals."""
    from falcon_core.backtesting.advisor import StrategyAdvisor

    db = _get_db()
    advisor = StrategyAdvisor(db, model=args.model)

    # Get strategies to analyze
    if args.strategy:
        rows = db.execute(
            '''SELECT strategy_name, strategy_code, backtest_sharpe,
                      backtest_win_rate, backtest_total_return
               FROM strategy_roster
               WHERE strategy_name = %s AND strategy_code IS NOT NULL''',
            (args.strategy,), fetch='all'
        )
    else:
        rows = db.execute(
            '''SELECT strategy_name, strategy_code, backtest_sharpe,
                      backtest_win_rate, backtest_total_return
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
        sharpe = row['backtest_sharpe'] if isinstance(row, dict) else row[2]
        win_rate = row['backtest_win_rate'] if isinstance(row, dict) else row[3]
        total_return = row['backtest_total_return'] if isinstance(row, dict) else row[4]

        if not code:
            continue

        backtest_results = {
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'total_return': total_return,
        }

        print(f"\nAnalyzing: {name}...")
        proposal = advisor.analyze_and_propose(
            strategy_name=name,
            strategy_code=code,
            backtest_results=backtest_results,
        )

        if proposal:
            print(f"  Proposal created (cost: ${proposal['api_cost_usd']:.4f})")
            print(f"  Type: {proposal['proposal_type']}")
            print(f"  Change: {proposal['change_description'][:80]}...")
            proposals_created += 1
        else:
            print(f"  No proposal generated (budget or validation issue)")

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
