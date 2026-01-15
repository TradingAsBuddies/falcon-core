#!/usr/bin/env python3
"""
Falcon Backtesting CLI

Command-line interface for running backtests, optimizing parameters,
and managing the feedback loop.

Usage:
    falcon-backtest run <strategy> <symbol> [--start=DATE] [--end=DATE]
    falcon-backtest optimize <strategy> <symbol> [--metric=sharpe_ratio]
    falcon-backtest list-strategies
    falcon-backtest feedback <strategy> [--days=30]
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_strategy_class(name: str):
    """Get strategy class by name"""
    from falcon_core.backtesting.strategies import get_available_strategies
    strategies = get_available_strategies()

    if name not in strategies:
        available = ', '.join(strategies.keys())
        raise ValueError(f"Unknown strategy: {name}. Available: {available}")

    return strategies[name]


def cmd_list_strategies(args):
    """List available strategies"""
    from falcon_core.backtesting.strategies import get_available_strategies

    strategies = get_available_strategies()

    print("\nAvailable Strategies:")
    print("=" * 60)

    for name, cls in strategies.items():
        print(f"\n{name}")
        print(f"  Description: {cls.description}")
        print(f"  Style: {cls.trading_style}")
        print(f"  Source: {cls.source_creator or 'N/A'}")

        # Show optimizable parameters
        params = cls.param_ranges()
        if params:
            print(f"  Optimizable params: {', '.join(params.keys())}")

    print()


def cmd_run(args):
    """Run a backtest"""
    from falcon_core.backtesting.engine import create_engine
    from falcon_core.backtesting.data_feed import DataFeed

    strategy_class = get_strategy_class(args.strategy)

    # Parse dates
    end_date = datetime.strptime(args.end, '%Y-%m-%d') if args.end else datetime.now()
    start_date = datetime.strptime(args.start, '%Y-%m-%d') if args.start else end_date - timedelta(days=365)

    print(f"\nRunning backtest: {strategy_class.name}")
    print(f"Symbol: {args.symbol}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print("=" * 60)

    # Load data
    data_feed = DataFeed()
    try:
        data = data_feed.get_historical_data(
            args.symbol,
            start_date,
            end_date,
            interval=args.interval,
        )
        print(f"Loaded {len(data)} bars")
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1

    # Create strategy and engine
    strategy = strategy_class()
    engine = create_engine(
        engine_type=args.engine,
        initial_capital=args.capital,
    )

    # Run backtest
    result = engine.run(strategy, data, args.symbol)

    # Print results
    print(result.summary())

    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"Results saved to: {args.output}")

    return 0


def cmd_optimize(args):
    """Optimize strategy parameters"""
    from falcon_core.backtesting.optimizer import ParameterOptimizer
    from falcon_core.backtesting.data_feed import DataFeed

    strategy_class = get_strategy_class(args.strategy)

    # Parse dates
    end_date = datetime.strptime(args.end, '%Y-%m-%d') if args.end else datetime.now()
    start_date = datetime.strptime(args.start, '%Y-%m-%d') if args.start else end_date - timedelta(days=365)

    print(f"\nOptimizing: {strategy_class.name}")
    print(f"Symbol: {args.symbol}")
    print(f"Metric: {args.metric}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print("=" * 60)

    # Load data
    data_feed = DataFeed()
    try:
        data = data_feed.get_historical_data(
            args.symbol,
            start_date,
            end_date,
            interval=args.interval,
        )
        print(f"Loaded {len(data)} bars")
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1

    # Run optimization
    optimizer = ParameterOptimizer()
    result = optimizer.optimize(
        strategy_class,
        data,
        args.symbol,
        metric=args.metric,
        max_combinations=args.max_combos,
        walk_forward=not args.no_walk_forward,
    )

    # Print results
    print(result.summary())

    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"Results saved to: {args.output}")

    return 0


def cmd_feedback(args):
    """Analyze live performance and generate feedback"""
    from falcon_core.backtesting.optimizer import ParameterOptimizer, FeedbackLoop

    print(f"\nAnalyzing feedback for: {args.strategy}")
    print(f"Days: {args.days}")
    print("=" * 60)

    # This requires database connection
    try:
        from falcon_core import get_db_manager
        db = get_db_manager()
    except Exception as e:
        print(f"Database not configured: {e}")
        print("Feedback loop requires database connection.")
        return 1

    optimizer = ParameterOptimizer(db_manager=db)
    feedback = FeedbackLoop(optimizer, db_manager=db)

    # Run analysis
    results = feedback.run_daily_analysis(args.strategy)

    print(f"\nLive Performance ({args.days} days):")
    perf = results.get('live_performance', {})
    if perf:
        print(f"  Trades: {perf.get('trades', 0)}")
        print(f"  Win Rate: {perf.get('win_rate', 0):.2%}")
        print(f"  Profit Factor: {perf.get('profit_factor', 0):.2f}")
        print(f"  Total P/L: ${perf.get('total_pnl', 0):.2f}")
    else:
        print("  No performance data available")

    if results.get('actions'):
        print("\nActions:")
        for action in results['actions']:
            print(f"  - {action}")

    if results.get('recommendation'):
        print(f"\nRecommendation: {results['recommendation']}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Falcon Backtesting CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # list-strategies command
    list_parser = subparsers.add_parser('list-strategies', help='List available strategies')
    list_parser.set_defaults(func=cmd_list_strategies)

    # run command
    run_parser = subparsers.add_parser('run', help='Run a backtest')
    run_parser.add_argument('strategy', help='Strategy name')
    run_parser.add_argument('symbol', help='Stock symbol')
    run_parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    run_parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    run_parser.add_argument('--interval', default='1d', help='Data interval (1d, 5m, etc)')
    run_parser.add_argument('--engine', default='auto', choices=['auto', 'simple', 'bt'])
    run_parser.add_argument('--capital', type=float, default=100000.0, help='Initial capital')
    run_parser.add_argument('--output', '-o', help='Output file for results (JSON)')
    run_parser.set_defaults(func=cmd_run)

    # optimize command
    opt_parser = subparsers.add_parser('optimize', help='Optimize strategy parameters')
    opt_parser.add_argument('strategy', help='Strategy name')
    opt_parser.add_argument('symbol', help='Stock symbol')
    opt_parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    opt_parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    opt_parser.add_argument('--interval', default='1d', help='Data interval')
    opt_parser.add_argument('--metric', default='sharpe_ratio',
                           choices=['sharpe_ratio', 'total_return', 'profit_factor', 'win_rate'])
    opt_parser.add_argument('--max-combos', type=int, default=500, help='Max combinations to test')
    opt_parser.add_argument('--no-walk-forward', action='store_true', help='Skip walk-forward validation')
    opt_parser.add_argument('--output', '-o', help='Output file for results (JSON)')
    opt_parser.set_defaults(func=cmd_optimize)

    # feedback command
    fb_parser = subparsers.add_parser('feedback', help='Analyze live performance feedback')
    fb_parser.add_argument('strategy', help='Strategy name')
    fb_parser.add_argument('--days', type=int, default=30, help='Days to analyze')
    fb_parser.set_defaults(func=cmd_feedback)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
