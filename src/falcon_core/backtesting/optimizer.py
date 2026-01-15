"""
Parameter Optimizer with Feedback Loop

Optimizes strategy parameters using:
1. Grid search / random search over parameter space
2. Walk-forward analysis for robustness
3. Feedback from live trading performance

The optimizer connects backtest results to the trading bot configuration,
creating a continuous improvement loop.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import numpy as np
import pandas as pd

from falcon_core.backtesting.engine import BacktestEngine, BacktestResult, create_engine
from falcon_core.backtesting.strategies.base import BaseStrategy, StrategyParams
from falcon_core.backtesting.data_feed import DataFeed

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Results from parameter optimization"""
    strategy_name: str
    best_params: Dict[str, Any]
    best_score: float
    metric_optimized: str

    # All results
    all_results: List[Dict] = field(default_factory=list)

    # Walk-forward results
    in_sample_score: float = 0.0
    out_of_sample_score: float = 0.0
    robustness_ratio: float = 0.0  # OOS/IS ratio

    # Optimization metadata
    param_ranges: Dict[str, Tuple] = field(default_factory=dict)
    total_combinations: int = 0
    combinations_tested: int = 0
    optimization_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "metric_optimized": self.metric_optimized,
            "in_sample_score": self.in_sample_score,
            "out_of_sample_score": self.out_of_sample_score,
            "robustness_ratio": self.robustness_ratio,
            "total_combinations": self.total_combinations,
            "combinations_tested": self.combinations_tested,
        }

    def summary(self) -> str:
        return f"""
Optimization Results: {self.strategy_name}
{'=' * 50}
Metric: {self.metric_optimized}
Best Score: {self.best_score:.4f}

Best Parameters:
{json.dumps(self.best_params, indent=2)}

Walk-Forward Analysis:
  In-Sample: {self.in_sample_score:.4f}
  Out-of-Sample: {self.out_of_sample_score:.4f}
  Robustness: {self.robustness_ratio:.2%}

Tested {self.combinations_tested}/{self.total_combinations} combinations
"""


class ParameterOptimizer:
    """
    Optimize strategy parameters with feedback loop integration.

    Supports:
    - Grid search optimization
    - Walk-forward validation
    - Live performance feedback
    - Automatic parameter adjustment
    """

    # Scoring metrics and their optimization direction
    METRICS = {
        'sharpe_ratio': 'maximize',
        'total_return': 'maximize',
        'profit_factor': 'maximize',
        'win_rate': 'maximize',
        'max_drawdown': 'minimize',
        'sortino_ratio': 'maximize',
        'expectancy': 'maximize',
    }

    def __init__(
        self,
        engine: Optional[BacktestEngine] = None,
        data_feed: Optional[DataFeed] = None,
        db_manager=None,
    ):
        self.engine = engine or create_engine()
        self.data_feed = data_feed or DataFeed(db_manager)
        self.db = db_manager

    def optimize(
        self,
        strategy_class: Type[BaseStrategy],
        data: pd.DataFrame,
        symbol: str,
        metric: str = 'sharpe_ratio',
        param_ranges: Optional[Dict[str, Tuple]] = None,
        max_combinations: int = 1000,
        walk_forward: bool = True,
        train_pct: float = 0.7,
    ) -> OptimizationResult:
        """
        Optimize strategy parameters.

        Args:
            strategy_class: Strategy class to optimize
            data: Historical data for backtesting
            symbol: Symbol being tested
            metric: Metric to optimize (sharpe_ratio, total_return, etc.)
            param_ranges: Parameter ranges to test (default: from strategy)
            max_combinations: Max parameter combinations to test
            walk_forward: Whether to do walk-forward validation
            train_pct: Percentage of data for training (if walk_forward)

        Returns:
            OptimizationResult with best parameters and scores
        """
        start_time = datetime.now()

        # Get parameter ranges
        if param_ranges is None:
            param_ranges = strategy_class.param_ranges()

        if not param_ranges:
            logger.warning(f"No parameter ranges for {strategy_class.name}")
            return self._empty_result(strategy_class.name)

        # Generate parameter combinations
        param_names = list(param_ranges.keys())
        param_values = []
        for name in param_names:
            min_val, max_val, step = param_ranges[name]
            values = np.arange(min_val, max_val + step, step)
            param_values.append(values)

        # Calculate total combinations
        total_combos = 1
        for values in param_values:
            total_combos *= len(values)

        logger.info(f"Optimizing {strategy_class.name}: {total_combos} combinations")

        # Split data for walk-forward
        if walk_forward:
            split_idx = int(len(data) * train_pct)
            train_data = data.iloc[:split_idx]
            test_data = data.iloc[split_idx:]
        else:
            train_data = data
            test_data = None

        # Run optimization
        all_results = []
        best_score = float('-inf') if self.METRICS.get(metric, 'maximize') == 'maximize' else float('inf')
        best_params = {}
        combinations_tested = 0

        # Grid search (with optional random sampling for large spaces)
        if total_combos > max_combinations:
            # Random sampling
            logger.info(f"Using random sampling ({max_combinations} of {total_combos})")
            param_combos = self._random_sample_params(param_ranges, max_combinations)
        else:
            # Full grid
            param_combos = list(product(*param_values))

        for combo in param_combos:
            # Build params dict
            params_dict = dict(zip(param_names, combo))

            # Create strategy with these params
            try:
                base_params = strategy_class.default_params()
                params = base_params.update(**params_dict)
                strategy = strategy_class(params)

                # Run backtest
                result = self.engine.run(strategy, train_data, symbol)

                # Get score
                score = getattr(result, metric, 0.0)
                if pd.isna(score):
                    score = 0.0

                all_results.append({
                    'params': params_dict,
                    'score': score,
                    'result': result,
                })

                # Check if best
                is_better = (
                    (self.METRICS.get(metric) == 'maximize' and score > best_score) or
                    (self.METRICS.get(metric) == 'minimize' and score < best_score)
                )

                if is_better:
                    best_score = score
                    best_params = params_dict.copy()

            except Exception as e:
                logger.warning(f"Failed to test params {params_dict}: {e}")

            combinations_tested += 1

            if combinations_tested % 100 == 0:
                logger.info(f"Tested {combinations_tested}/{len(param_combos)} combinations")

        # Walk-forward validation on test data
        in_sample_score = best_score
        out_of_sample_score = 0.0
        robustness_ratio = 0.0

        if walk_forward and test_data is not None and best_params:
            try:
                base_params = strategy_class.default_params()
                params = base_params.update(**best_params)
                strategy = strategy_class(params)

                test_result = self.engine.run(strategy, test_data, symbol)
                out_of_sample_score = getattr(test_result, metric, 0.0)

                if in_sample_score != 0:
                    robustness_ratio = out_of_sample_score / in_sample_score
            except Exception as e:
                logger.warning(f"Walk-forward validation failed: {e}")

        elapsed = (datetime.now() - start_time).total_seconds()

        return OptimizationResult(
            strategy_name=strategy_class.name,
            best_params=best_params,
            best_score=best_score,
            metric_optimized=metric,
            all_results=all_results,
            in_sample_score=in_sample_score,
            out_of_sample_score=out_of_sample_score,
            robustness_ratio=robustness_ratio,
            param_ranges=param_ranges,
            total_combinations=total_combos,
            combinations_tested=combinations_tested,
            optimization_time=elapsed,
        )

    def _random_sample_params(
        self,
        param_ranges: Dict[str, Tuple],
        n_samples: int,
    ) -> List[Tuple]:
        """Generate random parameter combinations"""
        samples = []
        param_names = list(param_ranges.keys())

        for _ in range(n_samples):
            combo = []
            for name in param_names:
                min_val, max_val, step = param_ranges[name]
                # Random value within range, snapped to step
                val = np.random.uniform(min_val, max_val)
                val = round(val / step) * step
                combo.append(val)
            samples.append(tuple(combo))

        return samples

    def _empty_result(self, strategy_name: str) -> OptimizationResult:
        return OptimizationResult(
            strategy_name=strategy_name,
            best_params={},
            best_score=0.0,
            metric_optimized="none",
        )

    def analyze_live_performance(
        self,
        strategy_name: str,
        days: int = 30,
    ) -> Dict[str, Any]:
        """
        Analyze live trading performance for a strategy.

        Compares actual trades to backtest expectations.
        """
        if self.db is None:
            logger.warning("Database not configured for performance analysis")
            return {}

        # Query recent trades for this strategy
        query = """
            SELECT o.*, p.name as profile_name
            FROM orders o
            LEFT JOIN profile_runs pr ON DATE(o.timestamp) = DATE(pr.run_timestamp)
            LEFT JOIN screener_profiles p ON pr.profile_id = p.id
            WHERE o.timestamp >= %s
            ORDER BY o.timestamp DESC
        """
        cutoff = datetime.now() - timedelta(days=days)
        trades = self.db.execute(query, (cutoff,), fetch='all')

        if not trades:
            return {'trades': 0, 'performance': {}}

        # Calculate live metrics
        df = pd.DataFrame(trades)

        if 'pnl' in df.columns:
            winning = df[df['pnl'] > 0]
            losing = df[df['pnl'] <= 0]

            return {
                'trades': len(df),
                'win_rate': len(winning) / len(df) if len(df) > 0 else 0,
                'avg_win': winning['pnl'].mean() if len(winning) > 0 else 0,
                'avg_loss': losing['pnl'].mean() if len(losing) > 0 else 0,
                'total_pnl': df['pnl'].sum(),
                'profit_factor': (
                    winning['pnl'].sum() / abs(losing['pnl'].sum())
                    if len(losing) > 0 and losing['pnl'].sum() != 0
                    else 0
                ),
            }

        return {'trades': len(df), 'performance': {}}

    def generate_adjustments(
        self,
        strategy_name: str,
        backtest_result: BacktestResult,
        live_performance: Dict[str, Any],
        max_adjustment: float = 0.10,
    ) -> Dict[str, float]:
        """
        Generate parameter adjustments based on live vs backtest comparison.

        Returns suggested parameter changes.
        """
        adjustments = {}

        if not live_performance or live_performance.get('trades', 0) < 10:
            logger.info("Insufficient live trades for adjustment calculation")
            return adjustments

        # Compare key metrics
        backtest_win_rate = backtest_result.win_rate
        live_win_rate = live_performance.get('win_rate', 0)

        backtest_pf = backtest_result.profit_factor
        live_pf = live_performance.get('profit_factor', 0)

        # If live performance significantly differs, suggest adjustments
        win_rate_diff = live_win_rate - backtest_win_rate
        pf_diff = live_pf - backtest_pf

        # Win rate lower than expected -> consider tighter entry criteria
        if win_rate_diff < -0.1:
            adjustments['signal_confidence_threshold'] = 0.05  # Increase required confidence
            adjustments['volume_filter_mult'] = 0.1  # Increase volume requirement

        # Profit factor lower -> consider better exit management
        if pf_diff < -0.5:
            adjustments['stop_loss_pct'] = -0.005  # Tighter stops
            adjustments['take_profit_mult'] = 0.1  # Higher targets

        # Cap adjustments
        for key in adjustments:
            adjustments[key] = max(-max_adjustment, min(max_adjustment, adjustments[key]))

        return adjustments

    def save_optimization_result(
        self,
        result: OptimizationResult,
        strategy_id: Optional[int] = None,
    ) -> int:
        """Save optimization result to database"""
        if self.db is None:
            logger.warning("Database not configured")
            return -1

        query = """
            INSERT INTO strategy_backtests
            (strategy_id, ticker, start_date, end_date, total_return,
             sharpe_ratio, max_drawdown, win_rate, total_trades,
             backtest_data, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        backtest_data = json.dumps({
            'best_params': result.best_params,
            'metric': result.metric_optimized,
            'robustness_ratio': result.robustness_ratio,
        })

        result_id = self.db.execute(
            query,
            (
                strategy_id,
                'MULTI',  # Multiple symbols
                datetime.now(),
                datetime.now(),
                result.best_score if result.metric_optimized == 'total_return' else 0,
                result.best_score if result.metric_optimized == 'sharpe_ratio' else 0,
                0,  # max_drawdown
                0,  # win_rate
                result.combinations_tested,
                backtest_data,
                datetime.now(),
            ),
            fetch='none'
        )

        logger.info(f"Saved optimization result: {result_id}")
        return result_id


class FeedbackLoop:
    """
    Continuous feedback loop between backtesting and live trading.

    Daily cycle:
    1. Collect live trading results
    2. Compare to backtest expectations
    3. Identify parameter drift
    4. Re-optimize if needed
    5. Update trading bot configuration
    """

    def __init__(
        self,
        optimizer: ParameterOptimizer,
        db_manager=None,
        config_path: str = "/etc/falcon/strategy_params.json",
    ):
        self.optimizer = optimizer
        self.db = db_manager
        self.config_path = config_path

    def run_daily_analysis(self, strategy_name: str) -> Dict[str, Any]:
        """
        Run daily performance analysis and update parameters if needed.

        Returns analysis results and any adjustments made.
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy_name,
            'actions': [],
        }

        # Get live performance
        live_perf = self.optimizer.analyze_live_performance(strategy_name, days=7)
        results['live_performance'] = live_perf

        # Compare to recent backtest
        # (would load from database in production)

        # Check for significant drift
        if live_perf.get('win_rate', 0) < 0.35:  # Example threshold
            results['actions'].append('ALERT: Win rate below threshold')
            results['recommendation'] = 're-optimize'

        if live_perf.get('profit_factor', 0) < 0.8:
            results['actions'].append('ALERT: Profit factor below threshold')
            results['recommendation'] = 're-optimize'

        return results

    def update_bot_config(
        self,
        strategy_name: str,
        new_params: Dict[str, Any],
    ) -> bool:
        """
        Update the trading bot's configuration with new parameters.

        Writes to config file that the bot reads on next run.
        """
        try:
            # Load existing config
            config = {}
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)

            # Update strategy params
            if 'strategies' not in config:
                config['strategies'] = {}

            config['strategies'][strategy_name] = {
                'params': new_params,
                'updated_at': datetime.now().isoformat(),
            }

            # Write back
            import os
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)

            logger.info(f"Updated bot config for {strategy_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to update bot config: {e}")
            return False

    def get_current_params(self, strategy_name: str) -> Dict[str, Any]:
        """Get current parameters for a strategy from config"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    return config.get('strategies', {}).get(strategy_name, {}).get('params', {})
        except Exception as e:
            logger.warning(f"Could not read config: {e}")

        return {}


# Import os at module level (needed by FeedbackLoop)
import os
