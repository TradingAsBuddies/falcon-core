"""Sentinel: Backtest engine — can load and execute a strategy end-to-end."""

import logging
from datetime import datetime, timedelta
from falcon_core.sentinel.base import BaseSentinel, SentinelResult, SentinelStatus

logger = logging.getLogger(__name__)

PROBE_SYMBOL = "SPY"


class BacktestEngineSentinel(BaseSentinel):

    name = "backtest-engine"
    description = "Load a strategy from DB and run a backtest against flat file data"

    def check(self) -> SentinelResult:
        # 1. Get a strategy from the roster
        try:
            from falcon_core import get_db_manager
            db = get_db_manager()
            row = db.execute(
                "SELECT strategy_name, strategy_code FROM strategy_roster "
                "WHERE strategy_code IS NOT NULL AND status = 'backtest' "
                "ORDER BY id LIMIT 1",
                fetch="one",
            )
        except Exception as e:
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.FAIL,
                reason=f"Cannot query strategy_roster: {e}",
            )

        if not row:
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.WARN,
                reason="No strategies with code in roster (status=backtest)",
            )

        strategy_name = row["strategy_name"]

        # 2. Load the strategy from code
        try:
            from falcon_core.backtesting.strategy_loader import load_strategy_from_code
            strategy_cls = load_strategy_from_code(row["strategy_code"], strategy_name)
            if strategy_cls is None:
                return SentinelResult(
                    name=self.name,
                    status=SentinelStatus.FAIL,
                    reason=f"Strategy '{strategy_name}' code parsed but produced no strategy class",
                )
        except Exception as e:
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.FAIL,
                reason=f"Failed to load strategy '{strategy_name}': {e}",
            )

        # 3. Fetch data from flat files
        try:
            from falcon_core.backtesting.data_feed import DataFeed
            feed = DataFeed()
            # Use a 5-day window from a recent period
            end_date = datetime.now() - timedelta(days=3)
            while end_date.weekday() >= 5:
                end_date -= timedelta(days=1)
            start_date = end_date - timedelta(days=7)

            interval = "1d"
            data = feed.get_historical_data(
                PROBE_SYMBOL,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                interval=interval,
            )
        except Exception as e:
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.FAIL,
                reason=f"Data fetch failed for probe: {e}",
            )

        if data.empty:
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.WARN,
                reason="Probe data returned empty (holiday week?)",
            )

        # 4. Run the backtest engine
        try:
            from falcon_core.backtesting.engine import create_engine
            engine = create_engine()
            strategy = strategy_cls()
            result = engine.run(strategy, data, PROBE_SYMBOL)
        except Exception as e:
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.FAIL,
                reason=f"Backtest engine crashed: {e}",
                details={"strategy": strategy_name},
            )

        # 5. Report
        engine_type = type(engine).__name__
        total_return = getattr(result, "total_return", None)
        total_trades = getattr(result, "total_trades", None)

        return SentinelResult(
            name=self.name,
            status=SentinelStatus.PASS,
            reason=f"Engine={engine_type}, strategy='{strategy_name}', "
                   f"bars={len(data)}, trades={total_trades}, return={total_return}",
            details={
                "engine": engine_type,
                "strategy": strategy_name,
                "bars": len(data),
                "total_trades": total_trades,
                "total_return": float(total_return) if total_return is not None else None,
            },
        )
