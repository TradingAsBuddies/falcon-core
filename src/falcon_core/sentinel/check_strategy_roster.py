"""Sentinel: Strategy roster — code integrity and parameter validation."""

import logging
from falcon_core.sentinel.base import BaseSentinel, SentinelResult, SentinelStatus

logger = logging.getLogger(__name__)


class StrategyRosterSentinel(BaseSentinel):

    name = "strategy-roster"
    description = "Verify all rostered strategies have valid code and parseable params"

    def check(self) -> SentinelResult:
        try:
            from falcon_core import get_db_manager
            db = get_db_manager()
            rows = db.execute(
                "SELECT id, strategy_name, status, strategy_code, params "
                "FROM strategy_roster ORDER BY id",
                fetch="all",
            )
        except Exception as e:
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.FAIL,
                reason=f"Cannot query strategy_roster: {e}",
            )

        if not rows:
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.WARN,
                reason="Strategy roster is empty",
            )

        try:
            from falcon_core.backtesting.strategy_loader import (
                load_strategy_from_code,
                validate_strategy_code,
            )
        except ImportError as e:
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.FAIL,
                reason=f"Cannot import strategy_loader: {e}",
            )

        total = len(rows)
        issues = []

        for row in rows:
            name = row["strategy_name"]
            code = row["strategy_code"]
            params = row["params"]

            # Check code exists
            if not code or not code.strip():
                issues.append(f"{name}: no strategy_code")
                continue

            # AST validation
            is_valid, reason = validate_strategy_code(code)
            if not is_valid:
                issues.append(f"{name}: code validation failed — {reason}")
                continue

            # Try loading
            try:
                strategy_cls = load_strategy_from_code(code, name)
                if strategy_cls is None:
                    issues.append(f"{name}: code parsed but no BaseStrategy subclass found")
            except Exception as e:
                issues.append(f"{name}: load error — {e}")

            # Check params is a dict (JSONB should deserialize)
            if params is not None and not isinstance(params, dict):
                issues.append(f"{name}: params is not a dict (got {type(params).__name__})")

        if issues:
            return SentinelResult(
                name=self.name,
                status=SentinelStatus.WARN,
                reason=f"{len(issues)} issue(s) in {total} strategies",
                details={"total": total, "issues": issues},
            )

        return SentinelResult(
            name=self.name,
            status=SentinelStatus.PASS,
            reason=f"All {total} strategies have valid code and params",
            details={"total": total},
        )
